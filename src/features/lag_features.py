"""
src/features/lag_features.py

Appends lag features derived from `units_sold` to the encoded DataFrame.
This is the first of four feature-engineering modules (lag → rolling →
calendar → price).  All four modules only append columns; none drop or
rename anything from their input.

Why lag features are useful
----------------------------
A lag feature is the value of the target variable at a fixed number of time
steps in the past.  For demand forecasting:

  lag_1   — yesterday's units sold.  The single strongest short-term signal:
             a store that sold 120 units yesterday is likely to sell a similar
             number today.  Captures day-to-day continuity.

  lag_7   — same day last week.  Weekly periodicity is one of the most robust
             patterns in retail demand.  A Monday in week N looks most like the
             Monday in week N-1 for a given store-product combination.

  lag_14  — two weeks prior.  Bi-weekly cycles exist for products tied to
             paydays (many employers pay fortnightly).  Also smooths out
             one-off spikes that would otherwise dominate lag_7.

  lag_28  — four weeks (roughly one month) prior.  Captures monthly
             seasonality and replenishment-driven demand cycles.  Provides
             the model with a "same time last month" reference.

Together these four lags give the model a compressed history of demand at
multiple time scales without requiring the full time-series context.  XGBoost
treats each lag as an independent numeric feature and learns which combination
is most predictive for each store-product pair.

Why partition by (store_id, product_id)
-----------------------------------------
Demand behaviour is entity-specific.  Product P0001 at store S001 has its
own demand pattern that is independent of P0001 at S002 or P0002 at S001.
Partitioning the window by both keys ensures that `lag_1` for (S001, P0001)
on a given date looks back to the previous row of that exact store-product
series — not to a row from a different store or product.

Without the partition, lag_1 for the first date of (S001, P0002) would
incorrectly use the last row of (S001, P0001), producing a nonsensical value
and silently corrupting the feature for that row.

Why ordering by date matters
------------------------------
`F.lag(col, N)` returns the value N rows before the current row within the
window partition.  "Before" is defined by the ORDER BY clause.  Without
ordering by `date`, the rows within each partition have an arbitrary order
(determined by Spark's internal task scheduling) and `lag_1` would return a
random previous row rather than the chronologically preceding one.

Ordering by `date` guarantees the lag offset is a calendar-day offset:
  - lag_1  → 1 row back  = 1 calendar day back  (grain is daily, no gaps)
  - lag_7  → 7 rows back = 7 calendar days back
  - lag_14 → 14 rows back
  - lag_28 → 28 rows back

The dataset grain is one record per (date, store_id, product_id) with no
missing dates, so row offset = calendar day offset holds exactly.

Null handling
-------------
Rows that have fewer than N prior rows in their partition (i.e., the first N
rows chronologically for each store-product) receive null for lag_N.  These
nulls are left in place here.  trainer.py handles them downstream (XGBoost
natively supports null/NaN features; the tree learns to route null-lag rows
to a separate branch).  Filling nulls with zeros or means at this stage would
introduce a false signal — zero is a valid sales count and should not be used
as a sentinel for "not enough history".

Input schema vs output schema
------------------------------
Input (from encode()) — 20 columns:

  date, store_id, product_id, category, region,
  inventory_level, units_sold, units_ordered, demand_forecast,
  is_forecast_anomaly, price, discount, weather_condition,
  is_holiday_or_promo, competitor_pricing, seasonality,
  category_encoded, region_encoded, weather_encoded, seasonality_encoded

Output — 24 columns (20 input + 4 lag columns appended):

  … all 20 input columns unchanged …
  lag_1    DoubleType   units_sold 1 day prior  for this (store_id, product_id)
  lag_7    DoubleType   units_sold 7 days prior
  lag_14   DoubleType   units_sold 14 days prior
  lag_28   DoubleType   units_sold 28 days prior

  Rows with insufficient history receive null for the corresponding lag(s).

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean
    from src.data_preparation.encoder import encode
    from src.features.lag_features import add_lag_features

    spark  = get_spark_session()
    df     = add_lag_features(encode(clean(load_raw_data(spark))))

    df.select("date", "store_id", "product_id",
              "units_sold", "lag_1", "lag_7").show(10)
"""

import logging
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window specification constants
#
# Defined at module level so they can be referenced in tests and docstrings
# without re-entering the function.  The window is shared across all lag
# columns — there is no need to create a new Window per lag offset.
# ---------------------------------------------------------------------------
_PARTITION_COLS: list[str] = ["store_id", "product_id"]
_ORDER_COL: str = "date"
_SOURCE_COL: str = "units_sold"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_lag_windows(config_path: Path) -> list[int]:
    """
    Load the `lag_windows` list from the feature_engineering section of config.yaml.

    Reading window sizes from config rather than hardcoding them means adding
    a new lag (e.g., lag_56) requires only a config change, not a code change.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml.

    Returns
    -------
    list[int]
        Ordered list of lag offsets in days, e.g. [1, 7, 14, 28].

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist.
    KeyError
        If the expected config keys are absent.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["feature_engineering"]["lag_windows"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_lag_features(
    df: DataFrame,
    config_path: str = "config/config.yaml",
) -> DataFrame:
    """
    Append lag features of `units_sold` for each window in config.lag_windows.

    A single Window specification is built once and reused for every lag
    offset.  Each lag column is appended via withColumn so the operation is
    lazy — Spark does not compute anything until an action (count, show, write)
    is triggered downstream.

    The Window:
        partitionBy("store_id", "product_id")
            Isolates each store-product time series so lags do not bleed
            across entity boundaries.
        orderBy("date")
            Establishes chronological order within each partition so that
            row offset N equals exactly N calendar days back.

    Parameters
    ----------
    df : DataFrame
        Encoded DataFrame from encode() — 20 columns.
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location.

    Returns
    -------
    DataFrame
        Input DataFrame with lag columns appended (one per configured window).
        Column names follow the pattern `lag_{N}`, e.g. lag_1, lag_7.
        All lag columns are DoubleType (cast from IntegerType units_sold) to
        be consistent with other derived numeric feature columns.
        Row count is unchanged.  Early rows in each partition receive null.

    Example
    -------
    >>> df = add_lag_features(encode(clean(load_raw_data(spark))))
    >>> df.select("store_id", "product_id", "date",
    ...           "units_sold", "lag_1", "lag_7").show(5)
    """
    lag_windows = _load_lag_windows(Path(config_path))

    logger.info(
        "Adding lag features | source_col=%s | windows=%s | "
        "partition=%s | order=%s",
        _SOURCE_COL,
        lag_windows,
        _PARTITION_COLS,
        _ORDER_COL,
    )

    # Build the window specification once.  All lag columns share the same
    # partition and ordering; only the row offset passed to F.lag() differs.
    window_spec = (
        Window
        .partitionBy(*_PARTITION_COLS)
        .orderBy(_ORDER_COL)
    )

    for lag in lag_windows:
        col_name = f"lag_{lag}"
        df = df.withColumn(
            col_name,
            # F.lag(col, offset) returns the value `offset` rows before the
            # current row within the window.  Default value is None (null),
            # which is the correct behaviour for rows with insufficient history.
            F.lag(_SOURCE_COL, lag).over(window_spec).cast(DoubleType()),
        )
        logger.debug("Appended column '%s' (lag=%d days)", col_name, lag)

    lag_col_names = [f"lag_{w}" for w in lag_windows]
    logger.info(
        "Lag features complete | columns_added=%s | total_columns=%d",
        lag_col_names,
        len(df.columns),
    )

    return df
