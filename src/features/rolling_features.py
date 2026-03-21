"""
src/features/rolling_features.py

Appends rolling mean and rolling standard deviation features derived from
`units_sold` to the lag-featured DataFrame.  This is the second of four
feature-engineering modules (lag → rolling → calendar → price).

Why rolling features are useful
---------------------------------
A rolling (moving) statistic summarises the recent history of a time series
into a single number.  Unlike a lag feature — which is a point-in-time
snapshot at an exact offset — a rolling feature aggregates over a window of
observations, which makes it more robust to single-day noise.

  Rolling mean   captures the demand level (trend) over a period.  A sudden
                 spike on one day shifts lag_1 dramatically but shifts
                 rolling_mean_7 only by 1/7 of that spike.  This stability
                 makes the mean a reliable signal of the underlying demand rate.

  Rolling std    captures demand volatility over a period.  A product with
                 rolling_std_7 ≈ 0 has very stable, predictable demand; one
                 with rolling_std_7 = 40 is erratic.  The model uses this to
                 widen or narrow its confidence implicitly: high-volatility
                 series require more conservative forecasts.  It also helps
                 the model distinguish between products whose average demand is
                 the same but whose day-to-day variability is very different.

How rolling mean differs from lag
-----------------------------------
  lag_7          The exact value of units_sold 7 days ago.  Sensitive to what
                 happened on that one specific day (e.g., a flash sale).

  rolling_mean_7 The average of units_sold over the past 7 days including
                 today.  Smooths out single-day anomalies.  If lag_7 was a
                 spike day, rolling_mean_7 dilutes it across the window.

Together they give the model complementary information:
  - "What was demand exactly N days ago?" (lag)
  - "What has demand been like recently on average?" (rolling mean)
  - "How variable has demand been recently?" (rolling std)

How rolling std helps capture demand volatility
-------------------------------------------------
Standard deviation of units_sold over a window quantifies how much daily
demand fluctuates around its mean within that window.

Business interpretation:
  - Low rolling_std_7 → stable product (e.g., daily staple grocery item).
                        The model can make a confident point forecast.
  - High rolling_std_7 → volatile product (e.g., seasonal toy, impulse item).
                          The model should be less certain; safety stock
                          calculations in recommender.py explicitly use
                          demand volatility.

For inventory planning, std dev also feeds the safety stock formula:
    safety_stock = Z × σ × √lead_time_days
Rolling std over a longer window provides the σ estimate.

Rolling window frame: rowsBetween(-(N-1), currentRow)
-------------------------------------------------------
Each window size requires its own frame specification.  The frame
`rowsBetween(-(N-1), Window.currentRow)` includes the current row and the
N-1 rows immediately before it — exactly N rows for a complete window.

`rowsBetween` (row-count based) is chosen over `rangeBetween` (value based)
because the dataset grain is uniform daily — one row per date — so row
distance and calendar distance are equivalent.  `rowsBetween` is also
unambiguous when there are ties in the ORDER BY column, though that cannot
happen here since `(date, store_id, product_id)` is the unique key.

Partial window behaviour
--------------------------
  rolling_mean_N  When fewer than N rows exist before the current row,
                  Spark computes the mean over however many rows are available
                  rather than returning null.  This is correct behaviour:
                  the first row of a series has rolling_mean_7 = units_sold
                  (mean of 1 value), which is a valid, unbiased estimate.

  rolling_std_N   Sample standard deviation requires at least 2 data points
                  (because it divides by N-1).  The first row of every
                  store-product partition returns null for all rolling_std
                  columns.  This is expected and left in place for the same
                  reason as lag nulls: filling with 0 would be misleading
                  (zero std implies perfect stability, not "unknown").

Input schema vs output schema
-------------------------------
Input (from add_lag_features()) — 24 columns:

  date, store_id, product_id, category, region,
  inventory_level, units_sold, units_ordered, demand_forecast,
  is_forecast_anomaly, price, discount, weather_condition,
  is_holiday_or_promo, competitor_pricing, seasonality,
  category_encoded, region_encoded, weather_encoded, seasonality_encoded,
  lag_1, lag_7, lag_14, lag_28

Output — 30 columns (24 input + 6 rolling feature columns appended):

  … all 24 input columns unchanged …
  rolling_mean_7   DoubleType   mean of units_sold over past 7 days  (partial window allowed)
  rolling_std_7    DoubleType   std dev of units_sold over past 7 days (null if < 2 rows)
  rolling_mean_14  DoubleType   mean over past 14 days
  rolling_std_14   DoubleType   std dev over past 14 days
  rolling_mean_28  DoubleType   mean over past 28 days
  rolling_std_28   DoubleType   std dev over past 28 days

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean
    from src.data_preparation.encoder import encode
    from src.features.lag_features import add_lag_features
    from src.features.rolling_features import add_rolling_features

    spark = get_spark_session()
    df = add_rolling_features(
             add_lag_features(
                 encode(clean(load_raw_data(spark)))))

    df.select("store_id", "product_id", "date",
              "units_sold", "rolling_mean_7", "rolling_std_7").show(10)
"""

import logging
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
#
# Identical partition and order keys as lag_features.py — both modules
# operate on the same (store_id, product_id) time-series grain ordered by
# date.  Defined at module level for import by tests.
# ---------------------------------------------------------------------------
_PARTITION_COLS: list[str] = ["store_id", "product_id"]
_ORDER_COL: str = "date"
_SOURCE_COL: str = "units_sold"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_rolling_windows(config_path: Path) -> list[int]:
    """
    Load the `rolling_windows` list from the feature_engineering config section.

    Keeping window sizes in config means adding or removing a window (e.g.,
    adding a 56-day rolling mean) requires only a config change, not a code
    change.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml.

    Returns
    -------
    list[int]
        Ordered list of window sizes in days, e.g. [7, 14, 28].

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

    return config["feature_engineering"]["rolling_windows"]


def _make_rolling_window_spec(window_size: int) -> "WindowSpec":
    """
    Build a Spark WindowSpec for a rolling window of `window_size` rows.

    The frame `rowsBetween(-(window_size - 1), Window.currentRow)` covers
    the current row plus the preceding window_size-1 rows — exactly
    window_size rows for a complete window.

    A new WindowSpec must be created per window size because the frame
    bounds differ for each N.  This is different from lag features where
    a single unbounded WindowSpec served all offsets.

    Parameters
    ----------
    window_size : int
        Number of rows (calendar days) in the rolling window.

    Returns
    -------
    pyspark.sql.window.WindowSpec
        Partitioned by (store_id, product_id), ordered by date, with a
        row-count frame of size window_size ending at the current row.
    """
    return (
        Window
        .partitionBy(*_PARTITION_COLS)
        .orderBy(_ORDER_COL)
        .rowsBetween(-(window_size - 1), Window.currentRow)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: DataFrame,
    config_path: str = "config/config.yaml",
) -> DataFrame:
    """
    Append rolling mean and rolling std columns for each configured window size.

    For each window size N in config.feature_engineering.rolling_windows, two
    columns are appended:
      rolling_mean_N  — F.avg(units_sold) over the past N rows (partial window
                        allowed; never null except when units_sold itself is null)
      rolling_std_N   — F.stddev(units_sold) over the past N rows (sample std
                        dev; null when fewer than 2 rows are in the window)

    Each window size requires a dedicated WindowSpec with its own rowsBetween
    frame.  The WindowSpec is built inside the loop for clarity; Spark's
    Catalyst optimiser plans the physical execution regardless of how many
    WindowSpec objects exist in the Python layer.

    Parameters
    ----------
    df : DataFrame
        Lag-featured DataFrame from add_lag_features() — 24 columns.
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location.

    Returns
    -------
    DataFrame
        Input DataFrame with 2 × len(rolling_windows) columns appended.
        Column names: rolling_mean_{N} and rolling_std_{N}.
        All rolling columns are DoubleType.
        Row count is unchanged.

    Example
    -------
    >>> df = add_rolling_features(add_lag_features(encode(clean(load_raw_data(spark)))))
    >>> df.select("store_id", "product_id", "date",
    ...           "units_sold", "rolling_mean_7", "rolling_std_7").show(5)
    """
    rolling_windows = _load_rolling_windows(Path(config_path))

    logger.info(
        "Adding rolling features | source_col=%s | windows=%s | "
        "partition=%s | order=%s",
        _SOURCE_COL,
        rolling_windows,
        _PARTITION_COLS,
        _ORDER_COL,
    )

    for window_size in rolling_windows:
        window_spec = _make_rolling_window_spec(window_size)

        mean_col = f"rolling_mean_{window_size}"
        std_col  = f"rolling_std_{window_size}"

        df = df.withColumn(
            mean_col,
            # F.avg returns the mean over all non-null rows in the frame.
            # Partial windows (fewer than N rows) are computed over whatever
            # rows are available — this is intentional, not a bug.
            F.avg(F.col(_SOURCE_COL)).over(window_spec),
        )
        df = df.withColumn(
            std_col,
            # F.stddev is sample standard deviation (divides by N-1).
            # Returns null when the window contains fewer than 2 rows.
            # This is the statistically correct behaviour: std dev is
            # undefined for a single observation.
            F.stddev(F.col(_SOURCE_COL)).over(window_spec),
        )

        logger.debug(
            "Appended '%s' and '%s' (window=%d rows)",
            mean_col, std_col, window_size,
        )

    new_cols = [
        name
        for w in rolling_windows
        for name in (f"rolling_mean_{w}", f"rolling_std_{w}")
    ]
    logger.info(
        "Rolling features complete | columns_added=%s | total_columns=%d",
        new_cols,
        len(df.columns),
    )

    return df
