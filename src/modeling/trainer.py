"""
src/modeling/trainer.py

Trains an XGBoost regression model to forecast `units_sold` using the
39-column feature-engineered DataFrame produced by price_features.py.
This module owns three responsibilities: selecting the training matrix,
performing a time-based train/validation split, and serialising the model
and metadata to disk.

─────────────────────────────────────────────────────────────
Feature selection: which columns are used and why
─────────────────────────────────────────────────────────────

INCLUDED (29 feature columns):

  Inventory / operations
    inventory_level      Current on-hand stock.  Low stock may constrain
                         observed sales (demand appears lower than actual);
                         the model can learn to account for this.
    units_ordered        Historical replenishment quantity.  Correlated with
                         expected future demand; the procurement team orders
                         based on their own forecast.

  Pricing — raw and derived columns are both included.  XGBoost can ignore
  redundant signals via low feature importance; including both lets the model
  decide which representation is most predictive rather than pre-deciding.
    price                Listed price level.
    discount             Discount percentage magnitude.
    competitor_pricing   Competitor's absolute price level.
    effective_price      Actual price paid (price * (1 − discount/100)).
    price_to_competitor_ratio  Competitive position (< 1 = cheaper).
    has_discount         Boolean promotional framing flag.

  Calendar / event context
    is_holiday_or_promo  Holiday or promotional day flag from the raw data.
    day_of_week          Captures weekday vs weekend demand patterns.
    month                Annual seasonality at monthly granularity.
    week_of_year         Sub-monthly seasonality (e.g. Black Friday week 47).
    quarter              Broad seasonal grouping; Q4 is the retail peak.
    day_of_month         Within-month patterns (payday, replenishment cycles).
    is_weekend           Explicit weekend signal alongside day_of_week.

  Categorical (encoded)
    category_encoded     Product category as integer (0–4).
    region_encoded       Store region as integer (0–3).
    weather_encoded      Weather condition as integer (0–3).
    seasonality_encoded  Seasonal label as integer (0–3).

  Lag features — point-in-time demand snapshots
    lag_1, lag_7, lag_14, lag_28

  Rolling features — demand level and volatility
    rolling_mean_7,  rolling_std_7
    rolling_mean_14, rolling_std_14
    rolling_mean_28, rolling_std_28

EXCLUDED and why:

  date              DateType key used to compute the split boundary; it is
                    not a numeric feature and must not be passed to the model.

  store_id          String identifier.  Entity-level signals are captured
  product_id        implicitly by the lag and rolling features, which are
                    computed per (store_id, product_id) partition.  Encoding
                    100 store-product combinations would require a separate
                    label-encoding step with high cardinality risk.

  category          Raw string; category_encoded carries the same information
  region            in a numeric form the model can use directly.
  weather_condition
  seasonality

  demand_forecast   The dataset's pre-built baseline forecast.  Excluding it
                    is essential for a fair comparison in evaluator.py: if the
                    model trained on demand_forecast it would partly predict
                    its own benchmark, making the measured improvement
                    meaningless.  demand_forecast also contains negative values
                    (data quality issue flagged by is_forecast_anomaly).

  is_forecast_anomaly  Audit flag derived from demand_forecast.  Not a demand
                    signal; carries no predictive information about units_sold.

  units_sold        The prediction target — never included as a feature.

─────────────────────────────────────────────────────────────
Why a strict time-based split is required
─────────────────────────────────────────────────────────────
A random train/test split is wrong for time-series data.

With a random split, a training row from 2021-11-15 might be evaluated
against a test row from 2021-11-10.  The model would have seen "future"
information — lag_7 for the training row is units_sold on 2021-11-08, which
the test row has not yet observed.  This is data leakage: the model appears
to generalise well but is actually memorising temporal patterns it would
never have access to at inference time.

The time-based split computes:
    cutoff_date = max(date) − holdout_days
All rows on or before cutoff_date go to train; all rows after go to validation.
This exactly mirrors the production scenario: the model is trained on all
history up to some point and evaluated on the immediately following period.

90 holdout days (≈ one quarter) is chosen because:
  - It is long enough to evaluate seasonal and promotional variation.
  - It represents a realistic business forecasting horizon.
  - It is easy to justify in interviews ("I held out the last quarter").

─────────────────────────────────────────────────────────────
Spark-to-pandas conversion
─────────────────────────────────────────────────────────────
XGBoost (via the sklearn API) operates on in-memory pandas DataFrames or
numpy arrays, not on Spark DataFrames.  The conversion happens after the
time split so that each subset is materialised separately:

  1. Apply the time filter on the Spark DataFrame (lazy, no data movement yet).
  2. Select only the feature + target columns (reduces data transferred from
     executors to the driver).
  3. Call .toPandas() — collects all rows to the driver.

At 73k total rows (≈64k train + ≈9k val), both DataFrames fit comfortably
in driver memory.  For very large datasets, one would sample or use a
distributed XGBoost variant (e.g., XGBoost with Spark/Dask), but that is
out of scope for this project.

BooleanType columns (is_holiday_or_promo, is_weekend, has_discount) become
Python bool dtype in pandas.  XGBoost's sklearn API accepts bool arrays and
converts them to float (True=1.0, False=0.0) internally.

Null handling: lag and rolling_std columns contain nulls for early rows in
each entity series.  XGBoost's tree_method="hist" treats missing values as a
special state and learns an optimal routing direction for null-feature rows
during training.  Nulls are left in place; no imputation is applied.

─────────────────────────────────────────────────────────────
Input schema vs training matrix schema
─────────────────────────────────────────────────────────────
Input (from add_price_features()) — 39 Spark columns:

  Retained as features (29):
    inventory_level, units_ordered,
    price, discount, competitor_pricing,
    effective_price, price_to_competitor_ratio, has_discount,
    is_holiday_or_promo,
    day_of_week, month, week_of_year, quarter, day_of_month, is_weekend,
    category_encoded, region_encoded, weather_encoded, seasonality_encoded,
    lag_1, lag_7, lag_14, lag_28,
    rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_std_14,
    rolling_mean_28, rolling_std_28

  Retained as target (1):
    units_sold

  Excluded — keys / strings / audit:
    date, store_id, product_id,
    category, region, weather_condition, seasonality,
    demand_forecast, is_forecast_anomaly

Training matrix (pandas):
    X_train   shape ≈ (64 100, 29)   float64 / bool columns
    y_train   shape ≈ (64 100,)      int64
    X_val     shape ≈ ( 9 000, 29)
    y_val     shape ≈ ( 9 000,)

Usage from another module
--------------------------
    from src.features.price_features import add_price_features
    from src.modeling.trainer import train

    feature_df = add_price_features(...)   # 39-column Spark DataFrame
    model      = train(feature_df)         # saves model + metadata, returns model
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training matrix definition
#
# FEATURE_COLUMNS is the single source of truth for which columns form the
# training matrix.  It is declared at module level so that predictor.py can
# import it directly — guaranteeing that inference uses exactly the same
# feature set as training.
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "units_sold"

FEATURE_COLUMNS: list[str] = [
    # ── Inventory / operations ─────────────────────────────────────────────
    "inventory_level",
    "units_ordered",
    # ── Pricing — raw ─────────────────────────────────────────────────────
    "price",
    "discount",
    "competitor_pricing",
    # ── Pricing — derived ─────────────────────────────────────────────────
    "effective_price",
    "price_to_competitor_ratio",
    "has_discount",
    # ── Calendar / event context ───────────────────────────────────────────
    "is_holiday_or_promo",
    "day_of_week",
    "month",
    "week_of_year",
    "quarter",
    "day_of_month",
    "is_weekend",
    # ── Categorical (integer-encoded) ─────────────────────────────────────
    "category_encoded",
    "region_encoded",
    "weather_encoded",
    "seasonality_encoded",
    # ── Lag features ──────────────────────────────────────────────────────
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    # ── Rolling features ──────────────────────────────────────────────────
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """
    Load and return the full config.yaml as a dict.

    trainer.py needs three sections: paths (model output), training
    (holdout_days, random_seed), and model.xgboost (hyperparameters).
    Loading the whole file at once avoids multiple file-open calls.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _compute_cutoff_date(df: DataFrame, holdout_days: int):
    """
    Compute the train/validation split boundary.

    cutoff_date = max(date) − holdout_days

    All rows on or before cutoff_date go to train; all rows after go to
    validation.  Computing the boundary from max(date) rather than from a
    fixed calendar date makes the split reproducible regardless of when the
    pipeline is run — the validation set is always the last N days of the
    dataset.

    Parameters
    ----------
    df : DataFrame
        Feature DataFrame containing a DateType column named "date".
    holdout_days : int
        Number of most-recent days to reserve for validation.

    Returns
    -------
    datetime.date
        The last date included in the training set.
    """
    max_date = df.agg(F.max("date")).collect()[0][0]   # → Python datetime.date
    return max_date - timedelta(days=holdout_days)


def _time_split(df: DataFrame, cutoff_date):
    """
    Split the DataFrame into training and validation sets by date.

    The split is strict and non-overlapping:
      train : date <= cutoff_date
      val   : date >  cutoff_date

    Both returned DataFrames are still Spark DataFrames (lazy); .toPandas()
    is called separately so that column selection can be applied first.

    Parameters
    ----------
    df : DataFrame
        Full feature DataFrame.
    cutoff_date : datetime.date
        Last date in the training set.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (train_spark_df, val_spark_df)
    """
    train_df = df.filter(F.col("date") <= F.lit(cutoff_date))
    val_df   = df.filter(F.col("date") >  F.lit(cutoff_date))
    return train_df, val_df


def _to_pandas(spark_df: DataFrame):
    """
    Select the training matrix columns and convert to a pandas DataFrame.

    Only FEATURE_COLUMNS + TARGET_COLUMN are selected before calling
    .toPandas().  This avoids transferring string/date key columns to the
    driver, reducing memory and network overhead during collection.

    BooleanType Spark columns (is_holiday_or_promo, is_weekend, has_discount)
    become Python bool dtype in pandas.  XGBoost's sklearn API accepts bool
    arrays and converts them to float internally — no explicit cast is needed.

    Null values in lag and rolling_std columns are preserved.  XGBoost's
    tree_method="hist" handles missing values natively by learning an optimal
    split direction for null-feature rows during training.

    Parameters
    ----------
    spark_df : DataFrame
        Train or validation Spark DataFrame with all 39 columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        (X, y) where X is the feature matrix and y is the target vector.
    """
    columns_to_select = FEATURE_COLUMNS + [TARGET_COLUMN]
    pdf = spark_df.select(columns_to_select).toPandas()

    X = pdf[FEATURE_COLUMNS]
    y = pdf[TARGET_COLUMN]
    return X, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(
    df: DataFrame,
    config_path: str = "config/config.yaml",
) -> XGBRegressor:
    """
    Train an XGBoost regressor on the feature-engineered DataFrame.

    Steps performed:
      1. Load config (paths, training params, XGBoost hyperparameters).
      2. Compute cutoff_date = max(date) − holdout_days.
      3. Split the Spark DataFrame into train / validation sets by date.
      4. Convert each split to a pandas (X, y) feature matrix.
      5. Instantiate and fit XGBRegressor using config hyperparameters.
      6. Save the trained model to disk via joblib.
      7. Save training metadata (cutoff_date, row counts, features, params)
         to a JSON file alongside the model.

    Parameters
    ----------
    df : DataFrame
        39-column feature DataFrame from add_price_features().
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location.

    Returns
    -------
    XGBRegressor
        The fitted model.  Also serialised to config.paths.model_dir /
        config.paths.model_filename via joblib.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If the training set is empty after the time split (which would
        indicate that holdout_days >= total days in the dataset).
    """
    cfg = _load_config(Path(config_path))

    training_cfg = cfg["training"]
    xgb_cfg      = cfg["model"]["xgboost"]
    paths_cfg    = cfg["paths"]

    holdout_days = training_cfg["holdout_days"]
    random_seed  = training_cfg["random_seed"]

    # ── 1. Compute split boundary ────────────────────────────────────────────
    cutoff_date = _compute_cutoff_date(df, holdout_days)
    logger.info(
        "Time split | holdout_days=%d | cutoff_date=%s | "
        "rows_before_split=%s (lazy)",
        holdout_days,
        cutoff_date,
        "computed after toPandas",
    )

    # ── 2. Split in Spark (still lazy) ───────────────────────────────────────
    train_spark, val_spark = _time_split(df, cutoff_date)

    # ── 3. Collect to pandas (triggers Spark execution) ──────────────────────
    logger.info("Converting train split to pandas…")
    X_train, y_train = _to_pandas(train_spark)

    logger.info("Converting validation split to pandas…")
    X_val, y_val = _to_pandas(val_spark)

    if len(X_train) == 0:
        raise ValueError(
            f"Training set is empty after time split at cutoff_date={cutoff_date}. "
            f"Check that holdout_days ({holdout_days}) is less than the total "
            "number of days in the dataset."
        )

    logger.info(
        "Split complete | train_rows=%d | val_rows=%d | features=%d",
        len(X_train),
        len(X_val),
        len(FEATURE_COLUMNS),
    )

    # ── 4. Build and fit XGBoost model ───────────────────────────────────────
    model = XGBRegressor(
        n_estimators     = xgb_cfg["n_estimators"],
        max_depth        = xgb_cfg["max_depth"],
        learning_rate    = xgb_cfg["learning_rate"],
        subsample        = xgb_cfg["subsample"],
        colsample_bytree = xgb_cfg["colsample_bytree"],
        min_child_weight = xgb_cfg["min_child_weight"],
        reg_alpha        = xgb_cfg["reg_alpha"],
        reg_lambda       = xgb_cfg["reg_lambda"],
        objective        = xgb_cfg["objective"],
        eval_metric      = xgb_cfg["eval_metric"],
        tree_method      = xgb_cfg["tree_method"],
        random_state     = random_seed,
        n_jobs           = -1,      # use all available CPU cores
    )

    logger.info(
        "Training XGBoost | n_estimators=%d | max_depth=%d | "
        "learning_rate=%s | tree_method=%s",
        xgb_cfg["n_estimators"],
        xgb_cfg["max_depth"],
        xgb_cfg["learning_rate"],
        xgb_cfg["tree_method"],
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        # verbose=False suppresses per-tree console output; progress is logged
        # at the INFO level by the logger above and below.
        verbose=False,
    )

    logger.info("Training complete.")

    # ── 5. Save model artifact ───────────────────────────────────────────────
    model_dir = Path(paths_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / paths_cfg["model_filename"]
    joblib.dump(model, model_path)
    logger.info("Model saved | path=%s", model_path)

    # ── 6. Save training metadata ────────────────────────────────────────────
    # Metadata provides a full audit trail: which data was used, which features
    # were selected, and which hyperparameters produced this model artifact.
    # predictor.py and evaluator.py can load this to verify consistency.
    metadata = {
        "training_timestamp": datetime.utcnow().isoformat(),
        "cutoff_date":        str(cutoff_date),
        "holdout_days":       holdout_days,
        "train_row_count":    len(X_train),
        "val_row_count":      len(X_val),
        "feature_count":      len(FEATURE_COLUMNS),
        "feature_columns":    FEATURE_COLUMNS,
        "target_column":      TARGET_COLUMN,
        "model_parameters":   xgb_cfg,
        "random_seed":        random_seed,
    }

    metadata_path = model_dir / paths_cfg["training_metadata_filename"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Training metadata saved | path=%s", metadata_path)

    return model
