"""
src/modeling/predictor.py

Loads a trained XGBoost model and runs batch inference on the feature-engineered
Spark DataFrame produced by price_features.py.

Returns a pandas DataFrame containing the original key columns alongside the
model's predicted_units_sold, ready for evaluator.py and recommender.py.

─────────────────────────────────────────────────────────────────────────────
Why train/inference feature parity matters
─────────────────────────────────────────────────────────────────────────────
A trained model encodes a fixed mapping from a specific vector of input features
to an output.  Each position in that vector corresponds to one column, in the
exact order that was present during training.

If inference constructs its feature matrix differently — different column count,
different column order, missing columns, or extra columns — the model silently
maps the wrong values to the wrong learned weights.  The result is not an error;
it is numerically wrong predictions that may look plausible and therefore go
undetected.

This is prevented here by importing FEATURE_COLUMNS directly from trainer.py:

    from src.modeling.trainer import get_feature_columns

Both training and inference execute `spark_df.select(FEATURE_COLUMNS)` from the
same list.  Adding, removing, or reordering a feature requires changing only one
line in trainer.py, and the change propagates automatically to predictor.py.

─────────────────────────────────────────────────────────────────────────────
Which key columns are preserved and why
─────────────────────────────────────────────────────────────────────────────
The output pandas DataFrame carries these columns in addition to
predicted_units_sold:

  date          Required by evaluator.py to slice predictions into time windows
                (e.g. weekly, monthly RMSE) and to plot actual vs. predicted
                demand over time.

  store_id      Allows evaluator.py and recommender.py to group results by
  product_id    individual store-product entities — the atomic unit of demand
                forecasting in this project.

  category      Human-readable labels kept for reporting and dashboard display.
  region        They are excluded from FEATURE_COLUMNS (raw strings are not
                usable by XGBoost) but are valuable downstream.

  units_sold    The ground-truth target.  Keeping it in the output DataFrame
                lets evaluator.py compute residuals (predicted − actual) without
                needing a separate join back to the source data.

  demand_forecast  The dataset's own pre-built baseline forecast.  Evaluator.py
                   compares the XGBoost model against this benchmark to
                   quantify improvement.  It is intentionally excluded from
                   FEATURE_COLUMNS (see trainer.py) so the comparison is fair.

─────────────────────────────────────────────────────────────────────────────
Spark-to-pandas conversion strategy
─────────────────────────────────────────────────────────────────────────────
XGBoost's sklearn API (XGBRegressor.predict) requires an in-memory pandas
DataFrame or numpy array.  The conversion is handled in two steps:

  1. Column selection happens in Spark (lazy), before any data is moved:
         spark_df.select(FEATURE_COLUMNS + KEY_COLUMNS)
     This limits the number of bytes transferred from executors to the driver
     to exactly the columns that are actually needed.  At ~73k rows with
     ~36 columns (29 features + 7 key columns), the driver memory footprint
     is small enough that a single .toPandas() call is appropriate.

  2. .toPandas() triggers Spark execution and collects all rows onto the driver.
     After collection, the feature matrix is extracted as:
         pdf[FEATURE_COLUMNS]
     This preserves the exact column order that matches the training matrix.

BooleanType Spark columns (is_holiday_or_promo, is_weekend, has_discount)
become Python bool dtype in pandas.  XGBoost's sklearn API accepts bool and
converts to float32 internally — no explicit cast is needed.

Null values in lag and rolling_std columns are preserved.  XGBoost's
tree_method="hist" handles missing values natively using the same optimal
routing direction that was learned during training.

─────────────────────────────────────────────────────────────────────────────
Input schema vs output schema
─────────────────────────────────────────────────────────────────────────────
Input — Spark DataFrame from add_price_features(), 39 columns:

  Key / reporting (7):
    date              DateType
    store_id          StringType
    product_id        StringType
    category          StringType
    region            StringType
    units_sold        IntegerType   ← ground truth kept for evaluation
    demand_forecast   DoubleType    ← baseline kept for benchmark comparison

  Features passed to model (29):
    inventory_level, units_ordered,
    price, discount, competitor_pricing,
    effective_price, price_to_competitor_ratio, has_discount,
    is_holiday_or_promo,
    day_of_week, month, week_of_year, quarter, day_of_month, is_weekend,
    category_encoded, region_encoded, weather_encoded, seasonality_encoded,
    lag_1, lag_7, lag_14, lag_28,
    rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_std_14,
    rolling_mean_28, rolling_std_28

  Audit / excluded from model (3):
    weather_condition, seasonality, is_forecast_anomaly

Output — pandas DataFrame, 8 columns:

  date                  object (datetime.date)
  store_id              object (str)
  product_id            object (str)
  category              object (str)
  region                object (str)
  units_sold            int64         ← actual demand (ground truth)
  demand_forecast       float64       ← dataset's baseline forecast
  predicted_units_sold  float64       ← XGBoost model output

Usage from another module
--------------------------
    from src.features.price_features import add_price_features
    from src.modeling.predictor import predict

    feature_df = add_price_features(...)        # 39-column Spark DataFrame
    predictions = predict(feature_df)           # pandas DataFrame, 8 columns

    print(predictions.head())
    #          date store_id product_id  category region  units_sold  \\
    # 0  2022-01-01     S001       P001  Clothing   East          42
    #    demand_forecast  predicted_units_sold
    # 0             38.5                 41.3
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml
from pyspark.sql import DataFrame

from src.modeling.trainer import get_feature_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Columns carried through to the output for evaluation and reporting.
# These are never passed to the model — they are preserved from the input
# Spark DataFrame so that downstream modules do not need to re-join.
# ---------------------------------------------------------------------------
KEY_COLUMNS: list[str] = [
    "date",
    "store_id",
    "product_id",
    "category",
    "region",
    "units_sold",
    "demand_forecast",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """
    Load and return config.yaml as a dict.

    Only the 'paths' section is used by predictor.py (to locate the model
    artifact), but the full file is loaded to stay consistent with trainer.py.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist at the given path.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_model(model_path: Path):
    """
    Deserialise the XGBoost model artifact from disk.

    trainer.py serialises the fitted XGBRegressor via joblib.dump().
    joblib.load() reconstructs the identical in-memory object, including
    all learned tree structures and hyperparameters.

    Parameters
    ----------
    model_path : Path
        Full path to the .joblib file written by trainer.py.

    Returns
    -------
    XGBRegressor
        The fitted model, ready to call .predict() on.

    Raises
    ------
    FileNotFoundError
        If the model artifact does not exist — most likely because
        trainer.py has not been run yet.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path.resolve()}. "
            "Run trainer.train() first to produce the model file."
        )
    model = joblib.load(model_path)
    logger.info("Model loaded | path=%s", model_path)
    return model


def _collect_to_pandas(spark_df: DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Select the required columns and collect the Spark DataFrame to pandas.

    Only FEATURE_COLUMNS + KEY_COLUMNS are selected before calling .toPandas().
    This reduces the data transferred from Spark executors to the driver to
    exactly the columns that are needed — omitting audit columns such as
    weather_condition, seasonality, and is_forecast_anomaly.

    Parameters
    ----------
    spark_df : DataFrame
        39-column feature DataFrame from add_price_features().

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with 36 columns (29 features + 7 key columns).
        Row count equals the input Spark DataFrame row count.
    """
    all_columns = KEY_COLUMNS + feature_columns
    columns_to_collect = []
    seen = set()
    for col in all_columns:
        if col not in seen:
            columns_to_collect.append(col)
            seen.add(col)
    logger.info(
        "Selecting %d columns before toPandas | features=%d | key_columns=%d",
        len(columns_to_collect),
        len(feature_columns),
        len(KEY_COLUMNS),
    )
    pdf = spark_df.select(columns_to_collect).toPandas()
    logger.info("Collected %d rows to pandas", len(pdf))
    return pdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    df: DataFrame,
    config_path: str = "config/config.yaml",
    use_baseline_feature: bool = False,
) -> pd.DataFrame:
    """
    Run batch inference on a feature-engineered Spark DataFrame.

    Steps performed:
      1. Load config to resolve the model artifact path.
      2. Load the trained XGBoost model from disk via joblib.
      3. Select KEY_COLUMNS + FEATURE_COLUMNS from the Spark DataFrame and
         collect to pandas — minimising data moved to the driver.
      4. Extract the feature matrix X = pdf[FEATURE_COLUMNS] (29 columns,
         same order as training).
      5. Call model.predict(X) to produce raw float predictions.
      6. Clip predictions to >= 0 (units sold cannot be negative).
      7. Append predicted_units_sold to the key-columns slice and return.

    Parameters
    ----------
    df : DataFrame
        39-column feature-engineered Spark DataFrame from add_price_features().
        Must contain all columns in FEATURE_COLUMNS and KEY_COLUMNS.
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with 8 columns:
          date, store_id, product_id, category, region,
          units_sold, demand_forecast, predicted_units_sold
        Row order matches the order of rows in the input Spark DataFrame
        after Spark's internal partition ordering (non-deterministic between
        runs; sort by date / store_id / product_id if a stable order is needed).

    Raises
    ------
    FileNotFoundError
        If config.yaml or the model artifact cannot be found.
    ValueError
        If the input DataFrame is missing any required feature or key column.
    """
    cfg = _load_config(Path(config_path))
    paths_cfg = cfg["paths"]

    # Determine feature columns based on the model variant
    feature_columns = get_feature_columns(use_baseline_feature)
    model_variant = "enhanced" if use_baseline_feature else "fair"

    # Construct the model filename based on the variant
    model_filename = f"{model_variant}_{paths_cfg['model_filename']}"
    model_path = Path(paths_cfg["model_dir"]) / model_filename
    model = _load_model(model_path)

    # ── Validate that required columns are present ────────────────────────────
    required_columns = set(KEY_COLUMNS + feature_columns)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input DataFrame is missing {len(missing)} required column(s): "
            f"{sorted(missing)}. "
            "Ensure the full add_price_features() pipeline has been applied."
        )

    # ── Collect feature + key columns to pandas ───────────────────────────────
    pdf = _collect_to_pandas(df, feature_columns)

    # ── Build the feature matrix in the exact training column order ───────────
    # pdf[FEATURE_COLUMNS] reproduces the same 29-column matrix that was passed
    # to model.fit() in trainer.py — same columns, same order.
    X = pdf[feature_columns].to_numpy()

    # ── Run inference ─────────────────────────────────────────────────────────
    logger.info(
        "Running inference | rows=%d | features=%d",
        len(X),
        len(feature_columns),
    )
    raw_predictions = model.predict(X)   # numpy array, float32

    # Clip to zero: XGBoost's reg:squarederror objective can output small
    # negative values for low-demand rows.  Units sold cannot be negative.
    predictions = raw_predictions.clip(min=0)

    logger.info(
        "Inference complete | min=%.2f | max=%.2f | mean=%.2f",
        predictions.min(),
        predictions.max(),
        predictions.mean(),
    )

    # ── Assemble the output DataFrame ─────────────────────────────────────────
    # Start from only the KEY_COLUMNS slice — the feature columns are not
    # needed downstream and would bloat the output unnecessarily.
    result = pdf[KEY_COLUMNS].copy()
    result["predicted_units_sold"] = predictions

    return result
