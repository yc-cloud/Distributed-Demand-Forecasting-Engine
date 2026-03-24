"""
src/modeling/evaluator.py

Evaluates the XGBoost demand forecast against two benchmarks:

  1. The ground truth: units_sold (actual observed demand).
  2. The dataset's pre-built baseline: demand_forecast.

Reports RMSE, MAE, and MAPE globally and broken down by product category.
Saves the full metrics report to a JSON file for reproducibility and portfolio
presentation.

─────────────────────────────────────────────────────────────────────────────
What each metric measures and why three metrics are reported together
─────────────────────────────────────────────────────────────────────────────

RMSE — Root Mean Squared Error
    sqrt( mean( (predicted − actual)² ) )

    Squares each error before averaging, so large individual errors contribute
    disproportionately to the final value.  This matters in retail: a single
    stock-out event or massive overstock causes outsized business harm.  RMSE
    is reported in the same unit as units_sold (number of units), making it
    directly interpretable ("the model is off by X units on average, with
    large errors penalised more").

    Weakness: sensitive to outliers — a handful of very wrong predictions can
    make RMSE look bad even if typical accuracy is good.  MAE is the antidote.

MAE — Mean Absolute Error
    mean( |predicted − actual| )

    Averages the absolute magnitude of every error equally, regardless of
    direction or size.  Because every row is weighted identically, MAE
    reflects typical per-day-per-product accuracy.  "On a typical day, the
    model is off by X units."

    Reported alongside RMSE: if RMSE >> MAE, there are a small number of very
    large errors inflating RMSE.  If RMSE ≈ MAE, errors are uniformly
    distributed.  This comparison guides where to invest in improvement.

MAPE — Mean Absolute Percentage Error
    mean( |predicted − actual| / actual ) × 100

    Scales each error by the actual demand level, producing a percentage.
    A MAPE of 15% means the model is off by 15% of actual demand on average.
    MAPE is scale-invariant: a 2-unit error on a product that sells 10/day
    (20% error) is treated more seriously than the same 2-unit error on a
    product that sells 200/day (1% error).  This is the right weighting for a
    multi-category assortment with very different volume levels.

    Weakness: undefined when actual = 0.  Rows where units_sold = 0 are
    excluded from MAPE computation and the count is logged; RMSE and MAE
    still include them.

Why three metrics?  Because no single metric captures the full picture:
  - RMSE catches dangerous outlier errors.
  - MAE reflects typical day-to-day accuracy.
  - MAPE enables comparison across categories with different demand scales.

─────────────────────────────────────────────────────────────────────────────
Why baseline comparison matters
─────────────────────────────────────────────────────────────────────────────
Reporting model metrics in isolation is meaningless without a reference point.
A RMSE of 12 units could be excellent or terrible depending on the scale of
demand and what a naive alternative would achieve.

The dataset's demand_forecast column provides a pre-built baseline — the
simplest available benchmark.  If the XGBoost model cannot beat this baseline,
it offers no value over using the provided forecast directly.

Baseline comparison answers the key business question:
    "How much better is the ML model than what we already had?"

The improvement is reported as a delta for each metric:
    improvement_rmse = baseline_rmse − model_rmse

A positive delta means the model reduced error.  If the delta is near zero or
negative, the model needs more feature engineering or tuning before deployment.

─────────────────────────────────────────────────────────────────────────────
How anomalous demand_forecast rows are handled
─────────────────────────────────────────────────────────────────────────────
The demand_forecast column is known to contain negative values (documented in
trainer.py: "demand_forecast also contains negative values — data quality issue
flagged by is_forecast_anomaly").

A negative forecast is not a valid demand prediction — demand cannot be
negative by definition.  Including these rows in baseline metrics would unfairly
inflate baseline errors, making the XGBoost model look better than it is.

Handling strategy: transparent exclusion with logging.

  1. Rows where demand_forecast < 0 are identified and counted before any
     metric is computed.
  2. Baseline metrics (RMSE, MAE, MAPE for demand_forecast) are computed on
     the CLEAN subset: rows where demand_forecast >= 0.
  3. Model metrics (RMSE, MAE, MAPE for predicted_units_sold) are computed on
     ALL rows — the XGBoost model does not produce negative forecasts because
     predictions are clipped to >= 0 in predictor.py.
  4. The anomaly count and the two different row counts are recorded in the
     JSON output so the reader can see exactly what was excluded.

This approach is honest: it does not discard anomalies silently, and it does
not corrupt the baseline benchmark by leaving obviously invalid rows in.

─────────────────────────────────────────────────────────────────────────────
Example output JSON structure
─────────────────────────────────────────────────────────────────────────────
{
  "evaluation_timestamp": "2024-03-01T14:32:07",
  "row_counts": {
    "total_rows": 9000,
    "model_eval_rows": 9000,
    "baseline_eval_rows": 8750,
    "baseline_anomaly_rows": 250,
    "mape_excluded_rows_model": 45,
    "mape_excluded_rows_baseline": 42
  },
  "global": {
    "model": {
      "rmse": 11.42,
      "mae": 7.83,
      "mape": 14.61
    },
    "baseline": {
      "rmse": 18.97,
      "mae": 13.55,
      "mape": 24.30
    },
    "improvement": {
      "rmse_delta": 7.55,
      "mae_delta": 5.72,
      "mape_delta": 9.69
    }
  },
  "by_category": {
    "Clothing": {
      "total_rows": 1800,
      "model_eval_rows": 1800,
      "baseline_eval_rows": 1755,
      "model": { "rmse": 10.11, "mae": 6.94, "mape": 13.02 },
      "baseline": { "rmse": 17.44, "mae": 12.01, "mape": 22.17 },
      "improvement": { "rmse_delta": 7.33, "mae_delta": 5.07, "mape_delta": 9.15 }
    },
    ...
  }
}

Usage from another module
--------------------------
    from src.modeling.predictor import predict
    from src.modeling.evaluator import evaluate

    predictions_df = predict(feature_df)        # 8-column pandas DataFrame
    metrics = evaluate(predictions_df)          # dict; also saved to JSON
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Filename written into model_dir alongside the trained model artifact.
# Keeping evaluation metadata next to the model makes the models/ directory a
# self-contained audit bundle: artifact + training metadata + eval metrics.
EVALUATION_METRICS_FILENAME: str = "evaluation_metrics.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> dict:
    """Load config.yaml; raise FileNotFoundError if absent."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error. Returns float rounded to 4 decimal places."""
    return round(math.sqrt(np.mean((predicted - actual) ** 2)), 4)


def _mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error. Returns float rounded to 4 decimal places."""
    return round(float(np.mean(np.abs(predicted - actual))), 4)


def _mape(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, int]:
    """
    Mean Absolute Percentage Error.

    Rows where actual = 0 are excluded because MAPE is undefined at zero
    (division by zero).  This is not imputation — those rows simply cannot
    contribute to a percentage-error metric and their exclusion is reported
    in the caller.

    Returns
    -------
    tuple[float, int]
        (mape_value_percent, n_excluded_rows)
        mape_value_percent is rounded to 4 decimal places.
        If all rows are excluded (all actuals = 0), returns (nan, n_excluded).
    """
    nonzero_mask = actual != 0
    n_excluded = int((~nonzero_mask).sum())

    if nonzero_mask.sum() == 0:
        return float("nan"), n_excluded

    actual_nz    = actual[nonzero_mask]
    predicted_nz = predicted[nonzero_mask]
    value = round(float(np.mean(np.abs(predicted_nz - actual_nz) / actual_nz) * 100), 4)
    return value, n_excluded


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[dict, int]:
    """
    Compute RMSE, MAE, and MAPE for one (actual, predicted) pair.

    Parameters
    ----------
    actual : np.ndarray
        Ground-truth values (units_sold).
    predicted : np.ndarray
        Forecast values (predicted_units_sold or demand_forecast).

    Returns
    -------
    tuple[dict, int]
        (metrics_dict, mape_excluded_count)
        metrics_dict has keys: rmse, mae, mape.
        mape_excluded_count is the number of rows excluded from MAPE.
    """
    mape_value, mape_excluded = _mape(actual, predicted)
    metrics = {
        "rmse": _rmse(actual, predicted),
        "mae":  _mae(actual, predicted),
        "mape": mape_value,
    }
    return metrics, mape_excluded


def _improvement(baseline_metrics: dict, model_metrics: dict) -> dict:
    """
    Compute per-metric delta: baseline − model.

    A positive delta means the model reduced error relative to the baseline.
    NaN is returned when either value is NaN (e.g. MAPE excluded all rows).
    """
    def delta(b, m):
        if math.isnan(b) or math.isnan(m):
            return float("nan")
        return round(b - m, 4)

    return {
        "rmse_delta": delta(baseline_metrics["rmse"], model_metrics["rmse"]),
        "mae_delta":  delta(baseline_metrics["mae"],  model_metrics["mae"]),
        "mape_delta": delta(baseline_metrics["mape"], model_metrics["mape"]),
    }


def _evaluate_group(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Compute model and baseline metrics for a single group (global or one category).

    Anomalous baseline rows (demand_forecast < 0) are filtered out before
    computing baseline metrics.  Model metrics always use the full group.

    Parameters
    ----------
    df : pd.DataFrame
        Slice of the predictions DataFrame for one group.
        Must contain: units_sold, predicted_units_sold, demand_forecast.

    Returns
    -------
    tuple[dict, dict]
        (group_counts, group_metrics) where:
          group_counts  — row counts and exclusion tallies for this group.
          group_metrics — model, baseline, and improvement metric dicts.
    """
    actual = df["units_sold"].to_numpy(dtype=float)
    predicted = df["predicted_units_sold"].to_numpy(dtype=float)

    # ── Baseline anomaly filter ───────────────────────────────────────────────
    # Exclude rows with negative demand_forecast from all baseline calculations.
    # Model metrics are unaffected (predicted_units_sold is always >= 0).
    clean_baseline_mask = df["demand_forecast"] >= 0
    n_anomalies = int((~clean_baseline_mask).sum())

    df_clean = df[clean_baseline_mask]
    actual_clean    = df_clean["units_sold"].to_numpy(dtype=float)
    baseline_clean  = df_clean["demand_forecast"].to_numpy(dtype=float)

    # ── Compute metrics ───────────────────────────────────────────────────────
    model_metrics,    mape_excl_model    = _compute_metrics(actual,       predicted)
    baseline_metrics, mape_excl_baseline = _compute_metrics(actual_clean, baseline_clean)

    counts = {
        "total_rows":             len(df),
        "model_eval_rows":        len(df),
        "baseline_eval_rows":     len(df_clean),
        "baseline_anomaly_rows":  n_anomalies,
        "mape_excluded_rows_model":    mape_excl_model,
        "mape_excluded_rows_baseline": mape_excl_baseline,
    }
    metrics = {
        "model":       model_metrics,
        "baseline":    baseline_metrics,
        "improvement": _improvement(baseline_metrics, model_metrics),
    }
    return counts, metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    predictions_df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> dict[str, Any]:
    """
    Evaluate demand forecasts against the ground truth and the baseline.

    Computes RMSE, MAE, and MAPE for both the XGBoost model and the dataset's
    pre-built baseline (demand_forecast), reporting results globally and per
    product category.  Saves the full report to a JSON file in model_dir.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of predictor.predict() — 8 columns:
          date, store_id, product_id, category, region,
          units_sold, demand_forecast, predicted_units_sold.
    config_path : str
        Path to config/config.yaml.

    Returns
    -------
    dict
        The full metrics report (same structure as the saved JSON).
        Returned so the caller can inspect or log metrics without re-reading
        the file.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If predictions_df is missing required columns.
    """
    # ── Validate input columns ────────────────────────────────────────────────
    required = {"units_sold", "predicted_units_sold", "demand_forecast", "category"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(
            f"predictions_df is missing required column(s): {sorted(missing)}. "
            "Ensure the DataFrame was produced by predictor.predict()."
        )

    cfg = _load_config(Path(config_path))
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting evaluation | total_rows=%d | categories=%s",
        len(predictions_df),
        sorted(predictions_df["category"].unique().tolist()),
    )

    # ── Global metrics ────────────────────────────────────────────────────────
    global_counts, global_metrics = _evaluate_group(predictions_df)

    logger.info(
        "Global | model RMSE=%.4f MAE=%.4f MAPE=%.4f | "
        "baseline RMSE=%.4f MAE=%.4f MAPE=%.4f | "
        "baseline_anomalies=%d",
        global_metrics["model"]["rmse"],
        global_metrics["model"]["mae"],
        global_metrics["model"]["mape"],
        global_metrics["baseline"]["rmse"],
        global_metrics["baseline"]["mae"],
        global_metrics["baseline"]["mape"],
        global_counts["baseline_anomaly_rows"],
    )

    # ── Per-category metrics ──────────────────────────────────────────────────
    by_category: dict[str, Any] = {}

    for category, group_df in predictions_df.groupby("category", sort=True):
        cat_counts, cat_metrics = _evaluate_group(group_df)

        by_category[category] = {**cat_counts, **cat_metrics}

        logger.info(
            "Category %-12s | model RMSE=%.4f MAE=%.4f | "
            "baseline RMSE=%.4f MAE=%.4f | rows=%d anomalies=%d",
            category,
            cat_metrics["model"]["rmse"],
            cat_metrics["model"]["mae"],
            cat_metrics["baseline"]["rmse"],
            cat_metrics["baseline"]["mae"],
            cat_counts["total_rows"],
            cat_counts["baseline_anomaly_rows"],
        )

    # ── Assemble report ───────────────────────────────────────────────────────
    report: dict[str, Any] = {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "row_counts": global_counts,
        "global":      global_metrics,
        "by_category": by_category,
    }

    # ── Save to JSON ──────────────────────────────────────────────────────────
    output_path = model_dir / EVALUATION_METRICS_FILENAME
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation metrics saved | path=%s", output_path)

    return report
