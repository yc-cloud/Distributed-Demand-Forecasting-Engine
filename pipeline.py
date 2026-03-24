"""
pipeline.py

End-to-end orchestration for the Distributed Demand Forecasting Engine.

Runs all 14 pipeline stages in sequence, from raw CSV ingestion through
model training, evaluation, inventory recommendation, and final file output.
Stops the Spark session in a finally block so that resources are always
released — even when a stage raises an exception.

─────────────────────────────────────────────────────────────────────────────
Pipeline flow
─────────────────────────────────────────────────────────────────────────────
Each numbered stage maps directly to one public function from one module.
The dependency graph is strictly linear — each stage consumes the output of
the previous one — so there is no parallelism and no branching.

  Stage  Module                         Input               Output
  ─────  ──────────────────────────────────────────────────────────────────
   1     spark_session.get_spark_session  config            SparkSession
   2     loader.load_raw_data             SparkSession      raw Spark DF  (15 cols)
   3     cleaner.clean                    raw DF            cleaned DF    (16 cols)
   4     encoder.encode                   cleaned DF        encoded DF    (20 cols)
   5     lag_features.add_lag_features    encoded DF        DF            (24 cols)
   6     rolling_features.add_rolling_   lag DF            DF            (30 cols)
   7     calendar_features.add_calendar_ rolling DF        DF            (36 cols)
   8     price_features.add_price_        calendar DF       feature DF    (39 cols)
   9     trainer.train                    feature DF        XGBRegressor  + artifacts on disk
  10     predictor.predict                feature DF        predictions   pandas (8 cols)
  11     evaluator.evaluate               predictions       metrics dict  + JSON on disk
  12     recommender.recommend            predictions +     recommendations pandas (16 cols)
                                          feature DF
  13     writer.write_forecast            predictions       paths dict
  14     writer.write_inventory           recommendations   paths dict

Note: feature_df (stage 8 output) is intentionally kept in scope through
stages 9–12.  It is referenced by:
  - trainer.train    (Spark DF → time-split → toPandas)
  - predictor.predict (Spark DF → select + toPandas)
  - recommender.recommend (Spark DF → extract latest inventory_level)

─────────────────────────────────────────────────────────────────────────────
Why Spark session cleanup belongs in a finally block
─────────────────────────────────────────────────────────────────────────────
SparkSession.stop() shuts down the JVM process, releases all executor memory,
and closes any open file handles or network connections.  If this call is
placed after the pipeline logic without a finally guard:

  - A Python exception in any stage would skip the stop() call entirely.
  - The JVM process would linger, holding memory and ports.
  - On a shared cluster this starves other jobs; on a laptop it causes
    confusing "port already in use" errors on the next run.

Placing stop() in a finally block guarantees it runs regardless of whether
the pipeline succeeded or failed.  The exception is still re-raised (we do
not catch and swallow it) so the caller sees the real error while the JVM is
still cleaned up.

─────────────────────────────────────────────────────────────────────────────
How output artifacts are tracked
─────────────────────────────────────────────────────────────────────────────
Every stage that writes to disk returns the path(s) of what it wrote.
run_pipeline() collects all of these into a single "artifacts" dict and
returns it to the caller.  This means:

  1. The caller (CLI, notebook, or test) can assert which files were created.
  2. Every path is logged at INFO level so a pipeline run produces a readable
     summary of outputs without the caller having to know the config.
  3. No path is hardcoded in pipeline.py — all paths come from the modules
     that wrote the files, which in turn read them from config.yaml.

The artifacts dict structure:
  {
    "model_path":        Path  — serialised XGBRegressor (.joblib)
    "metadata_path":     Path  — training run metadata (.json)
    "evaluation_path":   Path  — evaluation metrics (.json)
    "forecast_paths":    dict  — {"parquet": Path, "csv": Path}
    "inventory_paths":   dict  — {"parquet": Path, "csv": Path}
  }

─────────────────────────────────────────────────────────────────────────────
How to run the pipeline from the command line
─────────────────────────────────────────────────────────────────────────────
Run from the project root (the directory that contains config/ and src/):

    python pipeline.py

Optional: override the config file path:

    python pipeline.py --config path/to/other_config.yaml

The pipeline prints structured log lines at INFO level.  To see only warnings
and errors, set the log level in config.yaml:

    logging:
      level: "WARNING"

Expected output files after a successful run:

    models/xgboost_demand_forecast.joblib
    models/training_run_metadata.json
    models/evaluation_metrics.json
    data/predictions/demand_forecast.parquet
    data/predictions/demand_forecast.csv
    data/predictions/inventory_recommendations.parquet
    data/predictions/inventory_recommendations.csv
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

from src.utils.spark_session import get_spark_session, stop_spark_session
from src.data_preparation.loader import load_raw_data
from src.data_preparation.cleaner import clean
from src.data_preparation.encoder import encode
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.features.calendar_features import add_calendar_features
from src.features.price_features import add_price_features
from src.modeling.trainer import train
from src.modeling.predictor import predict
from src.modeling.evaluator import evaluate, EVALUATION_METRICS_FILENAME
from src.inventory.recommender import recommend
from src.output.writer import write_forecast, write_inventory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging(config_path: Path) -> None:
    """
    Configure the root logger from config.yaml.

    Called once at pipeline startup, before any module emits log lines.
    Falls back to INFO / a sensible format if the config is missing or
    the logging section is absent.
    """
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        log_cfg = cfg.get("logging", {})
        level  = log_cfg.get("level",  "INFO")
        fmt    = log_cfg.get("format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        datefmt = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")
    except FileNotFoundError:
        level, fmt, datefmt = "INFO", "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt=datefmt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str = "config/config.yaml", use_baseline_feature: bool = False) -> dict:
    """
    Execute the full demand forecasting pipeline end-to-end.

    All 14 stages run in sequence.  The Spark session is stopped in a
    finally block so it is always released, even if a stage raises.

    Parameters
    ----------
    config_path : str
        Path to config/config.yaml.  Passed through to every module that
        reads configuration (each module resolves paths relative to the
        project root, so this should be run from the project root directory).

    Returns
    -------
    dict
        Artifact paths for every file written during the run:
          model_path, metadata_path, evaluation_path,
          forecast_paths, inventory_paths.

    Raises
    ------
    Any exception raised by a pipeline stage is logged at ERROR level and
    re-raised so the caller / CLI exit code reflects the failure.
    """
    cfg_path = Path(config_path)
    _configure_logging(cfg_path)

    logger.info("=" * 70)
    logger.info("Distributed Demand Forecasting Engine — pipeline starting")
    logger.info("Config: %s", cfg_path.resolve())
    logger.info("=" * 70)

    # Read paths config once here so we can derive artifact paths for the
    # return dict without duplicating config-reading logic.
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(cfg["paths"]["model_dir"])
    artifacts: dict = {}

    spark = None
    try:
        # ── Stage 1: Spark session ────────────────────────────────────────────
        logger.info("Stage 1/14 | Creating Spark session")
        spark = get_spark_session(config_path=config_path)

        # ── Stage 2: Load raw data ────────────────────────────────────────────
        logger.info("Stage 2/14 | Loading raw data")
        raw_df = load_raw_data(spark, config_path=config_path)

        # ── Stage 3: Clean ────────────────────────────────────────────────────
        logger.info("Stage 3/14 | Cleaning data")
        cleaned_df = clean(raw_df)

        # ── Stage 4: Encode categorical features ─────────────────────────────
        logger.info("Stage 4/14 | Encoding categorical features")
        encoded_df = encode(cleaned_df, config_path=config_path)

        # ── Stage 5: Lag features ─────────────────────────────────────────────
        logger.info("Stage 5/14 | Adding lag features")
        lag_df = add_lag_features(encoded_df, config_path=config_path)

        # ── Stage 6: Rolling features ─────────────────────────────────────────
        logger.info("Stage 6/14 | Adding rolling features")
        rolling_df = add_rolling_features(lag_df, config_path=config_path)

        # ── Stage 7: Calendar features ────────────────────────────────────────
        logger.info("Stage 7/14 | Adding calendar features")
        calendar_df = add_calendar_features(rolling_df)

        # ── Stage 8: Price features ───────────────────────────────────────────
        logger.info("Stage 8/14 | Adding price features")
        feature_df = add_price_features(calendar_df)

        # ── Stage 9: Train model ──────────────────────────────────────────────
        logger.info("Stage 9/14 | Training XGBoost model")
        train(feature_df, config_path=config_path, use_baseline_feature=use_baseline_feature)

        model_variant = "enhanced" if use_baseline_feature else "fair"
        artifacts["model_path"]    = model_dir / f"{model_variant}_{cfg["paths"]["model_filename"]}"
        artifacts["metadata_path"] = model_dir / f"{model_variant}_{cfg["paths"]["training_metadata_filename"]}"
        logger.info("Model artifact  : %s", artifacts["model_path"])
        logger.info("Training metadata: %s", artifacts["metadata_path"])

        # ── Stage 10: Generate predictions ───────────────────────────────────
        logger.info("Stage 10/14 | Generating predictions")
        predictions_df = predict(feature_df, config_path=config_path, use_baseline_feature=use_baseline_feature)
        logger.info("Predictions shape: %s", predictions_df.shape)

        # ── Stage 11: Evaluate predictions ───────────────────────────────────
        logger.info("Stage 11/14 | Evaluating predictions")
        metrics = evaluate(predictions_df, config_path=config_path)

        artifacts["evaluation_path"] = model_dir / EVALUATION_METRICS_FILENAME
        logger.info("Evaluation metrics: %s", artifacts["evaluation_path"])
        logger.info(
            "Global RMSE  — model: %.4f  baseline: %.4f  improvement: %.4f",
            metrics["global"]["model"]["rmse"],
            metrics["global"]["baseline"]["rmse"],
            metrics["global"]["improvement"]["rmse_delta"],
        )
        logger.info(
            "Global MAE   — model: %.4f  baseline: %.4f  improvement: %.4f",
            metrics["global"]["model"]["mae"],
            metrics["global"]["baseline"]["mae"],
            metrics["global"]["improvement"]["mae_delta"],
        )
        logger.info(
            "Global MAPE  — model: %.4f  baseline: %.4f  improvement: %.4f",
            metrics["global"]["model"]["mape"],
            metrics["global"]["baseline"]["mape"],
            metrics["global"]["improvement"]["mape_delta"],
        )

        # ── Stage 12: Inventory recommendations ──────────────────────────────
        logger.info("Stage 12/14 | Generating inventory recommendations")
        recommendations_df = recommend(predictions_df, feature_df, config_path=config_path)
        logger.info("Recommendations shape: %s", recommendations_df.shape)
        logger.info(
            "SKUs flagged for reorder: %d / %d",
            int(recommendations_df["reorder_recommendation"].sum()),
            len(recommendations_df),
        )

        # ── Stage 13: Write forecast output ──────────────────────────────────
        logger.info("Stage 13/14 | Writing forecast output")
        forecast_paths = write_forecast(predictions_df, config_path=config_path)
        artifacts["forecast_paths"] = forecast_paths
        for fmt, path in forecast_paths.items():
            logger.info("Forecast %-8s: %s", fmt, path)

        # ── Stage 14: Write inventory output ─────────────────────────────────
        logger.info("Stage 14/14 | Writing inventory recommendations output")
        inventory_paths = write_inventory(recommendations_df, config_path=config_path)
        artifacts["inventory_paths"] = inventory_paths
        for fmt, path in inventory_paths.items():
            logger.info("Inventory %-7s: %s", fmt, path)

        logger.info("=" * 70)
        logger.info("Pipeline complete — all stages succeeded")
        logger.info("=" * 70)

        return artifacts

    except Exception:
        logger.error("Pipeline failed — see traceback above", exc_info=True)
        raise

    finally:
        # Always stop Spark, whether the pipeline succeeded or raised.
        # Omitting this would leave the JVM process running, holding ports
        # and memory until the OS reclaims them.
        if spark is not None:
            logger.info("Stopping Spark session")
            stop_spark_session()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Distributed Demand Forecasting Engine pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--use-baseline-feature",
        action="store_true",
        help="Use demand_forecast as a feature for the Enhanced model (default: False)",
    )
    args = parser.parse_args()

    artifacts = run_pipeline(config_path=args.config, use_baseline_feature=args.use_baseline_feature)

    print("\nArtifacts written:")
    print(f"  Model          : {artifacts.get('model_path')}")
    print(f"  Training meta  : {artifacts.get('metadata_path')}")
    print(f"  Eval metrics   : {artifacts.get('evaluation_path')}")
    for fmt, path in artifacts.get("forecast_paths", {}).items():
        print(f"  Forecast ({fmt:<8}): {path}")
    for fmt, path in artifacts.get("inventory_paths", {}).items():
        print(f"  Inventory ({fmt:<7}): {path}")

    sys.exit(0)
