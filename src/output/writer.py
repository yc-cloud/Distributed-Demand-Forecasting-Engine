"""
src/output/writer.py

Writes the two final pipeline outputs — demand forecast predictions and
inventory replenishment recommendations — to disk in CSV and/or Parquet
format.

All paths and filenames are read from config/config.yaml so that output
locations can be changed without touching any Python source file.

─────────────────────────────────────────────────────────────────────────────
Why both CSV and Parquet are supported
─────────────────────────────────────────────────────────────────────────────
The two formats serve different consumers and neither fully replaces the other.

Parquet
  - Columnar, compressed binary format.  A 73k-row predictions DataFrame that
    is ~4 MB as CSV compresses to ~400 KB as Parquet.
  - Read by pandas (.read_parquet), Spark (.read.parquet), Dask, Polars, and
    every modern data warehouse (BigQuery, Snowflake, Redshift).
  - Preserves dtypes exactly: float64, int64, bool, datetime.date all round-
    trip without any string parsing.  No silent type coercion on reload.
  - Correct choice for production pipelines and downstream programmatic use.

CSV
  - Human-readable plain text.  Open in Excel, Numbers, or any text editor.
  - Universal: no library, no schema, no binary decoder required.
  - Essential for portfolio presentation, stakeholder review, and quick manual
    inspection during development.
  - Weakness: all types become strings on disk; dates, booleans, and floats
    must be re-parsed on load.  Not suitable for high-volume production I/O.

Both are written by default (config.output.write_parquet and write_csv are
both true).  Either can be disabled with a single config change — useful when
running in a production environment that already has a data warehouse (no CSV
needed) or when debugging on a laptop where a quick CSV glance is enough.

─────────────────────────────────────────────────────────────────────────────
How output filenames are constructed
─────────────────────────────────────────────────────────────────────────────
config.yaml declares stem names without file extensions:

  output:
    forecast_filename:   "demand_forecast"
    inventory_filename:  "inventory_recommendations"
    predictions_dir:     set under paths.predictions_dir → "data/predictions"

The writer appends the correct extension for each format:

  data/predictions/demand_forecast.parquet
  data/predictions/demand_forecast.csv
  data/predictions/inventory_recommendations.parquet
  data/predictions/inventory_recommendations.csv

Keeping extensions out of the config avoids the config having to know about
format details (that is the writer's job), and allows the same stem to be
written in multiple formats without duplicating the name.

─────────────────────────────────────────────────────────────────────────────
Example: saving forecast output
─────────────────────────────────────────────────────────────────────────────
    from src.modeling.predictor import predict
    from src.output.writer import write_forecast

    predictions_df = predict(feature_df)        # 8-column pandas DataFrame
    paths = write_forecast(predictions_df)

    # paths is a dict of format → Path for every file actually written:
    # {
    #   "parquet": PosixPath("data/predictions/demand_forecast.parquet"),
    #   "csv":     PosixPath("data/predictions/demand_forecast.csv"),
    # }

─────────────────────────────────────────────────────────────────────────────
Example: saving inventory recommendation output
─────────────────────────────────────────────────────────────────────────────
    from src.inventory.recommender import recommend
    from src.output.writer import write_inventory

    recommendations_df = recommend(predictions_df, feature_df)
    paths = write_inventory(recommendations_df)

    # {
    #   "parquet": PosixPath("data/predictions/inventory_recommendations.parquet"),
    #   "csv":     PosixPath("data/predictions/inventory_recommendations.csv"),
    # }
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


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


def _write_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    write_parquet: bool,
    write_csv: bool,
) -> dict[str, Path]:
    """
    Write a pandas DataFrame to disk in the requested format(s).

    Creates output_dir if it does not already exist.  At least one of
    write_parquet or write_csv must be True; if both are False a ValueError
    is raised rather than silently writing nothing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    output_dir : Path
        Directory in which to write the output file(s).
    stem : str
        Filename without extension, e.g. "demand_forecast".
    write_parquet : bool
        Write a Parquet file when True.
    write_csv : bool
        Write a CSV file when True.

    Returns
    -------
    dict[str, Path]
        Mapping of format name to the Path of the written file.
        Keys are "parquet" and/or "csv" depending on which formats were
        written.

    Raises
    ------
    ValueError
        If both write_parquet and write_csv are False.
    """
    if not write_parquet and not write_csv:
        raise ValueError(
            "Both write_parquet and write_csv are False in config. "
            "Set at least one to true to produce output."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    if write_parquet:
        path = output_dir / f"{stem}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote Parquet | rows=%d | path=%s", len(df), path)
        written["parquet"] = path

    if write_csv:
        path = output_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        logger.info("Wrote CSV     | rows=%d | path=%s", len(df), path)
        written["csv"] = path

    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_forecast(
    predictions_df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> dict[str, Path]:
    """
    Save the demand forecast predictions DataFrame to disk.

    Reads the output directory, filename stem, and format toggles from
    config.yaml.  Creates the output directory if it does not exist.

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
    dict[str, Path]
        Format → Path mapping for every file written.
        Example: {"parquet": Path("data/predictions/demand_forecast.parquet"),
                  "csv":     Path("data/predictions/demand_forecast.csv")}

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If both output formats are disabled in the config.
    """
    cfg = _load_config(Path(config_path))

    output_dir = Path(cfg["paths"]["predictions_dir"])
    stem        = cfg["output"]["forecast_filename"]
    write_parquet = bool(cfg["output"]["write_parquet"])
    write_csv     = bool(cfg["output"]["write_csv"])

    logger.info(
        "Saving forecast output | stem=%s | rows=%d | parquet=%s | csv=%s",
        stem,
        len(predictions_df),
        write_parquet,
        write_csv,
    )

    return _write_dataframe(predictions_df, output_dir, stem, write_parquet, write_csv)


def write_inventory(
    recommendations_df: pd.DataFrame,
    config_path: str = "config/config.yaml",
) -> dict[str, Path]:
    """
    Save the inventory replenishment recommendations DataFrame to disk.

    Reads the output directory, filename stem, and format toggles from
    config.yaml.  Creates the output directory if it does not exist.

    Parameters
    ----------
    recommendations_df : pd.DataFrame
        Output of recommender.recommend() — 16 columns including store_id,
        product_id, safety_stock, reorder_point, recommended_order_qty, etc.
    config_path : str
        Path to config/config.yaml.

    Returns
    -------
    dict[str, Path]
        Format → Path mapping for every file written.
        Example: {"parquet": Path("data/predictions/inventory_recommendations.parquet"),
                  "csv":     Path("data/predictions/inventory_recommendations.csv")}

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If both output formats are disabled in the config.
    """
    cfg = _load_config(Path(config_path))

    output_dir    = Path(cfg["paths"]["predictions_dir"])
    stem          = cfg["output"]["inventory_filename"]
    write_parquet = bool(cfg["output"]["write_parquet"])
    write_csv     = bool(cfg["output"]["write_csv"])

    logger.info(
        "Saving inventory output | stem=%s | rows=%d | parquet=%s | csv=%s",
        stem,
        len(recommendations_df),
        write_parquet,
        write_csv,
    )

    return _write_dataframe(recommendations_df, output_dir, stem, write_parquet, write_csv)
