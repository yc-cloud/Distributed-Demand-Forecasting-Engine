"""
src/utils/spark_session.py

Centralised SparkSession factory for the Distributed Demand Forecasting Engine.

Why this module exists
----------------------
SparkSession creation is expensive: it initialises a JVM, allocates driver
memory, and sets up the Spark context. Creating a new session per module would
cause resource conflicts and slow the pipeline down significantly.

Centralising session creation here enforces three guarantees:
  1. Exactly one SparkSession exists for the lifetime of a pipeline run
     (singleton pattern backed by a module-level variable).
  2. All Spark configuration comes from config/config.yaml — no settings
     are scattered across individual pipeline modules.
  3. Any module that needs Spark imports one function: get_spark_session().
     The caller does not need to know about configuration at all.

Usage from another module
-------------------------
    from src.utils.spark_session import get_spark_session

    spark = get_spark_session()
    df = spark.read.parquet("data/processed/cleaned.parquet")

The session is created on the first call and reused on every subsequent call,
so it is safe to call get_spark_session() at the top of any pipeline module.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# Persists for the entire lifetime of the Python process. Initialised to None
# and set on the first call to get_spark_session(). All subsequent calls
# receive the same object without re-entering the builder chain.
# ---------------------------------------------------------------------------
_session: Optional[SparkSession] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_spark_config(config_path: Path) -> dict:
    """
    Load and return the 'spark' section from config.yaml.

    Kept private (underscore prefix) because callers interact only with
    get_spark_session(). Separating config loading from session creation
    makes each concern independently testable.

    Parameters
    ----------
    config_path : Path
        Absolute or relative path to config/config.yaml.

    Returns
    -------
    dict
        The contents of the top-level 'spark' key in the config file.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    KeyError
        If the config file exists but contains no 'spark' section.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if "spark" not in full_config:
        raise KeyError(
            f"'spark' section missing from {config_path}. "
            "Check that config/config.yaml matches the expected schema."
        )

    return full_config["spark"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_spark_session(
    config_path: str = "config/config.yaml",
) -> SparkSession:
    """
    Return a configured SparkSession, creating one if none exists yet.

    On the first call the session is built using settings from config.yaml
    and stored in the module-level singleton. Every subsequent call returns
    the cached session immediately without touching the builder or config.

    This function is the single entry point for Spark in the entire pipeline.
    No other module should call SparkSession.builder directly.

    Parameters
    ----------
    config_path : str
        Path to the project config file, relative to the working directory.
        Defaults to "config/config.yaml" so callers at the project root need
        not pass an argument.

    Returns
    -------
    SparkSession
        An active, fully configured SparkSession.

    Example
    -------
    >>> from src.utils.spark_session import get_spark_session
    >>> spark = get_spark_session()
    >>> spark.sql("SELECT 1").show()
    """
    global _session

    # Return the cached session if it already exists.
    # getOrCreate() on the builder also enforces a JVM-level singleton, but
    # checking _session first avoids re-parsing the config file on every call.
    if _session is not None:
        logger.debug("Returning existing SparkSession (already initialised).")
        return _session

    cfg = _load_spark_config(Path(config_path))

    logger.info(
        "Initialising SparkSession | app_name=%s | master=%s | "
        "shuffle_partitions=%s | driver_memory=%s",
        cfg["app_name"],
        cfg["master"],
        cfg["shuffle_partitions"],
        cfg["driver_memory"],
    )

    _session = (
        SparkSession.builder
        # Human-readable name shown in the Spark UI
        .appName(cfg["app_name"])
        # local[*] uses all available CPU cores on the local machine.
        # Replace with a cluster URL (e.g. spark://host:7077) to scale out.
        .master(cfg["master"])
        # Reduce shuffle partitions from the Spark default of 200.
        # For a 73k-row local dataset, 8 partitions avoids the overhead of
        # scheduling 200 near-empty tasks after every groupBy or window op.
        .config("spark.sql.shuffle.partitions", cfg["shuffle_partitions"])
        # In local mode driver == executor (single JVM), so both settings
        # control the same process. Kept separate to mirror cluster config.
        .config("spark.driver.memory", cfg["driver_memory"])
        .config("spark.executor.memory", cfg["executor_memory"])
        # getOrCreate() is a built-in Spark singleton guard: if a session
        # already exists in the JVM it is returned rather than a new one
        # being created. This is a safety net on top of the _session check.
        .getOrCreate()
    )

    # Suppress verbose Spark INFO logs that would otherwise drown out
    # application-level log output. Set in config.yaml (default: WARN).
    # This is applied after creation because setLogLevel operates on the
    # SparkContext, which does not exist until getOrCreate() completes.
    _session.sparkContext.setLogLevel(cfg["log_level"])

    logger.info("SparkSession ready.")
    return _session


def stop_spark_session() -> None:
    """
    Stop the active SparkSession and clear the module-level singleton.

    Called once at the end of pipeline.py to release JVM resources cleanly.
    Should not be called between pipeline stages — stopping and restarting
    a session mid-pipeline is expensive and unnecessary.

    Calling this function when no session exists is a no-op (safe to call
    unconditionally in a finally block).

    Example
    -------
    >>> from src.utils.spark_session import get_spark_session, stop_spark_session
    >>> spark = get_spark_session()
    >>> # ... run pipeline stages ...
    >>> stop_spark_session()
    """
    global _session

    if _session is None:
        logger.debug("stop_spark_session() called but no active session found.")
        return

    logger.info("Stopping SparkSession.")
    _session.stop()
    _session = None
