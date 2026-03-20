"""
src/data_preparation/loader.py

Reads the raw retail_store_inventory.csv into a Spark DataFrame using an
explicitly declared schema.  This is the first stage of the pipeline; it
produces a DataFrame whose column names exactly match the CSV header.  No
renaming, type casting, or feature engineering is performed here — those
concerns belong to cleaner.py and the feature modules that follow.

Why explicit schema instead of inferSchema=True
------------------------------------------------
inferSchema=True requires Spark to make a full extra pass over the file just
to guess types.  On a 73k-row CSV that is a tolerable cost in development, but
the approach has structural problems that make it unsuitable for a pipeline:

  1. Correctness risk — inference is heuristic.  A column like "Holiday/Promotion"
     that contains only 0 and 1 could be inferred as IntegerType or BooleanType
     depending on the sample seen.  The Date column could be inferred as a plain
     string if the format is not one Spark recognises automatically.  Any
     downstream code that relies on a type the inferred schema gets wrong will
     fail later — often with a cryptic cast error, not an obvious schema error.

  2. Instability — if the CSV producer changes a column (e.g., adds a decimal to
     Discount), the inferred schema changes silently.  An explicit schema fails
     fast at read time with a clear message.

  3. Performance — the extra scan is unnecessary once the schema is known.  For
     large files (hundreds of millions of rows) this doubles read time with no
     benefit.

  4. Readability and documentation — the StructType definition below is the single
     authoritative source of truth for what the raw file contains.  A reviewer
     reading this module immediately knows all 15 columns, their types, and the
     nullability contract without opening the CSV or the DESIGN doc.

Column type rationale (all 15 columns)
---------------------------------------
  Date              DateType        Native date arithmetic.  Enables window
                                    functions, lag computation, and calendar
                                    feature extraction without a manual cast.
                                    dateFormat="yyyy-MM-dd" is passed to the
                                    reader to ensure exact parsing.

  Store ID          StringType      Identifier like "S001".  No arithmetic;
                                    StringType is the correct choice for all
                                    opaque codes that will be used as partition
                                    keys or join keys.

  Product ID        StringType      Same reasoning as Store ID ("P0001"–"P0020").

  Category          StringType      Five nominal labels (Clothing, Electronics,
                                    Furniture, Groceries, Toys).  Remains a
                                    string through loading; encoder.py will add
                                    an ordinal integer column alongside it.

  Region            StringType      Four nominal labels (East, North, South,
                                    West).  Same reasoning as Category.

  Inventory Level   IntegerType     Whole-number on-hand stock count (50–500).
                                    Integer arithmetic is sufficient; no
                                    fractional inventory exists in this dataset.

  Units Sold        IntegerType     Forecasting target — whole-number unit
                                    count (0–499).  Integer; the model predicts
                                    a DoubleType, not this raw column.

  Units Ordered     IntegerType     Replenishment quantity (20–200).  Whole
                                    number; used in business logic, not as a
                                    continuous feature.

  Demand Forecast   DoubleType      Pre-computed forecast from the dataset
                                    (−9.99–518.55).  Fractional values and
                                    negative values are present, so DoubleType
                                    is required.  cleaner.py will flag rows
                                    where this is negative.

  Price             DoubleType      Monetary value (10.00–100.00).  Always
                                    fractional; DoubleType preserves cents.

  Discount          IntegerType     Percentage discount (0–20).  Whole numbers
                                    only in this dataset; IntegerType is
                                    correct.  price_features.py uses it in
                                    arithmetic that produces a Double result,
                                    so the cast happens there, not here.

  Weather Condition StringType      Four nominal labels (Sunny, Cloudy, Rainy,
                                    Snowy).  Categorical string; encoder.py
                                    will add a weather_encoded integer column.

  Holiday/Promotion IntegerType     Binary flag stored as 0/1 in the CSV.
                                    Loaded as IntegerType to match the on-disk
                                    representation exactly.  cleaner.py casts
                                    it to BooleanType and renames it to
                                    is_holiday_or_promo.

  Competitor Pricing DoubleType     Fractional monetary value (5.03–104.94).
                                    Requires DoubleType for decimal precision.
                                    Used in price_to_competitor_ratio feature.

  Seasonality       StringType      Four nominal labels (Spring, Summer,
                                    Autumn, Winter).  Categorical string;
                                    encoder.py will add seasonality_encoded.

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data

    spark = get_spark_session()
    raw_df = load_raw_data(spark)

    raw_df.printSchema()
    raw_df.show(5)
"""

import logging
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Explicit schema — the single source of truth for the raw CSV structure.
#
# nullable=False is declared for every column because the dataset is known to
# be complete (no missing values).  If nullable=False is violated at read
# time Spark surfaces a clear error rather than silently inserting nulls that
# corrupt downstream aggregations or window functions.
# ---------------------------------------------------------------------------
RAW_SCHEMA = StructType(
    [
        # ── Identity / key columns ──────────────────────────────────────────
        StructField("Date",               DateType(),    nullable=False),
        StructField("Store ID",           StringType(),  nullable=False),
        StructField("Product ID",         StringType(),  nullable=False),
        # ── Categorical descriptors ─────────────────────────────────────────
        StructField("Category",           StringType(),  nullable=False),
        StructField("Region",             StringType(),  nullable=False),
        # ── Core business metrics ───────────────────────────────────────────
        StructField("Inventory Level",    IntegerType(), nullable=False),
        StructField("Units Sold",         IntegerType(), nullable=False),  # forecast target
        StructField("Units Ordered",      IntegerType(), nullable=False),
        # ── Forecast baseline and pricing ───────────────────────────────────
        StructField("Demand Forecast",    DoubleType(),  nullable=False),
        StructField("Price",              DoubleType(),  nullable=False),
        StructField("Discount",           IntegerType(), nullable=False),
        # ── Exogenous / contextual features ─────────────────────────────────
        StructField("Weather Condition",  StringType(),  nullable=False),
        StructField("Holiday/Promotion",  IntegerType(), nullable=False),
        StructField("Competitor Pricing", DoubleType(),  nullable=False),
        StructField("Seasonality",        StringType(),  nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw_file_path(config_path: Path) -> str:
    """
    Read config.yaml and return the raw CSV file path string.

    Kept private because callers only need load_raw_data().  Separating this
    out makes it easy to unit-test config loading without spinning up Spark.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml, absolute or relative to the working dir.

    Returns
    -------
    str
        The value of paths.raw_data_file from the config.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist.
    KeyError
        If the expected keys are absent from the config.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config["paths"]["raw_data_file"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(
    spark: SparkSession,
    config_path: str = "config/config.yaml",
) -> DataFrame:
    """
    Read the raw CSV into a Spark DataFrame using the declared RAW_SCHEMA.

    The returned DataFrame preserves the original column names from the CSV
    header (e.g. "Store ID", "Units Sold") so that cleaner.py can do all
    renaming in one place.  No columns are added, dropped, or modified here.

    The CSV reader is configured with:
      - header=True    — first row contains column names
      - schema         — RAW_SCHEMA declared above; inferSchema is never used
      - dateFormat     — "yyyy-MM-dd" matches the dataset's date representation
      - mode=FAILFAST  — raise immediately on any row that violates the schema
                         rather than silently replacing bad values with nulls

    Parameters
    ----------
    spark : SparkSession
        An active SparkSession, typically from get_spark_session().
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location
        so pipeline.py callers need not pass an argument.

    Returns
    -------
    pyspark.sql.DataFrame
        Raw DataFrame with 15 columns and original CSV column names.
        Row count matches the source CSV exactly (no filtering applied).

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    pyspark.sql.utils.AnalysisException
        If the CSV file does not exist at the path specified in config.yaml.
    pyspark.sql.utils.SparkException
        If mode=FAILFAST is triggered by a row that cannot be parsed against
        RAW_SCHEMA (e.g., a null in a non-nullable column, a malformed date).

    Example
    -------
    >>> from src.utils.spark_session import get_spark_session
    >>> from src.data_preparation.loader import load_raw_data
    >>> spark = get_spark_session()
    >>> df = load_raw_data(spark)
    >>> df.printSchema()
    >>> df.show(5)
    """
    raw_file_path = _load_raw_file_path(Path(config_path))

    logger.info("Loading raw CSV | path=%s | schema=%d columns", raw_file_path, len(RAW_SCHEMA))

    df: DataFrame = (
        spark.read.format("csv")
        .option("header", True)
        # Explicit schema — no inferSchema; see module docstring for rationale
        .schema(RAW_SCHEMA)
        # Date format must match the source file exactly to avoid silent nulls
        .option("dateFormat", "yyyy-MM-dd")
        # Fail immediately on any unparseable row rather than coercing to null.
        # PERMISSIVE (the default) would silently corrupt the dataset if the
        # CSV changes; FAILFAST surfaces the problem at the earliest possible
        # point in the pipeline.
        .option("mode", "FAILFAST")
        .load(raw_file_path)
    )

    logger.info(
        "Raw DataFrame loaded | rows=%d | columns=%s",
        df.count(),
        df.columns,
    )

    return df
