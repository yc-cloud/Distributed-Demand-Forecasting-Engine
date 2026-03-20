"""
src/data_preparation/cleaner.py

Applies all cleaning transformations to the raw DataFrame produced by loader.py.
This is Stage 2 of the pipeline.  It is deliberately narrow in scope: rename
columns to snake_case, enforce one type cast that the CSV format could not
represent (boolean flag), and add one audit column that flags a known data
quality issue.  No encoding, no feature engineering, no row filtering.

Cleaning rules applied (and why each is justified)
---------------------------------------------------

1. Column rename to snake_case
   All 15 raw column names contain spaces or mixed case (e.g. "Store ID",
   "Weather Condition", "Holiday/Promotion").  Spark allows quoted access to
   such names but every downstream operation becomes error-prone: window
   specifications, groupBy keys, and feature engineering functions all work
   more reliably with simple identifiers.  snake_case is the Python/Spark
   convention and the name used in every stage after this one.

2. Cast `is_holiday_or_promo` from IntegerType to BooleanType
   The CSV represents this column as 0 or 1.  loader.py reads it as
   IntegerType — the correct representation of what is on disk.  Semantically
   the column is a flag: "was there a promotion or public holiday on this date?"
   Converting to BooleanType makes the intent explicit, prevents arithmetic
   misuse (e.g., accidentally summing it as a count), and eliminates the need
   for repeated `col("...") == 1` casts in every downstream module.

3. Add `is_forecast_anomaly` flag (demand_forecast < 0)
   The dataset ships with a pre-computed `Demand Forecast` column that contains
   negative values (minimum observed: −9.99).  Negative unit demand is
   physically impossible and is a data quality defect in the source system's
   forecasting model.

   Why flag rather than drop these rows:
     a) The ML model targets `units_sold`, not `demand_forecast`.  Dropping rows
        because the dataset's own baseline is wrong would silently shrink the
        training set without any justification in the target variable.
     b) `evaluator.py` benchmarks the ML model against this baseline.  Keeping
        flagged rows intact — while marking them — lets evaluator.py either
        report overall error including anomalies, or exclude them for a fair
        apples-to-apples comparison.  Both analyses are meaningful.
     c) Flagging is auditable: the pipeline produces a record of every anomalous
        row instead of silently discarding data.  This mirrors a production
        data-contract enforcement pattern where data engineers log violations
        without destroying the underlying record.

No other transformations are applied here.  Every other column is already the
correct type from loader.py's explicit schema, and there are no null values,
duplicates, or range violations to address in this dataset.

Input schema vs cleaned schema
-------------------------------
Input (from loader.py) — 15 columns, raw CSV names:

  Date               DateType
  Store ID           StringType
  Product ID         StringType
  Category           StringType
  Region             StringType
  Inventory Level    IntegerType
  Units Sold         IntegerType
  Units Ordered      IntegerType
  Demand Forecast    DoubleType
  Price              DoubleType
  Discount           IntegerType
  Weather Condition  StringType
  Holiday/Promotion  IntegerType     ← integer 0/1
  Competitor Pricing DoubleType
  Seasonality        StringType

Output (from clean()) — 16 columns, snake_case names:

  date               DateType
  store_id           StringType
  product_id         StringType
  category           StringType
  region             StringType
  inventory_level    IntegerType
  units_sold         IntegerType
  units_ordered      IntegerType
  demand_forecast    DoubleType
  is_forecast_anomaly BooleanType    ← NEW: demand_forecast < 0
  price              DoubleType
  discount           IntegerType
  weather_condition  StringType
  is_holiday_or_promo BooleanType   ← renamed + cast from Holiday/Promotion
  competitor_pricing DoubleType
  seasonality        StringType

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean

    spark  = get_spark_session()
    raw_df = load_raw_data(spark)
    clean_df = clean(raw_df)

    clean_df.printSchema()   # 16 columns, snake_case, is_holiday_or_promo is BooleanType
    clean_df.show(5)
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column rename mapping — raw CSV name → snake_case name.
#
# Defined at module level so it can be inspected or tested independently.
# Ordering follows the raw schema declaration in loader.py for traceability.
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP: dict[str, str] = {
    "Date":               "date",
    "Store ID":           "store_id",
    "Product ID":         "product_id",
    "Category":           "category",
    "Region":             "region",
    "Inventory Level":    "inventory_level",
    "Units Sold":         "units_sold",
    "Units Ordered":      "units_ordered",
    "Demand Forecast":    "demand_forecast",
    "Price":              "price",
    "Discount":           "discount",
    "Weather Condition":  "weather_condition",
    # The raw name contains a "/" which is invalid in most Spark SQL contexts.
    # Renaming to is_holiday_or_promo here; the cast to BooleanType follows.
    "Holiday/Promotion":  "is_holiday_or_promo",
    "Competitor Pricing": "competitor_pricing",
    "Seasonality":        "seasonality",
}

# ---------------------------------------------------------------------------
# Explicit column order for the cleaned DataFrame.
#
# Using a select() at the end of clean() enforces this order regardless of
# which Spark version reorders columns internally.  is_forecast_anomaly is
# placed immediately after demand_forecast because it is derived from it.
# ---------------------------------------------------------------------------
CLEANED_COLUMN_ORDER: list[str] = [
    "date",
    "store_id",
    "product_id",
    "category",
    "region",
    "inventory_level",
    "units_sold",
    "units_ordered",
    "demand_forecast",
    "is_forecast_anomaly",   # derived from demand_forecast; placed beside it
    "price",
    "discount",
    "weather_condition",
    "is_holiday_or_promo",   # renamed + cast from Holiday/Promotion
    "competitor_pricing",
    "seasonality",
]


# ---------------------------------------------------------------------------
# Private cleaning steps — one function per transformation.
#
# Keeping each step as its own function makes the pipeline in clean() easy
# to read, and makes each step independently unit-testable.
# ---------------------------------------------------------------------------

def _rename_columns(df: DataFrame) -> DataFrame:
    """
    Rename all raw CSV column names to their snake_case equivalents.

    Applies COLUMN_RENAME_MAP via withColumnRenamed so that only the names
    change — no data movement, no schema re-inference.  Columns not present
    in the map are left untouched (there are none in this pipeline, but the
    behaviour is safe by design).

    Parameters
    ----------
    df : DataFrame
        Raw DataFrame from loader.py with original CSV column names.

    Returns
    -------
    DataFrame
        Same data, all 15 columns renamed to snake_case.
    """
    for raw_name, clean_name in COLUMN_RENAME_MAP.items():
        df = df.withColumnRenamed(raw_name, clean_name)
    return df


def _cast_holiday_flag(df: DataFrame) -> DataFrame:
    """
    Cast `is_holiday_or_promo` from IntegerType (0/1) to BooleanType.

    Called after _rename_columns(), so the column is already named
    `is_holiday_or_promo` at this point.  The cast replaces the column
    in-place: 0 → False, 1 → True.

    Using BooleanType makes the semantic intent clear to any reader of the
    schema and prevents accidental integer arithmetic (e.g., SUM) on what
    is conceptually a yes/no flag.

    Parameters
    ----------
    df : DataFrame
        DataFrame with `is_holiday_or_promo` as IntegerType.

    Returns
    -------
    DataFrame
        Same DataFrame with `is_holiday_or_promo` as BooleanType.
    """
    return df.withColumn(
        "is_holiday_or_promo",
        F.col("is_holiday_or_promo").cast(BooleanType()),
    )


def _add_forecast_anomaly_flag(df: DataFrame) -> DataFrame:
    """
    Add boolean column `is_forecast_anomaly`: True where demand_forecast < 0.

    The dataset's pre-computed Demand Forecast column contains negative values
    (physically impossible unit forecasts).  This flag surfaces those rows for
    downstream use without removing them from the dataset.  See the module
    docstring for the full rationale on flagging vs dropping.

    Parameters
    ----------
    df : DataFrame
        DataFrame with `demand_forecast` as DoubleType.

    Returns
    -------
    DataFrame
        Input DataFrame with one additional BooleanType column appended.
    """
    return df.withColumn(
        "is_forecast_anomaly",
        F.col("demand_forecast") < 0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(df: DataFrame) -> DataFrame:
    """
    Apply all cleaning transformations to the raw DataFrame from loader.py.

    Transformation order:
      1. Rename all columns to snake_case (see COLUMN_RENAME_MAP).
      2. Cast `is_holiday_or_promo` from IntegerType to BooleanType.
      3. Add `is_forecast_anomaly` flag where demand_forecast < 0.
      4. Re-select columns in the canonical order defined by CLEANED_COLUMN_ORDER.

    No rows are added or removed.  The output row count equals the input row
    count exactly.

    Parameters
    ----------
    df : DataFrame
        Raw DataFrame produced by load_raw_data() — 15 columns, original CSV
        column names, Holiday/Promotion as IntegerType.

    Returns
    -------
    DataFrame
        Cleaned DataFrame with 16 columns in snake_case.  All original business
        data is preserved.  `is_holiday_or_promo` is BooleanType.
        `is_forecast_anomaly` is a new BooleanType audit column.

    Example
    -------
    >>> from src.utils.spark_session import get_spark_session
    >>> from src.data_preparation.loader import load_raw_data
    >>> from src.data_preparation.cleaner import clean
    >>> spark = get_spark_session()
    >>> clean_df = clean(load_raw_data(spark))
    >>> clean_df.printSchema()
    """
    logger.info(
        "Cleaning started | input_columns=%d | input_rows_approx=see_loader_log",
        len(df.columns),
    )

    df = _rename_columns(df)
    df = _cast_holiday_flag(df)
    df = _add_forecast_anomaly_flag(df)

    # Enforce canonical column order so all downstream modules can rely on a
    # fixed column position in addition to column names.
    df = df.select(CLEANED_COLUMN_ORDER)

    # Count anomalous rows once — the only aggregate computed in this stage.
    # This is a meaningful validation checkpoint: if the count is 0 the source
    # data may have changed; if it is unexpectedly high something is wrong.
    anomaly_count = df.filter(F.col("is_forecast_anomaly")).count()
    logger.info(
        "Cleaning complete | output_columns=%d | forecast_anomalies=%d",
        len(df.columns),
        anomaly_count,
    )

    return df
