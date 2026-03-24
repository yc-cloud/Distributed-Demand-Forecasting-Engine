"""
src/data_preparation/encoder.py

Adds integer label-encoded columns for the four nominal categorical variables.
This is Stage 3 of the pipeline.  It appends one `_encoded` column per
categorical column and preserves the original string columns so that
downstream reporting and groupBy operations remain human-readable.

Encoding strategy: config-driven integer label encoding
--------------------------------------------------------
Four columns require encoding before the data can be used in a numeric model:
`category`, `region`, `weather_condition`, and `seasonality`.

The chosen strategy maps each string value to a fixed integer index (0, 1, 2 …)
defined by the ordered lists in config.yaml's `encoding` section.  For example:

    category:  Clothing=0, Electronics=1, Furniture=2, Groceries=3, Toys=4
    region:    East=0, North=1, South=2, West=3
    weather:   Sunny=0, Cloudy=1, Rainy=2, Snowy=3
    seasonality: Spring=0, Summer=1, Autumn=2, Winter=3

Why NOT the Spark ML StringIndexer → OneHotEncoder pipeline
------------------------------------------------------------
The config.yaml comments reference StringIndexer/OneHotEncoder, which is the
natural Spark ML approach.  However, that pipeline produces a `Vector` column
(a sparse binary vector), not individual numeric columns.  This creates a
practical problem:

  - trainer.py converts the Spark DataFrame to a pandas DataFrame and trains
    XGBoost via the sklearn API.  Spark Vector columns do not convert to flat
    numeric columns automatically; they produce a single object-typed column
    holding a list.  Expanding them back to individual columns (using
    VectorSlicer or manual UDFs) adds substantial complexity with no gain.

  - XGBoost and other tree-based models do not require one-hot encoding.
    Trees split on `feature < threshold`.  On an integer-encoded nominal
    column (e.g., category ∈ {0, 1, 2, 3, 4}), the tree can isolate
    individual categories with at most ⌈log₂(k)⌉ splits.  The false ordinal
    relationship implied by the integers does not bias a tree the way it
    biases a linear model, because trees never perform arithmetic on feature
    values — only comparisons.

  - Linear models (logistic regression, SVM) would require true OHE because
    they treat integer encodings as continuous, implying Clothing < Electronics
    < Furniture which is semantically meaningless.  XGBoost does not have this
    problem.

Why these four columns are nominal, not ordinal
-----------------------------------------------
  category        Clothing / Electronics / Furniture / Groceries / Toys.
                  No natural ranking exists between product categories.
                  Assigning integers 0–4 imposes an artificial order that has
                  no business meaning, but trees ignore that ordering.

  region          East / North / South / West.  Geographic compass directions
                  are not rankable.  "West > North" is meaningless.

  weather_condition  Sunny / Cloudy / Rainy / Snowy.  One might argue a
                  severity gradient (Sunny→Snowy), but that is context-
                  dependent (Snowy is not always worse than Rainy for demand),
                  so treating it as ordinal would encode an unjustified
                  assumption.  Nominal is the safe, defensible choice.

  seasonality     Spring / Summer / Autumn / Winter.  These are cyclical, not
                  linear.  Winter is not "greater than" Autumn; it wraps back
                  to Spring.  True ordinal encoding would be misleading;
                  cyclical encoding (sin/cos transforms) is theoretically
                  correct but adds complexity unnecessary for a tree model.
                  Simple label encoding is a practical, defensible default.

How the encoded columns are used downstream
-------------------------------------------
  trainer.py      Selects numeric columns (including `*_encoded`) to build the
                  feature matrix.  Keeps the original strings for later joins.

  evaluator.py    Groups results by `category` and `region` (the original
                  strings) for per-category error reporting.  The encoded
                  columns are not needed there.

  recommender.py  Groups by `store_id` and `product_id`; uses `category` and
                  `region` strings for output readability.

Stability guarantee
-------------------
The integer indices come from config.yaml, not from the frequency order of
values observed in the data (which is what StringIndexer does by default).
This means the mapping is identical across every run, even if the data
distribution shifts.  A new run with different row counts will never silently
reassign Clothing from 0 to 2.

Input schema vs encoded schema
-------------------------------
Input (from cleaner.py) — 16 columns, snake_case:

  date                DateType
  store_id            StringType
  product_id          StringType
  category            StringType   ← to be encoded
  region              StringType   ← to be encoded
  inventory_level     IntegerType
  units_sold          IntegerType
  units_ordered       IntegerType
  demand_forecast     DoubleType
  is_forecast_anomaly BooleanType
  price               DoubleType
  discount            IntegerType
  weather_condition   StringType   ← to be encoded
  is_holiday_or_promo BooleanType
  competitor_pricing  DoubleType
  seasonality         StringType   ← to be encoded

Output (from encode()) — 20 columns (16 original + 4 encoded):

  … all 16 input columns unchanged …
  category_encoded    IntegerType  Clothing=0 … Toys=4
  region_encoded      IntegerType  East=0 … West=3
  weather_encoded     IntegerType  Sunny=0 … Snowy=3
  seasonality_encoded IntegerType  Spring=0 … Winter=3

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean
    from src.data_preparation.encoder import encode

    spark    = get_spark_session()
    raw_df   = load_raw_data(spark)
    clean_df = clean(raw_df)
    enc_df   = encode(clean_df)

    enc_df.select("category", "category_encoded",
                  "weather_condition", "weather_encoded").show(5)
"""

import logging
from pathlib import Path

import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encoding targets
#
# Each tuple: (config_key, source_column, output_column).
#   config_key    — key under `encoding` in config.yaml whose `values` list
#                   defines the fixed string→integer mapping.
#   source_column — the snake_case column name in the cleaned DataFrame.
#   output_column — the new integer column appended by encode().
#
# `weather_condition` maps to `weather_encoded` (not `weather_condition_encoded`)
# to match the column names specified in DESIGN.md Stage 2 schema.
# ---------------------------------------------------------------------------
ENCODING_TARGETS: list[tuple[str, str, str]] = [
    ("category",          "category",          "category_encoded"),
    ("region",            "region",            "region_encoded"),
    ("weather_condition", "weather_condition", "weather_encoded"),
    ("seasonality",       "seasonality",       "seasonality_encoded"),
]


# ---------------------------------------------------------------------------
# Dynamic Encoding targets
#
# These columns will be encoded dynamically based on their distinct values
# in the DataFrame. This is suitable for high-cardinality IDs like store_id
# and product_id where pre-defining all values in config.yaml is impractical.
# ---------------------------------------------------------------------------
DYNAMIC_ENCODING_TARGETS: list[tuple[str, str]] = [
    ("store_id", "store_id_encoded"),
    ("product_id", "product_id_encoded"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_encoding_config(config_path: Path) -> dict[str, list[str]]:
    """
    Load and return the `encoding` section from config.yaml.

    Returns a dict mapping each config key to its ordered list of string
    values, e.g. {"category": ["Clothing", "Electronics", ...], ...}.

    Parameters
    ----------
    config_path : Path
        Path to config/config.yaml.

    Returns
    -------
    dict[str, list[str]]
        Mapping of config key → ordered list of category values.

    Raises
    ------
    FileNotFoundError
        If config.yaml does not exist.
    KeyError
        If the `encoding` section is absent.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}. "
            "Ensure config/config.yaml exists at the project root."
        )

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if "encoding" not in full_config:
        raise KeyError(
            f"'encoding' section missing from {config_path}. "
            "Check that config/config.yaml matches the expected schema."
        )

    # Extract only the `values` list from each group.
    return {
        key: group["values"]
        for key, group in full_config["encoding"].items()
    }


def _make_label_encoder(column_name: str, values: list[str]) -> "Column":
    """
    Build a Spark Column expression that maps string values to integer indices.

    Uses `create_map` to build a Spark SQL map literal — a key/value store
    embedded directly in the query plan.  The map is then looked up against
    the source column with `getItem`.

    Example for column_name="category", values=["Clothing", "Electronics"]:
        create_map("Clothing", 0, "Electronics", 1).getItem(col("category"))
        → 0 for "Clothing", 1 for "Electronics", null for any unknown value

    Returning null for unknown values (rather than raising) is intentional:
    schema_validator.py (a later module) is responsible for detecting unknown
    category values before this stage runs.  Silently returning null here keeps
    this function's responsibility narrow.

    Parameters
    ----------
    column_name : str
        Name of the source column in the DataFrame (e.g. "category").
    values : list[str]
        Ordered list of valid string values from config.yaml.  The index of
        each value in this list becomes its integer encoding.

    Returns
    -------
    pyspark.sql.Column
        A Column expression of IntegerType.
    """
    # Flatten [(value, index), ...] into [lit(value), lit(index), ...]
    # as required by F.create_map(*args).
    flat_pairs = [
        item
        for idx, value in enumerate(values)
        for item in (F.lit(value), F.lit(idx))
    ]
    return F.create_map(*flat_pairs).getItem(F.col(column_name)).cast(IntegerType())


def _make_dynamic_label_encoder(column_name: str) -> "Column":
    """
    Build a Spark Column expression that maps string values to integer indices
    dynamically based on their occurrence in the DataFrame.

    Uses dense_rank() over a window partitioned by the column to be encoded.
    This assigns a unique, consecutive integer ID to each distinct value.

    Parameters
    ----------
    column_name : str
        Name of the source column in the DataFrame (e.g. "store_id").

    Returns
    -------
    pyspark.sql.Column
        A Column expression of IntegerType.
    """
    window_spec = Window.orderBy(column_name)
    return F.dense_rank().over(window_spec) - 1 # -1 to make it 0-indexed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode(df: DataFrame, config_path: str = "config/config.yaml") -> DataFrame:
    """
    Append integer label-encoded columns for all four nominal categoricals.

    Reads the fixed value ordering from config.yaml so that the integer
    mapping is identical on every pipeline run regardless of row distribution.
    The four original string columns are preserved unchanged.

    Transformation applied per target (see ENCODING_TARGETS):
      category          → category_encoded    (IntegerType, 0–4)
      region            → region_encoded      (IntegerType, 0–3)
      weather_condition → weather_encoded     (IntegerType, 0–3)
      seasonality       → seasonality_encoded (IntegerType, 0–3)

    Parameters
    ----------
    df : DataFrame
        Cleaned DataFrame from clean() — 16 columns, snake_case.
    config_path : str
        Path to config/config.yaml.  Defaults to the project root location.

    Returns
    -------
    DataFrame
        Input DataFrame with 4 additional IntegerType columns appended.
        Row count is unchanged.  All 16 input columns are preserved.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    KeyError
        If the `encoding` section or a required config key is absent.

    Example
    -------
    >>> enc_df = encode(clean_df)
    >>> enc_df.select("category", "category_encoded").distinct().show()
    +-----------+----------------+
    |   category|category_encoded|
    +-----------+----------------+
    |   Clothing|               0|
    |Electronics|               1|
    |  Furniture|               2|
    |  Groceries|               3|
    |       Toys|               4|
    +-----------+----------------+
    """
    encoding_cfg = _load_encoding_config(Path(config_path))

    all_encoded_cols = [output_col for _, _, output_col in ENCODING_TARGETS] + \
                       [output_col for _, output_col in DYNAMIC_ENCODING_TARGETS]
    logger.info(
        "Encoding started | input_columns=%d | config_targets=%s | dynamic_targets=%s",
        len(df.columns),
        [output_col for _, _, output_col in ENCODING_TARGETS],
        [output_col for _, output_col in DYNAMIC_ENCODING_TARGETS],
    )

    for config_key, source_col, output_col in ENCODING_TARGETS:
        if config_key not in encoding_cfg:
            raise KeyError(
                f"Encoding config key '{config_key}' not found in {config_path}. "
                f"Expected keys: {list(encoding_cfg.keys())}"
            )

        values = encoding_cfg[config_key]
        encoder_expr = _make_label_encoder(source_col, values)
        df = df.withColumn(output_col, encoder_expr)

        logger.debug(
            "Encoded '%s' → '%s' (config-driven) | mapping=%s",
            source_col,
            output_col,
            {v: i for i, v in enumerate(values)},
        )

    for source_col, output_col in DYNAMIC_ENCODING_TARGETS:
        encoder_expr = _make_dynamic_label_encoder(source_col)
        df = df.withColumn(output_col, encoder_expr)

        logger.debug(
            "Encoded '%s' → '%s' (dynamic)",
            source_col,
            output_col,
        )

    logger.info(
        "Encoding complete | output_columns=%d | encoded_columns=%s",
        len(df.columns),
        all_encoded_cols,
    )

    return df
