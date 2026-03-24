"""
src/features/price_features.py

Appends three price-derived features built from the `price`, `discount`, and
`competitor_pricing` columns.  This is the fourth and final feature-engineering
module (lag → rolling → calendar → price).

Like calendar_features.py, every column here is a stateless row-level
expression — no Window functions, no aggregation, no config dependency.
Spark evaluates all three withColumn calls in a single pass.

Why price-related features are useful
---------------------------------------
Price is one of the strongest levers in retail demand.  Raw `price` alone is
a weak signal because:

  - A price of $50 means very different things for a toy vs. a piece of
    furniture.  The absolute value is hard to interpret without context.
  - The presence of a discount changes consumer behaviour independently of the
    final price paid: promotional framing drives purchases even when the
    effective price equals the usual price elsewhere.
  - Competitive context matters: $50 is attractive if competitors charge $70
    but unattractive if competitors charge $40.

The three derived features each capture a distinct dimension of pricing:

  effective_price            the actual price the consumer pays after discount
  price_to_competitor_ratio  how this store's price compares to competitors
  has_discount               whether any promotion is active (binary signal)

Explanation of each derived feature
--------------------------------------

effective_price
    Formula:  price * (1 - discount / 100)
    Type:     DoubleType
    Range:    $10.00 – $100.00 (discount shrinks from 0% to 20%)

    `price` is the listed price; `discount` is an integer percentage (0–20).
    A 20% discount on a $100 item gives effective_price = $80.  This is the
    figure that directly influences the consumer's purchase decision — the
    model should see what was actually paid, not just the sticker price.

    `discount / 100` uses Spark's floating-point column division (not integer
    division), so the result is always a DoubleType fraction (e.g., 20 / 100
    = 0.20).

price_to_competitor_ratio
    Formula:  price / competitor_pricing
    Type:     DoubleType
    Expected range: ~0.10 – ~19.9  (price 10.00 / max_comp 104.94  to
                                     price 100.00 / min_comp 5.03)

    Values < 1.0  → this store is cheaper than the competitor (demand ↑)
    Values = 1.0  → price parity
    Values > 1.0  → this store is more expensive (demand ↓)

    This ratio is a domain-driven feature: competitive pricing is a known
    demand driver in retail.  It also makes the price signal scale-invariant
    — a $50 item at a $100 competitor price (ratio 0.5) is equivalent to a
    $10 item at a $20 competitor price (also ratio 0.5), even though the
    absolute prices differ by 5×.

    Division-by-zero guard: `competitor_pricing` ranges from 5.03 to 104.94
    in this dataset, so zero is impossible in practice.  A defensive
    F.when(...).otherwise(null) guard is included anyway so the function
    behaves correctly if the source data ever changes.  Returning null (rather
    than infinity or zero) is the correct choice: it signals missing information
    rather than a false numeric value, and XGBoost handles null natively.

has_discount
    Formula:  discount > 0
    Type:     BooleanType

    `discount` is an integer in the range 0–20.  has_discount = True whenever
    any discount is applied, regardless of its magnitude.

    A dedicated boolean captures the promotional framing effect separately from
    the magnitude captured in effective_price.  In retail, even a 1% discount
    can trigger promotional behaviour (the "on sale" label drives traffic).
    The model can learn this binary threshold directly rather than having to
    discover from `effective_price` that "prices slightly below the list price
    behave differently".

Input schema vs output schema
-------------------------------
Input (from add_calendar_features()) — 36 columns:

  date, store_id, product_id, category, region,
  inventory_level, units_sold, units_ordered, demand_forecast,
  is_forecast_anomaly, price, discount, weather_condition,
  is_holiday_or_promo, competitor_pricing, seasonality,
  category_encoded, region_encoded, weather_encoded, seasonality_encoded,
  lag_1, lag_7, lag_14, lag_28,
  rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_std_14,
  rolling_mean_28, rolling_std_28,
  day_of_week, month, week_of_year, quarter, day_of_month, is_weekend

Output — 39 columns (36 input + 3 price-derived columns appended):

  … all 36 input columns unchanged …
  effective_price            DoubleType   price * (1 − discount / 100)
  price_to_competitor_ratio  DoubleType   price / competitor_pricing
                                          (null if competitor_pricing = 0)
  has_discount               BooleanType  True when discount > 0

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean
    from src.data_preparation.encoder import encode
    from src.features.lag_features import add_lag_features
    from src.features.rolling_features import add_rolling_features
    from src.features.calendar_features import add_calendar_features
    from src.features.price_features import add_price_features

    spark = get_spark_session()
    df = add_price_features(
             add_calendar_features(
                 add_rolling_features(
                     add_lag_features(
                         encode(clean(load_raw_data(spark)))))))

    df.select("price", "discount", "competitor_pricing",
              "effective_price", "price_to_competitor_ratio",
              "has_discount").show(5)
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source column names referenced by all three feature expressions.
# Defined at module level so tests can import them without calling the function.
# ---------------------------------------------------------------------------
_PRICE_COL: str = "price"
_DISCOUNT_COL: str = "discount"
_COMPETITOR_PRICING_COL: str = "competitor_pricing"


def add_price_features(df: DataFrame) -> DataFrame:
    """
    Append effective_price, price_to_competitor_ratio, and has_discount.

    All three are stateless row-level expressions derived from `price`,
    `discount`, and `competitor_pricing`.  No Window functions, no config
    file, and no inter-row state are required.

    Parameters
    ----------
    df : DataFrame
        Calendar-featured DataFrame from add_calendar_features() — 36 columns.

    Returns
    -------
    DataFrame
        Input DataFrame with 3 price feature columns appended.
        Row count is unchanged.  No nulls are introduced except in
        price_to_competitor_ratio when competitor_pricing = 0 (which cannot
        occur in this dataset but is guarded defensively).

    Example
    -------
    >>> df = add_price_features(add_calendar_features(...))
    >>> df.select("price", "discount", "effective_price",
    ...           "price_to_competitor_ratio", "has_discount").show(3)
    +-----+--------+---------------+-------------------------+-----------+
    |price|discount|effective_price|price_to_competitor_ratio|has_discount|
    +-----+--------+---------------+-------------------------+-----------+
    | 50.0|      10|           45.0|                     0.83|       true|
    | 80.0|       0|           80.0|                     1.14|      false|
    | 30.0|      20|           24.0|                     0.50|       true|
    +-----+--------+---------------+-------------------------+-----------+
    """
    logger.info(
        "Adding price features | sources=[%s, %s, %s] | input_columns=%d",
        _PRICE_COL,
        _DISCOUNT_COL,
        _COMPETITOR_PRICING_COL,
        len(df.columns),
    )

    # ── effective_price ───────────────────────────────────────────────────────
    # price * (1 - discount / 100)
    # discount is IntegerType; Spark's column division always produces
    # DoubleType, so no explicit cast is needed.  The result is the actual
    # price paid by the consumer after the percentage discount is applied.
    df = df.withColumn(
        "effective_price",
        F.col(_PRICE_COL) * (F.lit(1) - F.col(_DISCOUNT_COL) / F.lit(100)),
    )

    # ── price_to_competitor_ratio ─────────────────────────────────────────────
    # price / competitor_pricing
    # Defensive guard: return null when competitor_pricing = 0 rather than
    # producing infinity (which XGBoost cannot handle) or silently crashing.
    # In this dataset competitor_pricing ranges from 5.03 to 104.94, so the
    # guard will never trigger — but it costs nothing and keeps the function
    # safe if the source data ever changes.
    df = df.withColumn(
        "price_to_competitor_ratio",
        F.when(
            F.col(_COMPETITOR_PRICING_COL) != 0,
            F.col(_PRICE_COL) / F.col(_COMPETITOR_PRICING_COL),
        ).otherwise(F.lit(None).cast("double")),
    )

    # ── has_discount ──────────────────────────────────────────────────────────
    # True whenever any discount is applied (discount > 0).
    # Captures the promotional framing effect as a binary signal, separate
    # from the magnitude already encoded in effective_price.
    df = df.withColumn(
        "has_discount",
        F.col(_DISCOUNT_COL) > 0,
    )

    logger.info(
        "Price features complete | columns_added=%s | total_columns=%d",
        ["effective_price", "price_to_competitor_ratio", "has_discount"],
        len(df.columns),
    )

    return df
