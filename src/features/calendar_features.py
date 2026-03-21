"""
src/features/calendar_features.py

Appends calendar-derived features extracted from the `date` column.
This is the third of four feature-engineering modules (lag → rolling →
calendar → price).

Unlike the lag and rolling modules, calendar features require no Window
functions and no config file — every column is a deterministic, row-level
transformation of the existing `date` (DateType) column using Spark's
built-in date functions.  There is no aggregation, no inter-row dependency,
and no state carried between rows.

Why calendar features are useful
----------------------------------
Retail demand has strong periodic structure.  A model that sees only
`units_sold` history has no direct way to know that Fridays are heavier
shopping days, that December is a peak month, or that Q4 is categorically
different from Q1.  Calendar features make that structure explicit and
learnable:

  - Day-of-week effects are among the most reliable patterns in retail.
    Convenience items peak mid-week; big-ticket purchases concentrate on
    weekends.

  - Month and quarter encode seasonality that repeats annually.  Toy demand
    spikes in November–December; summer clothing peaks in June–July.

  - Week-of-year gives finer seasonal granularity than month alone.  It
    captures effects like "Back to School" (weeks 32–35) or "Black Friday"
    (week 47) that month boundaries would straddle.

  - is_weekend provides a binary signal that is immediately interpretable and
    often sufficient to capture the weekend vs weekday demand split without
    requiring the model to rediscover it from day_of_week alone.

XGBoost treats each calendar column as an independent numeric feature and
learns non-linear thresholds across them.  Even though month is cyclical
(December wraps to January), tree-based models handle this well because they
can create a split at "month >= 11" (holiday season) without assuming any
linear relationship.

Explanation of each derived feature
--------------------------------------

day_of_week     IntegerType, 0 = Monday … 6 = Sunday (Python / ISO 8601
                convention).  Spark's F.dayofweek() uses the SQL convention
                (1 = Sunday … 7 = Saturday), so the expression
                (F.dayofweek("date") + 5) % 7 converts to the ISO convention:

                    SQL → ISO
                    1 (Sun) → 6
                    2 (Mon) → 0
                    3 (Tue) → 1
                    4 (Wed) → 2
                    5 (Thu) → 3
                    6 (Fri) → 4
                    7 (Sat) → 5

                This mapping is verified in the inline comments below.

month           IntegerType, 1 – 12.  Encodes the month of year directly.
                January = 1, December = 12.  Captures annual seasonality at
                monthly granularity.

week_of_year    IntegerType, 1 – 53.  ISO 8601 week number.  ISO weeks start
                on Monday; the first week of the year is the week containing
                the first Thursday.  A small number of years have 53 weeks.
                Finer than month for capturing promotional events that fall
                in a specific week (e.g., Black Friday in week ~47).

quarter         IntegerType, 1 – 4.  Q1 = Jan–Mar, Q4 = Oct–Dec.  Retail
                demand often differs substantially between Q4 (holiday season)
                and Q1 (post-holiday slowdown).  Coarser than month but more
                stable across years.

day_of_month    IntegerType, 1 – 31.  The calendar day within the month.
                Captures within-month patterns: payday effects (end-of-month
                spending), early-month replenishment, and mid-month lulls.

is_weekend      BooleanType.  True when day_of_week is Saturday (5) or
                Sunday (6) — i.e., day_of_week >= 5.  Derived from the
                already-computed day_of_week column rather than from `date`
                again, making the dependency explicit and readable.

How is_weekend is defined
---------------------------
A weekend is Saturday or Sunday.  Using the ISO day_of_week encoding
(Mon=0 … Sun=6), weekends correspond to day_of_week values 5 and 6.
The expression `F.col("day_of_week") >= 5` captures both in a single
comparison.  This is equivalent to `day_of_week.isin(5, 6)` but more
concise and marginally faster (one comparison vs two).

The weekend / weekday distinction is kept as a separate boolean rather than
relying on the model to discover the threshold in day_of_week, because:
  a) It is an explicit business concept used in operations and reporting.
  b) It provides the model with a clean, directly interpretable feature
     alongside the more granular day_of_week.

Input schema vs output schema
-------------------------------
Input (from add_rolling_features()) — 30 columns:

  date, store_id, product_id, category, region,
  inventory_level, units_sold, units_ordered, demand_forecast,
  is_forecast_anomaly, price, discount, weather_condition,
  is_holiday_or_promo, competitor_pricing, seasonality,
  category_encoded, region_encoded, weather_encoded, seasonality_encoded,
  lag_1, lag_7, lag_14, lag_28,
  rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_std_14,
  rolling_mean_28, rolling_std_28

Output — 36 columns (30 input + 6 calendar columns appended):

  … all 30 input columns unchanged …
  day_of_week     IntegerType   0 = Mon … 6 = Sun
  month           IntegerType   1 – 12
  week_of_year    IntegerType   1 – 53
  quarter         IntegerType   1 – 4
  day_of_month    IntegerType   1 – 31
  is_weekend      BooleanType   True for Saturday (5) or Sunday (6)

Usage from another module
--------------------------
    from src.utils.spark_session import get_spark_session
    from src.data_preparation.loader import load_raw_data
    from src.data_preparation.cleaner import clean
    from src.data_preparation.encoder import encode
    from src.features.lag_features import add_lag_features
    from src.features.rolling_features import add_rolling_features
    from src.features.calendar_features import add_calendar_features

    spark = get_spark_session()
    df = add_calendar_features(
             add_rolling_features(
                 add_lag_features(
                     encode(clean(load_raw_data(spark))))))

    df.select("date", "day_of_week", "month",
              "week_of_year", "is_weekend").show(5)
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

# The single source column for all calendar derivations.
_DATE_COL: str = "date"


def add_calendar_features(df: DataFrame) -> DataFrame:
    """
    Append six calendar features derived from the `date` column.

    All six transformations are stateless row-level expressions — no Window
    functions, no aggregation, no config dependency.  Each withColumn call is
    lazy; Spark plans all six together in a single pass over the data.

    Transformation details
    ----------------------
    day_of_week
        (F.dayofweek("date") + 5) % 7
        Converts Spark's SQL convention (1=Sun … 7=Sat) to the ISO / Python
        convention (0=Mon … 6=Sun).  The offset +5 followed by mod 7 performs
        the circular shift without any conditional logic.

    month
        F.month("date")
        Returns the month number directly (1–12).  No conversion needed.

    week_of_year
        F.weekofyear("date")
        Returns ISO 8601 week number (1–53).  ISO weeks start on Monday.

    quarter
        F.quarter("date")
        Returns 1–4 directly.  No conversion needed.

    day_of_month
        F.dayofmonth("date")
        Returns 1–31 directly.  No conversion needed.

    is_weekend
        F.col("day_of_week") >= 5
        References the day_of_week column added earlier in this function.
        Sat=5, Sun=6 — both satisfy >= 5.  Using a single comparison rather
        than isin(5, 6) is equivalent and marginally more readable.

    Parameters
    ----------
    df : DataFrame
        Rolling-featured DataFrame from add_rolling_features() — 30 columns.

    Returns
    -------
    DataFrame
        Input DataFrame with 6 calendar columns appended.
        Row count is unchanged.  No nulls are introduced (date is non-nullable
        in the declared schema and all Spark date functions are null-safe).

    Example
    -------
    >>> df = add_calendar_features(add_rolling_features(add_lag_features(...)))
    >>> df.select("date", "day_of_week", "is_weekend").show(3)
    +----------+-----------+----------+
    |      date|day_of_week|is_weekend|
    +----------+-----------+----------+
    |2020-01-06|          0|     false|   ← Monday
    |2020-01-11|          5|      true|   ← Saturday
    |2020-01-12|          6|      true|   ← Sunday
    +----------+-----------+----------+
    """
    logger.info(
        "Adding calendar features | source_col=%s | input_columns=%d",
        _DATE_COL,
        len(df.columns),
    )

    # ── day_of_week ──────────────────────────────────────────────────────────
    # Spark's dayofweek(): 1=Sun, 2=Mon, 3=Tue, 4=Wed, 5=Thu, 6=Fri, 7=Sat
    # Target ISO convention: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    # Shift: (spark_value + 5) % 7
    #   Sun: (1+5)%7=6 ✓   Mon: (2+5)%7=0 ✓   Sat: (7+5)%7=5 ✓
    df = df.withColumn(
        "day_of_week",
        ((F.dayofweek(F.col(_DATE_COL)) + 5) % 7).cast("int"),
    )

    # ── month ─────────────────────────────────────────────────────────────────
    # 1 = January … 12 = December.  Direct Spark function, no conversion.
    df = df.withColumn(
        "month",
        F.month(F.col(_DATE_COL)),
    )

    # ── week_of_year ──────────────────────────────────────────────────────────
    # ISO 8601 week number: 1–53.  Weeks start on Monday.
    # The first week of a year is the one containing the year's first Thursday.
    df = df.withColumn(
        "week_of_year",
        F.weekofyear(F.col(_DATE_COL)),
    )

    # ── quarter ───────────────────────────────────────────────────────────────
    # 1 = Q1 (Jan–Mar) … 4 = Q4 (Oct–Dec).  Direct Spark function.
    df = df.withColumn(
        "quarter",
        F.quarter(F.col(_DATE_COL)),
    )

    # ── day_of_month ──────────────────────────────────────────────────────────
    # 1–31.  Captures within-month patterns (payday, month-end replenishment).
    df = df.withColumn(
        "day_of_month",
        F.dayofmonth(F.col(_DATE_COL)),
    )

    # ── is_weekend ────────────────────────────────────────────────────────────
    # Derived from day_of_week (already appended above).
    # Sat=5, Sun=6  →  day_of_week >= 5  →  True on weekends.
    df = df.withColumn(
        "is_weekend",
        F.col("day_of_week") >= 5,
    )

    logger.info(
        "Calendar features complete | columns_added=%s | total_columns=%d",
        ["day_of_week", "month", "week_of_year", "quarter",
         "day_of_month", "is_weekend"],
        len(df.columns),
    )

    return df
