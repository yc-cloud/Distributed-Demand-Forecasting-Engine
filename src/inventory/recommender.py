"""
src/inventory/recommender.py

Translates per-row demand forecasts from predictor.py into actionable per-SKU
inventory recommendations.  One output row per (store_id, product_id) pair
summarises how much stock is needed, whether a replenishment order should be
placed today, and how many days of supply remain on hand.

All inventory policy parameters (lead time, order cycle, service level) are
read from config/config.yaml so that business inputs can be tuned without
touching code.

─────────────────────────────────────────────────────────────────────────────
What each inventory metric means
─────────────────────────────────────────────────────────────────────────────

avg_daily_demand  (D̄)
    The mean of predicted_units_sold across all forecast rows for a
    (store_id, product_id) pair.  This is the expected number of units sold
    on a typical day.  It drives every downstream formula as the central
    estimate of demand rate.

demand_std_dev  (σ)
    The standard deviation of predicted_units_sold across the same rows.
    Measures how volatile demand is for this SKU.  A σ of 0 means demand is
    perfectly flat; a high σ means the SKU has spiky, unpredictable demand
    that requires a larger safety buffer.

current_inventory_level
    The inventory_level observed on the most recent date in the dataset for
    this (store_id, product_id) pair.  "Most recent" is determined by taking
    the row with max(date) per entity from the Spark feature DataFrame.  This
    is used as the proxy for today's on-hand stock — the stock level from
    which we decide whether to reorder and how much to order.

lead_time_days  (L)
    The number of calendar days between placing a purchase order and receiving
    the goods.  Sourced from config.inventory.lead_time_days.  During the lead
    time the store must sell from existing stock with no new arrivals, so this
    window is the critical exposure period that safety stock must cover.

order_cycle_days  (T)
    The intended interval between consecutive replenishment orders — how many
    days a single order batch is expected to last.  Sourced from
    config.inventory.order_cycle_days.  This determines the volume ordered:
    a 30-day cycle means one order should cover 30 days of demand.

service_level_z  (Z)
    The Z-score corresponding to the target in-stock probability (service
    level).  Z = 1.65 → 95% probability that stock does not run out during
    the lead time.  Sourced from config.inventory.service_level_z.  A higher
    Z means a larger safety buffer and fewer stock-outs, at the cost of more
    working capital tied up in inventory.

safety_stock  =  Z × σ × √L
    The buffer stock held above average expected demand to absorb demand
    variability and supply uncertainty during the lead-time window.

    Why √L?  Demand variability accumulates over time.  If daily demand has
    standard deviation σ, then over L independent days the cumulative
    variability is σ × √L (sum of independent variances, then square root).
    Multiplying by Z converts that cumulative variability into a confidence
    interval — "with Z standard deviations of buffer we meet the target
    service level even in above-average demand weeks."

reorder_point  =  D̄ × L  +  safety_stock
    The inventory level at which a new order should be placed so that it
    arrives just as stock reaches zero under average demand.  D̄ × L is the
    expected demand consumed during the lead time.  The safety_stock term
    extends this to handle above-average demand during the wait period.

    If current_inventory_level ≤ reorder_point, the reorder flag fires.

target_stock_level  =  D̄ × T  +  safety_stock
    The ideal on-hand quantity immediately after a replenishment arrives.
    D̄ × T covers one full order cycle at the average demand rate; safety_stock
    is added to provide the buffer above the cycle cover.

recommended_order_qty  =  max(target_stock_level − current_inventory_level, 0)
    The quantity to order now to top up from the current level to the target.
    Floored at 0: if current stock already exceeds the target there is nothing
    to order.

days_of_supply  =  current_inventory_level / D̄
    How many days the current stock will last at the average demand rate,
    assuming no new stock arrives.  A value less than lead_time_days is a
    critical signal: the store may run out before the next order arrives.
    Set to NaN when avg_daily_demand = 0 (undefined; division by zero).

reorder_recommendation  =  current_inventory_level ≤ reorder_point
    Boolean flag.  True means the store should place a replenishment order
    today.  This is the primary actionable output of the recommender.

─────────────────────────────────────────────────────────────────────────────
Difference between reorder_point and recommended_order_qty
─────────────────────────────────────────────────────────────────────────────
These answer two distinct questions:

  reorder_point        WHEN to order.
    "At what stock level should we place an order?"
    It is a threshold compared to current_inventory_level.  It controls
    timing — if you wait too long you will stock out during the lead time.
    Formula inputs: avg demand rate, lead time, demand variability.

  recommended_order_qty    HOW MUCH to order.
    "Once we decide to order, what quantity do we place?"
    It computes the top-up volume needed to reach target_stock_level.
    Formula inputs: avg demand rate, order cycle length, safety stock,
    current on-hand stock.

A store can be below its reorder_point (reorder now = True) but still have
a recommended_order_qty of 0 if its current inventory somehow exceeds the
target_stock_level — this happens only at the edge where the config
parameters are very conservative.  Conversely, a large recommended_order_qty
may be computed even when reorder_recommendation = False (the store is above
the reorder threshold but an opportunistic bulk order would be efficient).
Both outputs are reported so the operations team can apply their own policy.

─────────────────────────────────────────────────────────────────────────────
How current inventory level is chosen
─────────────────────────────────────────────────────────────────────────────
The Spark feature DataFrame contains one row per (store_id, product_id, date)
with an inventory_level column that records on-hand stock for that day.

To represent "today's stock", we take the row with max(date) per entity:

    Window.partitionBy("store_id", "product_id").orderBy(col("date").desc())
    filter rank == 1

This is the most recently observed stock reading available in the dataset.
In production this would be replaced by a live warehouse management system
query, but within the dataset this is the correct proxy for current state.

Using the latest date (rather than, say, an average) is correct because
inventory is a stock variable, not a flow variable — it is measured at a
point in time, not averaged over a period.

─────────────────────────────────────────────────────────────────────────────
Input schema vs output schema
─────────────────────────────────────────────────────────────────────────────
Input A — predictions_df from predictor.predict(), pandas DataFrame:

  date                  object (datetime.date)   — used to join context only
  store_id              object (str)              — grouping key
  product_id            object (str)              — grouping key
  category              object (str)              — carried through for reporting
  region                object (str)              — carried through for reporting
  units_sold            int64                     — not used here
  demand_forecast       float64                   — not used here
  predicted_units_sold  float64                   ← demand signal for all formulas

Input B — feature_df from add_price_features(), Spark DataFrame (39 columns):

  date              DateType    — used to find most-recent row per entity
  store_id          StringType  — entity key
  product_id        StringType  — entity key
  inventory_level   DoubleType  ← current stock level

Output — pandas DataFrame, one row per (store_id, product_id), 16 columns:

  store_id                object   entity key
  product_id              object   entity key
  category                object   for reporting
  region                  object   for reporting
  avg_daily_demand        float64  D̄
  demand_std_dev          float64  σ
  current_inventory_level float64  latest observed inventory_level
  lead_time_days          int      L  (from config)
  order_cycle_days        int      T  (from config)
  service_level_z         float    Z  (from config)
  safety_stock            float64  Z × σ × √L
  reorder_point           float64  D̄ × L + safety_stock
  target_stock_level      float64  D̄ × T + safety_stock
  recommended_order_qty   float64  max(target − current, 0)
  days_of_supply          float64  current / D̄  (NaN when D̄ = 0)
  reorder_recommendation  bool     current ≤ reorder_point

Usage from another module
--------------------------
    from src.modeling.predictor import predict
    from src.inventory.recommender import recommend

    predictions_df   = predict(feature_df)
    recommendations  = recommend(predictions_df, feature_df)

    print(recommendations[recommendations["reorder_recommendation"]].head())
"""

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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


def _extract_latest_inventory(feature_df: DataFrame) -> pd.DataFrame:
    """
    Return the most-recently-observed inventory_level per (store_id, product_id).

    Uses a descending-date window rank so that exactly one row per entity is
    kept — the row with max(date).  This is the closest available proxy for
    current on-hand stock.

    Parameters
    ----------
    feature_df : DataFrame
        39-column Spark DataFrame from add_price_features().
        Must contain: store_id, product_id, date, inventory_level.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns: store_id, product_id, inventory_level.
        One row per (store_id, product_id).
    """
    entity_keys = ["store_id", "product_id"]

    w = Window.partitionBy(*entity_keys).orderBy(F.col("date").desc())

    latest_pdf = (
        feature_df
        .withColumn("_date_rank", F.rank().over(w))
        .filter(F.col("_date_rank") == 1)
        .select(*entity_keys, "inventory_level")
        .toPandas()
    )

    logger.info(
        "Extracted latest inventory_level for %d (store_id, product_id) pairs",
        len(latest_pdf),
    )
    return latest_pdf


def _aggregate_demand(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute avg_daily_demand and demand_std_dev per (store_id, product_id).

    The standard deviation uses ddof=1 (sample std dev) to avoid
    underestimating variability on the finite forecast horizon.  For entities
    with only one forecast row, std dev is NaN — treated as 0 downstream so
    that safety stock is computed conservatively (no artificial buffer added
    on top of unknown variability, rather than crashing or inflating the
    estimate).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of predictor.predict(); must contain store_id, product_id,
        category, region, predicted_units_sold.

    Returns
    -------
    pd.DataFrame
        One row per (store_id, product_id) with columns:
        store_id, product_id, category, region,
        avg_daily_demand, demand_std_dev.
    """
    entity_keys = ["store_id", "product_id"]

    # category and region are constant per entity — take the first value.
    demand_agg = (
        predictions_df
        .groupby(entity_keys, sort=False)
        .agg(
            category         = ("category",            "first"),
            region           = ("region",               "first"),
            avg_daily_demand = ("predicted_units_sold", "mean"),
            demand_std_dev   = ("predicted_units_sold", "std"),   # ddof=1
        )
        .reset_index()
    )

    # Entities with a single forecast row produce NaN std dev.
    # Treat as 0: we have no evidence of variability, so no variance buffer.
    demand_agg["demand_std_dev"] = demand_agg["demand_std_dev"].fillna(0.0)

    return demand_agg


def _compute_inventory_metrics(
    row: pd.Series,
    lead_time_days: int,
    order_cycle_days: int,
    service_level_z: float,
) -> pd.Series:
    """
    Apply the inventory formulas to a single (store_id, product_id) row.

    All five derived metrics are computed here so they appear together and can
    be read as a coherent derivation chain rather than scattered assignments.

    Parameters
    ----------
    row : pd.Series
        Must contain: avg_daily_demand, demand_std_dev, current_inventory_level.
    lead_time_days : int
        L — days between order placement and receipt.
    order_cycle_days : int
        T — days one order batch is intended to cover.
    service_level_z : float
        Z — service-level Z-score.

    Returns
    -------
    pd.Series
        New columns: safety_stock, reorder_point, target_stock_level,
        recommended_order_qty, days_of_supply, reorder_recommendation.
    """
    d_bar   = row["avg_daily_demand"]        # D̄
    sigma   = row["demand_std_dev"]          # σ
    current = row["current_inventory_level"] # on-hand stock

    # ── Safety stock: buffer to absorb demand variability over the lead time ──
    # Z × σ × √L
    safety_stock = service_level_z * sigma * math.sqrt(lead_time_days)

    # ── Reorder point: trigger threshold ─────────────────────────────────────
    # D̄ × L + safety_stock
    reorder_point = d_bar * lead_time_days + safety_stock

    # ── Target stock level: ideal post-replenishment quantity ─────────────────
    # D̄ × T + safety_stock
    target_stock_level = d_bar * order_cycle_days + safety_stock

    # ── Recommended order quantity: top-up to target, floor at 0 ─────────────
    # max(target_stock_level − current, 0)
    recommended_order_qty = max(target_stock_level - current, 0.0)

    # ── Days of supply: how long current stock lasts at average demand ────────
    # Undefined (NaN) when avg_daily_demand = 0 — the store never sells this
    # SKU, so "days of supply" is meaningless rather than infinite.
    days_of_supply = current / d_bar if d_bar > 0 else float("nan")

    # ── Reorder recommendation: binary trigger flag ───────────────────────────
    reorder_recommendation = bool(current <= reorder_point)

    return pd.Series({
        "safety_stock":           round(safety_stock,           4),
        "reorder_point":          round(reorder_point,          4),
        "target_stock_level":     round(target_stock_level,     4),
        "recommended_order_qty":  round(recommended_order_qty,  4),
        "days_of_supply":         round(days_of_supply,         4) if not math.isnan(days_of_supply) else float("nan"),
        "reorder_recommendation": reorder_recommendation,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    predictions_df: pd.DataFrame,
    feature_df: DataFrame,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Generate per-SKU inventory recommendations from demand forecasts.

    Steps performed:
      1. Load inventory policy parameters from config.yaml.
      2. Extract the latest observed inventory_level per (store_id, product_id)
         from the Spark feature DataFrame.
      3. Aggregate predicted_units_sold into avg_daily_demand and demand_std_dev
         per entity.
      4. Join aggregated demand onto the latest inventory levels.
      5. Apply the inventory formulas row-by-row to produce all derived metrics.
      6. Return a clean, sorted output DataFrame (one row per entity).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Output of predictor.predict() — 8 columns including predicted_units_sold,
        store_id, product_id, category, region.
    feature_df : DataFrame
        39-column Spark DataFrame from add_price_features() — used solely to
        extract the latest inventory_level per (store_id, product_id).
    config_path : str
        Path to config/config.yaml.

    Returns
    -------
    pd.DataFrame
        One row per (store_id, product_id), 16 columns — see module docstring
        for the full output schema.

    Raises
    ------
    FileNotFoundError
        If config.yaml is missing.
    ValueError
        If predictions_df is missing required columns.
    """
    # ── Validate input columns ────────────────────────────────────────────────
    required = {"store_id", "product_id", "category", "region", "predicted_units_sold"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(
            f"predictions_df is missing required column(s): {sorted(missing)}. "
            "Ensure the DataFrame was produced by predictor.predict()."
        )

    # ── 1. Load policy parameters ─────────────────────────────────────────────
    cfg = _load_config(Path(config_path))
    inv_cfg = cfg["inventory"]

    lead_time_days   = int(inv_cfg["lead_time_days"])
    order_cycle_days = int(inv_cfg["order_cycle_days"])
    service_level_z  = float(inv_cfg["service_level_z"])

    logger.info(
        "Inventory parameters | lead_time_days=%d | order_cycle_days=%d | "
        "service_level_z=%.2f (%.0f%% service level)",
        lead_time_days,
        order_cycle_days,
        service_level_z,
        # Approximate service level percentage from Z-score for logging only
        100 * (0.5 + 0.5 * math.erf(service_level_z / math.sqrt(2))),
    )

    # ── 2. Extract latest inventory from the Spark feature DataFrame ──────────
    latest_inventory = _extract_latest_inventory(feature_df)

    # ── 3. Aggregate forecasted demand per entity ─────────────────────────────
    demand_agg = _aggregate_demand(predictions_df)

    logger.info(
        "Demand aggregated | entities=%d | "
        "global avg_daily_demand=%.2f | global demand_std_dev=%.2f",
        len(demand_agg),
        demand_agg["avg_daily_demand"].mean(),
        demand_agg["demand_std_dev"].mean(),
    )

    # ── 4. Join inventory onto demand aggregates ──────────────────────────────
    # Inner join: only entities present in both DataFrames are retained.
    # In a correctly run pipeline all entities appear in both; a mismatch
    # would indicate a pipeline ordering error and should be investigated.
    entity_keys = ["store_id", "product_id"]
    merged = demand_agg.merge(
        latest_inventory,
        on=entity_keys,
        how="inner",
        validate="1:1",   # assert each entity appears once in both sides
    )
    # Rename to match the expected column name in _compute_inventory_metrics()
    # and the output schema.  The source column is named inventory_level in
    # the Spark DataFrame; current_inventory_level makes its semantics
    # ("today's on-hand stock") explicit throughout all downstream code.
    merged = merged.rename(columns={"inventory_level": "current_inventory_level"})

    n_demand  = len(demand_agg)
    n_merged  = len(merged)
    if n_merged < n_demand:
        logger.warning(
            "%d entities in predictions_df had no matching inventory row in "
            "feature_df and were dropped from recommendations.",
            n_demand - n_merged,
        )

    # ── 5. Apply inventory formulas ───────────────────────────────────────────
    policy_metrics = merged.apply(
        _compute_inventory_metrics,
        axis=1,
        lead_time_days=lead_time_days,
        order_cycle_days=order_cycle_days,
        service_level_z=service_level_z,
    )

    result = pd.concat([merged, policy_metrics], axis=1)

    # ── 6. Add config columns for full auditability ───────────────────────────
    # Including the policy parameters in the output row means a reader can
    # verify every formula without having to cross-reference the config file.
    result["lead_time_days"]   = lead_time_days
    result["order_cycle_days"] = order_cycle_days
    result["service_level_z"]  = service_level_z

    # ── 7. Select and order output columns ───────────────────────────────────
    output_columns = [
        "store_id",
        "product_id",
        "category",
        "region",
        "avg_daily_demand",
        "demand_std_dev",
        "current_inventory_level",
        "lead_time_days",
        "order_cycle_days",
        "service_level_z",
        "safety_stock",
        "reorder_point",
        "target_stock_level",
        "recommended_order_qty",
        "days_of_supply",
        "reorder_recommendation",
    ]

    result = (
        result[output_columns]
        .sort_values(entity_keys)
        .reset_index(drop=True)
    )

    n_reorder = int(result["reorder_recommendation"].sum())
    logger.info(
        "Recommendations complete | total_skus=%d | reorder_flagged=%d (%.1f%%)",
        len(result),
        n_reorder,
        100 * n_reorder / len(result) if len(result) > 0 else 0.0,
    )

    return result
