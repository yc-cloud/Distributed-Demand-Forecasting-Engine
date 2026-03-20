# Distributed Demand Forecasting Engine — Project Design

> Portfolio project for resume and interviews.
> Data source: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) (Kaggle)

---

## 1. Dataset Overview

**File:** `data/raw/retail_store_inventory.csv`
**Rows:** 73,100
**Row grain:** one record per `(Date, Store ID, Product ID)` — already at daily level, no disaggregation needed
**Entities:** 5 stores × 20 products across 4 regions

### Confirmed Column Schema (15 columns)

| # | Column | Type | Range / Values | Role |
|---|--------|------|----------------|------|
| 1 | `Date` | DATE | Daily | Time axis |
| 2 | `Store ID` | STRING | S001–S005 | Entity key |
| 3 | `Product ID` | STRING | P0001–P0020 | Entity key |
| 4 | `Category` | STRING | Clothing, Electronics, Furniture, Groceries, Toys | Categorical feature |
| 5 | `Region` | STRING | East, North, South, West | Categorical feature |
| 6 | `Inventory Level` | INTEGER | 50–500 | Inventory input / business logic |
| 7 | `Units Sold` | INTEGER | 0–499 | **Forecasting target** |
| 8 | `Units Ordered` | INTEGER | 20–200 | Replenishment signal / business logic |
| 9 | `Demand Forecast` | DOUBLE | −9.99–518.55 | Baseline to benchmark against |
| 10 | `Price` | DOUBLE | 10.00–100.00 | Pricing feature |
| 11 | `Discount` | INTEGER | 0–20 (percent) | Pricing feature |
| 12 | `Weather Condition` | STRING | Sunny, Cloudy, Rainy, Snowy | Exogenous feature |
| 13 | `Holiday/Promotion` | INTEGER | 0 or 1 | Calendar / event feature |
| 14 | `Competitor Pricing` | DOUBLE | 5.03–104.94 | Competitive feature |
| 15 | `Seasonality` | STRING | Spring, Summer, Autumn, Winter | Calendar / exogenous feature |

---

## 2. Column Role Mapping

### Forecasting Target
| Column | Reason |
|--------|--------|
| `Units Sold` | Primary daily demand quantity. One value per store-product-date. |

### Time-Series Feature Engineering Sources
| Column | Derived Features |
|--------|-----------------|
| `Date` | `day_of_week`, `month`, `week_of_year`, `quarter`, `day_of_month`, `is_weekend` |
| `Units Sold` | Lag-1, 7, 14, 28 per (Store ID, Product ID); rolling mean/std/min/max 7/14/28 days |
| `Holiday/Promotion` | Used directly as binary feature |
| `Seasonality` | Ordinal-encoded season index |
| `Weather Condition` | Ordinal-encoded weather index |
| `Price` + `Discount` | `effective_price = Price × (1 − Discount/100)` |
| `Price` + `Competitor Pricing` | `price_to_competitor_ratio = Price / Competitor Pricing` |
| `Discount` | `has_discount = Discount > 0` (boolean) |

### Inventory Business Logic Inputs
| Column | Use |
|--------|-----|
| `Inventory Level` | Current on-hand stock; used for `days_of_supply` and `reorder_recommendation` |
| `Units Ordered` | Historical replenishment cadence; validates order quantity recommendations |
| `Demand Forecast` | Pre-existing baseline in dataset — benchmarked against ML model in `evaluator.py` |

**Key design insight:** The dataset ships with its own `Demand Forecast` column (with values ranging to −9.99,
indicating data quality issues). The project explicitly benchmarks the ML model against this baseline,
quantifying the improvement. This mirrors a realistic industry pattern: replacing a rule-based forecast
with an ML system and justifying the switch with measured error reduction.

---

## 3. Folder Structure

```
Distributed-Demand-Forecasting-Engine/
│
├── config/
│   ├── config.yaml                   # Paths, Spark settings, model params, encoding maps,
│   │                                 #   lead-time defaults, service level Z-score
│   └── logging.yaml                  # Log format, handlers, level
│
├── data/
│   ├── raw/                          # Kaggle CSV placed here manually (gitignored)
│   │   └── retail_store_inventory.csv
│   ├── processed/                    # Cleaned + encoded Parquet (gitignored)
│   ├── features/                     # Feature-engineered Parquet (gitignored)
│   └── predictions/                  # Forecast + recommendation outputs (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Distributions, missing values, category/region breakdowns
│   ├── 02_feature_exploration.ipynb  # Lag correlations, rolling stats, seasonality patterns
│   ├── 03_model_vs_baseline.ipynb    # ML forecast vs dataset's built-in Demand Forecast column
│   └── 04_inventory_analysis.ipynb   # Reorder point, safety stock, stockout risk by store/product
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_preparation/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Reads retail_store_inventory.csv with declared Spark schema
│   │   ├── cleaner.py                # snake_case rename, type casting, is_forecast_anomaly flag,
│   │   │                             #   range validation, Holiday/Promotion cast to boolean
│   │   └── encoder.py                # Ordinal-encodes Weather Condition, Category, Region,
│   │                                 #   Seasonality using maps from config.yaml
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── lag_features.py           # Lag 1/7/14/28 on units_sold per (store_id, product_id)
│   │   │                             #   using Spark Window partitioned by entity, ordered by date
│   │   ├── rolling_features.py       # Rolling 7/14/28-day mean, std, min, max on units_sold
│   │   ├── calendar_features.py      # day_of_week, month, week_of_year, quarter, is_weekend
│   │   └── price_features.py         # effective_price, price_to_competitor_ratio, has_discount
│   │
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Time-based train/val split at 80th percentile date;
│   │   │                             #   XGBoost training; joblib model serialization
│   │   ├── predictor.py              # Batch inference; appends predicted_units_sold column
│   │   └── evaluator.py              # RMSE, MAE, MAPE for ML model AND dataset's Demand Forecast
│   │                                 #   baseline; writes metrics.json; reports per category
│   │
│   ├── inventory/
│   │   ├── __init__.py
│   │   └── recommender.py            # Safety stock, reorder point, days_of_supply,
│   │                                 #   reorder_recommendation, recommended_order_qty
│   │
│   ├── output/
│   │   ├── __init__.py
│   │   └── writer.py                 # Writes forecast + recommendations to Parquet + CSV
│   │
│   └── utils/
│       ├── __init__.py
│       ├── spark_session.py           # SparkSession factory; config-driven, local mode
│       └── schema_validator.py        # Validates column presence and types at pipeline boundaries;
│                                      #   raises SchemaValidationError on mismatch
│
├── tests/
│   ├── unit/
│   │   ├── test_loader.py
│   │   ├── test_cleaner.py
│   │   ├── test_encoder.py
│   │   ├── test_lag_features.py
│   │   ├── test_rolling_features.py
│   │   ├── test_price_features.py
│   │   └── test_recommender.py
│   └── integration/
│       └── test_pipeline_end_to_end.py
│
├── pipeline.py                       # Top-level orchestrator; stage-by-stage with timing logs
├── DESIGN.md                         # This file
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `loader.py` | Reads `retail_store_inventory.csv` using an explicitly declared Spark schema (no schema inference). Returns a raw DataFrame with original column names. Raises on file-not-found. |
| `cleaner.py` | Renames all columns to `snake_case`. Enforces types: `Date→DateType`, `Holiday/Promotion→BooleanType`. Adds `is_forecast_anomaly` boolean where `demand_forecast < 0`. Filters rows with `units_sold < 0`. |
| `encoder.py` | Ordinal-encodes `weather_condition`, `seasonality`, `category`, and `region` using maps stored in `config.yaml`. Appends encoded integer columns alongside originals for readability. |
| `lag_features.py` | Generates `lag_1`, `lag_7`, `lag_14`, `lag_28` from `units_sold` using `Window.partitionBy('store_id','product_id').orderBy('date')`. Null-safe — early rows receive null lags. |
| `rolling_features.py` | Computes 7/14/28-day rolling mean, std, min, max of `units_sold` per `(store_id, product_id)` using Spark Window range frames. |
| `calendar_features.py` | Extracts `day_of_week`, `month`, `week_of_year`, `quarter`, `day_of_month`, `is_weekend` from `date` using Spark date functions. |
| `price_features.py` | Derives: `effective_price = price × (1 − discount / 100)`, `price_to_competitor_ratio = price / competitor_pricing`, `has_discount = discount > 0`. |
| `trainer.py` | Converts feature DataFrame to Pandas; performs strict time-based split at 80th percentile date; trains XGBoost regressor; serializes model via joblib. |
| `evaluator.py` | Computes RMSE, MAE, MAPE for both the ML model and the dataset's `demand_forecast` baseline — globally and per category. Writes `metrics.json`. |
| `predictor.py` | Loads serialized model; runs batch inference; returns DataFrame with `predicted_units_sold` appended. |
| `recommender.py` | Groups by `(store_id, product_id)` over forecast horizon; derives `avg_daily_demand`, `demand_std_dev`; computes `safety_stock`, `reorder_point`; compares against `inventory_level`; sets `reorder_recommendation` and `recommended_order_qty`. |
| `writer.py` | Writes demand forecast and inventory recommendation DataFrames to `data/predictions/` as Parquet and CSV. |
| `spark_session.py` | Returns singleton local SparkSession; app name, memory, and shuffle partitions from `config.yaml`. |
| `schema_validator.py` | Asserts column presence and declared types; raises `SchemaValidationError` with descriptive message on mismatch. Called at every major stage boundary. |
| `pipeline.py` | Imports and calls each stage sequentially; logs stage name and wall-clock duration; single CLI entry point. |

---

## 5. Input Schema

**File:** `data/raw/retail_store_inventory.csv`
**Grain:** `(Date, Store ID, Product ID)` — one row per store-product-date

| Column | Declared Spark Type | Nullable | Notes |
|--------|--------------------|---------:|-------|
| `Date` | `DateType` | No | Format: YYYY-MM-DD |
| `Store ID` | `StringType` | No | Values: S001–S005 |
| `Product ID` | `StringType` | No | Values: P0001–P0020 |
| `Category` | `StringType` | No | 5 distinct values |
| `Region` | `StringType` | No | 4 distinct values |
| `Inventory Level` | `IntegerType` | No | Range: 50–500 |
| `Units Sold` | `IntegerType` | No | Range: 0–499 |
| `Units Ordered` | `IntegerType` | No | Range: 20–200 |
| `Demand Forecast` | `DoubleType` | No | Range: −9.99–518.55 |
| `Price` | `DoubleType` | No | Range: 10.00–100.00 |
| `Discount` | `IntegerType` | No | Range: 0–20 |
| `Weather Condition` | `StringType` | No | Sunny, Cloudy, Rainy, Snowy |
| `Holiday/Promotion` | `IntegerType` | No | 0 or 1 |
| `Competitor Pricing` | `DoubleType` | No | Range: 5.03–104.94 |
| `Seasonality` | `StringType` | No | Spring, Summer, Autumn, Winter |

---

## 6. Intermediate Schemas

### Stage 1 — Cleaned Dataset
**Produced by:** `cleaner.py`

| Column | Type | Note |
|--------|------|------|
| `date` | `DateType` | Renamed; type enforced |
| `store_id` | `StringType` | Renamed |
| `product_id` | `StringType` | Renamed |
| `category` | `StringType` | Renamed |
| `region` | `StringType` | Renamed |
| `inventory_level` | `IntegerType` | Renamed |
| `units_sold` | `IntegerType` | Renamed |
| `units_ordered` | `IntegerType` | Renamed |
| `demand_forecast` | `DoubleType` | Renamed |
| `is_forecast_anomaly` | `BooleanType` | **New** — `true` where `demand_forecast < 0` |
| `price` | `DoubleType` | Renamed |
| `discount` | `IntegerType` | Renamed |
| `weather_condition` | `StringType` | Renamed |
| `is_holiday_or_promo` | `BooleanType` | Renamed; cast from integer |
| `competitor_pricing` | `DoubleType` | Renamed |
| `seasonality` | `StringType` | Renamed |

### Stage 2 — Encoded Dataset
**Produced by:** `encoder.py`

Adds encoded integer columns alongside original strings:

| New Column | Type | Encoding |
|------------|------|----------|
| `weather_encoded` | `IntegerType` | Sunny=0, Cloudy=1, Rainy=2, Snowy=3 |
| `category_encoded` | `IntegerType` | Config-driven ordinal map |
| `region_encoded` | `IntegerType` | Config-driven ordinal map |
| `seasonality_encoded` | `IntegerType` | Spring=0, Summer=1, Autumn=2, Winter=3 |

### Stage 3 — Feature-Engineered Dataset
**Produced by:** `lag_features.py`, `rolling_features.py`, `calendar_features.py`, `price_features.py`

| New Column | Type | Logic |
|------------|------|-------|
| `lag_1` | `DoubleType` | `units_sold` 1 day prior, per (store_id, product_id) |
| `lag_7` | `DoubleType` | 7 days prior |
| `lag_14` | `DoubleType` | 14 days prior |
| `lag_28` | `DoubleType` | 28 days prior |
| `rolling_mean_7` | `DoubleType` | 7-day rolling mean of `units_sold` |
| `rolling_std_7` | `DoubleType` | 7-day rolling std dev |
| `rolling_mean_14` | `DoubleType` | 14-day rolling mean |
| `rolling_std_14` | `DoubleType` | 14-day rolling std dev |
| `rolling_mean_28` | `DoubleType` | 28-day rolling mean |
| `rolling_std_28` | `DoubleType` | 28-day rolling std dev |
| `day_of_week` | `IntegerType` | 0=Mon … 6=Sun |
| `month` | `IntegerType` | 1–12 |
| `week_of_year` | `IntegerType` | 1–52 |
| `quarter` | `IntegerType` | 1–4 |
| `day_of_month` | `IntegerType` | 1–31 |
| `is_weekend` | `BooleanType` | Saturday or Sunday |
| `effective_price` | `DoubleType` | `price × (1 − discount / 100)` |
| `price_to_competitor_ratio` | `DoubleType` | `price / competitor_pricing` |
| `has_discount` | `BooleanType` | `discount > 0` |

---

## 7. Final Output Schemas

### Demand Forecast Output
**File:** `data/predictions/demand_forecast.parquet` + `.csv`

| Column | Type | Description |
|--------|------|-------------|
| `store_id` | `StringType` | Store identifier |
| `product_id` | `StringType` | Product identifier |
| `category` | `StringType` | Product category |
| `region` | `StringType` | Store region |
| `forecast_date` | `DateType` | The date being forecast |
| `predicted_units_sold` | `DoubleType` | XGBoost model point forecast |
| `actual_units_sold` | `IntegerType` | Ground truth (null for future dates) |
| `dataset_demand_forecast` | `DoubleType` | Baseline forecast from the raw dataset |
| `model_absolute_error` | `DoubleType` | `|predicted_units_sold − actual_units_sold|` |
| `baseline_absolute_error` | `DoubleType` | `|dataset_demand_forecast − actual_units_sold|` |
| `model_version` | `StringType` | ISO timestamp of training run |
| `run_id` | `StringType` | Pipeline execution UUID |

### Inventory Recommendation Output
**File:** `data/predictions/inventory_recommendations.parquet` + `.csv`

| Column | Type | Description |
|--------|------|-------------|
| `store_id` | `StringType` | Store identifier |
| `product_id` | `StringType` | Product identifier |
| `category` | `StringType` | Product category |
| `region` | `StringType` | Store region |
| `current_inventory_level` | `IntegerType` | Latest `inventory_level` from raw data |
| `avg_daily_demand` | `DoubleType` | Mean of `predicted_units_sold` over forecast horizon |
| `demand_std_dev` | `DoubleType` | Std dev of `predicted_units_sold` over forecast horizon |
| `lead_time_days` | `IntegerType` | Replenishment lead time from `config.yaml` |
| `service_level_z` | `DoubleType` | Z-score for target service level (e.g., 1.65 = 95%) |
| `safety_stock` | `DoubleType` | `Z × σ × √lead_time_days` |
| `reorder_point` | `DoubleType` | `avg_daily_demand × lead_time_days + safety_stock` |
| `days_of_supply` | `DoubleType` | `current_inventory_level / avg_daily_demand` |
| `reorder_recommendation` | `BooleanType` | `true` when `current_inventory_level ≤ reorder_point` |
| `recommended_order_qty` | `DoubleType` | One full replenishment cycle: `avg_daily_demand × lead_time_days + safety_stock` |
| `generated_at` | `TimestampType` | Pipeline run timestamp |

---

## 8. Recommended Implementation Order

```
Phase 1 — Foundation
  1. spark_session.py           SparkSession factory (local mode, config-driven)
  2. config.yaml                Paths, Spark config, encoding maps, lead-time/service-level defaults
  3. schema_validator.py        Reusable validation utility; called at every stage boundary

Phase 2 — Data Preparation
  4. loader.py                  Read retail_store_inventory.csv with declared Spark schema
  5. cleaner.py                 snake_case rename, type casting, is_forecast_anomaly flag
  6. encoder.py                 Categorical ordinal encoding via config.yaml maps

Phase 3 — Feature Engineering
  7. lag_features.py            Lag 1/7/14/28 with Spark Window per (store_id, product_id)
  8. rolling_features.py        Rolling 7/14/28-day stats (mean, std, min, max)
  9. calendar_features.py       DOW, month, week_of_year, quarter, is_weekend
 10. price_features.py          effective_price, price_to_competitor_ratio, has_discount

Phase 4 — Modeling
 11. trainer.py                 Time-based split (80th percentile date); XGBoost; joblib serialization
 12. evaluator.py               RMSE/MAE/MAPE for ML model AND dataset's Demand Forecast baseline
 13. predictor.py               Batch inference on full feature DataFrame

Phase 5 — Business Output
 14. recommender.py             Safety stock, reorder point, order quantity per store+product
 15. writer.py                  Parquet + CSV output for forecast and recommendations

Phase 6 — Orchestration & Tests
 16. pipeline.py                End-to-end runner with per-stage timing logs
 17. Unit tests                 One test file per src/ module
 18. Integration test           Full pipeline over a 500-row slice of the real CSV

Phase 7 — Documentation & Portfolio Polish
 19. Notebooks (EDA, feature exploration, model vs baseline, inventory analysis)
 20. README with architecture diagram, dataset setup instructions, sample output
```

---

## 9. Key Interview-Worthy Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Benchmark against built-in `Demand Forecast` column** | The dataset ships with a naive forecast that has negative values. Measuring ML improvement over it mirrors how real teams justify replacing rule-based systems. Strong interview story. |
| **`is_forecast_anomaly` flag** | Raw `Demand Forecast` has values down to −9.99. Surfacing this as a boolean flag rather than silently dropping rows demonstrates data quality awareness and auditability. |
| **Spark Window for lag/rolling features** | Correct distributed primitive for time-series features on partitioned entity data. Demonstrates PySpark depth beyond basic DataFrame API. |
| **Strict time-based train/test split** | No random shuffle. Prevents future data leakage — the most common time-series ML interview question. |
| **Config-driven encoding maps** | Categorical encoding logic lives in YAML, not hardcoded Python. Matches production ETL data contract patterns. |
| **`price_to_competitor_ratio` feature** | Domain-aware feature engineering. Competitive pricing is a known demand driver in retail. Shows business context beyond raw ML. |
| **Schema validation at every stage boundary** | Pipeline fails fast with a clear error instead of propagating corrupt data. Mirrors data contract enforcement in production pipelines. |
| **Parquet + CSV dual output** | Parquet for efficient downstream consumption; CSV for human-readable portfolio presentation. |
