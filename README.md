# Distributed Demand Forecasting & Inventory Optimization System

## Overview

This project implements an end-to-end demand forecasting and inventory optimization system using PySpark and XGBoost. It processes over 73,000 retail transaction records across more than 100 store-product (SKU) combinations.

The system not only predicts demand but also translates predictions into operational decisions, including safety stock estimation and reorder point calculation, connecting machine learning outputs to real-world supply chain applications.

---

## Core Insight

Improving prediction accuracy does not necessarily improve business outcomes.

Although the enhanced model significantly reduced prediction error, it increased stock-out risk, revealing a mismatch between traditional ML metrics (e.g., RMSE) and operational performance.

---

## Model Design

Two model variants were implemented:

### Fair Model
- Excludes `demand_forecast`
- Relies purely on engineered features
- Simulates a production ML system without external signals

### Enhanced Model
- Includes `demand_forecast`
- Learns to correct baseline forecast errors
- Represents integration of ML with existing business systems

---

## Results

| Metric | Fair Model | Baseline | Enhanced Model |
|--------|-----------|----------|----------------|
| RMSE   | ~70.5     | 10.0     | 7.97           |
| MAE    | ~54.8     | 8.35     | 6.74           |
| MAPE   | ~186%     | 22.7%    | 18.1%          |

The enhanced model improves RMSE by approximately 20 percent compared to the baseline.

---

## Business Impact

A business-oriented evaluation metric was introduced:

Stock-out occurs when predicted demand is lower than actual demand.

Results:

- Model stock-out rate: 49.76%
- Baseline stock-out rate: 33.23%

Despite improved RMSE, the model increased stock-out risk by 16.5 percentage points.

---

## Root Cause Analysis

The model tends to underpredict high-demand scenarios because:

- RMSE penalizes large errors symmetrically
- Underprediction is not penalized more heavily than overprediction
- The model lacks strong signals for baseline demand differences across entities

---

## Optimization Strategy

To address these issues:

### Entity Identity Features
- Added `store_id_encoded` and `product_id_encoded`
- Enables the model to learn store-level and product-level baseline demand

### Feature Engineering
- Lag features (1, 7, 14, 28 days)
- Rolling statistics (mean and standard deviation)
- Calendar features (weekday, month, seasonality)
- Pricing signals (discount, competitor pricing)

### Evaluation Shift
- Moved from pure ML metrics (RMSE, MAE)
- Toward business metrics (stock-out rate)

---

## Pipeline

End-to-end pipeline:

Data Loading → Cleaning → Encoding  
→ Feature Engineering (lag, rolling, calendar, pricing)  
→ Model Training (XGBoost)  
→ Prediction  
→ Evaluation  
→ Inventory Recommendation  
→ Output (CSV and Parquet)

---

## Inventory Optimization

The system converts predictions into actionable decisions:

Safety Stock = Z × σ × √L  
Reorder Point = Demand × Lead Time + Safety Stock  

Where:
- Z = service level factor
- σ = demand standard deviation
- L = lead time

---

## Tech Stack

- Python  
- PySpark  
- XGBoost  
- Pandas / NumPy  
- PyArrow  
- YAML configuration  

---

## How to Run

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### Run pipeline
#### Fair model:
```bash
python pipeline.py

#### Enhanced model:
```bash
python pipeline.py --use-baseline-feature


