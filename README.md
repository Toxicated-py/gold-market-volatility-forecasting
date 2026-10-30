# Forecasting Gold Market Volatility Using Machine Learning

## Project Overview

This project uses machine learning to forecast short-term gold market volatility using historical gold futures data.

The objective is to estimate the volatility of gold over the **next five observed trading sessions**, using information available at the current trading session's close.

This is a volatility forecasting project, not a trading bot. It does not generate buy or sell signals or attempt to predict market direction.

## Problem Statement

Gold prices can experience periods of both low and high volatility. Estimating future volatility can help quantify market uncertainty and understand changing risk conditions.

This project investigates whether historical price behaviour, recent realized volatility, momentum and daily price ranges contain useful information for forecasting future gold volatility.

## Dataset

- **Source:** Yahoo Finance
- **Ticker:** `GC=F`
- **Instrument:** COMEX Gold Futures continuous-contract ticker
- **Frequency:** Daily
- **Raw observations:** 6,568
- **Date range:** 30 August 2000 to 30 October 2026
- **Features used:** OHLC-derived price features and historical return features

Volume was excluded from model features because the historical volume data contained quality concerns.

**Important:** `GC=F` is a gold futures proxy, not the same instrument as spot XAUUSD. Futures contract rolls and settlement-price conventions may affect historical observations.

## Target Variable

The target is the realized volatility over the next five observed trading sessions:

\[
Target_t = 100 \times \sqrt{\sum_{i=1}^{5}r_{t+i}^{2}}
\]

where:

\[
r_t = \ln\left(\frac{Close_t}{Close_{t-1}}\right)
\]

The target uses future returns from the five sessions following the feature date. It is not a five-calendar-day forecast.

## Feature Engineering

A total of 17 features were created.

### Price and return features
- Current daily return
- Overnight gap
- Intraday price range
- Open-to-close return
- Lagged returns over 1, 2, 3, 5 and 10 sessions

### Historical volatility features
- 5-session realized volatility
- 10-session realized volatility
- 20-session realized volatility
- 60-session realized volatility

### Momentum features
- 5-session momentum
- 10-session momentum
- 20-session momentum
- 60-session momentum

All features are constructed using information available at the prediction date.

## Methodology

The project follows a chronological machine learning workflow.

1. Download historical gold futures data.
2. Perform exploratory data analysis.
3. Investigate missing values, duplicate dates and OHLC consistency.
4. Engineer historical return, volatility and momentum features.
5. Create a future five-session volatility target.
6. Split the dataset chronologically.
7. Establish baseline performance.
8. Train and compare machine learning models.
9. Tune Random Forest and XGBoost using time-series cross-validation.
10. Analyse model performance and feature importance.
11. Evaluate the final selected model on an untouched test dataset.

### Chronological data split

| Dataset | Observations | Period |
|---|---:|---|
| Training | 4,547 | 27 Nov 2000 – 16 Jan 2019 |
| Validation | 970 | 25 Jan 2019 – 29 Nov 2022 |
| Test | 976 | 7 Dec 2022 – 23 Oct 2026 |

Five observations were purged between each split to prevent overlapping five-session target windows across dataset boundaries.

## Models

The following models were evaluated:

- Mean baseline
- Five-day volatility persistence baseline
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

XGBoost was selected based on validation performance and tuned using `TimeSeriesSplit` with four folds and a five-observation gap.

### Final XGBoost configuration

- `n_estimators`: 500
- `max_depth`: 3
- `learning_rate`: 0.01
- `subsample`: 0.7
- `colsample_bytree`: 0.7
- `min_child_weight`: 10
- `reg_lambda`: 1

## Final Test Results

The final model was retrained using the combined training and validation datasets. The test dataset was kept separate for final evaluation.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Mean Baseline | 1.4683 | 0.9692 | -0.0370 |
| Five-day Persistence | 1.5002 | 1.0214 | -0.0826 |
| **Final XGBoost** | **1.1692** | **0.8071** | **0.3424** |

The final XGBoost model achieved a **20.37% reduction in RMSE** compared with the mean baseline.

The positive R² indicates that the model captures some useful variation in future volatility. However, the remaining unexplained variation means that its forecasts should not be treated as precise predictions.

## Feature Importance

Permutation importance on the validation dataset identified the following leading features:

1. `realized_vol_60d`
2. `realized_vol_20d`
3. `momentum_60d`
4. `intraday_range_pct`
5. `momentum_20d`
6. `realized_vol_10d`

Historical volatility features were particularly informative, suggesting that recent volatility regimes contain useful information about near-term volatility.

## Data Quality Considerations

The investigation identified 441 historical records with at least one OHLC consistency violation. These records occurred between 2001 and 2011.

A controlled experiment masked the three OHLC-derived features on those dates. This produced almost identical validation performance, so the original feature dataset was retained.

Large daily price movements were also investigated. They were not automatically removed because unusually large returns can represent genuine market events.

These findings are documented as limitations of the underlying historical data.

## Limitations

- Yahoo Finance's continuous gold futures ticker is not identical to spot gold.
- Historical futures data may contain contract-roll and settlement-price effects.
- Historical OHLC inconsistencies may affect some engineered features.
- The model has limited predictive power and does not explain all future volatility variation.
- Evaluation is based on one chronological test period.
- The model does not predict price direction or generate trading signals.
- No live deployment or real-time forecasting performance has been established.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Yahoo Finance data

## Project Outputs

- Final trained model: `models/gold_volatility_xgboost_final.joblib`
- Test predictions: `reports/test_predictions.csv`
- Final metrics: `reports/final_test_metrics.csv`
- Model comparison and feature importance: `data/processed/` and `reports/`
- Visualisations: `reports/figures/`

## Conclusion

This project demonstrates an end-to-end time-series regression workflow for forecasting gold market volatility.

The final XGBoost model outperformed both baseline approaches on the untouched test period, achieving a 20.37% reduction in RMSE against the mean baseline.

The results suggest that historical volatility and longer-term momentum contain useful information for forecasting short-term gold volatility. However, the model's performance is not sufficient to guarantee reliable forecasts under all market conditions.

The project is an experimental forecasting study and should not be interpreted as financial advice or a production-ready trading system.