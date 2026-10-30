# Gold Market Volatility Forecasting

This DataTalksClub ML Zoomcamp project forecasts **realized gold volatility over
the next five observed trading sessions**. It is a volatility and risk-estimation
model, not a price-direction predictor or trading bot.

## Data and target

The data comes from Yahoo Finance ticker `GC=F`, a continuous COMEX gold futures
proxy. It is not the same instrument as XAUUSD spot. The target at date `t` is:

`100 * sqrt(sum(log(Close[t+i] / Close[t+i-1]) ** 2 for i=1..5))`

The five sessions are observed trading sessions, not calendar days. The model
uses 17 OHLC-derived features: current and overnight returns, intraday range,
open-close return, return lags (1/2/3/5/10), realized volatility (5/10/20/60)
and momentum (5/10/20/60). The canonical implementation is
[`src/features.py`](src/features.py).

The raw data includes zero-volume records, flat OHLC records and historical OHLC
consistency violations. These observations were investigated rather than silently
deleted. Volume is excluded, and the OHLC masking experiment showed negligible
validation impact. Large returns were retained because they may be genuine market
events.

## Methodology and results

Data is split chronologically with a five-observation purge between partitions:

| Partition | Rows | Period |
|---|---:|---|
| Train | 4,547 | 2000-11-27 to 2019-01-16 |
| Validation | 970 | 2019-01-25 to 2022-11-29 |
| Test | 976 | 2022-12-07 to 2026-10-23 |

The selected model is XGBoost, retrained on train plus validation data:
500 estimators, depth 3, learning rate 0.01, subsample 0.7,
`colsample_bytree=0.7`, `min_child_weight=10`, and `reg_lambda=1`.

The current reproducible held-out test results are:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Mean baseline | 1.4683 | 0.9692 | -0.0370 |
| 5-day persistence | 1.5002 | 1.0214 | -0.0826 |
| Final XGBoost | **1.1692** | **0.8071** | **0.3424** |

The model reduces RMSE by 20.37% versus the mean baseline, but meaningful
unexplained variation remains. Results are not a guarantee of future performance.

## Repository structure

```text
data/                 Raw and processed datasets
models/               Final model and metadata
reports/              Metrics, predictions and figures
src/features.py       Shared feature engineering
src/predict.py        Cached model loading and inference
train.py              Reproducible final training/evaluation
app.py                FastAPI service
tests/                Feature, prediction and API tests
notebooks/            Executed analysis notebook
01_*.py ... 11_*.py   Original analysis and experiment scripts
Dockerfile            Container deployment
```

## Windows PowerShell setup

Python 3.11 is recommended because it is the deployment base image. From the
repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned dependency set is intended for Python 3.11. The local audit used
Python 3.14 where older pinned NumPy wheels are unavailable; use Python 3.11
for a clean installation.

## Train and test

The processed chronological splits are already included. To retrain the final
model and regenerate all final artifacts:

```powershell
python train.py
pytest -q
```

This writes `models/gold_volatility_xgboost_final.joblib`,
`models/model_metadata.json`, `reports/final_test_metrics.csv`,
`reports/test_predictions.csv`, and `reports/final_feature_importance.csv`.
The analysis notebook can be executed with:

```powershell
python -m jupyter nbconvert --to notebook --execute `
  notebooks/gold_volatility_analysis.ipynb --inplace `
  --ExecutePreprocessor.timeout=300
```

## API

Start the service:

```powershell
uvicorn app:app --reload
```

Open Swagger at <http://127.0.0.1:8000/docs>. `GET /health` checks that the
serialized model is available. `POST /predict` accepts at least 61 historical
observations because the 60-session feature needs 61 closes:

```json
{
  "history": [
    {
      "date": "2026-08-01",
      "open": 3350.0,
      "high": 3370.0,
      "low": 3340.0,
      "close": 3360.0
    }
  ]
}
```

The example above is intentionally abbreviated and is not sufficient by itself.
A successful response contains `as_of_date`, `observations_used`,
`predicted_future_5d_volatility_pct`, and the target definition. The API
calculates all features internally and never uses future observations.

## Docker

With Docker installed:

```powershell
docker build -t gold-volatility-api .
docker run --rm -p 8000:8000 gold-volatility-api
```

Then visit <http://localhost:8000/docs>. The image contains only the API,
shared code, pinned dependencies and final model artifact.

## Reproducibility and responsible use

`train.py` records feature order, target, parameters, date ranges and library
versions in `models/model_metadata.json`. The final test partition remains
held out during fitting. The notebook and original scripts preserve the EDA,
data-quality investigations, model comparisons, tuning, outlier analysis and
OHLC masking experiment.

This is an experimental study based on one historical test period and a
third-party futures proxy. Futures rolls, settlement conventions, data-quality
issues and regime changes can affect performance. Forecasts are not financial
advice and should not be used as a sole basis for investment decisions.
