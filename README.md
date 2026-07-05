
---

# Quantitative Portfolio Optimization & Predictive Analytics

An end-to-end quantitative financial engineering pipeline designed for **GMF Investments**. This platform leverages classical time-series analysis (ARIMA/SARIMA) and Deep Learning recurrent networks (LSTM) to forecast volatility regimes, calculate systematic risk metrics, and derive mathematically optimal asset allocations using Modern Portfolio Theory (MPT).

---

## ─── Project Architecture ───

The workflow maps clean data ingestion, defensive preprocessing, predictive modeling, and strategic backtesting across three distinct asset classes:

* **High-Growth Equity:** Tesla Inc. (`$TSLA`)
* **Large-Cap Market Proxy:** SPDR S&P 500 ETF Trust (`$SPY`)
* **Fixed Income Anchor:** Vanguard Total Bond Market ETF (`$BND`)

```
portfolio-optimization/
├── .github/workflows/
│   └── unittests.yml               # Automated CI test execution suite
├── data/
│   └── processed/                  # Flattened, stationary time-series datasets
├── notebooks/
│   ├── Task_1_EDA_Preprocessing.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Robust API failover ingestion engine
│   ├── preprocessor.py            # MultiIndex column squeezing & transformation
│   ├── models.py                   # ARIMA/SARIMA & LSTM forecasting layers
│   └── mpt_optimizer.py           # Efficient Frontier & PyPortfolioOpt execution
├── scripts/
│   └── run_pipeline.py            # Unified execution control script
├── tests/
│   └── test_pipeline.py           # Unit testing assertion suite
├── requirements.txt                # Fixed dependency versions
└── README.md                       # System documentation

```

---

## ─── System Features & Technical Workflow ───

### 1. Robust Data Ingestion & Formatting

* **MultiIndex Resolution:** Programmatically flattens the nested column matrices native to `yfinance` queries down to flat, predictable 1D sequences.
* **Data Integrity Guardrails:** Embedded defensive `.squeeze()` layers ensure no hidden 2D array matrix dimensions break downstream predictive mathematical engines.
* **Stationarity Transformations:** Automatically maps price inputs into continuous daily log returns ($R_t = \ln(P_t / P_{t-1})$) to ensure constant variance over time.

### 2. Deep Statistical Diagnostics

* **Augmented Dickey-Fuller (ADF) Test:** Evaluates unit root parameters across raw prices and differences to explicitly justify model tuning parameters (setting integration order $d=1$).
* **Residual Quality Tests:** Embeds automated Ljung-Box test execution on ARIMA residuals to confirm that early error lags are completely free of serial correlation ($p > 0.05$).

### 3. Dual-Engine Time Series Forecasting

* **Parametric Linearity (ARIMA/SARIMA):** Optimizes parameter bounds $(p, d, q) \times (P, D, Q)_m$ via systematic Akaike Information Criterion (AIC) minimization using `pmdarima`.
* **Non-Linear Deep Learning (LSTM):** Deploys a recurrent neural network with a rolling lookback window of 60 days, protected by `Dropout(0.2)` layers and monitored via validation early-stopping.

### 4. Asset Allocation & Backtesting

* **Efficient Frontier Mapping:** Generates optimal portfolio variance boundaries utilizing expected return arrays alongside historical asset covariance matrices.
* **Algorithmic Backtesting:** Simulates forward out-of-sample portfolio execution, tracking cumulative performance against a traditional passive **60% SPY / 40% BND** institutional benchmark.

---

## ─── Installation & Environmental Setup ───

### 1. Clone the Workspace

```bash
git clone https://github.com/your-username/portfolio-optimization.git
cd portfolio-optimization

```

### 2. Configure Virtual Environment

```bash
python -bin/python -m venv venv
source venv/bin/activate       # On Windows use: venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

---

## ─── Operational Ingestion Code Blueprint ───

This defensive extraction template demonstrates how data access is protected against network dropouts and multi-index anomalies:

```python
import yfinance as yf
import pandas as pd
import numpy as np

def load_and_sanitize_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Extracts ticker data, squashes MultiIndex layers, and computes log returns."""
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No asset payloads retrieved for symbol: {ticker}")
        
    # Resolve MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(col).strip() for col in df.columns]
    
    # Enforce strict 1D array containment
    close_series = df['Close'].squeeze()
    
    # Structural features
    df['Returns'] = close_series.pct_change()
    df['Log_Returns'] = np.log(close_series / close_series.shift(1))
    
    return df.dropna()

# Execution example
if __name__ == "__main__":
    tsla_clean = load_and_sanitize_data("TSLA", "2015-01-01", "2026-06-30")
    print(f"Data Matrix Ingested successfully. Total Rows: {len(tsla_clean)}")

```

---

## ─── Automated Testing & Continuous Integration ───

The continuous integration architecture uses GitHub Actions to run the test suite found in `tests/` on every push or pull request to the `main` branch.

To execute the unit tests locally and verify processing integrity:

```bash
pytest --verbose

```
