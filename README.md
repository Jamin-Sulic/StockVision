# StockVision – ML Stock Prediction Experiment

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red?logo=xgboost)
![FastAPI](https://img.shields.io/badge/FastAPI-Railway-green?logo=fastapi)
![Next.js](https://img.shields.io/badge/Next.js-Vercel-black?logo=next.js)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?logo=supabase)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Not financial advice.** This is a personal deep learning experiment for educational and research purposes only.

---

## About

StockVision explores whether technical indicators alone can predict short-term stock price direction using a two-model ML pipeline:

1. **Bidirectional LSTM** — learns temporal patterns from 60-day sequences of 22 engineered features (RSI, MACD, Bollinger Bands, momentum, volatility, ATR, stochastic oscillator)
2. **XGBoost Classifier** — takes the LSTM's predicted return as an additional input alongside raw features to generate a directional UP/DOWN signal
3. **Confidence Scorer** — combines LSTM magnitude, XGBoost probability, and model agreement to filter low-confidence trades

Models are trained per-ticker on 10+ years of OHLCV data sourced via **yfinance**.

---

## Features

- Live daily predictions for 8 major tickers (AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, V)
- Bidirectional LSTM + XGBoost ensemble with confidence scoring
- Historical backtesting with equity curves (Buy & Hold vs XGBoost vs Scorer)
- FastAPI backend on Railway with Supabase caching
- GitHub Actions cron job runs predictions every weekday at 22:00 UTC
- Next.js dashboard on Vercel with real-time signal visualization

---

## Tech Stack

### Machine Learning
- TensorFlow / Keras — Bidirectional LSTM for time series prediction
- XGBoost — Directional signal classifier
- Scikit-learn — Feature scaling (RobustScaler), metrics
- Pandas, NumPy — Feature engineering (22 technical indicators)
- yFinance — Historical OHLCV data
- Jupyter — Model development & experimentation

### Backend
- FastAPI — Prediction & backtest API
- Uvicorn — ASGI server
- Railway — Cloud deployment
- Supabase — PostgreSQL database for predictions & backtest cache
- GitHub Actions — Daily prediction cron job (22:00 UTC weekdays)

### Frontend
- Next.js (React + TypeScript) — Dashboard UI
- TailwindCSS — Styling
- Recharts — Equity curve & price charts
- Vercel — Frontend deployment

---

## Project Structure

```
StockVision/
├── models/
│   └── saved_models/{ticker}/
│       ├── lstm/        # model.h5 + scaler.pkl
│       ├── xgb/         # model.json
│       └── scorer/      # config.json (threshold, weights, vol_stats)
├── scripts/
│   ├── train_all_tickers.py   # Train LSTM + XGBoost + Scorer per ticker
│   ├── daily_predict.py       # Daily prediction pipeline → Supabase
│   └── upload_market_data.py  # Upload historical OHLCV to Supabase
├── backend/
│   └── main.py                # FastAPI: /predict/live, /predict/historical
├── frontend/
│   └── app/page.tsx           # Next.js dashboard
├── .github/
│   └── workflows/
│       └── daily_predict.yml  # GitHub Actions cron
└── notebooks/                 # Jupyter experimentation
```

---

## Setup

### 1. Clone & install
```bash
git clone https://github.com/Jamin-Sulic/StockVision.git
cd StockVision
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Environment variables
Create a `.env` file in the root:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key
SUPABASE_ANON_KEY=your_anon_key
```

### 3. Train models
```bash
python scripts/train_all_tickers.py --tickers AAPL MSFT GOOGL
```

### 4. Upload market data
```bash
python scripts/upload_market_data.py
```

### 5. Run daily predictions
```bash
python scripts/daily_predict.py --tickers AAPL MSFT GOOGL
```

### 6. Start backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### 7. Start frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Model Performance (Test Set)

| Ticker | Accuracy | AUC   | Win Rate |
|--------|----------|-------|----------|
| JPM    | 60.1%    | 0.549 | 56.6%    |
| AAPL   | 60.1%    | 0.521 | 53.2%    |
| NVDA   | 59.4%    | 0.516 | 53.5%    |
| V      | 57.8%    | 0.546 | 54.3%    |
| MSFT   | 57.4%    | 0.559 | 51.0%    |
| META   | 57.4%    | 0.484 | 51.5%    |
| AMZN   | 58.1%    | 0.579 | 51.5%    |
| GOOGL  | 56.0%    | 0.465 | 51.0%    |

---

## Live Demo

**Frontend:** [stockvision-alpha.vercel.app](https://stockvision-alpha.vercel.app)  
**API:** [stockvision-production.up.railway.app](https://stockvision-production.up.railway.app/health)

---

## Author

**Jamin Sulic** — [jaminsulic.com](https://jaminsulic.com) · [GitHub](https://github.com/Jamin-Sulic)

---

*For educational purposes only. Past model performance does not guarantee future results.*