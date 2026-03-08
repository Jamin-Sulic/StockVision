# ==========================================
# StockVision FastAPI Backend
# ==========================================
# Ausführen:
#   uvicorn backend.main:app --reload --port 8000
#
# Endpoints:
#   GET /tickers                          → verfügbare Ticker
#   GET /predict/live/{ticker}            → neueste Prediction
#   GET /predict/historical/{ticker}      → Backtest on-demand
#   GET /health                           → Health Check

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

# ==========================================
# Setup
# ==========================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT",
    r"C:\Users\jamin\Desktop\Coding\Projects\StockVision-clean"))
load_dotenv(PROJECT_ROOT / ".env")

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

app = FastAPI(
    title="StockVision API",
    description="LSTM + XGBoost Stock Prediction API",
    version="1.0.0"
)

# CORS für Next.js Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Next.js lokal
        "https://*.vercel.app",       # Vercel deployment
        "*"                           # Temporär für Development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Health Check
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

# ==========================================
# GET /tickers
# Gibt alle Ticker zurück die Predictions haben
# ==========================================
@app.get("/tickers")
def get_tickers():
    """
    Gibt verfügbare Ticker zurück mit letzter Prediction.
    """
    try:
        # Ticker mit eigenen Modellen
        models_res = supabase.table("available_models")\
            .select("ticker, test_accuracy, test_auc, test_sharpe, trained_at")\
            .eq("is_active", True)\
            .execute()

        # Alle Ticker die jemals eine Prediction hatten
        preds_res = supabase.table("predictions")\
            .select("ticker")\
            .execute()

        pred_tickers = list(set([r["ticker"] for r in preds_res.data]))

        return {
            "tickers_with_models": [r["ticker"] for r in models_res.data],
            "tickers_with_predictions": pred_tickers,
            "models": models_res.data,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# GET /predict/live/{ticker}
# Neueste Prediction für einen Ticker
# ==========================================
@app.get("/predict/live/{ticker}")
def get_live_prediction(ticker: str):
    """
    Gibt die neueste Prediction für einen Ticker zurück.
    Daten kommen aus der täglich aktualisierten predictions Tabelle.
    """
    ticker = ticker.upper()

    try:
        res = supabase.table("predictions")\
            .select("*")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()

        if not res.data:
            raise HTTPException(
                status_code=404,
                detail=f"Keine Prediction für {ticker} gefunden. "
                       f"Script ausführen: python scripts/daily_predict.py --tickers {ticker}"
            )

        pred = res.data[0]

        # Letzte 90 Tage Kursdaten für Chart
        chart_res = supabase.table("market_data")\
            .select("date, close, return_1d, rsi, macd_hist, volatility")\
            .eq("ticker", ticker)\
            .order("date", desc=True)\
            .limit(90)\
            .execute()

        chart_data = sorted(chart_res.data, key=lambda x: x["date"])

        # Signal leserlich machen
        direction_label = "UP" if pred["direction"] == 1 else "DOWN"
        signal_emoji    = "📈" if pred["xgb_signal"] == 1 else "📉"

        return {
            "ticker":           ticker,
            "date":             pred["date"],
            "current_price":    pred["current_price"],
            "predicted_price":  pred["predicted_price"],
            "predicted_return": pred["predicted_return"],
            "signal": {
                "direction":         direction_label,
                "emoji":             signal_emoji,
                "xgb_proba_up":      pred["xgb_proba_up"],
                "score":             pred["score"],
                "trade":             pred["trade_signal"],
                "confidence":        pred["confidence_label"],
                "lstm_magnitude":    pred["lstm_magnitude"],
                "xgb_confidence":    pred["xgb_confidence"],
                "agreement":         pred["agreement"],
            },
            "model": {
                "ticker":        pred["model_ticker"],
                "is_own_model":  pred["is_own_model"],
            },
            "chart_data": chart_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# GET /predict/historical/{ticker}
# Backtest für einen Zeitraum
# ==========================================
@app.get("/predict/historical/{ticker}")
def get_historical(
    ticker: str,
    start:  str = Query(..., description="Start Datum YYYY-MM-DD"),
    end:    str = Query(..., description="End Datum YYYY-MM-DD"),
):
    """
    Berechnet Backtest für Ticker + Zeitraum.
    Cached in backtest_results Tabelle.
    """
    ticker = ticker.upper()

    try:
        start_date = date.fromisoformat(start)
        end_date   = date.fromisoformat(end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Datum Format: YYYY-MM-DD")

    if (end_date - start_date).days < 30:
        raise HTTPException(status_code=400, detail="Mindestens 30 Tage Zeitraum")

    if (end_date - start_date).days > 365 * 5:
        raise HTTPException(status_code=400, detail="Maximal 5 Jahre Zeitraum")

    # Cache prüfen
    try:
        cached = supabase.table("backtest_results")\
            .select("*")\
            .eq("ticker", ticker)\
            .eq("start_date", str(start_date))\
            .eq("end_date", str(end_date))\
            .execute()

        if cached.data:
            print(f"Cache hit: {ticker} {start_date} → {end_date}")
            return _format_backtest_response(cached.data[0], from_cache=True)
    except Exception:
        pass

    # Immer frisch via yfinance berechnen (alle Features vorhanden)
    try:
        data_res_data = _fetch_and_compute(ticker, start_date, end_date)
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Keine Daten für {ticker} im Zeitraum {start} bis {end}. Fehler: {str(e)}"
        )

    # Backtest berechnen
    result = _run_backtest(ticker, data_res_data, start_date, end_date)

    # Cachen
    try:
        supabase.table("backtest_results").upsert(
            result, on_conflict="ticker,start_date,end_date"
        ).execute()
    except Exception as e:
        print(f"Cache save fehler: {e}")

    return _format_backtest_response(result, from_cache=False)


def _fetch_and_compute(ticker: str, start: date, end: date) -> list:
    """Lädt Daten via yfinance und berechnet alle Features inline."""
    import yfinance as yf

    raw = yf.download(ticker,
                      start=str(start - timedelta(days=150)),
                      end=str(end + timedelta(days=1)),
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"Keine yfinance Daten für {ticker}")

    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    df = raw[["Open","High","Low","Close","Volume"]].copy()

    # Alle 22 Features berechnen
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    df["return_1d"]  = close.pct_change(1)
    df["return_3d"]  = close.pct_change(3)
    df["return_5d"]  = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)
    df["return_21d"] = close.pct_change(21)

    MA7  = close.rolling(7).mean()
    MA30 = close.rolling(30).mean()
    MA90 = close.rolling(90).mean()
    df["dist_ma7"]    = (close - MA7)  / (MA7  + 1e-8)
    df["dist_ma30"]   = (close - MA30) / (MA30 + 1e-8)
    df["dist_ma90"]   = (close - MA90) / (MA90 + 1e-8)
    df["ma7_vs_ma30"] = (MA7   - MA30) / (MA30 + 1e-8)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_position"] = (close - bb_mid) / (2 * bb_std + 1e-8)
    df["bb_width"]    = (4 * bb_std)     / (bb_mid + 1e-8)

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    ps    = close.rolling(26).mean()
    df["macd_norm"]        = macd / (ps + 1e-8)
    df["macd_signal_norm"] = sig  / (ps + 1e-8)
    df["macd_hist"]        = (macd - sig) / (ps + 1e-8)

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi"] = rs / (1 + rs)

    df["volatility"]   = df["return_1d"].rolling(21).std()
    df["volume_ratio"] = volume / (volume.rolling(21).mean() + 1e-8)
    df["atr_norm"]     = (high - low).rolling(14).mean() / (close + 1e-8)
    df["hl_spread"]    = (high - low) / (close + 1e-8)

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df["stochastic"]  = (close - low14) / (high14 - low14 + 1e-8)
    df["mom_5_norm"]  = close.diff(5)  / (close.shift(5)  + 1e-8)
    df["mom_10_norm"] = close.diff(10) / (close.shift(10) + 1e-8)

    df["next_close"] = close.shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Auf Zeitraum einschränken
    df = df.loc[str(start):str(end)].dropna()

    records = []
    for dt, row in df.iterrows():
        record = {"ticker": ticker, "date": str(dt.date())}
        for col in df.columns:
            val = row[col]
            if not pd.isna(val):
                record[col.lower()] = float(val)
        records.append(record)
    return records


def _run_backtest(ticker: str, data: list, start_date: date, end_date: date) -> dict:
    """Berechnet Backtest mit XGBoost (kein TF nötig)."""
    import xgboost as xgb
    import pickle

    df = pd.DataFrame(data).sort_values("date").reset_index(drop=True)

    # next_close aus close berechnen falls nicht vorhanden
    if "next_close" not in df.columns or df["next_close"].isna().all():
        df["next_close"] = df["close"].shift(-1)
    df["next_close"] = df["next_close"].fillna(method="ffill")

    # XGBoost Modell laden
    xgb_dir = PROJECT_ROOT / "models" / "saved_models" / ticker.lower() / "xgb"
    if not (xgb_dir / "model.json").exists():
        xgb_dir = PROJECT_ROOT / "models" / "saved_models" / "aapl" / "xgb"

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_dir / "model.json"))

    # LSTM direkt berechnen für alle Backtest-Tage
    import tensorflow as tf
    from tensorflow import keras
    import pickle

    def directional_loss(y_true, y_pred):
        mse         = tf.reduce_mean(tf.square(y_true - y_pred))
        directional = tf.reduce_mean(tf.maximum(0.0, 1.0 - y_true * y_pred))
        return 0.5 * mse + 0.5 * directional

    lstm_dir = PROJECT_ROOT / "models" / "saved_models" / ticker.lower() / "lstm"
    if not (lstm_dir / "model.h5").exists():
        lstm_dir = PROJECT_ROOT / "models" / "saved_models" / "aapl" / "lstm"

    # Unterstützt h5 und keras
    model_path = lstm_dir / "model.h5"
    if not model_path.exists():
        model_path = lstm_dir / "model.keras"

    lstm_model = keras.models.load_model(
        model_path,
        custom_objects={"directional_loss": directional_loss}
    )
    with open(lstm_dir / "scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)

    feature_scaler = scaler_data["feature_scaler"]
    return_std     = scaler_data["return_std"]
    lookback       = scaler_data["lookback"]
    feat_cols_lstm = scaler_data["features"]

    # Rohdaten mit extra History für Lookback laden
    import yfinance as yf
    raw_full = yf.download(ticker,
                           start=str(start_date - timedelta(days=300)),
                           end=str(end_date + timedelta(days=1)),
                           auto_adjust=True, progress=False)
    raw_full.columns = [c[0] if isinstance(c, tuple) else c for c in raw_full.columns]
    df_full = raw_full[["Open","High","Low","Close","Volume"]].copy()

    # Alle Features berechnen
    c = df_full["Close"]; h = df_full["High"]; l = df_full["Low"]; v = df_full["Volume"]
    df_full["return_1d"]  = c.pct_change(1);  df_full["return_3d"]  = c.pct_change(3)
    df_full["return_5d"]  = c.pct_change(5);  df_full["return_10d"] = c.pct_change(10)
    df_full["return_21d"] = c.pct_change(21)
    MA7=c.rolling(7).mean(); MA30=c.rolling(30).mean(); MA90=c.rolling(90).mean()
    df_full["dist_ma7"]=(c-MA7)/(MA7+1e-8); df_full["dist_ma30"]=(c-MA30)/(MA30+1e-8)
    df_full["dist_ma90"]=(c-MA90)/(MA90+1e-8); df_full["ma7_vs_ma30"]=(MA7-MA30)/(MA30+1e-8)
    bb_mid=c.rolling(20).mean(); bb_std=c.rolling(20).std()
    df_full["bb_position"]=(c-bb_mid)/(2*bb_std+1e-8); df_full["bb_width"]=(4*bb_std)/(bb_mid+1e-8)
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean()
    macd=ema12-ema26; sig=macd.ewm(span=9).mean(); ps=c.rolling(26).mean()
    df_full["macd_norm"]=macd/(ps+1e-8); df_full["macd_signal_norm"]=sig/(ps+1e-8)
    df_full["macd_hist"]=(macd-sig)/(ps+1e-8)
    delta=c.diff(); gain=delta.clip(lower=0).rolling(14).mean()
    loss=(-delta.clip(upper=0)).rolling(14).mean(); rs=gain/(loss+1e-8)
    df_full["rsi"]=rs/(1+rs)
    df_full["volatility"]=df_full["return_1d"].rolling(21).std()
    df_full["volume_ratio"]=v/(v.rolling(21).mean()+1e-8)
    df_full["atr_norm"]=(h-l).rolling(14).mean()/(c+1e-8)
    df_full["hl_spread"]=(h-l)/(c+1e-8)
    low14=l.rolling(14).min(); high14=h.rolling(14).max()
    df_full["stochastic"]=(c-low14)/(high14-low14+1e-8)
    df_full["mom_5_norm"]=c.diff(5)/(c.shift(5)+1e-8)
    df_full["mom_10_norm"]=c.diff(10)/(c.shift(10)+1e-8)
    df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna()

    avail_lstm = [col for col in feat_cols_lstm if col in df_full.columns]
    X_scaled   = feature_scaler.transform(df_full[avail_lstm])
    dates_full = df_full.index.tolist()

    lstm_pred_map = {}
    for i in range(lookback, len(df_full)):
        dt_str = str(dates_full[i].date())
        if dt_str < str(start_date) or dt_str > str(end_date):
            continue
        seq = X_scaled[i-lookback:i].reshape(1, lookback, len(avail_lstm)).astype(np.float32)
        pred_scaled = lstm_model.predict(seq, verbose=0)[0][0]
        lstm_pred_map[dt_str] = float(pred_scaled * return_std)

    print(f"DEBUG: LSTM preds: {len(lstm_pred_map)}, sample: {list(lstm_pred_map.items())[:2]}")

    # Features für XGBoost
    feat_cols = [
        "return_1d","return_3d","return_5d","return_10d","return_21d",
        "dist_ma7","dist_ma30","dist_ma90","ma7_vs_ma30",
        "bb_position","bb_width","macd_norm","macd_signal_norm","macd_hist",
        "rsi","volatility","volume_ratio","atr_norm","hl_spread","stochastic",
        "mom_5_norm","mom_10_norm"
    ]
    available_feats = [c for c in feat_cols if c in df.columns]
    df["lstm_pred_return"] = df["date"].map(lstm_pred_map).fillna(0.0)

    xgb_feats = available_feats + ["lstm_pred_return"]
    X = df[xgb_feats].fillna(0).values

    df["xgb_signal"] = xgb_model.predict(X)
    df["xgb_proba"]  = xgb_model.predict_proba(X)[:, 1]
    print(f"DEBUG: xgb_signal counts: {df['xgb_signal'].value_counts().to_dict()}")
    print(f"DEBUG: xgb_proba sample: {df['xgb_proba'].describe()}")
    print(f"DEBUG: X shape: {X.shape}, n_feats expected: {xgb_model.n_features_in_}")

    # Scorer
    scorer_path = PROJECT_ROOT / "models" / "saved_models" / ticker.lower() / "scorer" / "config.json"
    if not scorer_path.exists():
        scorer_path = PROJECT_ROOT / "models" / "saved_models" / "aapl" / "scorer" / "config.json"
    scorer_cfg = json.loads(scorer_path.read_text())
    threshold  = scorer_cfg["threshold"]
    weights    = scorer_cfg["weights"]

    return_std_val = scaler_data["return_std"]

    lstm_magnitude = df["lstm_pred_return"].abs() / (return_std_val + 1e-8)
    lstm_magnitude = lstm_magnitude.clip(0, 1)
    xgb_confidence = (df["xgb_proba"] - 0.5).abs() * 2.0
    lstm_direction = df["lstm_pred_return"].apply(lambda x: 1 if x >= 0 else -1)
    xgb_direction  = df["xgb_proba"].apply(lambda x: 1 if x >= 0.5 else -1)
    agreement      = (lstm_direction == xgb_direction).astype(float)

    df["score"] = (
        weights["lstm_magnitude"] * lstm_magnitude +
        weights["xgb_confidence"] * xgb_confidence +
        weights["agreement"]      * agreement
    )
    df["trade"] = (df["score"] >= threshold) & (df["xgb_signal"] == 1)
    print(f"DEBUG: Scorer threshold={threshold}, trades={df['trade'].sum()}, score_mean={df['score'].mean():.4f}")

    # Backtest Engine
    close      = df["close"].values.astype(float)
    next_close = df["next_close"].values.astype(float)
    n          = len(df)
    COST       = 0.001

    def calc_equity(signals):
        cap    = 10_000.0
        equity = [cap]
        trades = []
        for i in range(n - 1):
            if signals[i] == 1 and not np.isnan(next_close[i]):
                ret  = (next_close[i] - close[i]) / close[i] - COST
                pnl  = cap * ret
                cap += pnl
                trades.append({
                    "date":    df["date"].iloc[i],
                    "return":  round(ret, 6),
                    "correct": ret > 0,
                })
            equity.append(cap)
        return np.array(equity), trades

    eq_bh,     tr_bh     = calc_equity(np.ones(n, dtype=int))
    eq_xgb,    tr_xgb    = calc_equity(df["xgb_signal"].values.astype(int))
    eq_scorer, tr_scorer = calc_equity(df["trade"].astype(int).values)

    def metrics(eq, trades):
        tr   = (eq[-1] - 10_000) / 10_000
        yr   = len(eq) / 252
        ar   = (1 + tr) ** (1 / yr) - 1 if yr > 0 else 0
        dr   = np.diff(eq) / eq[:-1]
        sh   = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0
        peak = np.maximum.accumulate(eq)
        mdd  = float(((eq - peak) / peak).min())
        wr   = float(np.mean([t["correct"] for t in trades])) if trades else 0
        wins = sum(t["return"] for t in trades if t["return"] > 0)
        loss = abs(sum(t["return"] for t in trades if t["return"] < 0))
        pf   = wins / loss if loss > 0 else 0
        return {
            "total_return":  round(tr, 4),
            "annual_return": round(ar, 4),
            "sharpe":        round(sh, 4),
            "max_drawdown":  round(mdd, 4),
            "win_rate":      round(wr, 4),
            "profit_factor": round(pf, 4),
            "n_trades":      len(trades),
            "final_capital": round(float(eq[-1]), 2),
        }

    m_bh     = metrics(eq_bh,     tr_bh)
    m_xgb    = metrics(eq_xgb,    tr_xgb)
    m_scorer = metrics(eq_scorer, tr_scorer)

    dates = df["date"].tolist()

    return {
        "ticker":      ticker,
        "start_date":  str(start_date),
        "end_date":    str(end_date),
        # Buy & Hold
        "bh_total_return":    m_bh["total_return"],
        "bh_annual_return":   m_bh["annual_return"],
        "bh_sharpe":          m_bh["sharpe"],
        "bh_max_drawdown":    m_bh["max_drawdown"],
        "bh_final_capital":   m_bh["final_capital"],
        # XGBoost
        "xgb_total_return":   m_xgb["total_return"],
        "xgb_annual_return":  m_xgb["annual_return"],
        "xgb_sharpe":         m_xgb["sharpe"],
        "xgb_max_drawdown":   m_xgb["max_drawdown"],
        "xgb_win_rate":       m_xgb["win_rate"],
        "xgb_profit_factor":  m_xgb["profit_factor"],
        "xgb_n_trades":       m_xgb["n_trades"],
        "xgb_final_capital":  m_xgb["final_capital"],
        # Scorer
        "scorer_total_return":  m_scorer["total_return"],
        "scorer_annual_return": m_scorer["annual_return"],
        "scorer_sharpe":        m_scorer["sharpe"],
        "scorer_max_drawdown":  m_scorer["max_drawdown"],
        "scorer_win_rate":      m_scorer["win_rate"],
        "scorer_profit_factor": m_scorer["profit_factor"],
        "scorer_n_trades":      m_scorer["n_trades"],
        "scorer_final_capital": m_scorer["final_capital"],
        # Charts
        "equity_bh":     json.dumps(eq_bh.tolist()),
        "equity_xgb":    json.dumps(eq_xgb.tolist()),
        "equity_scorer": json.dumps(eq_scorer.tolist()),
        "trade_dates":   json.dumps(dates),
    }


def _format_backtest_response(data: dict, from_cache: bool) -> dict:
    """Formatiert Backtest Ergebnis für Frontend."""
    return {
        "ticker":     data["ticker"],
        "period":     {"start": data["start_date"], "end": data["end_date"]},
        "from_cache": from_cache,
        "strategies": {
            "buy_and_hold": {
                "total_return":  data["bh_total_return"],
                "annual_return": data["bh_annual_return"],
                "sharpe":        data["bh_sharpe"],
                "max_drawdown":  data["bh_max_drawdown"],
                "final_capital": data["bh_final_capital"],
            },
            "xgboost": {
                "total_return":  data["xgb_total_return"],
                "annual_return": data["xgb_annual_return"],
                "sharpe":        data["xgb_sharpe"],
                "max_drawdown":  data["xgb_max_drawdown"],
                "win_rate":      data["xgb_win_rate"],
                "profit_factor": data["xgb_profit_factor"],
                "n_trades":      data["xgb_n_trades"],
                "final_capital": data["xgb_final_capital"],
            },
            "xgboost_scorer": {
                "total_return":  data["scorer_total_return"],
                "annual_return": data["scorer_annual_return"],
                "sharpe":        data["scorer_sharpe"],
                "max_drawdown":  data["scorer_max_drawdown"],
                "win_rate":      data["scorer_win_rate"],
                "profit_factor": data["scorer_profit_factor"],
                "n_trades":      data["scorer_n_trades"],
                "final_capital": data["scorer_final_capital"],
            },
        },
        "charts": {
            "equity_bh":     json.loads(data["equity_bh"])     if data.get("equity_bh")     else [],
            "equity_xgb":    json.loads(data["equity_xgb"])    if data.get("equity_xgb")    else [],
            "equity_scorer": json.loads(data["equity_scorer"]) if data.get("equity_scorer") else [],
            "dates":         json.loads(data["trade_dates"])   if data.get("trade_dates")   else [],
        },
    }