# ==========================================
# Daily Prediction Script
# ==========================================
# Täglich ausführen (z.B. 18:00 nach Marktschluss)
# Berechnet LSTM + XGBoost + Scorer für alle Ticker
# und speichert in Supabase predictions Tabelle
#
# Ausführen:
#   python scripts/daily_predict.py
#   python scripts/daily_predict.py --tickers AAPL MSFT GOOGL
#   python scripts/daily_predict.py --date 2024-12-20  (historisch)

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

# ==========================================
# Argumente
# ==========================================
parser = argparse.ArgumentParser(description="StockVision Daily Prediction")
parser.add_argument("--tickers", nargs="+", default=None,
                    help="Ticker Liste z.B. AAPL MSFT GOOGL")
parser.add_argument("--date", default=None,
                    help="Datum für historische Predictions (YYYY-MM-DD)")
parser.add_argument("--force", action="store_true",
                    help="Bestehende Predictions überschreiben")
args = parser.parse_args()

# ==========================================
# Setup
# ==========================================
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT",
    r"C:\Users\jamin\Desktop\Coding\Projects\StockVision-clean"))
load_dotenv(PROJECT_ROOT / ".env")

from supabase import create_client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
DATA_DIR   = PROJECT_ROOT / "data" / "processed"

# Ticker Liste
DEFAULT_TICKERS = os.environ.get("DEFAULT_TICKERS", "AAPL").split(",")
TICKERS = args.tickers if args.tickers else DEFAULT_TICKERS

# Datum
PRED_DATE = date.fromisoformat(args.date) if args.date else date.today()

print(f"{'='*55}")
print(f"StockVision Daily Prediction")
print(f"  Datum:   {PRED_DATE}")
print(f"  Ticker:  {TICKERS}")
print(f"{'='*55}")

# ==========================================
# Modell Loader
# ==========================================
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb

def directional_loss(y_true, y_pred):
    mse         = tf.reduce_mean(tf.square(y_true - y_pred))
    directional = tf.reduce_mean(tf.maximum(0.0, 1.0 - y_true * y_pred))
    return 0.5 * mse + 0.5 * directional

def load_models(ticker: str):
    """
    Lädt Modelle für einen Ticker.
    Fallback: AAPL-Modell falls kein eigenes vorhanden.
    """
    # Eigenes Modell suchen
    ticker_dir = MODELS_DIR / ticker.lower()
    aapl_dir   = MODELS_DIR / "lstm_price"

    if (ticker_dir / "lstm" / "model.keras").exists():
        model_dir    = ticker_dir
        lstm_dir     = ticker_dir / "lstm"
        xgb_dir      = ticker_dir / "xgb"
        scorer_dir   = ticker_dir / "scorer"
        is_own_model = True
        model_ticker = ticker
        print(f"  ✅ Eigenes Modell gefunden: {ticker}")
    else:
        # Fallback: AAPL Modell
        lstm_dir     = MODELS_DIR / "lstm_price"
        xgb_dir      = MODELS_DIR / "xgb_classifier"
        scorer_dir   = MODELS_DIR / "confidence_scorer"
        is_own_model = False
        model_ticker = "AAPL"
        print(f"  ⚠️  Kein eigenes Modell für {ticker} → verwende AAPL-Modell")

    # LSTM laden
    lstm_model = keras.models.load_model(
        lstm_dir / "model.keras",
        custom_objects={"directional_loss": directional_loss}
    )

    # Scaler + Metadata
    with open(lstm_dir / "scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)

    feature_scaler    = scaler_data["feature_scaler"]
    return_std        = scaler_data["return_std"]
    lookback          = scaler_data["lookback"]
    feature_cols      = scaler_data["features"]

    # XGBoost laden
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(xgb_dir / "model.json"))
    xgb_feat_cols = feature_cols + ["lstm_pred_return"]

    # Scorer Config
    scorer_cfg = json.loads((scorer_dir / "config.json").read_text())

    return {
        "lstm_model":      lstm_model,
        "xgb_model":       xgb_model,
        "feature_scaler":  feature_scaler,
        "return_std":      return_std,
        "lookback":        lookback,
        "feature_cols":    feature_cols,
        "xgb_feat_cols":   xgb_feat_cols,
        "scorer_cfg":      scorer_cfg,
        "is_own_model":    is_own_model,
        "model_ticker":    model_ticker,
    }

# ==========================================
# Feature Berechnung
# ==========================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet alle 22 stationären Features aus OHLCV."""
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
    df["bb_width"]    = (4 * bb_std) / (bb_mid + 1e-8)

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

    df["Next_Close"] = close.shift(-1)

    return df.replace([np.inf, -np.inf], np.nan)

# ==========================================
# Prediction für einen Ticker
# ==========================================
def predict_ticker(ticker: str, pred_date: date, models: dict) -> dict | None:
    """
    Lädt Daten, berechnet Features, gibt Prediction zurück.
    """
    import yfinance as yf

    # Lookback 90 + Feature-Berechnung braucht ~120 Tage minimum
    # 500 Tage zurück = sicher genug saubere Rows
    start = pred_date - timedelta(days=500)
    end   = pred_date + timedelta(days=1)

    print(f"  Lade Daten {start} → {end}...")
    raw = yf.download(ticker, start=str(start), end=str(end),
                      auto_adjust=True, progress=False)

    if raw.empty or len(raw) < models["lookback"] + 30:
        print(f"  ❌ Nicht genug Daten für {ticker}")
        return None

    # Spalten normalisieren
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw = raw[["Open","High","Low","Close","Volume"]].copy()

    # Features berechnen
    df = compute_features(raw)
    df = df.dropna(subset=models["feature_cols"]).copy()

    if len(df) < models["lookback"]:
        print(f"  ❌ Zu wenig saubere Rows für {ticker}")
        return None

    # Aktueller Kurs (letzter verfügbarer Tag)
    current_price = float(df["Close"].iloc[-1])
    current_date  = df.index[-1].date()

    print(f"  Aktueller Kurs ({current_date}): ${current_price:.2f}")

    # ---- LSTM Prediction ----
    feature_scaler = models["feature_scaler"]
    feature_cols   = models["feature_cols"]
    lookback       = models["lookback"]
    return_std     = models["return_std"]

    X_scaled = feature_scaler.transform(df[feature_cols])
    # Letzte `lookback` Tage als Sequenz
    X_seq    = X_scaled[-lookback:].reshape(1, lookback, len(feature_cols))
    X_seq    = X_seq.astype(np.float32)

    pred_return_scaled = models["lstm_model"].predict(X_seq, verbose=0)[0][0]
    pred_return        = float(pred_return_scaled * return_std)
    predicted_price    = float(current_price * (1 + pred_return))

    print(f"  LSTM pred_return: {pred_return*100:+.3f}%  → ${predicted_price:.2f}")

    # ---- XGBoost Prediction ----
    df["lstm_pred_return"] = pred_return  # gleicher Wert für letzten Tag
    X_xgb    = df[models["xgb_feat_cols"]].iloc[-1:].values
    xgb_pred = int(models["xgb_model"].predict(X_xgb)[0])
    xgb_prob = float(models["xgb_model"].predict_proba(X_xgb)[0][1])

    print(f"  XGBoost: {'UP' if xgb_pred==1 else 'DOWN'}  P(UP)={xgb_prob:.3f}")

    # ---- Confidence Scorer ----
    cfg          = models["scorer_cfg"]
    vol_mean     = cfg["vol_stats"]["mean"]
    vol_std_val  = cfg["vol_stats"]["std"]
    threshold    = cfg["threshold"]
    weights      = cfg["weights"]

    current_vol    = float(df["volatility"].iloc[-1])
    vol_zscore     = (current_vol - vol_mean) / (vol_std_val + 1e-8)
    vol_penalty    = float(np.clip(1.0 - max(0, vol_zscore - 1.5) * 0.2, 0.5, 1.0))

    lstm_magnitude = float(np.clip(abs(pred_return) / return_std, 0, 1))
    xgb_confidence = float(abs(xgb_prob - 0.5) * 2.0)
    lstm_direction = int(np.sign(pred_return))
    xgb_direction  = 1 if xgb_prob >= 0.5 else -1
    agreement      = lstm_direction == xgb_direction

    score = (
        weights["lstm_magnitude"] * lstm_magnitude +
        weights["xgb_confidence"] * xgb_confidence +
        weights["agreement"]      * float(agreement)
    ) * vol_penalty

    trade_signal = score >= threshold
    direction    = xgb_direction

    if score >= 0.30:
        confidence_label = "HIGH"
    elif score >= 0.20:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    print(f"  Scorer: score={score:.4f}  trade={'✅ TRADE' if trade_signal else '⏭ SKIP'}  "
          f"confidence={confidence_label}")

    return {
        "ticker":            ticker,
        "date":              str(current_date),
        "current_price":     round(current_price, 4),
        "predicted_price":   round(predicted_price, 4),
        "predicted_return":  round(pred_return, 6),
        "xgb_signal":        xgb_pred,
        "xgb_proba_up":      round(xgb_prob, 4),
        "score":             round(score, 4),
        "trade_signal":      bool(trade_signal),
        "direction":         direction,
        "confidence_label":  confidence_label,
        "lstm_magnitude":    round(lstm_magnitude, 4),
        "xgb_confidence":    round(xgb_confidence, 4),
        "agreement":         bool(agreement),
        "model_ticker":      models["model_ticker"],
        "is_own_model":      models["is_own_model"],
    }

# ==========================================
# Main Loop
# ==========================================
results  = []
errors   = []

for ticker in TICKERS:
    print(f"\n{'─'*40}")
    print(f"Processing: {ticker}")

    # Bereits vorhanden?
    if not args.force:
        existing = supabase.table("predictions")\
            .select("id")\
            .eq("ticker", ticker)\
            .eq("date", str(PRED_DATE))\
            .execute()
        if existing.data:
            print(f"  ⏭  Bereits vorhanden (--force zum Überschreiben)")
            continue

    try:
        # Modelle laden
        models = load_models(ticker)

        # Prediction berechnen
        pred = predict_ticker(ticker, PRED_DATE, models)

        if pred is None:
            errors.append(ticker)
            continue

        # In Supabase speichern
        supabase.table("predictions").upsert(
            pred, on_conflict="ticker,date"
        ).execute()

        results.append(pred)
        print(f"  ✅ Gespeichert in Supabase")

    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        errors.append(ticker)
        import traceback
        traceback.print_exc()

# ==========================================
# Zusammenfassung
# ==========================================
print(f"\n{'='*55}")
print(f"Daily Prediction Zusammenfassung — {PRED_DATE}")
print(f"{'='*55}")
print(f"  Erfolgreich: {len(results)}/{len(TICKERS)}")
if errors:
    print(f"  Fehler:      {errors}")

print(f"\n{'Ticker':<8} {'Kurs':>8} {'Pred':>8} {'Return':>8} {'Signal':<6} {'Score':>6} {'Conf':<8}")
print("─"*55)
for r in results:
    signal = "UP ✅" if r["xgb_signal"] == 1 else "DOWN ❌"
    print(f"  {r['ticker']:<6} "
          f"${r['current_price']:>7.2f} "
          f"${r['predicted_price']:>7.2f} "
          f"{r['predicted_return']*100:>+7.2f}% "
          f"{signal:<6} "
          f"{r['score']:>5.3f}  "
          f"{r['confidence_label']}")

print(f"\n✅ Fertig. Predictions in Supabase gespeichert.")
print(f"   Nächster Schritt: FastAPI Backend liest diese Daten.")