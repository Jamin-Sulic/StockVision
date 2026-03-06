"""
train_all_tickers.py
====================
Trainiert LSTM + XGBoost + Confidence Scorer für alle angegebenen Ticker.
Speichert Modelle unter models/saved_models/{ticker}/

Usage:
    python scripts/train_all_tickers.py
    python scripts/train_all_tickers.py --tickers MSFT GOOGL AMZN
    python scripts/train_all_tickers.py --skip-existing  # Bereits trainierte überspringen
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR   = PROJECT_ROOT / "models" / "saved_models"

TICKERS = [
    "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "TSLA", "NFLX", "JPM", "V", "AMD"
]

LOOKBACK    = 60
EPOCHS      = 100
BATCH_SIZE  = 32
TEST_SPLIT  = 0.35  # letzte 35% als Test

# ─── Feature Engineering ──────────────────────────────────────────────────────
def compute_features(df_raw):
    df = df_raw[["Open","High","Low","Close","Volume"]].copy()
    c = df["Close"]; h = df["High"]; l = df["Low"]; v = df["Volume"]

    df["return_1d"]  = c.pct_change(1)
    df["return_3d"]  = c.pct_change(3)
    df["return_5d"]  = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)
    df["return_21d"] = c.pct_change(21)

    MA7  = c.rolling(7).mean()
    MA30 = c.rolling(30).mean()
    MA90 = c.rolling(90).mean()
    df["dist_ma7"]    = (c - MA7)  / (MA7  + 1e-8)
    df["dist_ma30"]   = (c - MA30) / (MA30 + 1e-8)
    df["dist_ma90"]   = (c - MA90) / (MA90 + 1e-8)
    df["ma7_vs_ma30"] = (MA7 - MA30) / (MA30 + 1e-8)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_position"] = (c - bb_mid) / (2 * bb_std + 1e-8)
    df["bb_width"]    = (4 * bb_std) / (bb_mid + 1e-8)

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    ps    = c.rolling(26).mean()
    df["macd_norm"]        = macd         / (ps + 1e-8)
    df["macd_signal_norm"] = sig          / (ps + 1e-8)
    df["macd_hist"]        = (macd - sig) / (ps + 1e-8)

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-8)
    df["rsi"] = rs / (1 + rs)

    df["volatility"]   = df["return_1d"].rolling(21).std()
    df["volume_ratio"] = v / (v.rolling(21).mean() + 1e-8)
    df["atr_norm"]     = (h - l).rolling(14).mean() / (c + 1e-8)
    df["hl_spread"]    = (h - l) / (c + 1e-8)

    low14  = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stochastic"]  = (c - low14)   / (high14 - low14 + 1e-8)
    df["mom_5_norm"]  = c.diff(5)     / (c.shift(5)  + 1e-8)
    df["mom_10_norm"] = c.diff(10)    / (c.shift(10) + 1e-8)

    df["target_return"] = c.pct_change(1).shift(-1)
    df["next_close"]    = c.shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

FEAT_COLS = [
    "return_1d","return_3d","return_5d","return_10d","return_21d",
    "dist_ma7","dist_ma30","dist_ma90","ma7_vs_ma30",
    "bb_position","bb_width","macd_norm","macd_signal_norm","macd_hist",
    "rsi","volatility","volume_ratio","atr_norm","hl_spread","stochastic",
    "mom_5_norm","mom_10_norm"
]

# ─── LSTM Training ────────────────────────────────────────────────────────────
def train_lstm(ticker, df, save_dir):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.preprocessing import RobustScaler

    print(f"  [LSTM] Training...")

    features      = FEAT_COLS
    return_std    = df["target_return"].std()
    target_scaled = df["target_return"] / (return_std + 1e-8)

    scaler = RobustScaler()
    X_all  = scaler.fit_transform(df[features])
    y_all  = target_scaled.values

    # Sequences
    X_seq, y_seq = [], []
    for i in range(LOOKBACK, len(X_all)):
        X_seq.append(X_all[i - LOOKBACK:i])
        y_seq.append(y_all[i])
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_tr, X_te = X_seq[:split], X_seq[split:]
    y_tr, y_te = y_seq[:split], y_seq[split:]

    # Model
    def directional_loss(y_true, y_pred):
        mse  = tf.reduce_mean(tf.square(y_true - y_pred))
        dirn = tf.reduce_mean(tf.maximum(0.0, 1.0 - y_true * y_pred))
        return 0.5 * mse + 0.5 * dirn

    inp = keras.Input(shape=(LOOKBACK, len(features)))
    x   = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(inp)
    x   = keras.layers.Dropout(0.3)(x)
    x   = keras.layers.Bidirectional(keras.layers.LSTM(32))(x)
    x   = keras.layers.Dropout(0.2)(x)
    x   = keras.layers.Dense(16, activation="relu")(x)
    out = keras.layers.Dense(1)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss=directional_loss)

    cb = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5)
    ]
    model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_te, y_te), callbacks=cb, verbose=0)

    # Directional accuracy
    preds  = model.predict(X_te, verbose=0).flatten()
    actual = y_te
    dir_acc = np.mean(np.sign(preds) == np.sign(actual))
    print(f"  [LSTM] Directional Accuracy: {dir_acc:.4f}")

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(save_dir / "model.keras")
    with open(save_dir / "scaler.pkl", "wb") as f:
        pickle.dump({
            "feature_scaler": scaler,
            "return_std":     float(return_std),
            "lookback":       LOOKBACK,
            "features":       features
        }, f)
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "ticker":      ticker,
            "dir_acc":     float(dir_acc),
            "trained_at":  datetime.now().isoformat(),
            "n_train":     len(X_tr),
            "n_test":      len(X_te)
        }, f, indent=2)

    return model, scaler, return_std, dir_acc


# ─── XGBoost Training ─────────────────────────────────────────────────────────
def train_xgboost(ticker, df, lstm_model, lstm_scaler, lstm_return_std, save_dir):
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    print(f"  [XGB] Computing LSTM predictions for features...")

    features   = FEAT_COLS
    X_all_raw  = lstm_scaler.transform(df[features])

    # LSTM preds für alle Tage
    lstm_preds = np.zeros(len(df))
    for i in range(LOOKBACK, len(df)):
        seq = X_all_raw[i - LOOKBACK:i].reshape(1, LOOKBACK, len(features)).astype(np.float32)
        lstm_preds[i] = lstm_model.predict(seq, verbose=0)[0][0] * lstm_return_std

    df = df.copy()
    df["lstm_pred_return"] = lstm_preds

    # Target: 5-day forward return positiv?
    df["target_5d"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    df = df.dropna()

    xgb_features = features + ["lstm_pred_return"]
    X = df[xgb_features].values
    y = df["target_5d"].values

    split  = int(len(X) * (1 - TEST_SPLIT))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.01, n_estimators=400,
        min_child_weight=8, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
        early_stopping_rounds=30
    )
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              verbose=False)

    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    acc   = accuracy_score(y_te, preds)
    auc   = roc_auc_score(y_te, proba)
    print(f"  [XGB] Accuracy: {acc:.4f}  AUC: {auc:.4f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(save_dir / "model.json"))
    with open(save_dir / "metadata.json", "w") as f:
        json.dump({
            "ticker":     ticker,
            "accuracy":   float(acc),
            "auc":        float(auc),
            "trained_at": datetime.now().isoformat(),
            "n_train":    len(X_tr),
            "n_test":     len(X_te),
            "features":   xgb_features
        }, f, indent=2)

    # Return test predictions for scorer
    return model, acc, auc, df.iloc[split:].copy(), proba, preds


# ─── Scorer Training ──────────────────────────────────────────────────────────
def train_scorer(ticker, df_test, xgb_proba, xgb_preds, lstm_preds_test,
                 lstm_return_std, save_dir):
    print(f"  [Scorer] Calibrating confidence scorer...")

    weights = {"lstm_magnitude": 0.35, "xgb_confidence": 0.40, "agreement": 0.25}

    lstm_mag  = np.abs(lstm_preds_test) / (lstm_return_std + 1e-8)
    lstm_mag  = np.clip(lstm_mag, 0, 1)
    xgb_conf  = np.abs(xgb_proba - 0.5) * 2.0
    lstm_dir  = np.sign(lstm_preds_test)
    xgb_dir   = np.where(xgb_proba >= 0.5, 1, -1)
    agreement = (lstm_dir == xgb_dir).astype(float)

    scores = (
        weights["lstm_magnitude"] * lstm_mag +
        weights["xgb_confidence"] * xgb_conf +
        weights["agreement"]      * agreement
    )

    # Threshold: 60th percentile der scores
    threshold = float(np.percentile(scores, 60))
    trade_signals = (scores >= threshold) & (xgb_preds == 1)

    # Win rate bei trades
    actual_returns = df_test["target_return"].values[:len(scores)]
    if trade_signals.sum() > 0:
        win_rate = np.mean(actual_returns[trade_signals] > 0)
        agreement_rate = agreement.mean()
    else:
        win_rate = 0.5
        agreement_rate = 0.5

    print(f"  [Scorer] Threshold: {threshold:.4f}  Agreement: {agreement_rate:.4f}  Win Rate: {win_rate:.4f}")

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump({
            "ticker":         ticker,
            "threshold":      threshold,
            "weights":        weights,
            "agreement_rate": float(agreement_rate),
            "win_rate":       float(win_rate),
            "trained_at":     datetime.now().isoformat()
        }, f, indent=2)

    # Save test predictions
    pd.DataFrame({
        "score":      scores,
        "trade":      trade_signals,
        "xgb_signal": xgb_preds,
        "xgb_proba":  xgb_proba
    }).to_csv(save_dir / "test_predictions.csv", index=False)

    return threshold, win_rate


# ─── Supabase Update ─────────────────────────────────────────────────────────
def register_in_supabase(ticker, acc, auc, sharpe, n_rows):
    try:
        from dotenv import load_dotenv
        from supabase import create_client
        load_dotenv(PROJECT_ROOT / ".env")
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            print(f"  [Supabase] Skipped (no .env)")
            return
        sb = create_client(url, key)
        sb.table("available_models").upsert({
            "ticker":       ticker,
            "trained_at":   datetime.now().isoformat(),
            "test_accuracy": float(acc),
            "test_auc":     float(auc),
            "test_sharpe":  float(sharpe),
            "is_active":    True
        }, on_conflict="ticker").execute()
        print(f"  [Supabase] Registered {ticker}")
    except Exception as e:
        print(f"  [Supabase] Error: {e}")


# ─── Main Training Loop ───────────────────────────────────────────────────────
def train_ticker(ticker, skip_existing=False):
    import yfinance as yf

    ticker_dir   = MODELS_DIR / ticker.lower()
    lstm_dir     = ticker_dir / "lstm"
    xgb_dir      = ticker_dir / "xgb"
    scorer_dir   = ticker_dir / "scorer"

    if skip_existing and (lstm_dir / "model.keras").exists() and (xgb_dir / "model.json").exists():
        print(f"  Skipping {ticker} (already trained)")
        return True

    print(f"\n{'='*55}")
    print(f"  Training {ticker}")
    print(f"{'='*55}")

    # Download data
    print(f"  Downloading data...")
    try:
        raw = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        if len(raw) < 500:
            print(f"  Not enough data ({len(raw)} rows) – skipping")
            return False
        print(f"  {len(raw)} rows downloaded ({raw.index[0].date()} → {raw.index[-1].date()})")
    except Exception as e:
        print(f"  Download failed: {e}")
        return False

    # Features
    df = compute_features(raw)
    print(f"  {len(df)} rows after feature engineering")

    try:
        # LSTM
        lstm_model, lstm_scaler, lstm_return_std, lstm_acc = train_lstm(
            ticker, df, lstm_dir
        )

        # XGBoost
        xgb_model, xgb_acc, xgb_auc, df_test, xgb_proba, xgb_preds = train_xgboost(
            ticker, df, lstm_model, lstm_scaler, lstm_return_std, xgb_dir
        )

        # LSTM preds für test split
        split = int(len(df) * (1 - TEST_SPLIT))
        df_test_feats = df.iloc[split:].copy()
        X_test_raw    = lstm_scaler.transform(df_test_feats[FEAT_COLS])
        lstm_preds_test = np.zeros(len(df_test_feats))
        for i in range(LOOKBACK, len(X_test_raw)):
            seq = X_test_raw[i - LOOKBACK:i].reshape(1, LOOKBACK, len(FEAT_COLS)).astype(np.float32)
            lstm_preds_test[i] = lstm_model.predict(seq, verbose=0)[0][0] * lstm_return_std
        lstm_preds_test = lstm_preds_test[LOOKBACK:]

        # Align lengths
        min_len = min(len(lstm_preds_test), len(xgb_proba))
        df_scorer = df_test_feats.iloc[LOOKBACK:LOOKBACK + min_len]

        # Scorer
        threshold, win_rate = train_scorer(
            ticker, df_scorer,
            xgb_proba[:min_len], xgb_preds[:min_len],
            lstm_preds_test[:min_len], lstm_return_std,
            scorer_dir
        )

        # Register in Supabase
        register_in_supabase(ticker, xgb_acc, xgb_auc, win_rate * 2 - 1, len(df))

        print(f"\n  ✅ {ticker} trained successfully")
        print(f"     LSTM Dir.Acc: {lstm_acc:.4f}")
        print(f"     XGB Accuracy: {xgb_acc:.4f}  AUC: {xgb_auc:.4f}")
        print(f"     Scorer Win Rate: {win_rate:.4f}")
        return True

    except Exception as e:
        import traceback
        print(f"  ❌ {ticker} failed: {e}")
        traceback.print_exc()
        return False


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    print(f"\nStockVision — Multi-Ticker Training")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Models dir: {MODELS_DIR}")

    results = {}
    for ticker in args.tickers:
        success = train_ticker(ticker, skip_existing=args.skip_existing)
        results[ticker] = "✅" if success else "❌"

    print(f"\n{'='*55}")
    print(f"Training Summary:")
    for ticker, status in results.items():
        print(f"  {status} {ticker}")
    print(f"{'='*55}")
    print(f"\nNext step: run daily predictions for all tickers:")
    print(f"  python scripts/daily_predict.py --tickers {' '.join(args.tickers)}")