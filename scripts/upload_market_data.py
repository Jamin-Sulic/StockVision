"""
upload_market_data.py
=====================
Lädt historische Marktdaten für alle Ticker in Supabase hoch.
Wird für die Price Charts im Frontend benötigt.

Usage:
    python scripts/upload_market_data.py
    python scripts/upload_market_data.py --tickers MSFT GOOGL
"""

import os
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from supabase import create_client
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V"]
BATCH_SIZE = 500

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
    df["macd_norm"]        = macd / (ps + 1e-8)
    df["macd_signal_norm"] = sig  / (ps + 1e-8)
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

    df["next_close"] = c.shift(-1)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def upload_ticker(ticker: str):
    import yfinance as yf

    print(f"\n{'─'*40}")
    print(f"Uploading: {ticker}")

    raw = yf.download(ticker, start="2015-01-01", auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    if raw.empty:
        print(f"  ❌ No data found")
        return False

    df = compute_features(raw)
    print(f"  {len(df)} rows computed")

    records = []
    for dt, row in df.iterrows():
        record = {"ticker": ticker, "date": str(dt.date())}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            if col == "Volume":
                record["volume"] = int(val)
            elif col in ["Open","High","Low","Close"]:
                record[col.lower()] = round(float(val), 4)
            else:
                record[col.lower()] = round(float(val), 6)
        records.append(record)

    uploaded = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("market_data").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
            uploaded += len(batch)
            print(f"  {uploaded}/{len(records)} rows...", end="\r")
        except Exception as e:
            print(f"\n  ❌ Batch error: {e}")

    print(f"  ✅ {ticker}: {uploaded} rows uploaded        ")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    args = parser.parse_args()

    print(f"StockVision — Market Data Upload")
    print(f"Tickers: {', '.join(args.tickers)}")

    success = []
    failed  = []
    for ticker in args.tickers:
        if upload_ticker(ticker):
            success.append(ticker)
        else:
            failed.append(ticker)

    print(f"\n{'='*40}")
    print(f"Done: {len(success)}/{len(args.tickers)} uploaded")
    if failed:
        print(f"Failed: {failed}")