# ==========================================
# Supabase Setup & Connection Test
# ==========================================
# pip install supabase python-dotenv
#
# .env Datei erstellen in PROJECT_ROOT:
#   SUPABASE_URL=https://xxxx.supabase.co
#   SUPABASE_SERVICE_KEY=eyJ...  (service_role key)
#   SUPABASE_ANON_KEY=eyJ...     (anon key)

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# Setup
# ==========================================
PROJECT_ROOT = Path(r"C:\Users\jamin\Desktop\Coding\Projects\StockVision-clean")
load_dotenv(PROJECT_ROOT / ".env")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "SUPABASE_URL und SUPABASE_SERVICE_KEY fehlen in .env\n"
        "➡️ Supabase Dashboard → Settings → API → Keys kopieren"
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"✅ Supabase verbunden: {SUPABASE_URL}")

# ==========================================
# Test 1: Tabellen prüfen
# ==========================================
print("\nTest 1: Tabellen prüfen...")
tables = ["predictions", "market_data", "backtest_results", "available_models"]
for table in tables:
    try:
        res = supabase.table(table).select("id").limit(1).execute()
        print(f"  ✅ {table}")
    except Exception as e:
        print(f"  ❌ {table}: {e}")

# ==========================================
# Test 2: AAPL Market Data hochladen
# ==========================================
print("\nTest 2: AAPL Market Data hochladen...")

DATA_DIR   = PROJECT_ROOT / "data" / "processed"
feat_path  = DATA_DIR / "AAPL_features_stationary.csv"

if not feat_path.exists():
    print(f"  ⚠️  {feat_path} nicht gefunden → skip")
else:
    feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)

    # Spalten die wir hochladen
    upload_cols = [
        "Open", "High", "Low", "Close", "Volume", "Next_Close",
        "return_1d", "return_3d", "return_5d", "return_10d", "return_21d",
        "dist_ma7", "dist_ma30", "dist_ma90", "ma7_vs_ma30",
        "bb_position", "bb_width", "macd_norm", "macd_signal_norm", "macd_hist",
        "rsi", "stochastic", "mom_5_norm", "mom_10_norm",
        "volatility", "volume_ratio", "atr_norm", "hl_spread",
    ]
    existing_cols = [c for c in upload_cols if c in feat.columns]
    feat_upload   = feat[existing_cols].copy()
    feat_upload   = feat_upload.replace([np.inf, -np.inf], np.nan).dropna()

    # Batch Upload (Supabase max 1000 rows per request)
    records = []
    for dt, row in feat_upload.iterrows():
        record = {
            "ticker": "AAPL",
            "date":   str(dt.date()),
            "close":  float(row.get("Close", 0)),
        }
        # OHLCV
        for col in ["Open","High","Low","Volume","Next_Close"]:
            if col in row:
                key = col.lower()
                record[key] = float(row[col]) if col != "Volume" else int(row[col])

        # Features (lowercase mapping)
        feature_map = {
            "return_1d": "return_1d", "return_3d": "return_3d",
            "return_5d": "return_5d", "return_10d": "return_10d",
            "return_21d": "return_21d", "dist_ma7": "dist_ma7",
            "dist_ma30": "dist_ma30", "dist_ma90": "dist_ma90",
            "ma7_vs_ma30": "ma7_vs_ma30", "bb_position": "bb_position",
            "bb_width": "bb_width", "macd_norm": "macd_norm",
            "macd_signal_norm": "macd_signal_norm", "macd_hist": "macd_hist",
            "rsi": "rsi", "stochastic": "stochastic",
            "mom_5_norm": "mom_5_norm", "mom_10_norm": "mom_10_norm",
            "volatility": "volatility", "volume_ratio": "volume_ratio",
            "atr_norm": "atr_norm", "hl_spread": "hl_spread",
        }
        for src, dst in feature_map.items():
            if src in row:
                record[dst] = round(float(row[src]), 6)

        records.append(record)

    # In Batches hochladen
    BATCH_SIZE = 500
    total      = len(records)
    uploaded   = 0

    for i in range(0, total, BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
        try:
            supabase.table("market_data").upsert(
                batch, on_conflict="ticker,date"
            ).execute()
            uploaded += len(batch)
            print(f"  Uploaded {uploaded}/{total} rows...", end="\r")
        except Exception as e:
            print(f"\n  ❌ Batch {i}-{i+BATCH_SIZE}: {e}")

    print(f"\n  ✅ AAPL Market Data: {uploaded} Rows hochgeladen")

# ==========================================
# Test 3: Verify
# ==========================================
print("\nTest 3: Daten verifizieren...")
res = supabase.table("market_data")\
    .select("ticker, date, close")\
    .eq("ticker", "AAPL")\
    .order("date", desc=True)\
    .limit(5)\
    .execute()

print(f"  Letzte 5 AAPL Einträge:")
for row in res.data:
    print(f"    {row['date']}  ${row['close']}")

# ==========================================
# Test 4: Available Models
# ==========================================
print("\nTest 4: Available Models...")
res = supabase.table("available_models").select("*").execute()
for m in res.data:
    print(f"  {m['ticker']}: acc={m['test_accuracy']}, aktiv={m['is_active']}")

print("\n" + "="*50)
print("✅ Supabase Setup komplett!")
print("="*50)
print("\nNächste Schritte:")
print("  1. .env Datei erstellen (siehe oben)")
print("  2. Dieses Script ausführen")
print("  3. Schritt 2: Daily Prediction Script")