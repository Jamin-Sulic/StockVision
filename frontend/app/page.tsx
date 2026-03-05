"use client";

import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid
} from "recharts";

const API = "http://localhost:8000";

const TICKERS = [
  "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA",
  "NFLX", "AMD", "INTC", "ORCL", "CRM", "ADBE", "PYPL",
  "BABA", "V", "MA", "JPM", "GS", "BAC", "WMT", "TGT",
  "COST", "DIS", "NIKE", "BA", "GE", "F", "GM", "XOM",
];

function TickerSearch({ value, onChange }: { value: string; onChange: (t: string) => void }) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);

  const matches = query.length === 0
    ? TICKERS
    : TICKERS.filter(t => t.startsWith(query.toUpperCase()));

  function select(t: string) {
    setQuery(t);
    onChange(t);
    setOpen(false);
  }

  return (
    <div className="relative w-56">
      <div className="flex items-center bg-zinc-900/80 border border-zinc-700/60 rounded-xl px-3 py-2 gap-2 focus-within:border-blue-500/60 transition-colors">
        <svg className="w-3.5 h-3.5 text-zinc-500 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 111 11a6 6 0 0116 0z" />
        </svg>
        <input
          value={query}
          onChange={e => { setQuery(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search ticker..."
          className="bg-transparent text-xs font-bold text-zinc-100 placeholder-zinc-500 focus:outline-none w-full uppercase tracking-widest"
        />
        {query && (
          <button onClick={() => { setQuery(""); setOpen(true); }} className="text-zinc-600 hover:text-zinc-400 text-xs">✕</button>
        )}
      </div>
      {open && matches.length > 0 && (
        <div className="absolute top-full mt-1.5 left-0 w-full bg-zinc-900 border border-zinc-700/60 rounded-xl overflow-hidden shadow-2xl z-50 max-h-52 overflow-y-auto">
          {matches.map((t, i) => (
            <button key={t} onMouseDown={() => select(t)}
              className={`w-full text-left px-4 py-2.5 text-xs font-bold tracking-widest transition-colors flex items-center justify-between
                ${t === value ? "bg-blue-500/20 text-blue-400" : "text-zinc-300 hover:bg-zinc-800/60 hover:text-zinc-100"}
                ${i !== matches.length - 1 ? "border-b border-zinc-800/40" : ""}`}>
              <span>{t}</span>
              {t === value && <span className="text-blue-400 text-xs">●</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface LiveData {
  ticker: string;
  date: string;
  current_price: number;
  predicted_price: number;
  predicted_return: number;
  signal: {
    direction: string;
    emoji: string;
    xgb_proba_up: number;
    score: number;
    trade: boolean;
    confidence: string;
    lstm_magnitude: number;
    xgb_confidence: number;
    agreement: boolean;
  };
  model: { ticker: string; is_own_model: boolean };
  chart_data: { date: string; close: number; rsi: number; macd_hist: number }[];
}

interface HistoricalData {
  ticker: string;
  period: { start: string; end: string };
  from_cache: boolean;
  strategies: {
    buy_and_hold: { total_return: number; sharpe: number; max_drawdown: number; final_capital: number };
    xgboost: { total_return: number; sharpe: number; max_drawdown: number; win_rate: number; n_trades: number; final_capital: number };
    xgboost_scorer: { total_return: number; sharpe: number; max_drawdown: number; win_rate: number; n_trades: number; final_capital: number };
  };
  charts: { equity_bh: number[]; equity_xgb: number[]; equity_scorer: number[]; dates: string[] };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const fmt = (n: number, d = 2) => n?.toFixed(d) ?? "—";
const pct = (n: number) => `${n >= 0 ? "+" : ""}${fmt(n * 100)}%`;

// ─── Components ───────────────────────────────────────────────────────────────
function SignalBadge({ signal }: { signal: LiveData["signal"] }) {
  const isUp = signal.direction === "UP";
  const isTrade = signal.trade;
  return (
    <div className="flex items-center gap-3">
      <span className={`text-5xl font-black tracking-tighter ${isUp ? "text-emerald-400" : "text-rose-400"}`}>
        {isUp ? "▲" : "▼"} {signal.direction}
      </span>
      <div className="flex flex-col gap-1">
        <span className={`text-xs font-bold px-2 py-0.5 rounded uppercase tracking-widest ${
          isTrade ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                  : "bg-zinc-700/50 text-zinc-400 border border-zinc-600/30"
        }`}>
          {isTrade ? "● TRADE" : "○ SKIP"}
        </span>
        <span className={`text-xs font-bold px-2 py-0.5 rounded uppercase tracking-widest ${
          signal.confidence === "HIGH" ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
          : signal.confidence === "MEDIUM" ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
          : "bg-zinc-700/50 text-zinc-400 border border-zinc-600/30"
        }`}>
          {signal.confidence}
        </span>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-4 flex flex-col gap-1">
      <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">{label}</span>
      <span className="text-2xl font-black text-zinc-100 tracking-tight">{value}</span>
      {sub && <span className="text-xs text-zinc-500">{sub}</span>}
    </div>
  );
}

function PriceChart({ data }: { data: LiveData["chart_data"] }) {
  const chartData = data.map(d => ({ date: d.date.slice(5), price: d.close }));
  const min = Math.min(...data.map(d => d.close));
  const max = Math.max(...data.map(d => d.close));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 10 }} tickLine={false} axisLine={false}
          interval={Math.floor(data.length / 6)} />
        <YAxis domain={[min * 0.98, max * 1.02]} tick={{ fill: "#71717a", fontSize: 10 }}
          tickLine={false} axisLine={false} tickFormatter={v => `$${v.toFixed(0)}`} width={50} />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#a1a1aa" }}
          formatter={(v: number | undefined) => [`$${(v ?? 0).toFixed(2)}`, "Price"] as [string, string]}
        />
        <Line type="monotone" dataKey="price" stroke="#60a5fa" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function EquityChart({ data, dates }: { data: { bh: number[]; xgb: number[]; scorer: number[] }; dates: string[] }) {
  const chartData = dates.map((d, i) => ({
    date: d.slice(5),
    "Buy & Hold": +data.bh[i]?.toFixed(2),
    "XGBoost": +data.xgb[i]?.toFixed(2),
    "Scorer": +data.scorer[i]?.toFixed(2),
  }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
        <XAxis dataKey="date" tick={{ fill: "#71717a", fontSize: 10 }} tickLine={false} axisLine={false}
          interval={Math.floor(dates.length / 6)} />
        <YAxis tick={{ fill: "#71717a", fontSize: 10 }} tickLine={false} axisLine={false}
          tickFormatter={v => `$${v.toFixed(0)}`} width={60} />
        <ReferenceLine y={10000} stroke="#3f3f46" strokeDasharray="4 4" />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", borderRadius: 8, fontSize: 12 }}
          formatter={(v: number | undefined) => [`$${(v ?? 0).toFixed(2)}`] as [string]}
        />
        <Line type="monotone" dataKey="Buy & Hold" stroke="#71717a" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="XGBoost" stroke="#60a5fa" strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="Scorer" stroke="#34d399" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function Home() {
  const [ticker, setTicker] = useState("AAPL");
  const [mode, setMode] = useState<"live" | "historical">("live");
  const [liveData, setLiveData] = useState<LiveData | null>(null);
  const [histData, setHistData] = useState<HistoricalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState("2024-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");

  function translateError(msg: string): string {
    if (msg.includes("Keine Prediction") || msg.includes("404") || msg.includes("not found"))
      return `No prediction found for ${ticker}. Run: python scripts/daily_predict.py --tickers ${ticker}`;
    if (msg.includes("Keine Daten") || msg.includes("No data"))
      return `No market data available for ${ticker} in the selected date range.`;
    if (msg.includes("Failed to fetch") || msg.includes("NetworkError"))
      return "Cannot connect to API. Make sure the FastAPI server is running on port 8000.";
    if (msg.includes("Mindestens"))
      return "Please select a date range of at least 30 days.";
    if (msg.includes("Maximal"))
      return "Date range cannot exceed 5 years.";
    return msg;
  }

  async function fetchLive(t = ticker) {
    setLoading(true);
    setError(null);
    setLiveData(null);
    try {
      const res = await fetch(`${API}/predict/live/${t}`);
      if (!res.ok) throw new Error(translateError((await res.json()).detail));
      setLiveData(await res.json());
    } catch (e: any) {
      setError(translateError(e.message));
    } finally {
      setLoading(false);
    }
  }

  async function fetchHistorical() {
    setLoading(true);
    setError(null);
    setHistData(null);
    try {
      const res = await fetch(`${API}/predict/historical/${ticker}?start=${startDate}&end=${endDate}`);
      if (!res.ok) throw new Error(translateError((await res.json()).detail));
      setHistData(await res.json());
    } catch (e: any) {
      setError(translateError(e.message));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (mode === "live") fetchLive(ticker);
    else { setHistData(null); setError(null); }
  }, [ticker, mode]);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100" style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <header className="border-b border-zinc-800/60 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
            <span className="text-white text-sm font-black">SV</span>
          </div>
          <div>
            <h1 className="text-sm font-bold text-zinc-100 tracking-tight">STOCKVISION</h1>
            <p className="text-xs text-zinc-500">LSTM + XGBoost Prediction Engine</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs text-zinc-400">API LIVE</span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <TickerSearch value={ticker} onChange={t => { setTicker(t); setHistData(null); }} />

          {/* Mode Toggle */}
          <div className="flex gap-1 bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-1 ml-auto">
            {(["live", "historical"] as const).map(m => (
              <button key={m} onClick={() => setMode(m)}
                className={`px-4 py-1.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all ${
                  mode === m ? "bg-zinc-700 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"
                }`}>
                {m === "live" ? "Live" : "Historical"}
              </button>
            ))}
          </div>
        </div>

        {/* Historical Date Picker */}
        {mode === "historical" && (
          <div className="flex items-center gap-3 flex-wrap">
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-blue-500/60" />
            <span className="text-zinc-600 text-xs">→</span>
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="bg-zinc-900/60 border border-zinc-800/60 rounded-lg px-3 py-2 text-xs text-zinc-200 focus:outline-none focus:border-blue-500/60" />
            <button onClick={fetchHistorical}
              className="px-4 py-2 bg-blue-500 hover:bg-blue-400 text-white text-xs font-bold rounded-lg transition-colors">
              RUN BACKTEST
            </button>
            {histData?.from_cache && (
              <span className="text-xs text-zinc-500 border border-zinc-700/40 rounded px-2 py-1">● CACHED</span>
            )}
          </div>
        )}

        {/* Loading / Error */}
        {loading && (
          <div className="flex items-center gap-3 py-12 justify-center text-zinc-500">
            <div className="w-5 h-5 border-2 border-blue-500/40 border-t-blue-500 rounded-full animate-spin" />
            <span className="text-sm">{mode === "historical" ? "Computing backtest (LSTM + XGBoost)..." : "Loading prediction..."}</span>
          </div>
        )}
        {error && !loading && (
          <div className="bg-zinc-900/60 border border-zinc-700/40 rounded-xl p-8 flex flex-col items-center gap-3 text-center">
            <span className="text-3xl">🚧</span>
            <p className="text-sm font-bold text-zinc-200">{ticker} — Coming Soon</p>
            <p className="text-xs text-zinc-500">This ticker is not available yet. Check back later.</p>
          </div>
        )}

        {/* Live View */}
        {mode === "live" && liveData && !loading && (
          <div className="space-y-4">
            {/* Top Row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Signal Card */}
              <div className="md:col-span-1 bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">Signal · {liveData.date}</span>
                  {!liveData.model.is_own_model && (
                    <span className="text-xs text-amber-400/70 border border-amber-500/20 rounded px-1.5 py-0.5">AAPL model</span>
                  )}
                </div>
                <SignalBadge signal={liveData.signal} />
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-zinc-800/40 rounded-lg p-3">
                    <div className="text-xs text-zinc-500 mb-1">P(UP)</div>
                    <div className="text-lg font-black text-zinc-100">{fmt(liveData.signal.xgb_proba_up * 100)}%</div>
                  </div>
                  <div className="bg-zinc-800/40 rounded-lg p-3">
                    <div className="text-xs text-zinc-500 mb-1">Score</div>
                    <div className="text-lg font-black text-zinc-100">{fmt(liveData.signal.score, 4)}</div>
                  </div>
                </div>
              </div>

              {/* Price Card */}
              <div className="md:col-span-2 bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-5 space-y-3">
                <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">{ticker} · Price Prediction</span>
                <div className="flex items-end gap-4">
                  <div>
                    <div className="text-xs text-zinc-500 mb-1">Current</div>
                    <div className="text-4xl font-black text-zinc-100">${fmt(liveData.current_price)}</div>
                  </div>
                  <div className="text-2xl text-zinc-600 mb-1">→</div>
                  <div>
                    <div className="text-xs text-zinc-500 mb-1">Tomorrow (LSTM)</div>
                    <div className={`text-4xl font-black ${liveData.predicted_return >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                      ${fmt(liveData.predicted_price)}
                    </div>
                  </div>
                  <div className={`text-lg font-bold mb-1 ${liveData.predicted_return >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                    {pct(liveData.predicted_return)}
                  </div>
                </div>
                <PriceChart data={liveData.chart_data} />
              </div>
            </div>

            {/* Score Breakdown */}
            <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-5">
              <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium block mb-4">Score Breakdown</span>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: "LSTM Magnitude", value: liveData.signal.lstm_magnitude, weight: "35%" },
                  { label: "XGB Confidence", value: liveData.signal.xgb_confidence, weight: "40%" },
                  { label: "Agreement", value: liveData.signal.agreement ? 1 : 0, weight: "25%" },
                ].map(item => (
                  <div key={item.label} className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-zinc-400">{item.label}</span>
                      <span className="text-zinc-500">{item.weight}</span>
                    </div>
                    <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 rounded-full transition-all"
                        style={{ width: `${Math.min(item.value * 100, 100)}%` }} />
                    </div>
                    <div className="text-xs font-bold text-zinc-300">{fmt(item.value, 4)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Historical View */}
        {mode === "historical" && histData && !loading && (
          <div className="space-y-4">
            {/* Metrics Table */}
            <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-zinc-800/60">
                <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">
                  Backtest · {histData.period.start} → {histData.period.end}
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800/60">
                      <th className="text-left px-5 py-3 text-zinc-500 font-medium uppercase tracking-wider">Strategy</th>
                      <th className="text-right px-4 py-3 text-zinc-500 font-medium uppercase tracking-wider">Return</th>
                      <th className="text-right px-4 py-3 text-zinc-500 font-medium uppercase tracking-wider">Sharpe</th>
                      <th className="text-right px-4 py-3 text-zinc-500 font-medium uppercase tracking-wider">Max DD</th>
                      <th className="text-right px-4 py-3 text-zinc-500 font-medium uppercase tracking-wider">Win Rate</th>
                      <th className="text-right px-4 py-3 text-zinc-500 font-medium uppercase tracking-wider">Trades</th>
                      <th className="text-right px-5 py-3 text-zinc-500 font-medium uppercase tracking-wider">Capital</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { name: "Buy & Hold", color: "text-zinc-400", s: histData.strategies.buy_and_hold, trades: null, wr: null },
                      { name: "XGBoost", color: "text-blue-400", s: histData.strategies.xgboost, trades: histData.strategies.xgboost.n_trades, wr: histData.strategies.xgboost.win_rate },
                      { name: "XGB + Scorer", color: "text-emerald-400", s: histData.strategies.xgboost_scorer, trades: histData.strategies.xgboost_scorer.n_trades, wr: histData.strategies.xgboost_scorer.win_rate },
                    ].map(row => (
                      <tr key={row.name} className="border-b border-zinc-800/40 hover:bg-zinc-800/20 transition-colors">
                        <td className={`px-5 py-3 font-bold ${row.color}`}>{row.name}</td>
                        <td className={`px-4 py-3 text-right font-bold ${row.s.total_return >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                          {pct(row.s.total_return)}
                        </td>
                        <td className="px-4 py-3 text-right text-zinc-300">{fmt(row.s.sharpe)}</td>
                        <td className="px-4 py-3 text-right text-rose-400">{pct(row.s.max_drawdown)}</td>
                        <td className="px-4 py-3 text-right text-zinc-300">{row.wr != null ? pct(row.wr) : "—"}</td>
                        <td className="px-4 py-3 text-right text-zinc-300">{row.trades ?? "—"}</td>
                        <td className="px-5 py-3 text-right text-zinc-200 font-bold">${row.s.final_capital.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Equity Chart */}
            <div className="bg-zinc-900/60 border border-zinc-800/60 rounded-xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-500 uppercase tracking-widest font-medium">Equity Curves · $10,000 Starting Capital</span>
                <div className="flex gap-4 text-xs">
                  <span className="text-zinc-500">— Buy & Hold</span>
                  <span className="text-blue-400">— XGBoost</span>
                  <span className="text-emerald-400">— Scorer</span>
                </div>
              </div>
              <EquityChart
                data={{ bh: histData.charts.equity_bh, xgb: histData.charts.equity_xgb, scorer: histData.charts.equity_scorer }}
                dates={histData.charts.dates}
              />
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <StatCard label="XGB vs BH" value={pct(histData.strategies.xgboost.total_return - histData.strategies.buy_and_hold.total_return)} sub="Alpha generated" />
              <StatCard label="XGB Sharpe" value={fmt(histData.strategies.xgboost.sharpe)} sub="Risk-adjusted return" />
              <StatCard label="Max Drawdown" value={pct(histData.strategies.xgboost.max_drawdown)} sub="XGBoost strategy" />
              <StatCard label="Win Rate" value={pct(histData.strategies.xgboost.win_rate)} sub={`${histData.strategies.xgboost.n_trades} trades`} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}