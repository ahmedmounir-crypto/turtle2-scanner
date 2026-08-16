"""
Turtle — Live SWRSI Signal Scanner (with position tracking)
--------------------------------------------------------------
Same SWRSI formula and same crossover rule used in the backtest
(pool_edge_scanner.py, --simple_crossover, buy_level=20, sell_level=80).

This mirrors the backtest's actual trade logic, not just a bare
signal check:
- A fresh buy/sell crossover on a coin with no open position -> OPEN
  that position (long or short) and alert.
- While a position is open, the SAME-direction signal is ignored
  (no pyramiding, same as the backtest).
- The OPPOSITE signal closes the open position, reports the realized
  return, and immediately opens the new position in the new direction
  (same reversal behavior as --side both in the backtest).
- Positions persist across runs via a JSON state file, since each
  GitHub Actions run starts a fresh container with no memory.

State file (default: turtle_state.json) holds one entry per
"symbol|timeframe" key:
{"dir": "long"/"short", "entry_price": ..., "entry_time": "...", "entry_swrsi": ...}

Usage:
  python turtle.py --all_coins --timeframe 1h --telegram
  python turtle.py --all_coins --timeframe 1d --telegram
  python turtle.py --symbol BTC/USDT --timeframe 1h
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import requests
except ImportError:
    requests = None

# ---- indicator math (identical to the backtest) ----

def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100
    out[(avg_gain == 0) & (avg_loss == 0)] = 50
    return out

def stoch(src_high: pd.Series, src_low: pd.Series, src_close: pd.Series, length: int) -> pd.Series:
    hh = src_high.rolling(length).max()
    ll = src_low.rolling(length).min()
    return 100 * (src_close - ll) / (hh - ll).replace(0, np.nan)

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()

def compute_swrsi(df: pd.DataFrame, length_rsi=14, length_stoch=14,
                   buy_level=20, sell_level=80) -> pd.DataFrame:
    """Same formula and same simple_crossover rule as the backtest — no smoothing,
    to stay identical to what was actually backtested."""
    close = df["close"]
    r = rsi(close, length_rsi)
    k = sma(stoch(r, r, r, length_stoch), 3)
    d = sma(k, 3)
    swrsi = (5 * r + d + k) / 7

    prev = swrsi.shift(1)
    buy_signal = (prev < buy_level) & (swrsi >= buy_level)
    sell_signal = (prev > sell_level) & (swrsi <= sell_level)

    out = df.copy()
    out["swrsi"] = swrsi
    out["buy_signal"] = buy_signal.fillna(False)
    out["sell_signal"] = sell_signal.fillna(False)
    return out

# ---- exchange plumbing ----

def make_exchange(exchange_id: str):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})

def fetch_ohlcv_ccxt(exchange, symbol: str, timeframe: str, limit: int = 300,
                      retries: int = 3, retry_delay: float = 2.0) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df[["time", "open", "high", "low", "close"]]
        except Exception as e:
            last_err = e
            time.sleep(retry_delay * (attempt + 1))
    raise last_err

def get_all_usdt_symbols(exchange, quote: str = "USDT", spot_only: bool = True) -> list:
    markets = exchange.load_markets()
    symbols = []
    for sym, m in markets.items():
        if m.get("quote") != quote:
            continue
        if spot_only and not m.get("spot", True):
            continue
        if not m.get("active", True):
            continue
        symbols.append(sym)
    return sorted(symbols)

def send_telegram(message: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id or requests is None:
        print("[telegram] Skipped — missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID or requests lib.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print(f"[telegram] Failed to send: {e}")

# ---- state persistence ----

def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except Exception as e:
            # File exists but is corrupt/unreadable — do NOT silently return {} and
            # let the caller wipe out real open positions. Fail loudly instead.
            raise RuntimeError(
                f"State file '{path}' exists but could not be parsed ({e}). "
                f"Refusing to continue with a blank state, since that would silently "
                f"drop every open position. Fix or restore the file manually."
            )
    return {}

def save_state(path: str, state: dict):
    # Write to a temp file first, then atomically replace — avoids a half-written
    # (corrupt) state file if the process is killed mid-write.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, path)

# ---- signal + position logic ----

def get_latest_bar(exchange, symbol: str, timeframe: str, buy_level: float, sell_level: float,
                    limit: int = 300, only_last_closed: bool = True):
    df = fetch_ohlcv_ccxt(exchange, symbol, timeframe, limit)
    if len(df) < 60:
        return None
    df = compute_swrsi(df, buy_level=buy_level, sell_level=sell_level)
    idx = -2 if only_last_closed and len(df) > 1 else -1
    row = df.iloc[idx]
    return {
        "time": str(row["time"]),
        "price": float(row["close"]),
        "swrsi": float(row["swrsi"]),
        "buy_signal": bool(row["buy_signal"]),
        "sell_signal": bool(row["sell_signal"]),
    }

def process_symbol(exchange, symbol: str, timeframe: str, buy_level: float, sell_level: float,
                    state: dict, limit: int = 300):
    """Returns an event dict (OPEN/CLOSE/None) and mutates state in place,
    exactly mirroring the backtest's open-hold-close-on-reversal logic."""
    bar = get_latest_bar(exchange, symbol, timeframe, buy_level, sell_level, limit)
    if bar is None:
        return None

    key = f"{symbol}|{timeframe}"
    pos = state.get(key)  # None, or {"dir": "long"/"short", "entry_price", "entry_time", "entry_swrsi"}

    if pos is None:
        if bar["buy_signal"]:
            state[key] = {"dir": "long", "entry_price": bar["price"], "entry_time": bar["time"], "entry_swrsi": bar["swrsi"]}
            return {"event": "OPEN", "dir": "long", "symbol": symbol, "price": bar["price"], "swrsi": bar["swrsi"]}
        if bar["sell_signal"]:
            state[key] = {"dir": "short", "entry_price": bar["price"], "entry_time": bar["time"], "entry_swrsi": bar["swrsi"]}
            return {"event": "OPEN", "dir": "short", "symbol": symbol, "price": bar["price"], "swrsi": bar["swrsi"]}
        return None

    # Position already open — only the OPPOSITE signal does anything (same as backtest: no pyramiding)
    opposite_fired = (pos["dir"] == "long" and bar["sell_signal"]) or (pos["dir"] == "short" and bar["buy_signal"])
    if not opposite_fired:
        return None

    entry = pos["entry_price"]
    exit_price = bar["price"]
    ret_pct = ((exit_price - entry) / entry * 100) if pos["dir"] == "long" else ((entry - exit_price) / entry * 100)

    closed = {
        "event": "CLOSE", "dir": pos["dir"], "symbol": symbol,
        "entry_price": entry, "exit_price": exit_price, "return_pct": ret_pct,
        "entry_time": pos["entry_time"], "exit_time": bar["time"],
    }

    # Reversal: immediately open the new position in the opposite direction (matches backtest side=both)
    new_dir = "short" if pos["dir"] == "long" else "long"
    state[key] = {"dir": new_dir, "entry_price": exit_price, "entry_time": bar["time"], "entry_swrsi": bar["swrsi"]}
    closed["reversed_into"] = new_dir
    return closed

def scan_all_coins(exchange_id: str, timeframe: str, buy_level: float, sell_level: float,
                    state: dict, quote: str = "USDT", sleep_between: float = 0.3, limit: int = 300):
    exchange = make_exchange(exchange_id)
    symbols = get_all_usdt_symbols(exchange, quote=quote)
    print(f"Found {len(symbols)} {quote} spot pairs on {exchange_id}. Checking latest bar on each...")

    events = []
    n_errors = 0
    for i, symbol in enumerate(symbols, 1):
        try:
            ev = process_symbol(exchange, symbol, timeframe, buy_level, sell_level, state, limit)
            if ev:
                events.append(ev)
        except Exception as e:
            n_errors += 1
            print(f"  [{i}/{len(symbols)}] {symbol}: ERROR — {type(e).__name__}: {e}")
        time.sleep(sleep_between)
        if i % 25 == 0:
            print(f"  ...checked {i}/{len(symbols)} (events={len(events)}, errors={n_errors})")

    n_open_positions = len(state)
    print(f"\nScan complete: {len(events)} new event(s), {n_open_positions} position(s) currently open, {n_errors} fetch errors.")
    return events

def format_event(ev: dict) -> str:
    if ev["event"] == "OPEN":
        arrow = "🟢 OPEN LONG" if ev["dir"] == "long" else "🔴 OPEN SHORT"
        return f"{arrow} {ev['symbol']} @ {ev['price']} (swrsi {round(ev['swrsi'], 1)})"
    else:
        sign = "+" if ev["return_pct"] >= 0 else ""
        arrow = "✅ CLOSE LONG" if ev["dir"] == "long" else "✅ CLOSE SHORT"
        return (f"{arrow} {ev['symbol']} entry {ev['entry_price']} -> exit {ev['exit_price']} "
                f"({sign}{round(ev['return_pct'], 2)}%) -> reversing into {ev['reversed_into'].upper()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="okx")
    parser.add_argument("--symbol", default="BTC/USDT", help="Used when --all_coins is not set")
    parser.add_argument("--timeframe", default="1h", help="e.g. 1h, 1d")
    parser.add_argument("--limit", type=int, default=300, help="Bars to fetch per coin for indicator warm-up")
    parser.add_argument("--buy_level", type=float, default=20, help="Same as backtest --buy_level")
    parser.add_argument("--sell_level", type=float, default=80, help="Same as backtest --sell_level")
    parser.add_argument("--all_coins", action="store_true", help="Scan every USDT spot pair instead of one symbol")
    parser.add_argument("--telegram", action="store_true", help="Send results to Telegram")
    parser.add_argument("--state_file", default="turtle_state.json", help="Where open positions are persisted between runs")
    args = parser.parse_args()

    state = load_state(args.state_file)
    print(f"Loaded state: {len(state)} position(s) currently open (from {args.state_file}).")

    if args.all_coins:
        events = scan_all_coins(args.exchange, args.timeframe, args.buy_level, args.sell_level,
                                 state, limit=args.limit)
        save_state(args.state_file, state)

        if not events:
            print("No new OPEN/CLOSE events this run.")
            if args.telegram:
                send_telegram(f"*Turtle SWRSI {args.timeframe} ({args.exchange})*\n"
                               f"No new events. {len(state)} position(s) currently open.")
            return

        print(f"\n{len(events)} event(s):")
        for ev in events:
            print(format_event(ev))

        if args.telegram:
            lines = [f"*Turtle SWRSI {args.timeframe} ({args.exchange})*",
                     f"{len(events)} event(s), {len(state)} position(s) open\n"]
            for ev in events:
                lines.append(format_event(ev))
            send_telegram("\n".join(lines))
        return

    exchange = make_exchange(args.exchange)
    ev = process_symbol(exchange, args.symbol, args.timeframe, args.buy_level, args.sell_level, state, args.limit)
    save_state(args.state_file, state)
    msg = format_event(ev) if ev else f"{args.symbol} {args.timeframe}: no new event this run."
    print(msg)
    if args.telegram:
        send_telegram(f"*Turtle SWRSI*\n{msg}")

if __name__ == "__main__":
    main()
  
