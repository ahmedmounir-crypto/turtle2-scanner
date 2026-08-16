"""
SWRSI (Stochastic Weighted RSI) Backtest
-----------------------------------------
Reimplements the Pine Script v4 "SWRSI" indicator logic in Python and
backtests the buy/sell signal logic it defines:

swrsi = (5*rsi + stochD + stochK) / 7   # RSI(14) weighted with its own Stoch %K/%D
swrsiema = EMA(swrsi, 2)

bullish = swrsi < 30 and swrsi > swrsiema
bearish = swrsi > 70 and swrsi < swrsiema

buySignal  = bullish and shortEMA(20) < longEMA(50)  -> enter LONG
sellSignal = bearish and shortEMA(20) > longEMA(50)  -> enter SHORT / exit LONG

Trade rule used here (the Pine script itself only plots signals, it has
no built-in exit/TP/SL):
- Long position opened on buySignal, closed on the next sellSignal
  (or end of data).
- Short position opened on sellSignal, closed on the next buySignal
  (or end of data).
- No pyramiding: a new signal in the same direction while already
  in that direction is ignored.

Usage:
  python swrsi_backtest.py path/to/ohlcv.csv --side long
  python swrsi_backtest.py path/to/ohlcv.csv --side short
  python swrsi_backtest.py path/to/ohlcv.csv --side both

CSV must have columns (case-insensitive): timestamp/date/time, open, high, low, close
(volume optional, not used by this indicator).
"""

import argparse
import os
import sys
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

def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder's smoothing (matches Pine's built-in rsi())
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

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    ts_col = next((c for c in df.columns if c in ("timestamp", "date", "time", "datetime", "open_time")), None)
    if ts_col is None:
        raise ValueError(f"Couldn't find a timestamp/date column. Columns found: {list(df.columns)}")

    # Handle both epoch ms/s and ISO strings
    if pd.api.types.is_numeric_dtype(df[ts_col]):
        unit = "ms" if df[ts_col].iloc[0] > 10**12 else "s"
        df["time"] = pd.to_datetime(df[ts_col], unit=unit)
    else:
        df["time"] = pd.to_datetime(df[ts_col])

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Columns found: {list(df.columns)}")

    df = df.sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]]

def compute_swrsi(df: pd.DataFrame, length_rsi=14, length_stoch=14, length_swrsi_ema=2,
                   short_ema=20, long_ema=50, buy_level=20, sell_level=80,
                   simple_crossover=False) -> pd.DataFrame:
    """
    simple_crossover=False (default): original strict logic —
        bullish = swrsi < 30 and swrsi > swrsi_ema
        bearish = swrsi > 70 and swrsi < swrsi_ema
        buy  = bullish and shortEMA < longEMA
        sell = bearish and shortEMA > longEMA

    simple_crossover=True: your actual rule —
        BUY  = swrsi crosses UP through buy_level (default 20), coming from below
        SELL = swrsi crosses DOWN through sell_level (default 80), coming from above
        No trend filter, no EMA condition. Stay in until the opposite signal.
    """
    close = df["close"]

    r = rsi(close, length_rsi)
    k = sma(stoch(r, r, r, length_stoch), 3)  # stoch() on rsi applied to rsi itself, as in the source
    d = sma(k, 3)

    swrsi = (5 * r + d + k) / 7
    swrsi_ema = ema(swrsi, length_swrsi_ema)

    out = df.copy()
    out["swrsi"] = swrsi
    out["swrsi_ema"] = swrsi_ema

    if simple_crossover:
        prev = swrsi.shift(1)
        buy_signal = (prev < buy_level) & (swrsi >= buy_level)
        sell_signal = (prev > sell_level) & (swrsi <= sell_level)
    else:
        short_ma = ema(close, short_ema)
        long_ma = ema(close, long_ema)
        bullish = (swrsi < 30) & (swrsi > swrsi_ema)
        bearish = (swrsi > 70) & (swrsi < swrsi_ema)
        buy_signal = bullish & (short_ma < long_ma)
        sell_signal = bearish & (short_ma > long_ma)

    out["buy_signal"] = buy_signal.fillna(False)
    out["sell_signal"] = sell_signal.fillna(False)
    return out

def backtest(df: pd.DataFrame, side: str = "both", fee_pct: float = 0.0, leverage: float = 1.0):
    """
    side: 'long' (only buySignal trades), 'short' (only sellSignal trades), 'both'
    fee_pct: round-trip fee/slippage as a fraction, e.g. 0.001 = 0.1% total
    leverage: multiplier applied to each trade's raw price-move return, e.g. 5.0 for 5x.
              A raw adverse move of >= 1/leverage (e.g. 20% against you at 5x) is flagged
              as a liquidation — the trade is capped at -100% and marked liquidated.
    """
    trades = []
    position = None  # dict: {'dir': 'long'/'short', 'entry_price', 'entry_time'}

    def close_trade(entry, exit_price, direction, entry_time, exit_time):
        raw_ret = (exit_price - entry) / entry if direction == "long" else (entry - exit_price) / entry
        levered_ret = raw_ret * leverage
        liquidated = levered_ret <= -1.0
        if liquidated:
            levered_ret = -1.0
        levered_ret -= fee_pct
        return {
            "dir": direction,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry,
            "exit_price": exit_price,
            "raw_return_pct": raw_ret * 100,
            "return_pct": levered_ret * 100,
            "liquidated": liquidated,
        }

    for i, row in df.iterrows():
        price = row["close"]
        time = row["time"]

        if position is None:
            if side in ("long", "both") and row["buy_signal"]:
                position = {"dir": "long", "entry_price": price, "entry_time": time}
            elif side in ("short", "both") and row["sell_signal"]:
                position = {"dir": "short", "entry_price": price, "entry_time": time}
        else:
            exit_now = False
            if position["dir"] == "long" and row["sell_signal"]:
                exit_now = True
            elif position["dir"] == "short" and row["buy_signal"]:
                exit_now = True

            if exit_now:
                trades.append(close_trade(position["entry_price"], price, position["dir"],
                                           position["entry_time"], time))
                # flip immediately into the new opposite signal
                if side == "both":
                    position = {"dir": "short" if position["dir"] == "long" else "long",
                                "entry_price": price, "entry_time": time}
else:
                    position = None

    # close any open position at the end of data
    if position is not None:
        price = df.iloc[-1]["close"]
        trades.append(close_trade(position["entry_price"], price, position["dir"],
                                   position["entry_time"], df.iloc[-1]["time"]))

    trades_df = pd.DataFrame(trades)
    return trades_df

def summarize(trades_df: pd.DataFrame, label: str):
    if trades_df.empty:
        print(f"\n[{label}] No trades generated — check signal frequency / data length.")
        return

    n = len(trades_df)
    wins = trades_df[trades_df["return_pct"] > 0]
    losses = trades_df[trades_df["return_pct"] <= 0]
    win_rate = len(wins) / n * 100
    avg_win = wins["return_pct"].mean() if len(wins) else 0
    avg_loss = losses["return_pct"].mean() if len(losses) else 0
    total_return = trades_df["return_pct"].sum()
    expectancy = trades_df["return_pct"].mean()
    profit_factor = (wins["return_pct"].sum() / abs(losses["return_pct"].sum())
                      if losses["return_pct"].sum() != 0 else np.inf)

    print(f"\n=== {label} ===")
    print(f"Trades:            {n}")
    print(f"Win rate:          {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg win:           {avg_win:.2f}%")
    print(f"Avg loss:          {avg_loss:.2f}%")
    print(f"Expectancy/trade:  {expectancy:.2f}%")
    print(f"Profit factor:     {profit_factor:.2f}")
    print(f"Total return:      {total_return:.2f}% (sum of per-trade % — not compounded)")
    if "liquidated" in trades_df.columns and trades_df["liquidated"].any():
        n_liq = int(trades_df["liquidated"].sum())
        print(f"⚠ Liquidated:      {n_liq}/{n} trades hit -100% (leverage wiped the position)")

def make_exchange(exchange_id: str):
    if ccxt is None:
        raise RuntimeError("ccxt not installed. pip install ccxt")
    return getattr(ccxt, exchange_id)({"enableRateLimit": True})

def fetch_ohlcv_ccxt(exchange, symbol: str, timeframe: str, limit: int = 1000,
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

def fetch_ohlcv_history(exchange, symbol: str, timeframe: str, lookback_days: float,
                         batch_limit: int = 300, retries: int = 3, retry_delay: float = 2.0,
                         pace: float = 0.25) -> pd.DataFrame:
    """
    Paginated fetch: loops fetch_ohlcv forward in time from (now - lookback_days) until
    reaching the present, since exchanges cap each single call at ~100-300 bars.
    """
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - int(lookback_days * 24 * 60 * 60 * 1000)
    now_ms = exchange.milliseconds()

    all_rows = []
    seen_ts = set()
    while since < now_ms:
        batch = None
        last_err = None
        for attempt in range(retries):
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
                break
            except Exception as e:
                last_err = e
                time.sleep(retry_delay * (attempt + 1))
        if batch is None:
            raise last_err

        if not batch:
            break

        new_rows = [r for r in batch if r[0] not in seen_ts]
        if not new_rows:
            break
        for r in new_rows:
            seen_ts.add(r[0])
        all_rows.extend(new_rows)

        last_ts = batch[-1][0]
        next_since = last_ts + tf_ms
        if next_since <= since:
            break  # exchange isn't advancing, avoid infinite loop
        since = next_since

        if len(batch) < batch_limit:
            break  # reached the end of available data

        time.sleep(pace)

    if not all_rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["time", "open", "high", "low", "close"]]

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

def summary_text(trades_df: pd.DataFrame, label: str) -> str:
    if trades_df.empty:
        return f"*{label}*\nNo trades generated."
    n = len(trades_df)
    wins = trades_df[trades_df["return_pct"] > 0]
    losses = trades_df[trades_df["return_pct"] <= 0]
    win_rate = len(wins) / n * 100
    expectancy = trades_df["return_pct"].mean()
    total_return = trades_df["return_pct"].sum()
    return (f"*{label}*\n"
            f"Trades: {n}\n"
            f"Win rate: {win_rate:.1f}%\n"
            f"Expectancy/trade: {expectancy:.2f}%\n"
            f"Total return: {total_return:.2f}%")

def get_all_usdt_symbols(exchange, quote: str = "USDT", spot_only: bool = True) -> list:
    """List all tradeable <coin>/<quote> spot symbols on the given exchange."""
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

def run_single(df: pd.DataFrame, label: str, side: str, fee_pct: float, out_stub: str,
               simple_crossover: bool = False, buy_level: float = 20, sell_level: float = 80,
               leverage: float = 1.0):
    df = compute_swrsi(df, buy_level=buy_level, sell_level=sell_level, simple_crossover=simple_crossover)
    trades = backtest(df, side=side, fee_pct=fee_pct, leverage=leverage)
    out_path = f"{out_stub}_swrsi_trades.csv"
    trades.to_csv(out_path, index=False)
    return trades

def scan_all_coins(exchange_id: str, timeframe: str, limit: int, side: str, fee_pct: float,
                    min_trades: int, quote: str = "USDT", sleep_between: float = 0.3,
                    simple_crossover: bool = False, buy_level: float = 20, sell_level: float = 80,
                    leverage: float = 1.0, lookback_days: float = None):
    exchange = make_exchange(exchange_id)
    symbols = get_all_usdt_symbols(exchange, quote=quote)
    print(f"Found {len(symbols)} {quote} spot pairs on {exchange_id}. Scanning...")

    rows = []
    all_trade_counts = []
    pooled_trades = []  # every trade from every coin, for the aggregate view
    n_success, n_skipped_thin, n_errors = 0, 0, 0
    for i, symbol in enumerate(symbols, 1):
        try:
            if lookback_days:
                df = fetch_ohlcv_history(exchange, symbol, timeframe, lookback_days)
            else:
                df = fetch_ohlcv_ccxt(exchange, symbol, timeframe, limit)
            if len(df) < 60:
                n_skipped_thin += 1
                continue
            trades = run_single(df, symbol, side, fee_pct, out_stub=f"{symbol.replace('/', '')}_{timeframe}",
                                 simple_crossover=simple_crossover, buy_level=buy_level, sell_level=sell_level,
                                 leverage=leverage)
            n = len(trades)
            all_trade_counts.append(n)
            if n > 0:
                t = trades.copy()
                t["symbol"] = symbol
                pooled_trades.append(t)
            if n < min_trades:
                n_skipped_thin += 1
                continue
            n_success += 1
            wins = trades[trades["return_pct"] > 0]
            win_rate = len(wins) / n * 100
            expectancy = trades["return_pct"].mean()
            total_return = trades["return_pct"].sum()
            n_liquidated = int(trades["liquidated"].sum()) if "liquidated" in trades.columns else 0
            rows.append({
                "symbol": symbol,
                "trades": n,
                "win_rate_pct": round(win_rate, 1),
                "expectancy_pct": round(expectancy, 2),
                "total_return_pct": round(total_return, 2),
                "liquidations": n_liquidated,
            })
        except Exception as e:
            n_errors += 1
            print(f"  [{i}/{len(symbols)}] {symbol}: ERROR — {type(e).__name__}: {e}")
        time.sleep(sleep_between)
        if i % 25 == 0:
            print(f"  ...scanned {i}/{len(symbols)} (ok={n_success}, thin={n_skipped_thin}, errors={n_errors})")

    print(f"\nScan complete: {n_success} coins with enough trades, "
          f"{n_skipped_thin} skipped (< {min_trades} trades or < 60 bars), {n_errors} fetch errors.")

    if all_trade_counts:
        import statistics
        print(f"Trade-count distribution across coins with usable data: "
              f"min={min(all_trade_counts)}, max={max(all_trade_counts)}, "
              f"median={statistics.median(all_trade_counts)}, "
              f"mean={statistics.mean(all_trade_counts):.1f}")

    pooled_df = pd.concat(pooled_trades, ignore_index=True) if pooled_trades else pd.DataFrame()

    if not rows:
        result = pd.DataFrame(columns=["symbol", "trades", "win_rate_pct", "expectancy_pct", "total_return_pct", "liquidations"])
    else:
        result = pd.DataFrame(rows).sort_values("expectancy_pct", ascending=False).reset_index(drop=True)

    return result, pooled_df

def summarize_pool(pooled_df: pd.DataFrame, label: str) -> str:
    """Aggregate every trade from every coin into one combined result."""
    if pooled_df.empty:
        return f"*{label} — POOLED*\nNo trades to pool."
    n = len(pooled_df)
    n_coins = pooled_df["symbol"].nunique()
    wins = pooled_df[pooled_df["return_pct"] > 0]
    losses = pooled_df[pooled_df["return_pct"] <= 0]
    win_rate = len(wins) / n * 100
    expectancy = pooled_df["return_pct"].mean()
    total_return = pooled_df["return_pct"].sum()
    profit_factor = (wins["return_pct"].sum() / abs(losses["return_pct"].sum())
                      if len(losses) and losses["return_pct"].sum() != 0 else float("inf"))
    n_liq = int(pooled_df["liquidated"].sum()) if "liquidated" in pooled_df.columns else 0

    lines = [
        f"=== {label} — POOLED across {n_coins} coins ===",
        f"Total trades:      {n}",
        f"Win rate:          {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)",
        f"Expectancy/trade:  {expectancy:.2f}%",
        f"Profit factor:     {profit_factor:.2f}",
        f"Total return:      {total_return:.2f}% (sum of per-trade %, not compounded, not equal-weighted across coins)",
    ]
    if n_liq:
        lines.append(f"⚠ Liquidated:      {n_liq}/{n} trades")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", help="Path to OHLCV CSV file. Omit to fetch live via ccxt instead.")
    parser.add_argument("--exchange", default="okx", help="ccxt exchange id, used when csv_path is omitted")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="e.g. 1h, 1d")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--side", choices=["long", "short", "both"], default="both")
    parser.add_argument("--fee_pct", type=float, default=0.0008, help="Round-trip fee/slippage fraction, default 0.08%")
    parser.add_argument("--telegram", action="store_true", help="Send summary to Telegram (needs TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID env vars)")
    parser.add_argument("--all_coins", action="store_true", help="Scan every USDT spot pair on the exchange instead of one symbol")
    parser.add_argument("--min_trades", type=int, default=8, help="Minimum trades for a coin to be included in --all_coins results (avoids overfitting on thin samples)")
    parser.add_argument("--simple_crossover", action="store_true",
                         help="Use the simple crossover rule instead of the strict EMA-filtered rule: "
                              "BUY when swrsi crosses up through --buy_level, SELL when it crosses down through --sell_level")
    parser.add_argument("--buy_level", type=float, default=20, help="Crossover buy threshold (only used with --simple_crossover)")
    parser.add_argument("--sell_level", type=float, default=80, help="Crossover sell threshold (only used with --simple_crossover)")
    parser.add_argument("--leverage", type=float, default=1.0, help="Leverage multiplier applied to each trade's return, e.g. 5 for 5x. A move against you of 1/leverage liquidates the trade (capped at -100%%).")
    parser.add_argument("--lookback_days", type=float, default=None,
                         help="Fetch this many days of full history via pagination instead of a single --limit call. E.g. 365 for 1 year, 1095 for 3 years.")
    args = parser.parse_args()

    if args.all_coins:
        results, pooled = scan_all_coins(args.exchange, args.timeframe, args.limit, args.side,
                                          args.fee_pct, args.min_trades,
                                          simple_crossover=args.simple_crossover,
                                          buy_level=args.buy_level, sell_level=args.sell_level,
                                          leverage=args.leverage, lookback_days=args.lookback_days)
        out_path = f"swrsi_all_coins_{args.timeframe}_summary.csv"
        results.to_csv(out_path, index=False)
        print(f"\n=== SWRSI scan across {len(results)} coins ({args.exchange}, {args.timeframe}) ===")
        print(results.to_string(index=False))
        print(f"\nFull summary saved to {out_path}")

        pool_label = f"{args.timeframe} ({args.exchange})"
        pool_summary = summarize_pool(pooled, label=pool_label)
        print(f"\n{pool_summary}")
        pool_path = f"swrsi_pooled_trades_{args.timeframe}.csv"
        pooled.to_csv(pool_path, index=False)
        print(f"Pooled trade log saved to {pool_path}")

        if args.telegram:
            if results.empty:
                send_telegram(f"*SWRSI scan {args.timeframe} ({args.exchange})*\n"
                               f"No coins reached min_trades={args.min_trades}. Try lowering --min_trades or increasing --limit.")
            else:
                top = results.head(25)
                lines = [f"*SWRSI scan {args.timeframe} ({args.exchange})*",
                         f"{len(results)} coins with >= {args.min_trades} trades\n"]
                for _, r in top.iterrows():
                    liq = f", {r['liquidations']} liq" if r.get("liquidations", 0) else ""
                    lines.append(f"{r['symbol']}: {r['trades']}t, {r['win_rate_pct']}% WR, {r['expectancy_pct']}% exp{liq}")
                send_telegram("\n".join(lines))
            send_telegram(pool_summary)
        return

    if args.csv_path:
        df = load_ohlcv(args.csv_path)
        label = f"{args.csv_path} | {args.side}"
        out_stub = args.csv_path.rsplit(".", 1)[0]
    else:
        exchange = make_exchange(args.exchange)
        if args.lookback_days:
            df = fetch_ohlcv_history(exchange, args.symbol, args.timeframe, args.lookback_days)
        else:
            df = fetch_ohlcv_ccxt(exchange, args.symbol, args.timeframe, args.limit)
        label = f"{args.symbol} {args.timeframe} ({args.exchange}) | {args.side}"
        out_stub = f"{args.symbol.replace('/', '')}_{args.timeframe}"

    trades = run_single(df, label, args.side, args.fee_pct, out_stub,
                         simple_crossover=args.simple_crossover,
                         buy_level=args.buy_level, sell_level=args.sell_level,
                         leverage=args.leverage)
    summarize(trades, label=label)
    print(f"\nTrade log saved to {out_stub}_swrsi_trades.csv")

    if args.telegram:
        send_telegram(summary_text(trades, label))

if __name__ == "__main__":
    main()
  
