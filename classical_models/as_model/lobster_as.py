"""LOBSTER data backend for the Avellaneda-Stoikov baseline.

The LOBSTER files used here are headerless 10-level order book snapshots plus
message rows. Prices are stored as dollars * 10000. Time is seconds after local
midnight, with regular trading hours encoded in the filename.
"""

from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from avellaneda_stoikov_amzn import (
    ASParams,
    ArrivalCalibration,
    PriceVolumeCurve,
    _probability_to_intensity,
    _queue_fill_size,
    _fit_exp_decay,
    as_quotes,
)


LOBSTER_PRICE_SCALE = 10_000
FIXED_PRICE_SCALE = 100_000
DEFAULT_LOBSTER_LEVELS = 10
REGULAR_START_SECONDS = 34_200.0
REGULAR_END_SECONDS = 57_600.0

LOBSTER_FILE_RE = re.compile(
    r"^(?P<symbol>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<kind>message|orderbook)_"
    r"(?P<levels>\d+)\.csv$"
)

MESSAGE_COLUMNS = ["time", "type", "order_id", "size", "price", "direction"]


@dataclass(frozen=True)
class LobsterDayFiles:
    date: date
    message_path: Path
    orderbook_path: Path
    symbol: str = "AMZN"
    levels: int = DEFAULT_LOBSTER_LEVELS
    start_seconds: float = REGULAR_START_SECONDS
    end_seconds: float = REGULAR_END_SECONDS


@dataclass
class LobsterTradeFlow:
    """Visible execution curves for each sampled decision interval."""

    buy_aggressor: list[PriceVolumeCurve]
    sell_aggressor: list[PriceVolumeCurve]

    def __len__(self) -> int:
        return len(self.buy_aggressor)

    @staticmethod
    def fixed_from_lobster_int(price: int) -> int:
        return int(price) * FIXED_PRICE_SCALE

    def buy_reaching_ask(self, interval: int, ask_lobster_int: int) -> int:
        if interval < 0 or interval >= len(self.buy_aggressor):
            return 0
        fixed_price = self.fixed_from_lobster_int(ask_lobster_int)
        return self.buy_aggressor[interval].volume_at_or_above(fixed_price)

    def sell_reaching_bid(self, interval: int, bid_lobster_int: int) -> int:
        if interval < 0 or interval >= len(self.sell_aggressor):
            return 0
        fixed_price = self.fixed_from_lobster_int(bid_lobster_int)
        return self.sell_aggressor[interval].volume_at_or_below(fixed_price)


@dataclass
class LobsterDayData:
    files: LobsterDayFiles
    bbo: pd.DataFrame
    trade_extrema: pd.DataFrame
    trade_flow: LobsterTradeFlow


def is_lobster_file(path: Path) -> bool:
    return LOBSTER_FILE_RE.match(path.name) is not None


def is_lobster_dir(data_dir: Path, levels: int = DEFAULT_LOBSTER_LEVELS) -> bool:
    root = Path(data_dir)
    return any(root.glob(f"*_message_{levels}.csv")) or any(root.rglob(f"*_message_{levels}.csv"))


def _parse_lobster_filename(path: Path) -> dict[str, object]:
    match = LOBSTER_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"not a LOBSTER filename: {path}")
    groups = match.groupdict()
    return {
        "symbol": groups["symbol"],
        "date": date.fromisoformat(groups["date"]),
        "start_seconds": _milliseconds_after_midnight(groups["start"]),
        "end_seconds": _milliseconds_after_midnight(groups["end"]),
        "kind": groups["kind"],
        "levels": int(groups["levels"]),
    }


def _milliseconds_after_midnight(value: str) -> float:
    return int(value) / 1000.0


def discover_lobster_pairs(
    data_dir: Path | str,
    *,
    levels: int = DEFAULT_LOBSTER_LEVELS,
) -> list[LobsterDayFiles]:
    """Discover paired LOBSTER message/orderbook files and validate coverage."""

    root = Path(data_dir)
    messages: dict[date, Path] = {}
    orderbooks: dict[date, Path] = {}
    metadata: dict[date, dict[str, object]] = {}

    for path in sorted(root.glob(f"*_message_{levels}.csv")):
        info = _parse_lobster_filename(path)
        messages[info["date"]] = path
        metadata[info["date"]] = info
    for path in sorted(root.glob(f"*_orderbook_{levels}.csv")):
        info = _parse_lobster_filename(path)
        orderbooks[info["date"]] = path
        metadata.setdefault(info["date"], info)

    missing_orderbooks = sorted(set(messages) - set(orderbooks))
    missing_messages = sorted(set(orderbooks) - set(messages))
    if missing_orderbooks or missing_messages:
        details: list[str] = []
        if missing_orderbooks:
            details.append(
                "missing orderbook for "
                + ", ".join(day.isoformat() for day in missing_orderbooks[:5])
            )
        if missing_messages:
            details.append(
                "missing message for "
                + ", ".join(day.isoformat() for day in missing_messages[:5])
            )
        raise FileNotFoundError("; ".join(details))

    pairs: list[LobsterDayFiles] = []
    for day in sorted(messages):
        info = metadata[day]
        pairs.append(
            LobsterDayFiles(
                date=day,
                message_path=messages[day],
                orderbook_path=orderbooks[day],
                symbol=str(info["symbol"]),
                levels=int(info["levels"]),
                start_seconds=float(info["start_seconds"]),
                end_seconds=float(info["end_seconds"]),
            )
        )
    return pairs


def load_lobster_day(
    files: LobsterDayFiles,
    *,
    dt_seconds: float,
    cache_dir: Path | str | None = None,
) -> LobsterDayData:
    """Load one LOBSTER day into sampled BBO and interval trade-flow features."""

    cache_path = _cache_path(cache_dir, files, dt_seconds)
    cache_meta = _cache_meta(files, dt_seconds)
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            if payload.get("meta") == cache_meta:
                return payload["day"]
        except Exception:
            pass

    messages = _load_messages(files.message_path)
    orderbook = _load_orderbook(files.orderbook_path, files.levels)
    if len(messages) != len(orderbook):
        raise ValueError(
            f"message/orderbook row mismatch for {files.date}: "
            f"{len(messages)} messages vs {len(orderbook)} book rows"
        )

    times = messages["time"].to_numpy(dtype=np.float64)
    grid_seconds, sample_idx = _sample_indices(
        times,
        dt_seconds=dt_seconds,
        start_seconds=files.start_seconds,
        end_seconds=files.end_seconds,
    )
    if len(grid_seconds) == 0:
        raise ValueError(f"no sampled LOBSTER states for {files.date}")

    sampled_book = orderbook.iloc[sample_idx].reset_index(drop=True)
    bbo = _build_sampled_bbo(files, sampled_book, grid_seconds)
    trade_flow, trade_extrema = _build_interval_trade_flow(
        messages,
        grid_seconds=grid_seconds,
        dt_seconds=dt_seconds,
    )
    day = LobsterDayData(files=files, bbo=bbo, trade_extrema=trade_extrema, trade_flow=trade_flow)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump({"meta": cache_meta, "day": day}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return day


def calibrate_lobster_arrival_rates(
    bbo: pd.DataFrame,
    trade_flow: LobsterTradeFlow,
    *,
    deltas: Iterable[float],
    dt_seconds: float,
    tick_size: float,
    order_size: int,
) -> ArrivalCalibration:
    """Estimate A/k using displayed depth and visible execution volume."""

    deltas = np.asarray(list(deltas), dtype=float)
    mids = bbo["mid"].to_numpy(dtype=float)
    levels = _levels_from_bbo(bbo)
    bid_prices = bbo[[f"bid_price_int_{level}" for level in levels]].to_numpy(dtype=np.int64)
    bid_sizes = bbo[[f"bid_size_{level}" for level in levels]].to_numpy(dtype=np.int64)
    ask_prices = bbo[[f"ask_price_int_{level}" for level in levels]].to_numpy(dtype=np.int64)
    ask_sizes = bbo[[f"ask_size_{level}" for level in levels]].to_numpy(dtype=np.int64)

    lambda_ask = np.full(len(deltas), np.nan)
    lambda_bid = np.full(len(deltas), np.nan)
    ask_posted = np.zeros(len(deltas), dtype=np.int64)
    bid_posted = np.zeros(len(deltas), dtype=np.int64)
    ask_fill_count = np.zeros(len(deltas), dtype=np.int64)
    bid_fill_count = np.zeros(len(deltas), dtype=np.int64)
    ask_full_fill_count = np.zeros(len(deltas), dtype=np.int64)
    bid_full_fill_count = np.zeros(len(deltas), dtype=np.int64)
    ask_queue_total = np.zeros(len(deltas), dtype=np.float64)
    bid_queue_total = np.zeros(len(deltas), dtype=np.float64)
    ask_reaching_total = np.zeros(len(deltas), dtype=np.float64)
    bid_reaching_total = np.zeros(len(deltas), dtype=np.float64)
    ask_fill_size_total = np.zeros(len(deltas), dtype=np.float64)
    bid_fill_size_total = np.zeros(len(deltas), dtype=np.float64)

    for j, delta in enumerate(deltas):
        delta = float(delta)
        bid_quote = _floor_to_lobster_int(mids - delta, tick_size)
        ask_quote = _ceil_to_lobster_int(mids + delta, tick_size)

        for idx in range(len(mids)):
            bid_queue = _level_volume_at(bid_prices[idx], bid_sizes[idx], int(bid_quote[idx]))
            ask_queue = _level_volume_at(ask_prices[idx], ask_sizes[idx], int(ask_quote[idx]))

            sell_volume = trade_flow.sell_reaching_bid(idx, int(bid_quote[idx]))
            buy_volume = trade_flow.buy_reaching_ask(idx, int(ask_quote[idx]))

            bid_fill = _queue_fill_size(bid_queue, sell_volume, order_size)
            ask_fill = _queue_fill_size(ask_queue, buy_volume, order_size)

            bid_posted[j] += 1
            ask_posted[j] += 1
            bid_queue_total[j] += bid_queue
            ask_queue_total[j] += ask_queue
            bid_reaching_total[j] += sell_volume
            ask_reaching_total[j] += buy_volume
            bid_fill_size_total[j] += bid_fill
            ask_fill_size_total[j] += ask_fill
            if bid_fill > 0:
                bid_fill_count[j] += 1
            if bid_fill >= order_size:
                bid_full_fill_count[j] += 1
            if ask_fill > 0:
                ask_fill_count[j] += 1
            if ask_fill >= order_size:
                ask_full_fill_count[j] += 1

        if bid_posted[j] > 0:
            lambda_bid[j] = _probability_to_intensity(bid_fill_count[j] / bid_posted[j], dt_seconds)
        if ask_posted[j] > 0:
            lambda_ask[j] = _probability_to_intensity(ask_fill_count[j] / ask_posted[j], dt_seconds)

    A_ask, k_ask = _fit_exp_decay(deltas, lambda_ask)
    A_bid, k_bid = _fit_exp_decay(deltas, lambda_bid)
    p_ask = np.divide(ask_fill_count, ask_posted, out=np.full(len(deltas), np.nan), where=ask_posted > 0)
    p_bid = np.divide(bid_fill_count, bid_posted, out=np.full(len(deltas), np.nan), where=bid_posted > 0)
    full_p_ask = np.divide(ask_full_fill_count, ask_posted, out=np.full(len(deltas), np.nan), where=ask_posted > 0)
    full_p_bid = np.divide(bid_full_fill_count, bid_posted, out=np.full(len(deltas), np.nan), where=bid_posted > 0)
    diagnostics = pd.DataFrame(
        {
            "p_ask": p_ask,
            "p_bid": p_bid,
            "full_p_ask": full_p_ask,
            "full_p_bid": full_p_bid,
            "ask_posted_count": ask_posted,
            "bid_posted_count": bid_posted,
            "ask_fill_count": ask_fill_count,
            "bid_fill_count": bid_fill_count,
            "ask_full_fill_count": ask_full_fill_count,
            "bid_full_fill_count": bid_full_fill_count,
            "mean_ask_queue_ahead": np.divide(
                ask_queue_total, ask_posted, out=np.full(len(deltas), np.nan), where=ask_posted > 0
            ),
            "mean_bid_queue_ahead": np.divide(
                bid_queue_total, bid_posted, out=np.full(len(deltas), np.nan), where=bid_posted > 0
            ),
            "mean_ask_reaching_volume": np.divide(
                ask_reaching_total, ask_posted, out=np.full(len(deltas), np.nan), where=ask_posted > 0
            ),
            "mean_bid_reaching_volume": np.divide(
                bid_reaching_total, bid_posted, out=np.full(len(deltas), np.nan), where=bid_posted > 0
            ),
            "mean_ask_fill_size": np.divide(
                ask_fill_size_total, ask_posted, out=np.full(len(deltas), np.nan), where=ask_posted > 0
            ),
            "mean_bid_fill_size": np.divide(
                bid_fill_size_total, bid_posted, out=np.full(len(deltas), np.nan), where=bid_posted > 0
            ),
        }
    )
    return ArrivalCalibration(
        float(A_ask),
        float(k_ask),
        float(A_bid),
        float(k_bid),
        deltas,
        lambda_ask,
        lambda_bid,
        diagnostics,
    )


def simulate_lobster_replay(
    bbo: pd.DataFrame,
    trade_flow: LobsterTradeFlow,
    params: ASParams,
    *,
    strategy: str = "as",
) -> pd.DataFrame:
    """Replay AS or symmetric MM quotes against LOBSTER visible trade flow."""

    if len(bbo) == 0:
        return pd.DataFrame()

    levels = _levels_from_bbo(bbo)
    bid_level_prices = bbo[[f"bid_price_int_{level}" for level in levels]].to_numpy(dtype=np.int64)
    bid_level_sizes = bbo[[f"bid_size_{level}" for level in levels]].to_numpy(dtype=np.int64)
    ask_level_prices = bbo[[f"ask_price_int_{level}" for level in levels]].to_numpy(dtype=np.int64)
    ask_level_sizes = bbo[[f"ask_size_{level}" for level in levels]].to_numpy(dtype=np.int64)

    q = 0
    cash = 0.0
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    times = bbo.index
    t0 = times[0]

    for idx, (timestamp, row) in enumerate(bbo.iterrows()):
        mid = float(row["mid"])
        if not math.isfinite(mid) or mid <= 0:
            continue
        elapsed = (timestamp - t0).total_seconds()
        if strategy == "symmetric":
            half_spread = max(params.tick_size, 0.5 * (2.0 / max(params.k, 1e-9)))
            bid = _floor_price(mid - half_spread, params.tick_size)
            ask = _ceil_price(mid + half_spread, params.tick_size)
            quotes = {
                "reservation": mid,
                "delta_bid": mid - bid,
                "delta_ask": ask - mid,
                "tau": params.horizon_seconds if params.rolling_horizon else max(0.0, params.horizon_seconds - elapsed),
            }
        else:
            quotes = as_quotes(mid, q, elapsed, params)
            bid = quotes["bid"]
            ask = quotes["ask"]

        quote_bid = q + params.order_size <= params.q_max
        quote_ask = q - params.order_size >= -params.q_max

        bid_fill_size = 0
        ask_fill_size = 0
        bid_queue = 0
        ask_queue = 0
        bid_reaching = 0
        ask_reaching = 0
        if quote_bid and math.isfinite(bid):
            bid_lobster_int = int(round(bid * LOBSTER_PRICE_SCALE))
            bid_queue = _level_volume_at(bid_level_prices[idx], bid_level_sizes[idx], bid_lobster_int)
            bid_reaching = trade_flow.sell_reaching_bid(idx, bid_lobster_int)
            bid_fill_size = _queue_fill_size(bid_queue, bid_reaching, params.order_size)
        if quote_ask and math.isfinite(ask):
            ask_lobster_int = int(round(ask * LOBSTER_PRICE_SCALE))
            ask_queue = _level_volume_at(ask_level_prices[idx], ask_level_sizes[idx], ask_lobster_int)
            ask_reaching = trade_flow.buy_reaching_ask(idx, ask_lobster_int)
            ask_fill_size = _queue_fill_size(ask_queue, ask_reaching, params.order_size)

        if bid_fill_size:
            cash -= bid_fill_size * bid
            q += bid_fill_size
        else:
            bid_fill_size = 0
        if ask_fill_size:
            cash += ask_fill_size * ask
            q -= ask_fill_size
        else:
            ask_fill_size = 0

        pnl = cash + q * mid
        rows.append(
            {
                "ts": timestamp,
                "mid": mid,
                "reservation": quotes["reservation"],
                "bid": bid,
                "ask": ask,
                "spread": ask - bid,
                "quote_bid": int(quote_bid),
                "quote_ask": int(quote_ask),
                "delta_bid": quotes["delta_bid"],
                "delta_ask": quotes["delta_ask"],
                "tau": quotes["tau"],
                "inventory": q,
                "cash": cash,
                "pnl": pnl,
                "bid_fill": int(bid_fill_size > 0),
                "ask_fill": int(ask_fill_size > 0),
                "bid_fill_size": bid_fill_size,
                "ask_fill_size": ask_fill_size,
                "bid_queue_ahead": bid_queue,
                "ask_queue_ahead": ask_queue,
                "bid_reaching_volume": bid_reaching,
                "ask_reaching_volume": ask_reaching,
                "fill_model": "lobster_level",
                "strategy": strategy,
            }
        )

    replay = pd.DataFrame(rows)
    if not replay.empty:
        replay = replay.set_index("ts")
    return replay


def _load_messages(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype={
            "time": "float64",
            "type": "int8",
            "order_id": "int64",
            "size": "int64",
            "price": "int64",
            "direction": "int8",
        },
    )


def _load_orderbook(path: Path, levels: int) -> pd.DataFrame:
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_int_{level}",
                f"ask_size_{level}",
                f"bid_price_int_{level}",
                f"bid_size_{level}",
            ]
        )
    frame = pd.read_csv(path, header=None, usecols=range(levels * 4), dtype="int64")
    frame.columns = columns
    return frame


def _sample_indices(
    times: np.ndarray,
    *,
    dt_seconds: float,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(times) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    effective_end = min(end_seconds, float(times[-1]) + dt_seconds)
    grid = np.arange(start_seconds, effective_end, dt_seconds, dtype=np.float64)
    idx = np.searchsorted(times, grid, side="right") - 1
    valid = idx >= 0
    return grid[valid], idx[valid].astype(np.int64)


def _build_sampled_bbo(
    files: LobsterDayFiles,
    sampled_book: pd.DataFrame,
    grid_seconds: np.ndarray,
) -> pd.DataFrame:
    local_midnight = pd.Timestamp(files.date.isoformat(), tz="America/New_York")
    local_times = local_midnight + pd.to_timedelta(grid_seconds, unit="s")
    index = local_times.tz_convert("UTC")

    bbo = pd.DataFrame(index=index)
    for level in range(1, files.levels + 1):
        bbo[f"ask_price_int_{level}"] = sampled_book[f"ask_price_int_{level}"].to_numpy(dtype=np.int64)
        bbo[f"ask_size_{level}"] = sampled_book[f"ask_size_{level}"].to_numpy(dtype=np.int64)
        bbo[f"bid_price_int_{level}"] = sampled_book[f"bid_price_int_{level}"].to_numpy(dtype=np.int64)
        bbo[f"bid_size_{level}"] = sampled_book[f"bid_size_{level}"].to_numpy(dtype=np.int64)

    bbo["best_ask"] = bbo["ask_price_int_1"] / LOBSTER_PRICE_SCALE
    bbo["best_bid"] = bbo["bid_price_int_1"] / LOBSTER_PRICE_SCALE
    bbo["ask_size"] = bbo["ask_size_1"]
    bbo["bid_size"] = bbo["bid_size_1"]
    bbo["mid"] = (bbo["best_bid"] + bbo["best_ask"]) / 2.0
    bbo["spread"] = bbo["best_ask"] - bbo["best_bid"]
    return bbo


def _build_interval_trade_flow(
    messages: pd.DataFrame,
    *,
    grid_seconds: np.ndarray,
    dt_seconds: float,
) -> tuple[LobsterTradeFlow, pd.DataFrame]:
    n = len(grid_seconds)
    buy_buckets: list[dict[int, int]] = [dict() for _ in range(n)]
    sell_buckets: list[dict[int, int]] = [dict() for _ in range(n)]
    buy_max = np.full(n, np.nan, dtype=float)
    sell_min = np.full(n, np.nan, dtype=float)
    buy_volume = np.zeros(n, dtype=np.int64)
    sell_volume = np.zeros(n, dtype=np.int64)

    if n == 0:
        flow = LobsterTradeFlow([], [])
        extrema = pd.DataFrame(
            {
                "max_buy_trade": [],
                "min_sell_trade": [],
                "buy_trade_volume": [],
                "sell_trade_volume": [],
            }
        )
        return flow, extrema

    visible = messages[(messages["type"] == 4) & (messages["price"] > 0)]
    if not visible.empty:
        bins = np.floor((visible["time"].to_numpy(dtype=float) - grid_seconds[0]) / dt_seconds).astype(np.int64)
        prices = visible["price"].to_numpy(dtype=np.int64)
        sizes = visible["size"].to_numpy(dtype=np.int64)
        directions = visible["direction"].to_numpy(dtype=np.int8)
        valid = (bins >= 0) & (bins < n)
        for bin_idx, price, size, direction in zip(bins[valid], prices[valid], sizes[valid], directions[valid]):
            fixed_price = int(price) * FIXED_PRICE_SCALE
            dollar_price = int(price) / LOBSTER_PRICE_SCALE
            if direction < 0:
                bucket = buy_buckets[int(bin_idx)]
                bucket[fixed_price] = bucket.get(fixed_price, 0) + int(size)
                buy_volume[int(bin_idx)] += int(size)
                if not math.isfinite(buy_max[int(bin_idx)]) or dollar_price > buy_max[int(bin_idx)]:
                    buy_max[int(bin_idx)] = dollar_price
            elif direction > 0:
                bucket = sell_buckets[int(bin_idx)]
                bucket[fixed_price] = bucket.get(fixed_price, 0) + int(size)
                sell_volume[int(bin_idx)] += int(size)
                if not math.isfinite(sell_min[int(bin_idx)]) or dollar_price < sell_min[int(bin_idx)]:
                    sell_min[int(bin_idx)] = dollar_price

    buy_curves = [_curve_from_bucket(bucket) for bucket in buy_buckets]
    sell_curves = [_curve_from_bucket(bucket) for bucket in sell_buckets]
    extrema = pd.DataFrame(
        {
            "max_buy_trade": buy_max,
            "min_sell_trade": sell_min,
            "buy_trade_volume": buy_volume,
            "sell_trade_volume": sell_volume,
        }
    )
    flow = LobsterTradeFlow(buy_aggressor=buy_curves, sell_aggressor=sell_curves)
    return flow, extrema


def _curve_from_bucket(bucket: dict[int, int]) -> PriceVolumeCurve:
    if not bucket:
        return PriceVolumeCurve.empty()
    prices = np.fromiter(bucket.keys(), dtype=np.int64)
    volumes = np.fromiter(bucket.values(), dtype=np.int64)
    return PriceVolumeCurve.from_arrays(prices, volumes)


def _floor_to_lobster_int(values: np.ndarray, tick_size: float) -> np.ndarray:
    ticks = np.floor(values / tick_size)
    return np.rint(ticks * tick_size * LOBSTER_PRICE_SCALE).astype(np.int64)


def _ceil_to_lobster_int(values: np.ndarray, tick_size: float) -> np.ndarray:
    ticks = np.ceil(values / tick_size)
    return np.rint(ticks * tick_size * LOBSTER_PRICE_SCALE).astype(np.int64)


def _floor_price(value: float, tick_size: float) -> float:
    return math.floor(value / tick_size) * tick_size


def _ceil_price(value: float, tick_size: float) -> float:
    return math.ceil(value / tick_size) * tick_size


def _level_volume_at(level_prices: np.ndarray, level_sizes: np.ndarray, price: int) -> int:
    matches = level_prices == price
    if not matches.any():
        return 0
    return int(level_sizes[matches].sum())


def _levels_from_bbo(bbo: pd.DataFrame) -> list[int]:
    levels: list[int] = []
    for column in bbo.columns:
        match = re.fullmatch(r"ask_price_int_(\d+)", str(column))
        if match:
            levels.append(int(match.group(1)))
    return sorted(levels)


def _cache_meta(files: LobsterDayFiles, dt_seconds: float) -> dict[str, object]:
    msg_stat = files.message_path.stat()
    book_stat = files.orderbook_path.stat()
    return {
        "date": files.date.isoformat(),
        "message_path": str(files.message_path),
        "message_size": msg_stat.st_size,
        "message_mtime_ns": msg_stat.st_mtime_ns,
        "orderbook_path": str(files.orderbook_path),
        "orderbook_size": book_stat.st_size,
        "orderbook_mtime_ns": book_stat.st_mtime_ns,
        "dt_seconds": float(dt_seconds),
        "levels": files.levels,
    }


def _cache_path(cache_dir: Path | str | None, files: LobsterDayFiles, dt_seconds: float) -> Path | None:
    if cache_dir is None:
        return None
    dt_ms = int(round(dt_seconds * 1000))
    return Path(cache_dir) / f"{files.symbol}_{files.date.isoformat()}_dt{dt_ms}_levels{files.levels}.pkl"
