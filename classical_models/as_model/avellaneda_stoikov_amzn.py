"""
Avellaneda--Stoikov market-making calibration and replay simulation.

This file is intended as a research scaffold for AMZN Databento MBO / L3 data.
It provides:
  1. A pure Avellaneda--Stoikov Poisson-fill simulator.
  2. Utilities to normalize Databento MBO data loaded from DBN, CSV, or Parquet.
  3. A simple L3 order-book reconstructor to produce sampled BBO/midprice data.
  4. Empirical calibration of sigma and lambda(delta) = A exp(-k delta).
  5. Historical replay simulation with trade-through or queue-aware fills.

Queue-aware simplification:
  The queue-aware model estimates displayed volume already resting at the AS
  quote price and fills only after historical opposite-side executions at that
  price or better exceed that queue. Cancellations ahead of the agent are not
  credited, so this is intentionally conservative.

Example:
  python avellaneda_stoikov_amzn.py --input amzn_2024-01-03.mbo.dbn.zst \
      --dt-ms 250 --gamma 0.002 --horizon-seconds 600 --order-size 10 --q-max 200

Dependencies:
  pip install numpy pandas
  Optional for DBN files: pip install databento
"""
from __future__ import annotations

import argparse
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from as_model import ASParams as PureASParams
from as_model import ASState, compute_as_quote

FIXED_PRICE_SCALE = 1_000_000_000.0
UNDEF_INT64_PRICE = np.iinfo(np.int64).max


def _price_to_int(price: float) -> int:
    """Convert a dollar price to Databento fixed-point integer units."""
    if not np.isfinite(price) or price <= 0:
        return UNDEF_INT64_PRICE
    scaled = price * FIXED_PRICE_SCALE
    if not np.isfinite(scaled) or scaled >= UNDEF_INT64_PRICE:
        return UNDEF_INT64_PRICE
    return int(round(scaled))


def _floor_to_tick(price_int: int, tick_int: int) -> int:
    return (price_int // tick_int) * tick_int


def _ceil_to_tick(price_int: int, tick_int: int) -> int:
    return ((price_int + tick_int - 1) // tick_int) * tick_int


def _candidate_quote_ints(mid: float, delta: float, tick_size: float) -> tuple[int, int]:
    """Return valid bid/ask price integers around mid for a delta grid point."""
    mid_int = _price_to_int(mid)
    tick_int = _price_to_int(tick_size)
    delta_int = max(0, int(round(delta * FIXED_PRICE_SCALE)))
    if mid_int == UNDEF_INT64_PRICE or tick_int <= 0 or tick_int == UNDEF_INT64_PRICE:
        return (UNDEF_INT64_PRICE, UNDEF_INT64_PRICE)
    bid_int = _floor_to_tick(mid_int - delta_int, tick_int)
    ask_int = _ceil_to_tick(mid_int + delta_int, tick_int)
    if bid_int <= 0 or ask_int <= 0:
        return (UNDEF_INT64_PRICE, UNDEF_INT64_PRICE)
    return (bid_int, ask_int)


def _queue_fill_size(queue_ahead: int | float, reaching_volume: int | float, order_size: int) -> int:
    """Conservative queue fill rule used by the queue-aware approximation."""
    queue = max(0, int(queue_ahead))
    reaching = max(0, int(reaching_volume))
    return int(min(max(0, order_size), max(0, reaching - queue)))


def _timestamp_series_ns(ts: pd.Series) -> pd.Series:
    """Return UTC timestamps as nanosecond integers regardless of pandas unit."""
    converted = pd.to_datetime(ts, utc=True)
    converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted.astype("datetime64[ns]").astype("int64")


def _datetime_index_ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Return a DatetimeIndex as nanosecond integers regardless of pandas unit."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.astype("datetime64[ns]").astype("int64").to_numpy()


def _char(x: Any) -> str:
    """Normalize bytes/strings/enums to one-character strings."""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    s = str(x)
    # Databento enum display may be like "Action.ADD" in some contexts.
    if len(s) > 1 and s[-1] in {"A", "B", "C", "F", "M", "N", "R", "T"}:
        return s[-1]
    return s[:1]


def load_mbo(path: str | Path) -> pd.DataFrame:
    """Load an MBO file from DBN/DBN.ZST, Parquet, or CSV into a DataFrame.

    The function attempts Databento DBNStore for DBN inputs. For half-year data,
    use this per day or per file rather than trying to load everything at once.
    """
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        df = pd.read_csv(path)
    elif ".dbn" in suffixes:
        try:
            import databento as db  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Reading DBN files requires `pip install databento`, or convert "
                "the file to CSV/Parquet first."
            ) from exc
        store = db.DBNStore.from_file(str(path))
        df = store.to_df()
        df = df.reset_index()
    else:
        raise ValueError(f"Unsupported input format: {path}")

    return normalize_mbo_df(df)


def normalize_mbo_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common MBO columns for modelling.

    Expected Databento-style columns include: ts_event, action, side, price,
    size, order_id, flags. The output has:
      ts_event: timezone-aware pandas datetime in UTC
      action, side: one-character strings
      price_int: integer price representation when possible
      price: dollar price as float
    """
    out = df.copy()

    if "ts_event" not in out.columns:
        # Databento to_df() can put ts_event in the index before reset_index().
        for candidate in ["index", "ts_recv"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "ts_event"})
                break
    if "ts_event" not in out.columns:
        raise ValueError("Input must contain a ts_event column or DatetimeIndex.")

    out["ts_event"] = pd.to_datetime(out["ts_event"], utc=True)

    if "action" not in out.columns:
        raise ValueError("Input must contain an action column.")
    if "side" not in out.columns:
        out["side"] = "N"

    out["action"] = out["action"].map(_char)
    out["side"] = out["side"].map(_char)

    if "price" not in out.columns:
        raise ValueError("Input must contain a price column.")

    price_raw = out["price"]
    price_values = pd.to_numeric(price_raw, errors="coerce").astype("float64").to_numpy()
    finite_price = price_values[np.isfinite(price_values)]
    max_abs_price = float(np.max(np.abs(finite_price))) if finite_price.size else 0.0

    price_int = np.full(len(out), UNDEF_INT64_PRICE, dtype=np.int64)
    price_float = np.full(len(out), np.nan, dtype=np.float64)

    # Databento fixed-point prices use 1 unit = 1e-9. Depending on how DBN is
    # decoded, undefined prices can force the column to float with NaN values.
    if max_abs_price > 10_000_000:
        valid_price = (
            np.isfinite(price_values)
            & (price_values > 0)
            & (price_values < UNDEF_INT64_PRICE)
        )
        price_int[valid_price] = np.rint(price_values[valid_price]).astype("int64")
        price_float[valid_price] = price_int[valid_price].astype("float64") / FIXED_PRICE_SCALE
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            scaled_price = price_values * FIXED_PRICE_SCALE
        valid_price = (
            np.isfinite(price_values)
            & (price_values > 0)
            & np.isfinite(scaled_price)
            & (scaled_price < UNDEF_INT64_PRICE)
        )
        price_float[valid_price] = price_values[valid_price]
        price_int[valid_price] = np.rint(scaled_price[valid_price]).astype("int64")

    out["price"] = price_float
    out["price_int"] = price_int

    if "size" not in out.columns:
        out["size"] = 0
    out["size"] = out["size"].fillna(0).astype("int64")

    if "order_id" not in out.columns:
        out["order_id"] = np.arange(len(out), dtype=np.int64)
    out["order_id"] = out["order_id"].fillna(0).astype("int64")

    if "sequence" in out.columns:
        out = out.sort_values(["ts_event", "sequence"], kind="mergesort")
    else:
        out = out.sort_values("ts_event", kind="mergesort")
    out = out.reset_index(drop=True)
    return out


@dataclass
class Order:
    side: str
    price_int: int
    size: int
    ts_event: pd.Timestamp


@dataclass
class SimpleMBOBook:
    """Minimal order-level book with lazy heaps for fast BBO extraction."""

    orders: dict[int, Order] = field(default_factory=dict)
    bid_levels: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    ask_levels: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    bid_heap: list[int] = field(default_factory=list)  # negative prices
    ask_heap: list[int] = field(default_factory=list)  # positive prices

    def clear(self) -> None:
        self.orders.clear()
        self.bid_levels.clear()
        self.ask_levels.clear()
        self.bid_heap.clear()
        self.ask_heap.clear()

    def _levels(self, side: str) -> dict[int, int]:
        return self.bid_levels if side == "B" else self.ask_levels

    def _add_level(self, side: str, price_int: int, size: int) -> None:
        if size <= 0 or price_int <= 0 or price_int == UNDEF_INT64_PRICE:
            return
        levels = self._levels(side)
        was_zero = levels.get(price_int, 0) <= 0
        levels[price_int] += size
        if was_zero:
            if side == "B":
                heapq.heappush(self.bid_heap, -price_int)
            elif side == "A":
                heapq.heappush(self.ask_heap, price_int)

    def _sub_level(self, side: str, price_int: int, size: int) -> None:
        if size <= 0:
            return
        levels = self._levels(side)
        levels[price_int] -= size
        if levels[price_int] <= 0:
            levels.pop(price_int, None)

    def apply_row(self, row: Any) -> None:
        action = _char(row.action)
        side = _char(row.side)
        order_id = int(row.order_id)
        price_int = int(row.price_int)
        size = int(row.size)
        ts_event = row.ts_event

        # Per Databento's MBO state-management convention, trades/fills/none do
        # not mutate resting state; fills are accompanied by cancel records.
        if action in {"T", "F", "N"}:
            return
        if action == "R":
            self.clear()
            return
        if action == "A":
            # Defensive replacement if an ID reappears.
            old = self.orders.pop(order_id, None)
            if old is not None:
                self._sub_level(old.side, old.price_int, old.size)
            if side in {"B", "A"} and size > 0 and price_int > 0 and price_int != UNDEF_INT64_PRICE:
                self.orders[order_id] = Order(side, price_int, size, ts_event)
                self._add_level(side, price_int, size)
            return
        if action == "C":
            old = self.orders.get(order_id)
            if old is None:
                return
            cancel_size = min(size, old.size)
            old.size -= cancel_size
            self._sub_level(old.side, old.price_int, cancel_size)
            if old.size <= 0:
                self.orders.pop(order_id, None)
            return
        if action == "M":
            old = self.orders.get(order_id)
            if old is None:
                if side in {"B", "A"} and size > 0:
                    self.orders[order_id] = Order(side, price_int, size, ts_event)
                    self._add_level(side, price_int, size)
                return
            self._sub_level(old.side, old.price_int, old.size)
            new_side = side if side in {"B", "A"} else old.side
            old.side = new_side
            old.price_int = price_int
            old.size = size
            old.ts_event = ts_event
            if size > 0:
                self._add_level(old.side, old.price_int, old.size)
            else:
                self.orders.pop(order_id, None)
            return

    def _best_bid_int(self) -> Optional[int]:
        while self.bid_heap:
            px = -self.bid_heap[0]
            if self.bid_levels.get(px, 0) > 0:
                return px
            heapq.heappop(self.bid_heap)
        return None

    def _best_ask_int(self) -> Optional[int]:
        while self.ask_heap:
            px = self.ask_heap[0]
            if self.ask_levels.get(px, 0) > 0:
                return px
            heapq.heappop(self.ask_heap)
        return None

    def bbo(self) -> tuple[float, float, int, int]:
        bid_int = self._best_bid_int()
        ask_int = self._best_ask_int()
        if bid_int is None or ask_int is None:
            return (np.nan, np.nan, 0, 0)
        return (
            bid_int / FIXED_PRICE_SCALE,
            ask_int / FIXED_PRICE_SCALE,
            int(self.bid_levels.get(bid_int, 0)),
            int(self.ask_levels.get(ask_int, 0)),
        )

    def level_volume(self, side: str, price_int: int) -> int:
        """Displayed same-side volume at an exact price level."""
        if price_int <= 0 or price_int == UNDEF_INT64_PRICE:
            return 0
        if side == "B":
            return int(self.bid_levels.get(price_int, 0))
        if side == "A":
            return int(self.ask_levels.get(price_int, 0))
        return 0


def build_sampled_bbo(
    mbo: pd.DataFrame,
    dt: str | pd.Timedelta = "250ms",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Reconstruct and sample BBO from MBO events.

    For correctness, the input should include the initial book snapshot or a
    complete replay from the start of the trading day. Then filter the sampled
    output to regular trading hours.
    """
    if isinstance(dt, str):
        dt = pd.Timedelta(dt)
    if len(mbo) == 0:
        raise ValueError("Empty MBO input.")

    start = pd.Timestamp(start, tz="UTC") if start is not None else mbo["ts_event"].iloc[0].ceil(dt)
    end = pd.Timestamp(end, tz="UTC") if end is not None else mbo["ts_event"].iloc[-1].floor(dt)

    book = SimpleMBOBook()
    rows: list[dict[str, Any]] = []
    next_sample = start

    for row in mbo.itertuples(index=False):
        ts = row.ts_event
        # Emit state up to this event time using state from previous events.
        while next_sample <= ts and next_sample <= end:
            bid, ask, bid_sz, ask_sz = book.bbo()
            if np.isfinite(bid) and np.isfinite(ask) and bid < ask:
                rows.append(
                    {
                        "ts": next_sample,
                        "best_bid": bid,
                        "best_ask": ask,
                        "bid_size": bid_sz,
                        "ask_size": ask_sz,
                        "mid": 0.5 * (bid + ask),
                        "spread": ask - bid,
                    }
                )
            next_sample += dt
        book.apply_row(row)
        if next_sample > end:
            break

    return pd.DataFrame(rows).set_index("ts") if rows else pd.DataFrame()


def filter_regular_hours(df: pd.DataFrame, start_time: str = "09:30", end_time: str = "16:00") -> pd.DataFrame:
    """Filter a UTC-indexed DataFrame or ts_event column to US regular trading hours."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        local_index = out.index.tz_convert("America/New_York")
        mask = (local_index.time >= pd.Timestamp(start_time).time()) & (
            local_index.time < pd.Timestamp(end_time).time()
        )
        return out.loc[mask]
    if "ts_event" not in out.columns:
        raise ValueError("DataFrame must have a DatetimeIndex or ts_event column.")
    local_ts = out["ts_event"].dt.tz_convert("America/New_York")
    mask = (local_ts.dt.time >= pd.Timestamp(start_time).time()) & (
        local_ts.dt.time < pd.Timestamp(end_time).time()
    )
    return out.loc[mask].reset_index(drop=True)


def estimate_sigma_from_mid(bbo: pd.DataFrame, winsorize_quantile: float = 0.999) -> float:
    """Estimate arithmetic Brownian sigma in dollars / sqrt(second)."""
    mids = bbo["mid"].dropna().astype(float)
    if len(mids) < 10:
        raise ValueError("Need more midprice samples to estimate sigma.")
    diffs = mids.diff().dropna()
    if winsorize_quantile is not None:
        cap = diffs.abs().quantile(winsorize_quantile)
        diffs = diffs.clip(-cap, cap)
    dt_seconds = np.median(np.diff(_datetime_index_ns(mids.index))) / 1e9
    return float(diffs.std(ddof=1) / math.sqrt(dt_seconds))


def resolve_replay_policy_sigma(daily_sigma: float, fixed_policy_sigma: float | None = None) -> float:
    """Return the sigma used by the replay policy, validating fixed held-out inputs."""
    sigma = daily_sigma if fixed_policy_sigma is None else fixed_policy_sigma
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Policy sigma must be finite and positive, got {sigma!r}.")
    return float(sigma)


def interval_trade_extrema(
    mbo: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    dt_seconds: float,
) -> pd.DataFrame:
    """For each decision interval, compute max buy-trade and min sell-trade price.

    Databento MBO uses action T for an aggressing order trade. The side field is
    interpreted here as the aggressor side: B means buy aggressor; A means sell
    aggressor. Verify this for your venue/dataset before relying on results.
    """
    n = len(grid_index)
    out = pd.DataFrame(
        {
            "max_buy_trade": np.full(n, np.nan),
            "min_sell_trade": np.full(n, np.nan),
            "buy_trade_volume": np.zeros(n),
            "sell_trade_volume": np.zeros(n),
        },
        index=grid_index,
    )
    if len(mbo) == 0 or n == 0:
        return out

    trades = mbo[mbo["action"].eq("T")].copy()
    if trades.empty:
        return out

    start_ns = grid_index[0].value
    dt_ns = int(dt_seconds * 1e9)
    bins = ((_timestamp_series_ns(trades["ts_event"]) - start_ns) // dt_ns).astype(int)
    trades = trades.assign(_bin=bins)
    trades = trades[(trades["_bin"] >= 0) & (trades["_bin"] < n)]
    if trades.empty:
        return out

    buy = trades[trades["side"].eq("B")]
    if not buy.empty:
        g = buy.groupby("_bin")
        idx = g["price"].max().index.to_numpy(dtype=int)
        out.iloc[idx, out.columns.get_loc("max_buy_trade")] = g["price"].max().to_numpy()
        out.iloc[idx, out.columns.get_loc("buy_trade_volume")] = g["size"].sum().to_numpy()

    sell = trades[trades["side"].eq("A")]
    if not sell.empty:
        g = sell.groupby("_bin")
        idx = g["price"].min().index.to_numpy(dtype=int)
        out.iloc[idx, out.columns.get_loc("min_sell_trade")] = g["price"].min().to_numpy()
        out.iloc[idx, out.columns.get_loc("sell_trade_volume")] = g["size"].sum().to_numpy()

    return out


@dataclass
class PriceVolumeCurve:
    """Sorted per-interval trade volume curve for one aggressor side."""

    prices: np.ndarray
    volumes: np.ndarray
    prefix: np.ndarray
    total_volume: int

    @classmethod
    def empty(cls) -> "PriceVolumeCurve":
        empty_prices = np.array([], dtype=np.int64)
        empty_volumes = np.array([], dtype=np.int64)
        return cls(empty_prices, empty_volumes, np.array([0], dtype=np.int64), 0)

    @classmethod
    def from_arrays(cls, prices: np.ndarray, volumes: np.ndarray) -> "PriceVolumeCurve":
        if len(prices) == 0:
            return EMPTY_PRICE_VOLUME
        order = np.argsort(prices, kind="mergesort")
        sorted_prices = prices[order].astype(np.int64, copy=False)
        sorted_volumes = volumes[order].astype(np.int64, copy=False)
        prefix = np.concatenate(([0], np.cumsum(sorted_volumes, dtype=np.int64)))
        return cls(sorted_prices, sorted_volumes, prefix, int(prefix[-1]))

    def volume_at_or_below(self, price_int: int) -> int:
        if self.total_volume <= 0 or price_int <= 0 or price_int == UNDEF_INT64_PRICE:
            return 0
        pos = int(np.searchsorted(self.prices, price_int, side="right"))
        return int(self.prefix[pos])

    def volume_at_or_above(self, price_int: int) -> int:
        if self.total_volume <= 0 or price_int <= 0 or price_int == UNDEF_INT64_PRICE:
            return 0
        pos = int(np.searchsorted(self.prices, price_int, side="left"))
        return int(self.total_volume - self.prefix[pos])


EMPTY_PRICE_VOLUME = PriceVolumeCurve.empty()


@dataclass
class IntervalTradeFlow:
    """Opposite-side trade volume curves aligned to the decision grid."""

    buy: list[PriceVolumeCurve]
    sell: list[PriceVolumeCurve]

    def buy_reaching_ask(self, interval: int, ask_int: int) -> int:
        return self.buy[interval].volume_at_or_above(ask_int)

    def sell_reaching_bid(self, interval: int, bid_int: int) -> int:
        return self.sell[interval].volume_at_or_below(bid_int)


def build_interval_trade_flow(
    mbo: pd.DataFrame,
    grid_index: pd.DatetimeIndex,
    dt_seconds: float,
) -> IntervalTradeFlow:
    """Build interval trade-volume curves for queue-aware fills.

    B aggressor trades are buyer-initiated and can fill asks. A aggressor trades
    are seller-initiated and can fill bids.
    """
    n = len(grid_index)
    buy = [EMPTY_PRICE_VOLUME] * n
    sell = [EMPTY_PRICE_VOLUME] * n
    if len(mbo) == 0 or n == 0:
        return IntervalTradeFlow(buy=buy, sell=sell)

    trades = mbo[mbo["action"].eq("T")].copy()
    if trades.empty:
        return IntervalTradeFlow(buy=buy, sell=sell)

    start_ns = grid_index[0].value
    dt_ns = int(dt_seconds * 1e9)
    bins = ((_timestamp_series_ns(trades["ts_event"]) - start_ns) // dt_ns).astype(int)
    trades = trades.assign(_bin=bins)
    trades = trades[(trades["_bin"] >= 0) & (trades["_bin"] < n)]
    trades = trades[
        trades["price_int"].gt(0)
        & trades["price_int"].ne(UNDEF_INT64_PRICE)
        & trades["side"].isin(["B", "A"])
        & trades["size"].gt(0)
    ]
    if trades.empty:
        return IntervalTradeFlow(buy=buy, sell=sell)

    agg = (
        trades.groupby(["_bin", "side", "price_int"], sort=False)["size"]
        .sum()
        .reset_index()
    )
    for (bin_idx, side), part in agg.groupby(["_bin", "side"], sort=False):
        curve = PriceVolumeCurve.from_arrays(
            part["price_int"].to_numpy(dtype=np.int64),
            part["size"].to_numpy(dtype=np.int64),
        )
        if side == "B":
            buy[int(bin_idx)] = curve
        elif side == "A":
            sell[int(bin_idx)] = curve

    return IntervalTradeFlow(buy=buy, sell=sell)


@dataclass
class ArrivalCalibration:
    A_ask: float
    k_ask: float
    A_bid: float
    k_bid: float
    deltas: np.ndarray
    lambda_ask: np.ndarray
    lambda_bid: np.ndarray
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def A(self) -> float:
        return float(np.nanmean([self.A_ask, self.A_bid]))

    @property
    def k(self) -> float:
        return float(np.nanmean([self.k_ask, self.k_bid]))

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "delta": self.deltas,
                "lambda_ask": self.lambda_ask,
                "lambda_bid": self.lambda_bid,
            }
        )
        if not self.diagnostics.empty:
            for column in self.diagnostics.columns:
                if column not in out.columns:
                    out[column] = self.diagnostics[column].to_numpy()
        return out


def _fit_exp_decay(deltas: np.ndarray, lambdas: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(lambdas) & (lambdas > 0) & np.isfinite(deltas)
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    slope, intercept = np.polyfit(deltas[mask], np.log(lambdas[mask]), 1)
    return float(math.exp(intercept)), float(max(1e-12, -slope))


def _probability_to_intensity(probability: float, dt_seconds: float) -> float:
    # Poisson conversion: P(fill in dt) = 1 - exp(-lambda dt)
    if 0 < probability < 1 and dt_seconds > 0:
        return float(-math.log1p(-probability) / dt_seconds)
    return float("nan")


def calibrate_arrival_rates(
    bbo: pd.DataFrame,
    trade_extrema: pd.DataFrame,
    deltas: Iterable[float],
    dt_seconds: float,
) -> ArrivalCalibration:
    """Calibrate lambda(delta) = A exp(-k delta) from trade-through events."""
    deltas = np.asarray(list(deltas), dtype=float)
    mids = bbo["mid"].to_numpy(dtype=float)
    buy_max = trade_extrema["max_buy_trade"].to_numpy(dtype=float)
    sell_min = trade_extrema["min_sell_trade"].to_numpy(dtype=float)

    n = min(len(mids), len(buy_max), len(sell_min))
    mids, buy_max, sell_min = mids[:n], buy_max[:n], sell_min[:n]

    lambda_ask = np.full(len(deltas), np.nan)
    lambda_bid = np.full(len(deltas), np.nan)
    p_ask_arr = np.full(len(deltas), np.nan)
    p_bid_arr = np.full(len(deltas), np.nan)
    ask_fill_counts = np.zeros(len(deltas), dtype=np.int64)
    bid_fill_counts = np.zeros(len(deltas), dtype=np.int64)
    posted_counts = np.full(len(deltas), n, dtype=np.int64)

    for j, delta in enumerate(deltas):
        ask_hits = np.isfinite(buy_max) & (buy_max >= mids + delta)
        bid_hits = np.isfinite(sell_min) & (sell_min <= mids - delta)
        p_ask = ask_hits.mean()
        p_bid = bid_hits.mean()
        p_ask_arr[j] = p_ask
        p_bid_arr[j] = p_bid
        ask_fill_counts[j] = int(ask_hits.sum())
        bid_fill_counts[j] = int(bid_hits.sum())
        lambda_ask[j] = _probability_to_intensity(p_ask, dt_seconds)
        lambda_bid[j] = _probability_to_intensity(p_bid, dt_seconds)

    A_ask, k_ask = _fit_exp_decay(deltas, lambda_ask)
    A_bid, k_bid = _fit_exp_decay(deltas, lambda_bid)
    diagnostics = pd.DataFrame(
        {
            "p_ask": p_ask_arr,
            "p_bid": p_bid_arr,
            "full_p_ask": p_ask_arr,
            "full_p_bid": p_bid_arr,
            "ask_posted_count": posted_counts,
            "bid_posted_count": posted_counts,
            "ask_fill_count": ask_fill_counts,
            "bid_fill_count": bid_fill_counts,
            "ask_full_fill_count": ask_fill_counts,
            "bid_full_fill_count": bid_fill_counts,
            "mean_ask_queue_ahead": np.full(len(deltas), np.nan),
            "mean_bid_queue_ahead": np.full(len(deltas), np.nan),
            "mean_ask_reaching_volume": np.full(len(deltas), np.nan),
            "mean_bid_reaching_volume": np.full(len(deltas), np.nan),
            "mean_ask_fill_size": np.full(len(deltas), np.nan),
            "mean_bid_fill_size": np.full(len(deltas), np.nan),
        }
    )
    return ArrivalCalibration(A_ask, k_ask, A_bid, k_bid, deltas, lambda_ask, lambda_bid, diagnostics)


def _safe_mean(total: float, count: int) -> float:
    return float(total / count) if count > 0 else float("nan")


def calibrate_queue_arrival_rates(
    mbo: pd.DataFrame,
    bbo: pd.DataFrame,
    trade_flow: IntervalTradeFlow,
    deltas: Iterable[float],
    dt_seconds: float,
    tick_size: float,
    order_size: int,
) -> ArrivalCalibration:
    """Calibrate arrival rates with a conservative displayed-queue fill rule."""
    deltas = np.asarray(list(deltas), dtype=float)
    n = len(bbo)
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

    if n == 0 or len(deltas) == 0:
        diagnostics = pd.DataFrame()
        return ArrivalCalibration(float("nan"), float("nan"), float("nan"), float("nan"), deltas, lambda_ask, lambda_bid, diagnostics)

    grid_index = bbo.index
    mids = bbo["mid"].to_numpy(dtype=float)
    book = SimpleMBOBook()
    sample_idx = 0

    def capture_sample(i: int) -> None:
        mid = mids[i]
        if not np.isfinite(mid):
            return
        for j, delta in enumerate(deltas):
            bid_int, ask_int = _candidate_quote_ints(mid, delta, tick_size)
            if bid_int != UNDEF_INT64_PRICE:
                bid_queue = book.level_volume("B", bid_int)
                bid_reaching = trade_flow.sell_reaching_bid(i, bid_int)
                bid_fill_size = _queue_fill_size(bid_queue, bid_reaching, order_size)
                bid_posted[j] += 1
                bid_queue_total[j] += bid_queue
                bid_reaching_total[j] += bid_reaching
                bid_fill_size_total[j] += bid_fill_size
                if bid_fill_size > 0:
                    bid_fill_count[j] += 1
                if bid_fill_size >= order_size:
                    bid_full_fill_count[j] += 1

            if ask_int != UNDEF_INT64_PRICE:
                ask_queue = book.level_volume("A", ask_int)
                ask_reaching = trade_flow.buy_reaching_ask(i, ask_int)
                ask_fill_size = _queue_fill_size(ask_queue, ask_reaching, order_size)
                ask_posted[j] += 1
                ask_queue_total[j] += ask_queue
                ask_reaching_total[j] += ask_reaching
                ask_fill_size_total[j] += ask_fill_size
                if ask_fill_size > 0:
                    ask_fill_count[j] += 1
                if ask_fill_size >= order_size:
                    ask_full_fill_count[j] += 1

    for row in mbo.itertuples(index=False):
        ts = row.ts_event
        while sample_idx < n and grid_index[sample_idx] <= ts:
            capture_sample(sample_idx)
            sample_idx += 1
        book.apply_row(row)
        if sample_idx >= n:
            break

    while sample_idx < n:
        capture_sample(sample_idx)
        sample_idx += 1

    p_ask = np.full(len(deltas), np.nan)
    p_bid = np.full(len(deltas), np.nan)
    full_p_ask = np.full(len(deltas), np.nan)
    full_p_bid = np.full(len(deltas), np.nan)
    mean_ask_queue = np.full(len(deltas), np.nan)
    mean_bid_queue = np.full(len(deltas), np.nan)
    mean_ask_reaching = np.full(len(deltas), np.nan)
    mean_bid_reaching = np.full(len(deltas), np.nan)
    mean_ask_fill_size = np.full(len(deltas), np.nan)
    mean_bid_fill_size = np.full(len(deltas), np.nan)

    for j in range(len(deltas)):
        if ask_posted[j] > 0:
            p_ask[j] = ask_fill_count[j] / ask_posted[j]
            full_p_ask[j] = ask_full_fill_count[j] / ask_posted[j]
            mean_ask_queue[j] = _safe_mean(ask_queue_total[j], int(ask_posted[j]))
            mean_ask_reaching[j] = _safe_mean(ask_reaching_total[j], int(ask_posted[j]))
            mean_ask_fill_size[j] = _safe_mean(ask_fill_size_total[j], int(ask_posted[j]))
            lambda_ask[j] = _probability_to_intensity(float(p_ask[j]), dt_seconds)
        if bid_posted[j] > 0:
            p_bid[j] = bid_fill_count[j] / bid_posted[j]
            full_p_bid[j] = bid_full_fill_count[j] / bid_posted[j]
            mean_bid_queue[j] = _safe_mean(bid_queue_total[j], int(bid_posted[j]))
            mean_bid_reaching[j] = _safe_mean(bid_reaching_total[j], int(bid_posted[j]))
            mean_bid_fill_size[j] = _safe_mean(bid_fill_size_total[j], int(bid_posted[j]))
            lambda_bid[j] = _probability_to_intensity(float(p_bid[j]), dt_seconds)

    A_ask, k_ask = _fit_exp_decay(deltas, lambda_ask)
    A_bid, k_bid = _fit_exp_decay(deltas, lambda_bid)
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
            "mean_ask_queue_ahead": mean_ask_queue,
            "mean_bid_queue_ahead": mean_bid_queue,
            "mean_ask_reaching_volume": mean_ask_reaching,
            "mean_bid_reaching_volume": mean_bid_reaching,
            "mean_ask_fill_size": mean_ask_fill_size,
            "mean_bid_fill_size": mean_bid_fill_size,
        }
    )
    return ArrivalCalibration(A_ask, k_ask, A_bid, k_bid, deltas, lambda_ask, lambda_bid, diagnostics)


@dataclass
class ASParams:
    gamma: float
    sigma: float
    k: float
    A: float
    horizon_seconds: float
    dt_seconds: float
    tick_size: float = 0.01
    order_size: int = 1
    q_max: int = 100
    min_spread_ticks: int = 1
    rolling_horizon: bool = True


def choose_gamma_for_target_skew(
    sigma: float,
    horizon_seconds: float,
    q_target: int,
    target_total_skew: float,
) -> float:
    """Choose gamma so q_target inventory moves reservation price by target_total_skew dollars."""
    denom = max(1e-12, abs(q_target) * sigma * sigma * horizon_seconds)
    return float(target_total_skew / denom)


def as_quotes(mid: float, q: int, t_elapsed: float, params: ASParams) -> dict[str, float]:
    """Compute Avellaneda--Stoikov reservation price, spread, bid, ask."""
    horizon_mode = "rolling" if params.rolling_horizon else "finite_episode"
    quote = compute_as_quote(
        PureASParams(
            sigma=params.sigma,
            k=params.k,
            gamma=params.gamma,
            horizon_seconds=params.horizon_seconds,
            tick_size=params.tick_size,
            min_spread_ticks=params.min_spread_ticks,
        ),
        ASState(mid=mid, inventory=q, elapsed_seconds=t_elapsed),
        horizon_mode=horizon_mode,
    )
    tau = params.horizon_seconds if params.rolling_horizon else max(0.0, params.horizon_seconds - t_elapsed)
    if quote.bid is None or quote.ask is None:
        raise RuntimeError("AS quote unexpectedly suppressed without an inventory limit.")

    return {
        "reservation": quote.reservation_price,
        "total_spread": quote.spread,
        "raw_total_spread": quote.raw_ask - quote.raw_bid,
        "bid": quote.bid,
        "ask": quote.ask,
        "delta_bid": mid - quote.bid,
        "delta_ask": quote.ask - mid,
        "tau": tau,
    }


def summarize_simulation(path: pd.DataFrame) -> dict[str, float]:
    if path.empty:
        return {}
    pnl = path["pnl"]
    inv = path["inventory"]
    dd = pnl - pnl.cummax()
    return {
        "final_pnl": float(pnl.iloc[-1]),
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std(ddof=1)),
        "max_drawdown": float(dd.min()),
        "final_inventory": float(inv.iloc[-1]),
        "max_abs_inventory": float(inv.abs().max()),
        "avg_sq_inventory": float((inv.astype(float) ** 2).mean()),
        "bid_fills": float(path["bid_fill"].sum()),
        "ask_fills": float(path["ask_fill"].sum()),
    }


def simulate_replay(
    bbo: pd.DataFrame,
    trade_extrema: pd.DataFrame,
    params: ASParams,
    strategy: str = "inventory",
    fill_model: str = "trade-through",
    mbo: Optional[pd.DataFrame] = None,
    trade_flow: Optional[IntervalTradeFlow] = None,
) -> pd.DataFrame:
    """Simulate AS strategy on sampled BBO using the selected historical fill model."""
    fill_model = fill_model.replace("_", "-")
    if fill_model == "queue-aware":
        fill_model = "queue"
    if fill_model == "queue":
        if mbo is None:
            raise ValueError("Queue-aware replay requires the normalized MBO dataframe.")
        if trade_flow is None:
            trade_flow = build_interval_trade_flow(mbo, bbo.index, params.dt_seconds)
        return simulate_replay_queue(mbo, bbo, trade_flow, params, strategy=strategy)
    if fill_model != "trade-through":
        raise ValueError(f"Unsupported fill model: {fill_model}")

    n = min(len(bbo), len(trade_extrema))
    if n == 0:
        return pd.DataFrame()

    cash = 0.0
    q = 0
    start_ts = bbo.index[0]
    rows: list[dict[str, Any]] = []

    mids = bbo["mid"].to_numpy(dtype=float)[:n]
    buy_max = trade_extrema["max_buy_trade"].to_numpy(dtype=float)[:n]
    sell_min = trade_extrema["min_sell_trade"].to_numpy(dtype=float)[:n]
    buy_volume = trade_extrema["buy_trade_volume"].to_numpy(dtype=float)[:n]
    sell_volume = trade_extrema["sell_trade_volume"].to_numpy(dtype=float)[:n]

    for i in range(n):
        ts = bbo.index[i]
        mid = mids[i]
        t_elapsed = (ts - start_ts).total_seconds()
        quotes = as_quotes(mid, 0 if strategy == "symmetric" else q, t_elapsed, params)
        bid = quotes["bid"]
        ask = quotes["ask"]

        quote_bid = q + params.order_size <= params.q_max
        quote_ask = q - params.order_size >= -params.q_max

        bid_fill = bool(quote_bid and np.isfinite(sell_min[i]) and sell_min[i] <= bid)
        ask_fill = bool(quote_ask and np.isfinite(buy_max[i]) and buy_max[i] >= ask)
        bid_fill_size = params.order_size if bid_fill else 0
        ask_fill_size = params.order_size if ask_fill else 0

        if bid_fill:
            cash -= bid * bid_fill_size
            q += bid_fill_size
        if ask_fill:
            cash += ask * ask_fill_size
            q -= ask_fill_size

        pnl = cash + q * mid
        rows.append(
            {
                "ts": ts,
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
                "bid_fill": int(bid_fill),
                "ask_fill": int(ask_fill),
                "bid_fill_size": bid_fill_size,
                "ask_fill_size": ask_fill_size,
                "bid_queue_ahead": 0,
                "ask_queue_ahead": 0,
                "bid_reaching_volume": float(sell_volume[i]) if bid_fill else 0.0,
                "ask_reaching_volume": float(buy_volume[i]) if ask_fill else 0.0,
                "fill_model": fill_model,
                "strategy": strategy,
            }
        )

    return pd.DataFrame(rows).set_index("ts")


def simulate_replay_queue(
    mbo: pd.DataFrame,
    bbo: pd.DataFrame,
    trade_flow: IntervalTradeFlow,
    params: ASParams,
    strategy: str = "inventory",
) -> pd.DataFrame:
    """Simulate AS replay using displayed queue ahead and reaching trade volume."""
    n = len(bbo)
    if n == 0:
        return pd.DataFrame()

    cash = 0.0
    q = 0
    start_ts = bbo.index[0]
    grid_index = bbo.index
    mids = bbo["mid"].to_numpy(dtype=float)
    book = SimpleMBOBook()
    rows: list[dict[str, Any]] = []
    sample_idx = 0

    def emit_sample(i: int) -> None:
        nonlocal cash, q

        ts = grid_index[i]
        mid = mids[i]
        t_elapsed = (ts - start_ts).total_seconds()
        quotes = as_quotes(mid, 0 if strategy == "symmetric" else q, t_elapsed, params)
        bid = quotes["bid"]
        ask = quotes["ask"]
        bid_int = _price_to_int(bid)
        ask_int = _price_to_int(ask)

        quote_bid = q + params.order_size <= params.q_max
        quote_ask = q - params.order_size >= -params.q_max

        bid_queue = book.level_volume("B", bid_int) if quote_bid else 0
        ask_queue = book.level_volume("A", ask_int) if quote_ask else 0
        bid_reaching = trade_flow.sell_reaching_bid(i, bid_int) if quote_bid else 0
        ask_reaching = trade_flow.buy_reaching_ask(i, ask_int) if quote_ask else 0
        bid_fill_size = _queue_fill_size(bid_queue, bid_reaching, params.order_size) if quote_bid else 0
        ask_fill_size = _queue_fill_size(ask_queue, ask_reaching, params.order_size) if quote_ask else 0
        bid_fill = bid_fill_size > 0
        ask_fill = ask_fill_size > 0

        if bid_fill:
            cash -= bid * bid_fill_size
            q += bid_fill_size
        if ask_fill:
            cash += ask * ask_fill_size
            q -= ask_fill_size

        pnl = cash + q * mid
        rows.append(
            {
                "ts": ts,
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
                "bid_fill": int(bid_fill),
                "ask_fill": int(ask_fill),
                "bid_fill_size": bid_fill_size,
                "ask_fill_size": ask_fill_size,
                "bid_queue_ahead": bid_queue,
                "ask_queue_ahead": ask_queue,
                "bid_reaching_volume": bid_reaching,
                "ask_reaching_volume": ask_reaching,
                "fill_model": "queue",
                "strategy": strategy,
            }
        )

    for row in mbo.itertuples(index=False):
        ts = row.ts_event
        while sample_idx < n and grid_index[sample_idx] <= ts:
            emit_sample(sample_idx)
            sample_idx += 1
        book.apply_row(row)
        if sample_idx >= n:
            break

    while sample_idx < n:
        emit_sample(sample_idx)
        sample_idx += 1

    return pd.DataFrame(rows).set_index("ts")


def simulate_poisson(
    params: ASParams,
    s0: float = 100.0,
    n_steps: int = 2_000,
    seed: int = 7,
    strategy: str = "inventory",
) -> pd.DataFrame:
    """Pure AS simulator: Brownian midprice + Poisson fills."""
    rng = np.random.default_rng(seed)
    cash = 0.0
    q = 0
    mid = s0
    rows: list[dict[str, Any]] = []

    for i in range(n_steps):
        t_elapsed = i * params.dt_seconds
        quotes = as_quotes(mid, 0 if strategy == "symmetric" else q, t_elapsed, params)
        delta_bid = max(0.0, quotes["delta_bid"])
        delta_ask = max(0.0, quotes["delta_ask"])
        lam_bid = params.A * math.exp(-params.k * delta_bid)
        lam_ask = params.A * math.exp(-params.k * delta_ask)
        p_bid = 1.0 - math.exp(-lam_bid * params.dt_seconds)
        p_ask = 1.0 - math.exp(-lam_ask * params.dt_seconds)

        quote_bid = q + params.order_size <= params.q_max
        quote_ask = q - params.order_size >= -params.q_max
        bid_fill = rng.random() < p_bid and quote_bid
        ask_fill = rng.random() < p_ask and quote_ask

        if bid_fill:
            cash -= quotes["bid"] * params.order_size
            q += params.order_size
        if ask_fill:
            cash += quotes["ask"] * params.order_size
            q -= params.order_size

        pnl = cash + q * mid
        rows.append(
            {
                "step": i,
                "mid": mid,
                "reservation": quotes["reservation"],
                "bid": quotes["bid"],
                "ask": quotes["ask"],
                "spread": quotes["ask"] - quotes["bid"],
                "quote_bid": int(quote_bid),
                "quote_ask": int(quote_ask),
                "delta_bid": quotes["delta_bid"],
                "delta_ask": quotes["delta_ask"],
                "tau": quotes["tau"],
                "inventory": q,
                "cash": cash,
                "pnl": pnl,
                "bid_fill": int(bid_fill),
                "ask_fill": int(ask_fill),
                "strategy": strategy,
            }
        )

        mid += params.sigma * math.sqrt(params.dt_seconds) * rng.normal()

    return pd.DataFrame(rows)


def run_synthetic_demo(args: argparse.Namespace) -> None:
    params = ASParams(
        gamma=args.gamma,
        sigma=args.sigma,
        k=args.k,
        A=args.A,
        horizon_seconds=args.horizon_seconds,
        dt_seconds=args.dt_ms / 1000.0,
        tick_size=args.tick_size,
        order_size=args.order_size,
        q_max=args.q_max,
    )
    inv = simulate_poisson(params, n_steps=args.steps, strategy="inventory")
    sym = simulate_poisson(params, n_steps=args.steps, strategy="symmetric")
    print("Synthetic Poisson simulation")
    print("Inventory AS:", summarize_simulation(inv))
    print("Symmetric   :", summarize_simulation(sym))
    out = pd.concat([inv, sym], ignore_index=False)
    out_path = Path(args.output)
    out.to_csv(out_path)
    print(f"Saved path to {out_path}")


def run_replay(args: argparse.Namespace) -> None:
    dt_seconds = args.dt_ms / 1000.0
    print(f"Loading {args.input} ...")
    mbo = load_mbo(args.input)

    # Reconstruct BBO using the whole input; filter RTH after reconstruction.
    print("Building sampled BBO ...")
    bbo_all = build_sampled_bbo(mbo, dt=pd.Timedelta(milliseconds=args.dt_ms))
    bbo = filter_regular_hours(bbo_all)
    mbo_rth = filter_regular_hours(mbo)
    if bbo.empty:
        raise RuntimeError("No BBO samples after filtering. Did the input include an initial snapshot?")

    print("Building interval trade extrema ...")
    extrema = interval_trade_extrema(mbo_rth, bbo.index, dt_seconds)
    trade_flow: Optional[IntervalTradeFlow] = None
    if args.fill_model == "queue":
        print("Building interval trade flow for queue-aware fills ...")
        trade_flow = build_interval_trade_flow(mbo_rth, bbo.index, dt_seconds)

    print(f"Calibrating sigma and lambda(delta) with {args.fill_model} fills ...")
    daily_sigma = estimate_sigma_from_mid(bbo)
    sigma_policy = resolve_replay_policy_sigma(daily_sigma, args.policy_sigma)
    deltas = np.arange(args.delta_start_ticks, args.delta_end_ticks + 1) * args.tick_size
    if args.fill_model == "queue":
        if trade_flow is None:
            raise RuntimeError("Queue-aware calibration requires interval trade flow.")
        cal = calibrate_queue_arrival_rates(
            mbo=mbo,
            bbo=bbo,
            trade_flow=trade_flow,
            deltas=deltas,
            dt_seconds=dt_seconds,
            tick_size=args.tick_size,
            order_size=args.order_size,
        )
    else:
        cal = calibrate_arrival_rates(bbo, extrema, deltas=deltas, dt_seconds=dt_seconds)
    k = args.k if args.k is not None else cal.k
    A = args.A if args.A is not None else cal.A
    gamma = args.gamma
    if gamma is None:
        gamma = choose_gamma_for_target_skew(
            sigma=sigma_policy,
            horizon_seconds=args.horizon_seconds,
            q_target=max(1, args.q_max),
            target_total_skew=args.target_skew,
        )

    params = ASParams(
        gamma=gamma,
        sigma=sigma_policy,
        k=k,
        A=A,
        horizon_seconds=args.horizon_seconds,
        dt_seconds=dt_seconds,
        tick_size=args.tick_size,
        order_size=args.order_size,
        q_max=args.q_max,
        rolling_horizon=not args.finite_episode_horizon,
    )

    print("Calibration:")
    print(f"  daily sigma        = {daily_sigma:.8f} dollars/sqrt(second)")
    print(f"  policy sigma used  = {sigma_policy:.8f} dollars/sqrt(second)")
    print(f"  A_ask, k_ask       = {cal.A_ask:.6f}, {cal.k_ask:.6f}")
    print(f"  A_bid, k_bid       = {cal.A_bid:.6f}, {cal.k_bid:.6f}")
    print(f"  A, k used          = {A:.6f}, {k:.6f}")
    print(f"  gamma used         = {gamma:.8g}")
    print(f"  fill model         = {args.fill_model}")

    print("Simulating replay ...")
    inv = simulate_replay(
        bbo,
        extrema,
        params,
        strategy="inventory",
        fill_model=args.fill_model,
        mbo=mbo,
        trade_flow=trade_flow,
    )
    sym = simulate_replay(
        bbo,
        extrema,
        params,
        strategy="symmetric",
        fill_model=args.fill_model,
        mbo=mbo,
        trade_flow=trade_flow,
    )
    print("Inventory AS:", summarize_simulation(inv))
    print("Symmetric   :", summarize_simulation(sym))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv.to_parquet(out_dir / "as_inventory_replay.parquet")
    sym.to_parquet(out_dir / "as_symmetric_replay.parquet")
    bbo.to_parquet(out_dir / "sampled_bbo.parquet")
    extrema.to_parquet(out_dir / "trade_extrema.parquet")
    cal.to_frame().to_csv(out_dir / "arrival_calibration.csv", index=False)
    pd.DataFrame(
        [
            {
                "daily_sigma": daily_sigma,
                "policy_sigma": sigma_policy,
                "A_calibrated": cal.A,
                "k_calibrated": cal.k,
                "A_used": A,
                "k_used": k,
                "gamma_used": gamma,
                "fill_model": args.fill_model,
            }
        ]
    ).to_csv(out_dir / "replay_parameters.csv", index=False)
    print(f"Saved outputs to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avellaneda--Stoikov AMZN MBO model scaffold")
    parser.add_argument("--input", type=str, default=None, help="DBN/CSV/Parquet MBO file. Omit for synthetic demo.")
    parser.add_argument("--output", type=str, default="as_outputs", help="Output CSV path for demo or directory for replay.")
    parser.add_argument("--dt-ms", type=int, default=250, help="Decision interval in milliseconds.")
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument("--order-size", type=int, default=10)
    parser.add_argument("--q-max", type=int, default=200)
    parser.add_argument("--horizon-seconds", type=float, default=600.0)
    parser.add_argument("--finite-episode-horizon", action="store_true", help="Use tau = T-t instead of rolling constant horizon.")
    parser.add_argument("--gamma", type=float, default=None, help="Risk aversion. If omitted in replay, calibrated by target skew.")
    parser.add_argument("--target-skew", type=float, default=0.05, help="Dollars of reservation-price shift at q_max when gamma is omitted.")
    parser.add_argument("--k", type=float, default=None, help="Arrival decay. If omitted in replay, fitted from data.")
    parser.add_argument("--A", type=float, default=None, help="Arrival scale. If omitted in replay, fitted from data.")
    parser.add_argument(
        "--policy-sigma",
        type=float,
        default=None,
        help="Fixed replay policy sigma. If omitted, use the same day's empirical sigma.",
    )
    parser.add_argument("--sigma", type=float, default=0.05, help="Synthetic demo sigma dollars/sqrt(second).")
    parser.add_argument("--steps", type=int, default=2_000, help="Synthetic demo steps.")
    parser.add_argument("--delta-start-ticks", type=int, default=1)
    parser.add_argument("--delta-end-ticks", type=int, default=20)
    parser.add_argument(
        "--fill-model",
        choices=["trade-through", "queue"],
        default="trade-through",
        help="Historical replay fill approximation for DBN inputs.",
    )
    args = parser.parse_args()

    # Synthetic defaults if user did not pass gamma/k/A.
    if args.input is None:
        if args.gamma is None:
            args.gamma = 0.002
        if args.k is None:
            args.k = 50.0  # per dollar; roughly 0.5 per cent-tick
        if args.A is None:
            args.A = 10.0
        if args.output == "as_outputs":
            args.output = "as_synthetic_paths.csv"
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.input is None:
        run_synthetic_demo(cli_args)
    else:
        run_replay(cli_args)
