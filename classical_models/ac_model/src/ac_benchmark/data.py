from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import Direction, EpisodeSpec


_PRICE_PATTERNS = ["price", "px", "p"]
_SIZE_PATTERNS = ["size", "sz", "qty", "q"]


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV or Parquet by file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".csv", ".txt", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path, **kwargs)
    raise ValueError(f"Unsupported file extension for {path}; use CSV or Parquet.")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Write CSV or Parquet by file extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix == ".csv" or path.name.endswith(".csv.gz"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension for {path}; use CSV or Parquet.")


def _find_side_level_columns(df: pd.DataFrame) -> Dict[Tuple[str, str, int], str]:
    """Map flexible input column names to (side, field, one-indexed level).

    Supports both one-indexed names such as bid_price_1 and zero-indexed names
    such as bid_px_00. If any recognised level is zero, all recognised levels
    are shifted by +1.
    """
    raw_matches: List[Tuple[str, str, int, str]] = []
    for original in df.columns:
        name = _normalise_name(original)
        candidates = [
            re.match(r"^(bid|ask)_(price|px|p|size|sz|qty|q)_?0*(\d+)$", name),
            re.match(r"^(bid|ask)_?0*(\d+)_(price|px|p|size|sz|qty|q)$", name),
            re.match(r"^(bid|ask)(price|px|p|size|sz|qty|q)_?0*(\d+)$", name),
        ]
        for m in candidates:
            if not m:
                continue
            g = m.groups()
            side = g[0]
            if g[1].isdigit():
                raw_level = int(g[1])
                token = g[2]
            else:
                token = g[1]
                raw_level = int(g[2])
            field = "price" if token in _PRICE_PATTERNS else "size"
            raw_matches.append((side, field, raw_level, original))
            break

    if not raw_matches:
        return {}
    shift = 1 if min(level for _, _, level, _ in raw_matches) == 0 else 0
    mapping: Dict[Tuple[str, str, int], str] = {}
    for side, field, raw_level, original in raw_matches:
        mapping[(side, field, raw_level + shift)] = original
    return mapping

def standardize_snapshot_columns(
    df: pd.DataFrame,
    *,
    price_scale: float = 1.0,
    depth_levels: Optional[int] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a canonical LOB snapshot DataFrame.

    Canonical level columns are:
        bid_price_1, bid_size_1, ask_price_1, ask_size_1, ...

    Also creates best_bid, best_ask, mid_price, and spread when possible.
    """
    if df.empty:
        raise ValueError("input DataFrame is empty")

    out = pd.DataFrame(index=df.index)
    norm_to_original = {_normalise_name(c): c for c in df.columns}

    if timestamp_col is not None:
        out["timestamp"] = df[timestamp_col].values
    else:
        for candidate in ["timestamp", "ts_event", "ts_recv", "time", "seconds", "datetime"]:
            if candidate in norm_to_original:
                out["timestamp"] = df[norm_to_original[candidate]].values
                break
        if "timestamp" not in out:
            out["timestamp"] = np.arange(len(df), dtype=np.int64)

    mapping = _find_side_level_columns(df)
    available_levels = sorted({lvl for (_, _, lvl) in mapping})
    if depth_levels is not None:
        available_levels = [lvl for lvl in available_levels if lvl <= depth_levels]

    for lvl in available_levels:
        for side in ["bid", "ask"]:
            for field in ["price", "size"]:
                key = (side, field, lvl)
                if key in mapping:
                    values = pd.to_numeric(df[mapping[key]], errors="coerce")
                    if field == "price" and price_scale not in (0, 1.0, 1):
                        values = values / float(price_scale)
                    out[f"{side}_{field}_{lvl}"] = values

    # Best bid/ask aliases if level columns were absent or named explicitly.
    best_aliases = {
        "best_bid": ["best_bid", "bid", "bb", "bid_price", "bid_px"],
        "best_ask": ["best_ask", "ask", "ba", "ask_price", "ask_px"],
    }
    for canonical, aliases in best_aliases.items():
        if canonical not in out:
            for alias in aliases:
                if alias in norm_to_original:
                    values = pd.to_numeric(df[norm_to_original[alias]], errors="coerce")
                    if price_scale not in (0, 1.0, 1):
                        values = values / float(price_scale)
                    out[canonical] = values
                    break

    if "bid_price_1" in out and "best_bid" not in out:
        out["best_bid"] = out["bid_price_1"]
    if "ask_price_1" in out and "best_ask" not in out:
        out["best_ask"] = out["ask_price_1"]

    if "best_bid" not in out or "best_ask" not in out:
        raise ValueError(
            "Could not infer best_bid/best_ask. Provide canonical level columns or best price columns."
        )

    out["mid_price"] = (pd.to_numeric(out["best_bid"], errors="coerce") + pd.to_numeric(out["best_ask"], errors="coerce")) / 2.0
    out["spread"] = pd.to_numeric(out["best_ask"], errors="coerce") - pd.to_numeric(out["best_bid"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["best_bid", "best_ask", "mid_price"])
    out = out[out["spread"] >= 0]
    return out.reset_index(drop=True)


def read_snapshots(
    path: str | Path,
    *,
    price_scale: float = 1.0,
    depth_levels: Optional[int] = None,
    timestamp_col: Optional[str] = None,
) -> pd.DataFrame:
    df = read_table(path)
    # Already canonical enough; still standardise to validate and add computed columns.
    return standardize_snapshot_columns(
        df, price_scale=price_scale, depth_levels=depth_levels, timestamp_col=timestamp_col
    )


def infer_depth(snapshot_df: pd.DataFrame) -> int:
    """Infer maximum complete bid/ask depth level in a canonical snapshot frame."""
    depths = []
    lvl = 1
    while True:
        required = [f"bid_price_{lvl}", f"bid_size_{lvl}", f"ask_price_{lvl}", f"ask_size_{lvl}"]
        if all(c in snapshot_df.columns for c in required):
            depths.append(lvl)
            lvl += 1
        else:
            break
    return max(depths) if depths else 0


def load_lobster_orderbook(
    orderbook_path: str | Path,
    *,
    message_path: Optional[str | Path] = None,
    levels: Optional[int] = None,
    price_scale: float = 10_000.0,
    header: bool = False,
) -> pd.DataFrame:
    """Load a LOBSTER orderbook CSV into canonical snapshots.

    LOBSTER orderbook files are commonly distributed without a header and with
    repeated columns: ask price, ask size, bid price, bid size for levels 1..K.
    The optional message file is used only for timestamps here.
    """
    orderbook_path = Path(orderbook_path)
    if header:
        raw = pd.read_csv(orderbook_path)
    else:
        raw = pd.read_csv(orderbook_path, header=None)
        n_cols = raw.shape[1]
        if n_cols % 4 != 0:
            raise ValueError(f"LOBSTER orderbook column count must be multiple of 4, got {n_cols}")
        inferred_levels = n_cols // 4
        if levels is None:
            levels = inferred_levels
        if levels > inferred_levels:
            raise ValueError(f"Requested levels={levels}, but file has only {inferred_levels}")
        names = []
        for lvl in range(1, inferred_levels + 1):
            names += [f"ask_price_{lvl}", f"ask_size_{lvl}", f"bid_price_{lvl}", f"bid_size_{lvl}"]
        raw.columns = names
        raw = raw.iloc[:, : 4 * levels]

    if message_path is not None:
        msg = pd.read_csv(message_path, header=None)
        if len(msg) != len(raw):
            raise ValueError(f"message rows ({len(msg)}) and orderbook rows ({len(raw)}) differ")
        raw.insert(0, "timestamp", msg.iloc[:, 0].values)
    else:
        raw.insert(0, "timestamp", np.arange(len(raw), dtype=np.int64))

    return standardize_snapshot_columns(raw, price_scale=price_scale, depth_levels=levels)


def generate_episode_starts(n_rows: int, spec: EpisodeSpec) -> np.ndarray:
    """Return start row indices for complete episodes."""
    if n_rows <= 0:
        return np.array([], dtype=int)
    last_needed_offset = (spec.episode_length - 1) * spec.step_stride_rows
    max_start = n_rows - 1 - last_needed_offset
    if max_start < 0:
        return np.array([], dtype=int)
    return np.arange(0, max_start + 1, spec.start_stride_rows, dtype=int)


def episode_indices(start_row: int, spec: EpisodeSpec) -> np.ndarray:
    return int(start_row) + np.arange(spec.episode_length, dtype=int) * spec.step_stride_rows


def assign_directions(num_episodes: int, mode: str, seed: int = 7) -> List[Direction]:
    if mode == "buy":
        return ["buy"] * num_episodes
    if mode == "sell":
        return ["sell"] * num_episodes
    if mode == "alternating":
        return ["buy" if i % 2 == 0 else "sell" for i in range(num_episodes)]
    if mode == "random":
        rng = np.random.default_rng(seed)
        return ["buy" if x else "sell" for x in rng.integers(0, 2, size=num_episodes)]
    raise ValueError(f"Unknown direction mode: {mode}")


def build_episode_plan(n_rows: int, spec: EpisodeSpec) -> pd.DataFrame:
    starts = generate_episode_starts(n_rows, spec)
    directions = assign_directions(len(starts), spec.directions, spec.random_seed)
    return pd.DataFrame(
        {
            "episode_id": np.arange(len(starts), dtype=int),
            "start_row": starts,
            "direction": directions,
            "episode_length": spec.episode_length,
            "messages_per_step": spec.messages_per_step,
        }
    )


def level_columns(side: Direction | str, depth: int) -> Tuple[List[str], List[str]]:
    """Return price and size column names for the book side consumed by side.

    side='buy' consumes ask levels; side='sell' consumes bid levels.
    """
    book_side = "ask" if side == "buy" else "bid"
    prices = [f"{book_side}_price_{lvl}" for lvl in range(1, depth + 1)]
    sizes = [f"{book_side}_size_{lvl}" for lvl in range(1, depth + 1)]
    return prices, sizes



def _first_existing(norm_to_original: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in norm_to_original:
            return norm_to_original[c]
    return None


def _map_mbo_side(value) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"b", "bid", "buy", "1"}:
        return "bid"
    if s in {"a", "ask", "offer", "sell", "-1"}:
        return "ask"
    return None


def _map_mbo_action(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    aliases = {
        "a": "add",
        "add": "add",
        "c": "cancel",
        "cancel": "cancel",
        "m": "modify",
        "modify": "modify",
        "d": "delete",
        "delete": "delete",
        "r": "clear",
        "clear": "clear",
        "f": "fill",
        "fill": "fill",
        "t": "trade",
        "trade": "trade",
    }
    return aliases.get(s, s)


def _book_snapshot_from_orders(orders: Dict[object, Tuple[str, float, int]], depth_levels: int) -> Optional[Dict[str, float]]:
    bid_levels: Dict[float, int] = {}
    ask_levels: Dict[float, int] = {}
    for side, price, size in orders.values():
        if size <= 0 or not np.isfinite(price):
            continue
        levels = bid_levels if side == "bid" else ask_levels
        levels[float(price)] = levels.get(float(price), 0) + int(size)
    if not bid_levels or not ask_levels:
        return None
    bid_prices = sorted(bid_levels.keys(), reverse=True)[:depth_levels]
    ask_prices = sorted(ask_levels.keys())[:depth_levels]
    if not bid_prices or not ask_prices:
        return None
    rec: Dict[str, float] = {
        "best_bid": float(bid_prices[0]),
        "best_ask": float(ask_prices[0]),
    }
    for i in range(depth_levels):
        lvl = i + 1
        if i < len(bid_prices):
            p = bid_prices[i]
            rec[f"bid_price_{lvl}"] = float(p)
            rec[f"bid_size_{lvl}"] = int(bid_levels[p])
        else:
            rec[f"bid_price_{lvl}"] = np.nan
            rec[f"bid_size_{lvl}"] = 0
        if i < len(ask_prices):
            p = ask_prices[i]
            rec[f"ask_price_{lvl}"] = float(p)
            rec[f"ask_size_{lvl}"] = int(ask_levels[p])
        else:
            rec[f"ask_price_{lvl}"] = np.nan
            rec[f"ask_size_{lvl}"] = 0
    rec["mid_price"] = (rec["best_bid"] + rec["best_ask"]) / 2.0
    rec["spread"] = rec["best_ask"] - rec["best_bid"]
    return rec


def reconstruct_mbo_snapshots(
    events: pd.DataFrame,
    *,
    depth_levels: int = 10,
    price_scale: float = 1.0,
    sample_every_events: int = 100,
    max_events: Optional[int] = None,
) -> pd.DataFrame:
    """Reconstruct top-K snapshots from a Databento-like MBO event table.

    This is a simple CPU fallback for quick thesis benchmarking. It expects
    order-level columns similar to: timestamp/ts_event, action, side, price,
    size, order_id. For large months of raw MBO data, a vendor/exported MBP
    snapshot file will usually be faster.
    """
    if events.empty:
        raise ValueError("events DataFrame is empty")
    norm_to_original = {_normalise_name(c): c for c in events.columns}
    ts_col = _first_existing(norm_to_original, ["timestamp", "ts_event", "ts_recv", "time", "datetime"])
    action_col = _first_existing(norm_to_original, ["action", "event_type", "type"])
    side_col = _first_existing(norm_to_original, ["side", "direction"])
    price_col = _first_existing(norm_to_original, ["price", "px"])
    size_col = _first_existing(norm_to_original, ["size", "qty", "quantity"])
    order_col = _first_existing(norm_to_original, ["order_id", "orderid", "order_ref", "order_reference_number"])
    missing = [name for name, col in [("timestamp", ts_col), ("action", action_col), ("side", side_col), ("price", price_col), ("size", size_col), ("order_id", order_col)] if col is None]
    if missing:
        raise ValueError(f"Missing MBO columns: {missing}. Available columns: {list(events.columns)}")

    df = events[[ts_col, action_col, side_col, price_col, size_col, order_col]].copy()
    df.columns = ["timestamp", "action", "side", "price", "size", "order_id"]
    if max_events is not None:
        df = df.iloc[: int(max_events)].copy()
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if price_scale not in (0, 1.0, 1):
        df["price"] = df["price"] / float(price_scale)
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0).astype(int)

    orders: Dict[object, Tuple[str, float, int]] = {}
    records: List[Dict[str, float]] = []
    sample_every_events = max(1, int(sample_every_events))

    for i, row in df.iterrows():
        action = _map_mbo_action(row["action"])
        order_id = row["order_id"]
        side = _map_mbo_side(row["side"])
        price = float(row["price"]) if np.isfinite(row["price"]) else np.nan
        size = int(row["size"])

        if action == "clear":
            orders.clear()
        elif action == "add":
            if side in {"bid", "ask"} and size > 0 and np.isfinite(price):
                orders[order_id] = (side, price, size)
        elif action == "modify":
            if order_id in orders:
                old_side, old_price, old_size = orders[order_id]
                orders[order_id] = (
                    side if side in {"bid", "ask"} else old_side,
                    price if np.isfinite(price) else old_price,
                    size if size > 0 else old_size,
                )
            elif side in {"bid", "ask"} and size > 0 and np.isfinite(price):
                orders[order_id] = (side, price, size)
        elif action in {"cancel", "fill", "trade"}:
            if order_id in orders:
                old_side, old_price, old_size = orders[order_id]
                new_size = old_size - size if size > 0 else 0
                if new_size > 0:
                    orders[order_id] = (old_side, old_price, new_size)
                else:
                    orders.pop(order_id, None)
        elif action == "delete":
            orders.pop(order_id, None)

        if i % sample_every_events == 0:
            rec = _book_snapshot_from_orders(orders, depth_levels)
            if rec is not None and rec["spread"] >= 0:
                rec["timestamp"] = row["timestamp"]
                records.append(rec)

    if not records:
        raise ValueError("No valid snapshots reconstructed; check columns, price scale, and event actions.")
    out = pd.DataFrame.from_records(records)
    ordered_cols = ["timestamp", "best_bid", "best_ask", "mid_price", "spread"]
    level_cols = []
    for lvl in range(1, depth_levels + 1):
        level_cols += [f"bid_price_{lvl}", f"bid_size_{lvl}", f"ask_price_{lvl}", f"ask_size_{lvl}"]
    return out[ordered_cols + level_cols].reset_index(drop=True)
