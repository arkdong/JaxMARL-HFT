from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


HorizonMode = Literal["rolling", "finite_episode"]


@dataclass(frozen=True)
class ASParams:
    sigma: float
    k: float
    gamma: float
    horizon_seconds: float
    tick_size: float
    min_spread_ticks: int = 1


@dataclass(frozen=True)
class ASState:
    mid: float
    inventory: int
    elapsed_seconds: float


@dataclass(frozen=True)
class ASQuote:
    reservation_price: float
    raw_bid: float
    raw_ask: float
    bid: float | None
    ask: float | None
    spread: float
    skew: float


def _validate_params(params: ASParams) -> None:
    if not math.isfinite(params.sigma) or params.sigma <= 0:
        raise ValueError(f"sigma must be finite and positive, got {params.sigma!r}")
    if not math.isfinite(params.k) or params.k <= 0:
        raise ValueError(f"k must be finite and positive, got {params.k!r}")
    if not math.isfinite(params.gamma) or params.gamma < 0:
        raise ValueError(f"gamma must be finite and non-negative, got {params.gamma!r}")
    if not math.isfinite(params.horizon_seconds) or params.horizon_seconds < 0:
        raise ValueError(f"horizon_seconds must be finite and non-negative, got {params.horizon_seconds!r}")
    if not math.isfinite(params.tick_size) or params.tick_size <= 0:
        raise ValueError(f"tick_size must be finite and positive, got {params.tick_size!r}")
    if params.min_spread_ticks < 1:
        raise ValueError(f"min_spread_ticks must be at least 1, got {params.min_spread_ticks!r}")


def remaining_horizon_seconds(
    horizon_seconds: float,
    elapsed_seconds: float,
    horizon_mode: HorizonMode,
) -> float:
    if horizon_mode == "rolling":
        return float(horizon_seconds)
    if horizon_mode == "finite_episode":
        return float(max(0.0, horizon_seconds - elapsed_seconds))
    raise ValueError(f"Unsupported AS horizon mode: {horizon_mode!r}")


def _round_bid(price: float, tick_size: float) -> float:
    return math.floor(price / tick_size) * tick_size


def _round_ask(price: float, tick_size: float) -> float:
    return math.ceil(price / tick_size) * tick_size


def compute_as_quote(
    params: ASParams,
    state: ASState,
    *,
    horizon_mode: HorizonMode = "rolling",
    inventory_limit: int | None = None,
    order_size: int = 1,
) -> ASQuote:
    """Compute one Avellaneda-Stoikov quote without data-loading side effects."""
    _validate_params(params)
    if not math.isfinite(state.mid) or state.mid <= 0:
        raise ValueError(f"mid must be finite and positive, got {state.mid!r}")
    if order_size <= 0:
        raise ValueError(f"order_size must be positive, got {order_size!r}")

    tau = remaining_horizon_seconds(params.horizon_seconds, state.elapsed_seconds, horizon_mode)
    sig2tau = params.sigma * params.sigma * tau
    reservation = state.mid - state.inventory * params.gamma * sig2tau

    if params.gamma <= 1e-12:
        total_spread = 2.0 / params.k
    else:
        total_spread = params.gamma * sig2tau + (2.0 / params.gamma) * math.log1p(params.gamma / params.k)

    min_spread = params.min_spread_ticks * params.tick_size
    total_spread = max(total_spread, min_spread)
    half_spread = total_spread / 2.0

    raw_bid = reservation - half_spread
    raw_ask = reservation + half_spread
    rounded_bid = _round_bid(raw_bid, params.tick_size)
    rounded_ask = _round_ask(raw_ask, params.tick_size)
    if rounded_ask - rounded_bid < min_spread:
        rounded_ask = rounded_bid + min_spread
    if rounded_bid >= rounded_ask:
        rounded_ask = rounded_bid + min_spread

    bid: float | None = rounded_bid
    ask: float | None = rounded_ask
    if inventory_limit is not None:
        if inventory_limit < 0:
            raise ValueError(f"inventory_limit must be non-negative, got {inventory_limit!r}")
        if state.inventory + order_size > inventory_limit:
            bid = None
        if state.inventory - order_size < -inventory_limit:
            ask = None

    return ASQuote(
        reservation_price=float(reservation),
        raw_bid=float(raw_bid),
        raw_ask=float(raw_ask),
        bid=None if bid is None else float(bid),
        ask=None if ask is None else float(ask),
        spread=float(rounded_ask - rounded_bid),
        skew=float(reservation - state.mid),
    )
