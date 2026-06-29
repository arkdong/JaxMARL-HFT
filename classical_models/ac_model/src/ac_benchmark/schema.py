from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

Direction = Literal["buy", "sell"]


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode construction parameters.

    The defaults match the simple JaxMARL-HFT execution setup used as the
    comparison target: a 600-share task over 64 environment steps. If your
    snapshot file is already sampled once per environment step, set
    messages_per_step=1.
    """

    task_size: int = 600
    episode_length: int = 64
    messages_per_step: int = 100
    episode_start_frequency_steps: int = 64
    lot_size: int = 10
    tick_size: float = 0.01
    directions: Literal["alternating", "random", "buy", "sell"] = "random"
    random_seed: int = 7

    @property
    def step_stride_rows(self) -> int:
        return int(self.messages_per_step)

    @property
    def start_stride_rows(self) -> int:
        return int(self.messages_per_step * self.episode_start_frequency_steps)


@dataclass(frozen=True)
class ACParams:
    """Calibrated quantities used for reporting and optional lambda conversion.

    The replay benchmark primarily uses the dimensionless kappa_T grid, because
    it is stable and immediately comparable across episode lengths. The
    calibrated volatility, spread, and eta are still useful diagnostics and can
    be reported in the thesis.
    """

    sigma_step: float
    half_spread: float
    eta_ac: float
    gamma_ac: float = 0.0
    q_grid: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 400, 600])
    n_obs: int = 0
    notes: str = "gamma_ac set to zero for the thesis benchmark."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ACParams":
        return ACParams(
            sigma_step=float(d.get("sigma_step", 0.0)),
            half_spread=float(d.get("half_spread", 0.0)),
            eta_ac=float(d.get("eta_ac", 0.0)),
            gamma_ac=float(d.get("gamma_ac", 0.0)),
            q_grid=list(map(int, d.get("q_grid", [10, 20, 50, 100, 200, 400, 600]))),
            n_obs=int(d.get("n_obs", 0)),
            notes=str(d.get("notes", "")),
        )


@dataclass(frozen=True)
class ACConfig:
    """Top-level experiment config."""

    train_path: Optional[Path] = None
    valid_path: Optional[Path] = None
    test_path: Optional[Path] = None
    out_dir: Path = Path("outputs/ac_benchmark")
    episode: EpisodeSpec = field(default_factory=EpisodeSpec)
    kappa_T_grid: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0)
    max_rows_calibration: Optional[int] = 250_000
    depth_levels: Optional[int] = None

