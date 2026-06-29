# gymnax_exchange/jaxrl/MARL/risk_neutral_export.py

import json
import os
from datetime import datetime
from typing import Any, Dict

import jax
import numpy as np


def _to_float_stats(x: Any) -> Dict[str, float]:
    arr = np.asarray(jax.device_get(x)).astype(float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _episode_sums(x: Any) -> np.ndarray:
    arr = np.asarray(jax.device_get(x)).astype(float).squeeze()
    if arr.ndim == 1:
        return arr
    # Expected shape after rollout is roughly [NUM_STEPS, NUM_ENVS].
    return np.nansum(arr, axis=0)


def export_risk_neutral_train_summary(
    *,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    checkpoint_dir: str,
    checkpoint_step: int,
    run_name: str | None = None,
) -> None:
    mm_cfg = config["dict_of_agents_configs"]["MarketMaking"]
    ex_cfg = config["dict_of_agents_configs"].get("Execution", {})
    world_cfg = config["world_config"]

    mm_info = metrics["traj_batch"][0].info["agent"]

    reward_base = mm_info.get("reward_base_mm")
    q2 = mm_info.get("squared_inventory")
    risk_penalty = mm_info.get("risk_penalty_mm")
    rho_mm = mm_info.get("rho_mm")

    train_metric_summary = {}

    if reward_base is not None:
        train_metric_summary["reward_base_mm"] = _to_float_stats(reward_base)
        train_metric_summary["sum_abs_reward_base_mm_per_env"] = {
            "mean": float(np.mean(_episode_sums(np.abs(jax.device_get(reward_base))))),
        }

    if q2 is not None:
        train_metric_summary["squared_inventory"] = _to_float_stats(q2)
        train_metric_summary["sum_squared_inventory_per_env"] = {
            "mean": float(np.mean(_episode_sums(q2))),
        }

    if risk_penalty is not None:
        train_metric_summary["risk_penalty_mm"] = _to_float_stats(risk_penalty)

    if rho_mm is not None:
        train_metric_summary["rho_mm_logged"] = _to_float_stats(rho_mm)

    manifest = {
        "schema_version": 1,
        "purpose": "risk_neutral_training_only",
        "created_at": datetime.utcnow().isoformat() + "Z",

        "do_not_use_this_file_for_rho_star": True,
        "rho_mm_star": None,
        "rho_mm_star_reason": (
            "rho_mm_star must be estimated later from frozen-policy "
            "validation trajectories, not from training rollouts."
        ),

        "run": {
            "wandb_project": config.get("PROJECT"),
            "wandb_run_name": run_name,
            "seed": config.get("SEED"),
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_step": int(checkpoint_step),
        },

        "training": {
            "num_envs": config.get("NUM_ENVS"),
            "num_steps": config.get("NUM_STEPS"),
            "total_timesteps": config.get("TOTAL_TIMESTEPS"),
            "num_updates": config.get("NUM_UPDATES"),
            "n_devices": config.get("N_DEVICES"),
            "calc_eval": config.get("CALC_EVAL"),
            "time_period": config.get("TimePeriod"),
        },

        "world_config": {
            "stock": world_cfg.get("stock"),
            "timePeriod": world_cfg.get("timePeriod"),
            "n_data_msg_per_step": world_cfg.get("n_data_msg_per_step"),
            "episode_time": world_cfg.get("episode_time"),
            "start_resolution": world_cfg.get("start_resolution"),
            "book_depth": world_cfg.get("book_depth"),
            "dataPath": world_cfg.get("dataPath"),
        },

        "market_making_config": {
            "action_space": mm_cfg.get("action_space"),
            "observation_space": mm_cfg.get("observation_space"),
            "reward_function": mm_cfg.get("reward_function"),
            "inv_penalty": mm_cfg.get("inv_penalty"),
            "rho_mm": mm_cfg.get("rho_mm"),
            "fixed_quant_value": mm_cfg.get("fixed_quant_value"),
            "reward_scaling_quo": mm_cfg.get("reward_scaling_quo"),
            "reference_price": mm_cfg.get("reference_price"),
            "unwind_price": mm_cfg.get("unwind_price"),
            "inventoryPnL_eta": mm_cfg.get("inventoryPnL_eta"),
            "inventoryPnL_gamma": mm_cfg.get("inventoryPnL_gamma"),
            "rebate_bps": mm_cfg.get("rebate_bps"),
        },

        "execution_config": {
            "action_space": ex_cfg.get("action_space"),
            "task": ex_cfg.get("task"),
            "task_size": ex_cfg.get("task_size"),
            "fixed_quant_value": ex_cfg.get("fixed_quant_value"),
            "reward_lambda": ex_cfg.get("reward_lambda"),
            "doom_price_penalty": ex_cfg.get("doom_price_penalty"),
            "reference_price": ex_cfg.get("reference_price"),
        },

        "training_sanity_metrics_last_update": train_metric_summary,

        "future_validation_formula": {
            "rho_mm_star": (
                "mean_env(sum_t(abs(reward_base_mm_t))) / "
                "(mean_env(sum_t(squared_inventory_t)) + eps)"
            ),
            "required_validation_fields": [
                "reward_base_mm",
                "squared_inventory",
                "risk_penalty_mm",
                "rho_mm",
                "inventory",
                "reward_portfolio_value",
            ],
        },
    }

    out_dir = os.path.join("results", "risk_neutral_training")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"risk_neutral_train_summary_step_{int(checkpoint_step)}.json",
    )

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[risk-neutral export] wrote {out_path}")