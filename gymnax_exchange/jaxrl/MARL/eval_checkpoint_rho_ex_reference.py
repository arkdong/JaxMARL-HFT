"""
Frozen validation evaluation for an old 2-player JaxMARL-HFT checkpoint.

Purpose
-------
Load a checkpoint trained before the V3 order-execution reward patch, evaluate
it under the current V3 code with rho_ex=0 on the validation data, and estimate
rho_ex_star from validation trajectories:

    rho_ex_star = mean_env(sum_t |reward_base_ex_t|) /
                  (mean_env(sum_t squared_remaining_qty_t) + eps)

This script performs NO PPO updates. It is intentionally separate from
ippo_rnn_JAXMARL.py so old checkpoints can be evaluated without retraining.
"""

from __future__ import annotations

import csv
import functools
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as oxcp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from gymnax_exchange.jaxen.marl_env import MARLEnv
from gymnax_exchange.jaxob.config_io import load_config_from_file
from gymnax_exchange.jaxob.jaxob_config import (
    CONFIG_OBJECT_DICT,
    MultiAgentConfig,
    World_EnvironmentConfig,
)
from gymnax_exchange.jaxrl.MARL.ippo_rnn_JAXMARL import (
    ActorCriticRNN,
    ScannedRNN,
    batchify,
    unbatchify,
)


os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")


def _to_container(x: Any) -> Dict[str, Any]:
    return OmegaConf.to_container(x, resolve=True)  # type: ignore[return-value]


def _create_agent_configs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Same precedence as ippo_rnn_JAXMARL.create_agent_configs.

    Lowest: dataclass defaults
    Middle: env JSON dict_of_agents_configs
    Highest: YAML AGENT_CONFIGS overrides
    """
    agent_configs: Dict[str, Any] = {}
    agent_overrides = config.get("AGENT_CONFIGS", {}) or {}

    for agent_type, cfg_dict in config.get("dict_of_agents_configs", {}).items():
        agent_config_class = CONFIG_OBJECT_DICT[agent_type]
        field_names = {f.name for f in fields(agent_config_class)}
        base_overrides = {k: v for k, v in cfg_dict.items() if k in field_names}
        sweep_overrides = {
            k: v for k, v in agent_overrides.get(agent_type, {}).items()
            if k in field_names
        }
        agent_configs[agent_type] = agent_config_class(**{**base_overrides, **sweep_overrides})

    # Allow a YAML-only agent override if it is not present in the JSON.
    for agent_type, cfg_dict in agent_overrides.items():
        if agent_type in agent_configs:
            continue
        agent_config_class = CONFIG_OBJECT_DICT[agent_type]
        field_names = {f.name for f in fields(agent_config_class)}
        overrides = {k: v for k, v in cfg_dict.items() if k in field_names}
        agent_configs[agent_type] = agent_config_class(**overrides)

    return agent_configs


def _merge_hydra_and_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if config.get("ENV_CONFIG") is None:
        raise ValueError("ENV_CONFIG must point to the old-checkpoint env JSON.")

    env_config = load_config_from_file(config["ENV_CONFIG"])
    merged = OmegaConf.merge(OmegaConf.create(config), OmegaConf.structured(env_config))
    out = _to_container(merged)

    # The standard eval harness passes the held-out split as EvalTimePeriod.
    # This evaluator builds the environment from TimePeriod, so keep them aligned.
    eval_period = str(out.get("EvalTimePeriod", out.get("TimePeriod", "val")))
    out["EvalTimePeriod"] = eval_period
    out["TimePeriod"] = eval_period
    out["world_config"]["timePeriod"] = eval_period
    return out


def _make_eval_env(config: Dict[str, Any], rng: jax.Array) -> Tuple[MARLEnv, Any]:
    agent_configs = _create_agent_configs(config)
    world_config_kwargs = {
        k: v
        for k, v in config["world_config"].items()
        if hasattr(World_EnvironmentConfig(), k) and k not in ["seed", "timePeriod"]
    }
    ma_config = MultiAgentConfig(
        number_of_agents_per_type=config["NUM_AGENTS_PER_TYPE"],
        dict_of_agents_configs=agent_configs,
        world_config=World_EnvironmentConfig(
            seed=int(config["SEED"]),
            timePeriod=str(config.get("TimePeriod", "val")),
            **world_config_kwargs,
        ),
    )
    env = MARLEnv(key=rng, multi_agent_config=ma_config)
    return env, env.default_params


def _linear_schedule(lr: float, config: Dict[str, Any], count: int) -> float:
    # Used only to reconstruct the TrainState optimizer tree. No updates happen.
    num_updates = max(1, int(config.get("NUM_UPDATES", 1)))
    frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
    return lr * frac


def _build_empty_train_states(
    env: MARLEnv,
    config: Dict[str, Any],
    rng: jax.Array,
) -> List[TrainState]:
    train_states: List[TrainState] = []

    for i, _instance in enumerate(env.instance_list):
        network = ActorCriticRNN(env.action_spaces[i].n, config=config)
        rng, init_rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, int(config["NUM_ENVS"]), env.observation_spaces[i].shape[0])),
            jnp.zeros((1, int(config["NUM_ENVS"]))),
        )
        init_hstate = ScannedRNN.initialize_carry(
            int(config["NUM_ENVS"]), int(config["GRU_HIDDEN_DIM"])
        )
        params = network.init(init_rng, init_hstate, init_x)

        if config["ANNEAL_LR"][i]:
            schedule = functools.partial(_linear_schedule, float(config["LR"][i]), config)
            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"][i])),
                optax.adam(learning_rate=schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"][i])),
                optax.adam(float(config["LR"][i]), eps=1e-5),
            )

        train_states.append(TrainState.create(apply_fn=network.apply, params=params, tx=tx))

    return train_states


def _restore_train_states(
    checkpoint_dir: str,
    checkpoint_step: int,
    target_train_states: List[TrainState],
) -> Tuple[List[TrainState], int]:
    checkpointer = oxcp.PyTreeCheckpointer()
    manager = oxcp.CheckpointManager(checkpoint_dir, checkpointer)

    step = int(checkpoint_step)
    if step < 0:
        step = manager.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint steps found in {checkpoint_dir}")

    # The training code saved {'model': runner_state[0], 'metrics': ...}.
    # Restore against the full top-level tree so Orbax 0.11 accepts the on-disk
    # metadata, while still only using the restored TrainStates below.
    target = {
        "model": target_train_states,
        "metrics": {
            "train_rewards": [np.nan for _ in target_train_states],
        },
    }
    try:
        restored = manager.restore(
            step,
            items=target,
            restore_kwargs={"restore_args": orbax_utils.restore_args_from_target(target)},
        )
    except TypeError:
        # Newer Orbax fallback.
        restored = manager.restore(step, args=oxcp.args.PyTreeRestore(target))

    if not isinstance(restored, dict) or "model" not in restored:
        raise RuntimeError(
            "Checkpoint restore did not return a {'model': train_states} tree. "
            "Check that CHECKPOINT_DIR points to a checkpoint saved by ippo_rnn_JAXMARL.py."
        )
    return restored["model"], step


def _tree_to_np(x: Any) -> np.ndarray:
    return np.asarray(jax.device_get(x), dtype=float).squeeze()


def _evaluate(
    env: MARLEnv,
    env_params: Any,
    train_states: List[TrainState],
    config: Dict[str, Any],
    rng: jax.Array,
) -> Any:
    num_envs = int(config["NUM_ENVS"])
    num_steps_eval = int(config["NUM_STEPS_EVAL"])

    reset_rng = jax.random.split(rng, num_envs)
    last_obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

    hstates = [
        ScannedRNN.initialize_carry(
            int(config["NUM_ACTORS_PERTYPE"][i]), int(config["GRU_HIDDEN_DIM"])
        )
        for i in range(len(train_states))
    ]
    last_done = [
        jnp.zeros((int(config["NUM_ACTORS_PERTYPE"][i]),), dtype=bool)
        for i in range(len(train_states))
    ]

    deterministic = bool(config.get("DETERMINISTIC_EVAL", False))

    def eval_step(carry, _unused):
        env_state, last_obs, last_done, hstates, rng = carry
        rng, policy_rng = jax.random.split(rng)

        actions = []
        values = []
        log_probs = []
        new_hstates = []

        for i, train_state in enumerate(train_states):
            obs_i = batchify(last_obs[i], int(config["NUM_ACTORS_PERTYPE"][i]))
            ac_in = (obs_i[jnp.newaxis, :], last_done[i][jnp.newaxis, :])
            h_i, pi, value = train_state.apply_fn(train_state.params, hstates[i], ac_in)

            if deterministic and hasattr(pi, "mode"):
                action = pi.mode()
            else:
                policy_rng, subkey = jax.random.split(policy_rng)
                action = pi.sample(seed=subkey)

            values.append(value)
            log_probs.append(pi.log_prob(action))
            new_hstates.append(h_i)
            action = unbatchify(
                action,
                num_envs,
                env.multi_agent_config.number_of_agents_per_type[i],
            )
            actions.append(action.squeeze())

        rng, step_rng = jax.random.split(rng)
        rng_step = jax.random.split(step_rng, num_envs)
        obsv, new_env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(rng_step, env_state, actions, env_params)

        transitions = []
        new_done = []
        for i, _train_state in enumerate(train_states):
            done_i = batchify(done["agents"][i], int(config["NUM_ACTORS_PERTYPE"][i])).squeeze()
            new_done.append(done_i)
            info_i = {
                "world": info["world"],
                "agent": jax.tree.map(
                    lambda x: x.reshape(int(config["NUM_ACTORS_PERTYPE"][i]), -1),
                    info["agents"][i],
                ),
            }
            transitions.append(
                {
                    "global_done": jnp.tile(done["__all__"], int(config["NUM_AGENTS_PER_TYPE"][i])),
                    "done": last_done[i],
                    "action": batchify(actions[i], int(config["NUM_ACTORS_PERTYPE"][i])).squeeze(),
                    "value": values[i].squeeze(),
                    "reward": batchify(reward[i], int(config["NUM_ACTORS_PERTYPE"][i])).squeeze(),
                    "log_prob": log_probs[i].squeeze(),
                    "info": info_i,
                }
            )

        return (new_env_state, obsv, new_done, new_hstates, rng), transitions

    carry = (env_state, last_obs, last_done, hstates, rng)
    _, traj = jax.lax.scan(eval_step, carry, None, length=num_steps_eval)
    return traj


def _first_existing(info: Dict[str, Any], names: List[str]) -> Any | None:
    for name in names:
        if name in info:
            return info[name]
    return None


def _summarise_and_write(
    traj: Any,
    config: Dict[str, Any],
    checkpoint_step: int,
) -> Dict[str, Any]:
    out_dir = Path(config.get("OUTPUT_DIR", "results/rho_ex_reference_validation"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agent order follows dict_of_agents_configs insertion order: MarketMaking then Execution.
    agent_names = list(config["dict_of_agents_configs"].keys())
    exec_index = agent_names.index("Execution")
    ex_info = traj[exec_index]["info"]["agent"]

    reward_base_tree = _first_existing(ex_info, ["reward_base_ex"])
    if reward_base_tree is not None:
        reward_base = _tree_to_np(reward_base_tree)
    else:
        advantage = _tree_to_np(ex_info["advantage"])
        drift = _tree_to_np(ex_info["drift"])
        reward_base = advantage + float(config["dict_of_agents_configs"]["Execution"]["reward_lambda"]) * drift

    if "squared_remaining_qty" in ex_info:
        q2 = _tree_to_np(ex_info["squared_remaining_qty"])
        x_before_terminal = _tree_to_np(ex_info["quant_left_before_terminal"])
    else:
        # Fallback for trajectories generated before the V3 logging fields existed.
        # The terminal doom trade can reduce final quant_left to zero, so reconstruct
        # pre-terminal remaining quantity with quant_left + doom_quant.
        q_left = _tree_to_np(ex_info["quant_left"])
        doom = _tree_to_np(ex_info["doom_quant"])
        x_before_terminal = np.maximum(q_left + doom, 0.0)
        q2 = x_before_terminal ** 2

    reward_base = np.reshape(reward_base, (int(config["NUM_STEPS_EVAL"]), -1))
    q2 = np.reshape(q2, (int(config["NUM_STEPS_EVAL"]), -1))
    x_before_terminal = np.reshape(x_before_terminal, (int(config["NUM_STEPS_EVAL"]), -1))

    sum_abs_reward_base = np.nansum(np.abs(reward_base), axis=0)
    sum_squared_remaining = np.nansum(q2, axis=0)
    eps = float(config.get("RHO_STAR_EPS", 1e-8))
    rho_episode = sum_abs_reward_base / (sum_squared_remaining + eps)

    rho_ratio_of_means = float(np.nanmean(sum_abs_reward_base) / (np.nanmean(sum_squared_remaining) + eps))
    rho_mean_episode = float(np.nanmean(rho_episode))
    rho_median_episode = float(np.nanmedian(rho_episode))

    # Optional extra metrics for sanity checks.
    def info_np(name: str, default: float = np.nan) -> np.ndarray:
        if name in ex_info:
            return np.reshape(_tree_to_np(ex_info[name]), (int(config["NUM_STEPS_EVAL"]), -1))
        return np.full_like(q2, default, dtype=float)

    slippage = info_np("slippage")
    doom_quant = info_np("doom_quant")
    quant_left = info_np("quant_left")

    rows_path = out_dir / f"rho_ex_reference_validation_episodes_step_{checkpoint_step}.csv"
    with rows_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env_index",
                "sum_abs_reward_base_ex",
                "sum_squared_remaining_qty",
                "rho_ex_star_episode",
                "final_quant_left_before_terminal",
                "final_quant_left_after_accounting",
                "terminal_doom_quant",
                "sum_slippage",
            ],
        )
        writer.writeheader()
        for i in range(sum_abs_reward_base.shape[0]):
            writer.writerow(
                {
                    "env_index": i,
                    "sum_abs_reward_base_ex": float(sum_abs_reward_base[i]),
                    "sum_squared_remaining_qty": float(sum_squared_remaining[i]),
                    "rho_ex_star_episode": float(rho_episode[i]),
                    "final_quant_left_before_terminal": float(x_before_terminal[-1, i]),
                    "final_quant_left_after_accounting": float(quant_left[-1, i]),
                    "terminal_doom_quant": float(doom_quant[-1, i]),
                    "sum_slippage": float(np.nansum(slippage[:, i])),
                }
            )

    summary = {
        "schema_version": 1,
        "purpose": "rho_ex_reference_from_frozen_old_checkpoint_validation",
        "checkpoint_dir": config["CHECKPOINT_DIR"],
        "checkpoint_step": int(checkpoint_step),
        "time_period": config.get("TimePeriod"),
        "num_envs": int(config["NUM_ENVS"]),
        "num_steps_eval": int(config["NUM_STEPS_EVAL"]),
        "execution_config": {
            "reward_lambda": config["dict_of_agents_configs"]["Execution"].get("reward_lambda"),
            "reward_scaling_quo": config["dict_of_agents_configs"]["Execution"].get("reward_scaling_quo"),
            "doom_price_penalty": config["dict_of_agents_configs"]["Execution"].get("doom_price_penalty"),
            "reference_price": config["dict_of_agents_configs"]["Execution"].get("reference_price"),
            "rho_ex": config["dict_of_agents_configs"]["Execution"].get("rho_ex"),
            "zeta_ex": config["dict_of_agents_configs"]["Execution"].get("zeta_ex"),
        },
        "formula": {
            "rho_ex_star_ratio_of_means": "mean_env(sum_t(abs(reward_base_ex_t))) / (mean_env(sum_t(squared_remaining_qty_t)) + eps)",
            "rho_ex_star_episode": "sum_t(abs(reward_base_ex_t)) / (sum_t(squared_remaining_qty_t) + eps)",
            "eps": eps,
            "recommended_for_sweeps": "rho_ex_star_ratio_of_means",
        },
        "rho_ex_star_ratio_of_means": rho_ratio_of_means,
        "rho_ex_star_mean_episode": rho_mean_episode,
        "rho_ex_star_median_episode": rho_median_episode,
        "validation_stats": {
            "mean_sum_abs_reward_base_ex": float(np.nanmean(sum_abs_reward_base)),
            "std_sum_abs_reward_base_ex": float(np.nanstd(sum_abs_reward_base)),
            "mean_sum_squared_remaining_qty": float(np.nanmean(sum_squared_remaining)),
            "std_sum_squared_remaining_qty": float(np.nanstd(sum_squared_remaining)),
            "mean_final_quant_left_before_terminal": float(np.nanmean(x_before_terminal[-1])),
            "mean_terminal_doom_quant": float(np.nanmean(doom_quant[-1])),
        },
        "episode_csv": str(rows_path),
    }

    summary_path = out_dir / f"rho_ex_reference_validation_summary_step_{checkpoint_step}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("[rho-ex reference eval] wrote", summary_path)
    print("[rho-ex reference eval] wrote", rows_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


@hydra.main(version_base="1.3", config_path="../../../config/rl_configs", config_name="eval_old_checkpoint_rho_ex_reference")
def main(cfg) -> None:
    config = _merge_hydra_and_env_config(_to_container(cfg))

    # Derived fields used by network/eval code.
    config["NUM_ACTORS_PERTYPE"] = [
        int(n) * int(config["NUM_ENVS"]) for n in config["NUM_AGENTS_PER_TYPE"]
    ]
    config["NUM_ACTORS_TOTAL"] = sum(config["NUM_ACTORS_PERTYPE"])
    # Needed only for reconstructing schedules/optimizer tree.
    config["NUM_UPDATES"] = max(
        1,
        int(float(config.get("TOTAL_TIMESTEPS", 1)) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])),
    )

    rng = jax.random.PRNGKey(int(config["SEED"]))
    rng, env_rng, init_rng, eval_rng = jax.random.split(rng, 4)

    env, env_params = _make_eval_env(config, env_rng)
    target_train_states = _build_empty_train_states(env, config, init_rng)
    train_states, restored_step = _restore_train_states(
        checkpoint_dir=config["CHECKPOINT_DIR"],
        checkpoint_step=int(config.get("CHECKPOINT_STEP", -1)),
        target_train_states=target_train_states,
    )

    print(f"[rho-ex reference eval] restored checkpoint step {restored_step}")
    print("[rho-ex reference eval] eval timePeriod:", config.get("TimePeriod"))
    print("[rho-ex reference eval] agent order:", list(config["dict_of_agents_configs"].keys()))

    traj = _evaluate(env, env_params, train_states, config, eval_rng)
    _summarise_and_write(traj, config, restored_step)


if __name__ == "__main__":
    main()
