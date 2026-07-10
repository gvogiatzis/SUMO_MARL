#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List
import atexit

_EVAL_POOLS: Dict[tuple, "EvalEnvPool"] = {}

# Import BOTH envs at top (as requested)
from sumo_marl_random_flow_env import SumoGridMARLRandomEnv
from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv

# =============================== Env builders (TRAIN=random, EVAL=fixed) ===============================

def build_train_env(args):
    """
    TRAINING: always use random-flow environment (SumoGridMARLFixedEnv).
    grid_n is taken explicitly from args.grid_n
    """
    return SumoGridMARLRandomEnv(
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        grid_n=args.grid_n,  # <— explicit
        episode_steps=args.episode_steps,
        sumo_steps_per_env_step=args.sumo_steps_per_env_step,
        regime=getattr(args, "regime", None),
        regime_intensity=getattr(args, "regime_intensity", 1.0),
        segment_steps=getattr(args, "segment_steps", 150),
        seed=args.seed,
        verbose=False,
        suppress_sumo_output=True,
    )

# def list_trip_files_for_grid(grid_n: int, env_ep_length: int) -> List[Path]:
#     trip_dir = Path(f"./eval_trips_grid_{grid_n}_len{env_ep_length}")
#     return sorted(trip_dir.glob("*.xml"))

def list_trip_files_for_grid(grid_n: int) -> List[Path]:
    trip_dir = Path(f"./eval_trips_grid_{grid_n}")
    return sorted(trip_dir.glob("*.xml"))

class EvalEnvPool:
    """
    Holds one SumoGridMARLFixedEnv per trip file, built once and reused
    across multiple evaluation calls.
    """
    def __init__(self, envs: List[SumoGridMARLFixedEnv], trip_paths: List[Path]):
        self.envs = envs
        self.trip_paths = trip_paths

    @classmethod
    def from_args(cls, args) -> "EvalEnvPool":
        # eval_ep_length = args.episode_steps * args.sumo_steps_per_env_step
        # trip_paths = list_trip_files_for_grid(args.grid_n, eval_ep_length)
        trip_paths = list_trip_files_for_grid(args.grid_n)
        if not trip_paths:
            raise FileNotFoundError(f"No *.xml trip files in {trips_dir_for_grid(args.grid_n).resolve()}")
        envs: List[SumoGridMARLFixedEnv] = []
        for tp in trip_paths:
            envs.append(
                SumoGridMARLFixedEnv(
                    gui=args.gui,
                    gui_delay_ms=args.gui_delay_ms,
                    grid_n=args.grid_n,
                    episode_steps=args.episode_steps,
                    sumo_steps_per_env_step=args.sumo_steps_per_env_step,
                    seed=args.seed + 999,   # eval seed offset
                    verbose=False,
                    suppress_sumo_output=True,
                    fixed_trips_file=str(tp),
                )
            )
        return cls(envs, trip_paths)

    def close_all(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass

    @atexit.register
    def _close_eval_pools_at_exit():
        # Close all cached eval envs on interpreter shutdown
        for pool in list(_EVAL_POOLS.values()):
            try:
                pool.close_all()
            except Exception:
                pass

def get_or_create_eval_pool(args) -> EvalEnvPool:
    """
    Cache key uses the settings that affect env construction.
    If any of these change, a new pool is built.
    """
    key = (args.grid_n, args.episode_steps, args.sumo_steps_per_env_step, bool(args.gui), int(args.gui_delay_ms), int(args.seed))
    pool = _EVAL_POOLS.get(key)
    if pool is None:
        pool = EvalEnvPool.from_args(args)
        _EVAL_POOLS[key] = pool
    return pool
