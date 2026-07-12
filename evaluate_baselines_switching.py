#!/usr/bin/env python3
"""Evaluate rule-based controllers (fixed-time, max-pressure) on the frozen
switching-schedule eval cases, matching evaluate_switching.py's protocol so the
rows are directly comparable to the learned-method main table.

Reuses the controller classes and episode runner from
evaluate_baselines_fixed_trips.py; iterates over the generic case_XX.xml files
in eval_trips_switching_v*_grid_N/ rather than regime-typed directories.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
from evaluate_baselines_fixed_trips import (
    build_controller,
    parse_fixed_time_cycle,
    controller_names,
    PhaseUseStats,
)


def run_case(controller_name, cfg, grid_n, trips_file, episode_steps, sspes, sumo_seed):
    env = SumoGridMARLFixedEnv(
        grid_n=grid_n, episode_steps=episode_steps, sumo_steps_per_env_step=sspes,
        fixed_trips_file=str(trips_file), seed=sumo_seed, suppress_sumo_output=True,
    )
    obs = env.reset()
    agent_ids = list(env.agent_ids)
    action_dim = int(env.action_spaces[agent_ids[0]].n)
    cycle = parse_fixed_time_cycle(cfg.fixed_time_cycle, action_dim)
    controller = build_controller(controller_name, cfg, agent_ids, action_dim, cycle)
    controller.reset(env=env)

    ep_return, final_kpis, done, steps = 0.0, {}, False, 0
    while not done and steps < env.episode_steps:
        actions = controller.act(obs, env=env)
        obs, rewards, done, info = env.step(actions)
        ep_return += float(np.sum(list(rewards.values())))
        final_kpis = info.get("network_kpis", {})
        steps += 1
    env.close()
    return {
        "throughput_vph": float(final_kpis.get("throughput_veh_per_hour", 0.0)),
        "mean_wait_s": float(final_kpis.get("mean_waiting_time_s", 0.0)),
        "mean_travel_s": float(final_kpis.get("mean_travel_time_s", 0.0)),
        "episode_return": float(ep_return),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--switching-dir", type=str, default=None)
    ap.add_argument("--controllers", type=str, default="fixed_time,max_pressure")
    ap.add_argument("--out-dir", type=str, default="eval_baselines_switching")
    ap.add_argument("--sumo-seed", type=int, default=12345)
    ap.add_argument("--n-sumo-seeds", type=int, default=3)
    ap.add_argument("--tipover-wait-s", type=float, default=60.0)
    # controller knobs (defaults mirror evaluate_baselines_fixed_trips.py)
    ap.add_argument("--fixed-time-hold-steps", type=int, default=4)
    ap.add_argument("--fixed-time-cycle", type=str, default="0,1,2,3")
    ap.add_argument("--tie-tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    sw_dir = Path(args.switching_dir or f"eval_trips_switching_v3_grid_{args.grid_n}")
    meta = json.loads((sw_dir / "schedules.json").read_text())
    episode_steps = int(meta["episode_steps"])
    sspes = int(meta["sumo_steps_per_env_step"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controllers = [c.strip() for c in args.controllers.split(",")]
    rows = []
    for controller_name in controllers:
        per = []
        for fname in sorted(meta["schedules"].keys()):
            for k in range(args.n_sumo_seeds):
                m = run_case(controller_name, args, args.grid_n, sw_dir / fname,
                             episode_steps, sspes, args.sumo_seed + 101 * k)
                per.append(m)
                rows.append({"controller": controller_name, "grid_n": args.grid_n,
                             "case": fname, "sumo_seed": args.sumo_seed + 101 * k, **m})
        w = np.array([m["mean_wait_s"] for m in per])
        tp = np.array([m["throughput_vph"] for m in per])
        print(f"{controller_name:14s} n={len(w):3d} | wait med {np.median(w):6.1f}s mean {w.mean():6.1f}s | "
              f"tip-over {100*(w > args.tipover_wait_s).mean():4.1f}% | TP med {np.median(tp):7.1f}")

    out_csv = out_dir / f"baselines_switching_grid{args.grid_n}.csv"
    with out_csv.open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
