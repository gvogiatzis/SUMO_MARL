#!/usr/bin/env python3
"""Evaluate all trained methods on the frozen switching-schedule eval cases.

Produces the Phase 3 main-comparison rows for the switching task: one CSV row
per (method, case) with episode KPIs, mean±std aggregated per method.
Reuses model construction/checkpoint/action machinery from
evaluate_fixed_trips_all.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
from evaluate_fixed_trips_all import build_model, checkpoint_path_for, load_checkpoint, run_single_episode

DEFAULT_METHODS = ["dqn_mlp", "drqn_lstm", "dqn_gnn", "drqn_gnn_lstm", "colight", "gated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="Model seed folder")
    ap.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--logs-base", type=str, default=".")
    ap.add_argument("--switching-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="eval_switching")
    ap.add_argument("--hidden-q", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--sumo-seed", type=int, default=12345)
    args = ap.parse_args()

    sw_dir = Path(args.switching_dir or f"eval_trips_switching_v2_grid_{args.grid_n}")
    meta = json.loads((sw_dir / "schedules.json").read_text())
    device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in args.methods:
        ckpt = checkpoint_path_for(method, args.grid_n, args.seed, Path(args.logs_base))
        if ckpt is None:
            print(f"[SKIP] {method}: checkpoint not found")
            continue

        model = None
        kpis_per_case = []
        for fname in sorted(meta["schedules"].keys()):
            env = SumoGridMARLFixedEnv(
                grid_n=args.grid_n,
                episode_steps=int(meta["episode_steps"]),
                sumo_steps_per_env_step=int(meta["sumo_steps_per_env_step"]),
                fixed_trips_file=str(sw_dir / fname),
                seed=args.sumo_seed,
                suppress_sumo_output=True,
            )
            if model is None:
                obs = env.reset()
                aid0 = list(env.agent_ids)[0]
                O = len(obs[aid0]); A = env.action_spaces[aid0].n
                model = build_model(method, O, A, args.hidden_q, args.hidden_q, args.gnn_layers).to(device)
                load_checkpoint(model, ckpt, device)
                model.eval()
            m = run_single_episode(env, model, device)
            env.close()
            kpis_per_case.append(m)
            rows.append({"method": method, "grid_n": args.grid_n, "seed": args.seed, "case": fname, **m})

        tp = [m["throughput_vph"] for m in kpis_per_case]
        wt = [m["mean_wait_s"] for m in kpis_per_case]
        rt = [m["episode_return"] for m in kpis_per_case]
        print(f"{method:16s} over {len(kpis_per_case)} cases | TP {np.mean(tp):7.1f}±{np.std(tp):6.1f} | "
              f"wait {np.mean(wt):6.1f}±{np.std(wt):5.1f}s | return {np.mean(rt):8.1f}±{np.std(rt):6.1f}")

    out_csv = out_dir / f"switching_eval_grid{args.grid_n}_seed{args.seed}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
