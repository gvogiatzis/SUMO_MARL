#!/usr/bin/env python3
"""Within-episode gate analysis on frozen switching-schedule eval cases (RQ2).

For each case in eval_trips_switching_grid_{N}/, runs the trained gated model
greedily, logs per-step hard gate rates, aligns them with the case's known
regime schedule, and reports per-regime gate means computed *within* episodes.
Saves per-case gate time series and a summary CSV.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
from marl_utils.models import GatedSpatioTemporalQ
from marl_utils.common import build_grid_edge_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="Model seed folder")
    ap.add_argument("--logs-base", type=str, default=".")
    ap.add_argument("--switching-dir", type=str, default=None,
                    help="Default: eval_trips_switching_grid_{N}")
    ap.add_argument("--out-dir", type=str, default="gate_analysis_switching")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--sumo-seed", type=int, default=12345)
    args = ap.parse_args()

    sw_dir = Path(args.switching_dir or f"eval_trips_switching_grid_{args.grid_n}")
    meta = json.loads((sw_dir / "schedules.json").read_text())
    episode_steps = int(meta["episode_steps"])
    sspes = int(meta["sumo_steps_per_env_step"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.logs_base) / f"logs_grid_{args.grid_n}" / f"seed{args.seed}" / \
        f"model_best_gated_shared_seqlen8_seed{args.seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model = None
    per_regime = defaultdict(lambda: {"g_mem": [], "g_com": []})
    rows = []

    for fname, schedule in sorted(meta["schedules"].items()):
        env = SumoGridMARLFixedEnv(
            grid_n=args.grid_n, episode_steps=episode_steps,
            sumo_steps_per_env_step=sspes,
            fixed_trips_file=str(sw_dir / fname), seed=args.sumo_seed,
            suppress_sumo_output=True,
        )
        obs = env.reset()
        agent_ids = list(env.agent_ids)
        edge_index = build_grid_edge_index(agent_ids)

        if model is None:
            O = len(obs[agent_ids[0]])
            A = env.action_spaces[agent_ids[0]].n
            model = GatedSpatioTemporalQ(node_dim=O, actions=A, hidden=args.hidden,
                                         gnn_layers=args.gnn_layers)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
            model.eval()

        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        gm_t, gc_t = [], []
        done = False
        while not done:
            X = np.stack([obs[a] for a in agent_ids], 0).astype(np.float32)
            with torch.no_grad():
                q, hidden, g = model.step(torch.as_tensor(X).unsqueeze(0), edge_index, hidden)
            gm_t.append(float((g[..., 0] > 0.5).float().mean()))
            gc_t.append(float((g[..., 1] > 0.5).float().mean()))
            acts = torch.argmax(q[0], -1).tolist()
            obs, _, done, info = env.step({a: int(acts[i]) for i, a in enumerate(agent_ids)})
        env.close()

        gm = np.array(gm_t)
        gc = np.array(gc_t)
        np.save(out_dir / f"{Path(fname).stem}_gmem_seed{args.seed}.npy", gm)
        np.save(out_dir / f"{Path(fname).stem}_gcom_seed{args.seed}.npy", gc)

        # align steps to schedule segments (env step t covers sim time [t, t+1)*sspes seconds)
        for (t0, t1, regime) in schedule:
            s0, s1 = int(t0 // sspes), min(int(t1 // sspes), len(gm))
            if s1 <= s0:
                continue
            per_regime[regime]["g_mem"].append(gm[s0:s1].mean())
            per_regime[regime]["g_com"].append(gc[s0:s1].mean())
            rows.append({
                "case": fname, "seed": args.seed, "regime": regime,
                "t0": t0, "t1": t1,
                "g_mem": float(gm[s0:s1].mean()), "g_com": float(gc[s0:s1].mean()),
            })
        wait = info.get("network_kpis", {}).get("mean_waiting_time_s", 0.0)
        seg_gm = [round(float(gm[int(a // sspes):min(int(b // sspes), len(gm))].mean()), 2) for a, b, _ in schedule]
        seg_gc = [round(float(gc[int(a // sspes):min(int(b // sspes), len(gc))].mean()), 2) for a, b, _ in schedule]
        print(f"{fname}: schedule={[r for _, _, r in schedule]} g_mem/seg={seg_gm} g_com/seg={seg_gc} wait={wait:.1f}s")

    print("\n=== Within-episode gate rates by regime (mean over segments) ===")
    for regime, d in sorted(per_regime.items()):
        print(f"{regime:10s} g_mem={np.mean(d['g_mem']):.3f}±{np.std(d['g_mem']):.3f} "
              f"g_com={np.mean(d['g_com']):.3f}±{np.std(d['g_com']):.3f} (n={len(d['g_mem'])} segments)")

    out_csv = out_dir / f"segments_grid{args.grid_n}_seed{args.seed}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
