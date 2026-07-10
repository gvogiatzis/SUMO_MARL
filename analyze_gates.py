#!/usr/bin/env python3
"""Gate-activation-by-regime analysis for the gated adaptive model (RQ2).

Loads a trained GatedSpatioTemporalQ checkpoint, runs greedy episodes on each
regime's fixed trip file, and records per-step hard gate activation rates.
Outputs a summary CSV plus per-regime gate time series (.npy).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
from marl_utils.models import GatedSpatioTemporalQ
from marl_utils.common import build_grid_edge_index
from evaluate_fixed_trips_all import resolve_trips_by_type

REGIMES = ["corridor", "cross", "platoons", "bursty"]


def main():
    ap = argparse.ArgumentParser("Gate activation by regime")
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="Model seed folder")
    ap.add_argument("--logs-base", type=str, default=".")
    ap.add_argument("--trips-root", type=str, default=".")
    ap.add_argument("--out-dir", type=str, default="gate_analysis")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sumo-steps-per-env-step", type=int, default=5)
    ap.add_argument("--n-evals", type=int, default=10)
    ap.add_argument("--eval-seed", type=int, default=12345)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(args.logs_base) / f"logs_grid_{args.grid_n}" / f"seed{args.seed}" / \
        f"model_best_gated_shared_seqlen8_seed{args.seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    rng = np.random.default_rng(args.eval_seed)
    sumo_seeds = rng.integers(0, 2**31 - 1, size=args.n_evals, dtype=np.int64).tolist()

    rows = []
    for regime in REGIMES:
        trips = resolve_trips_by_type(Path(args.trips_root), args.grid_n, regime)
        env = SumoGridMARLFixedEnv(
            grid_n=args.grid_n, episode_steps=args.steps,
            sumo_steps_per_env_step=args.sumo_steps_per_env_step,
            fixed_trips_file=str(trips), seed=int(sumo_seeds[0]),
            suppress_sumo_output=True,
        )
        obs = env.reset()
        agent_ids = list(env.agent_ids)
        O = len(obs[agent_ids[0]])
        A = env.action_spaces[agent_ids[0]].n
        edge_index = build_grid_edge_index(agent_ids)

        model = GatedSpatioTemporalQ(node_dim=O, actions=A, hidden=args.hidden,
                                     gnn_layers=args.gnn_layers).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()

        # per-step series averaged over evals: [n_evals, steps] for each gate
        gmem_series, gcom_series = [], []
        kpi_wait = []
        for s in sumo_seeds:
            env.seed = int(s)
            obs = env.reset()
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            g_mem_t, g_com_t = [], []
            done, info = False, {}
            while not done:
                X = np.stack([obs[a] for a in agent_ids], 0).astype(np.float32)
                X = torch.as_tensor(X).unsqueeze(0)
                with torch.no_grad():
                    q, hidden, g = model.step(X, edge_index, hidden)
                g_mem_t.append(float((g[..., 0] > 0.5).float().mean()))
                g_com_t.append(float((g[..., 1] > 0.5).float().mean()))
                acts = torch.argmax(q[0], -1).tolist()
                obs, _, done, info = env.step({a: int(acts[i]) for i, a in enumerate(agent_ids)})
            gmem_series.append(g_mem_t)
            gcom_series.append(g_com_t)
            kpi_wait.append(float(info.get("network_kpis", {}).get("mean_waiting_time_s", 0.0)))
        env.close()

        T = min(len(x) for x in gmem_series)
        gmem = np.array([x[:T] for x in gmem_series])  # [n_evals, T]
        gcom = np.array([x[:T] for x in gcom_series])
        np.save(out_dir / f"gmem_series_{regime}_grid{args.grid_n}_seed{args.seed}.npy", gmem)
        np.save(out_dir / f"gcom_series_{regime}_grid{args.grid_n}_seed{args.seed}.npy", gcom)

        rows.append({
            "regime": regime, "grid_n": args.grid_n, "seed": args.seed,
            "g_mem_rate": float(gmem.mean()), "g_mem_std": float(gmem.mean(1).std()),
            "g_com_rate": float(gcom.mean()), "g_com_std": float(gcom.mean(1).std()),
            "mean_wait_s": float(np.mean(kpi_wait)),
        })
        print(f"{regime:9s} g_mem={gmem.mean():.3f}±{gmem.mean(1).std():.3f} "
              f"g_com={gcom.mean():.3f}±{gcom.mean(1).std():.3f} wait={np.mean(kpi_wait):.1f}s")

    import csv
    out_csv = out_dir / f"gate_summary_grid{args.grid_n}_seed{args.seed}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
