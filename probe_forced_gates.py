#!/usr/bin/env python3
"""Forced-gate probe: marginal value of memory vs communication per regime.

Evaluates a trained GatedSpatioTemporalQ with gates FORCED to {both, mem-only,
com-only, neither} on pure-regime episodes (matched intensity), paired across
configs via identical env seeds. If mem-only ~ com-only everywhere, the demand
does not dissociate the pathways and no gate mechanism can differentiate them.
"""
from __future__ import annotations
import argparse
from itertools import product
from typing import Optional, Tuple

import numpy as np
import torch

from sumo_marl_random_flow_env import SumoGridMARLRandomEnv
from marl_utils.models import GatedSpatioTemporalQ
from marl_utils.common import build_grid_edge_index

CONFIGS = {"both": (1.0, 1.0), "mem_only": (1.0, 0.0), "com_only": (0.0, 1.0), "neither": (0.0, 0.0)}


def force_gates(model, g_mem: float, g_com: float):
    def _gate(logits):
        g = torch.zeros_like(logits)
        g[..., 0] = g_mem
        g[..., 1] = g_com
        return g, g
    model._gate = _gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--intensity", type=float, default=3.0)
    ap.add_argument("--intensity-map", type=str, default=None)
    ap.add_argument("--episode-steps", type=int, default=450)
    ap.add_argument("--env-seeds", type=str, default="901,902")
    args = ap.parse_args()

    env_seeds = [int(s) for s in args.env_seeds.split(",")]
    imap = {}
    if args.intensity_map:
        imap = {k.strip(): float(v) for k, v in (kv.split(":") for kv in args.intensity_map.split(","))}
    results = {}
    model = None

    for regime, env_seed in product(("corridor", "cross", "platoons", "bursty"), env_seeds):
        env = SumoGridMARLRandomEnv(gui=False, grid_n=args.grid_n, episode_steps=args.episode_steps,
                                    sumo_steps_per_env_step=5, seed=env_seed, regime=regime,
                                    regime_intensity=imap.get(regime, args.intensity),
                                    suppress_sumo_output=True)
        for cname, (gm, gc) in CONFIGS.items():
            obs = env.reset()  # same seed -> same trips for every config (rng re-seeded per env)
            agent_ids = list(env.agent_ids)
            edge_index = build_grid_edge_index(agent_ids)
            if model is None:
                O = len(obs[agent_ids[0]]); A = env.action_spaces[agent_ids[0]].n
                base = GatedSpatioTemporalQ(node_dim=O, actions=A, hidden=128, gnn_layers=2)
                base.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False))
                base.eval()
                model = base
            force_gates(model, gm, gc)
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            ep_ret, done = 0.0, False
            while not done:
                X = np.stack([obs[a] for a in agent_ids], 0).astype(np.float32)
                with torch.no_grad():
                    q, hidden, _ = model.step(torch.as_tensor(X).unsqueeze(0), edge_index, hidden)
                acts = torch.argmax(q[0], -1).tolist()
                obs, rew, done, info = env.step({a: int(acts[i]) for i, a in enumerate(agent_ids)})
                ep_ret += float(np.sum(list(rew.values())))
            results.setdefault((regime, cname), []).append(
                (ep_ret, info["network_kpis"]["mean_waiting_time_s"]))
            # fresh trips for next config need the same rng state: re-create env per config instead
            env._rng = np.random.default_rng(env_seed)  # reset episode RNG so trips repeat
        env.close()

    print(f"\n=== Forced-gate probe (return | wait_s), intensity {args.intensity} ===")
    print(f"{'regime':10s} {'both':>16s} {'mem_only':>16s} {'com_only':>16s} {'neither':>16s}")
    for regime in ("corridor", "cross", "platoons", "bursty"):
        cells = []
        for cname in ("both", "mem_only", "com_only", "neither"):
            vals = results[(regime, cname)]
            r = np.mean([v[0] for v in vals]); w = np.mean([v[1] for v in vals])
            cells.append(f"{r:7.0f}|{w:6.1f}")
        print(f"{regime:10s} {cells[0]:>16s} {cells[1]:>16s} {cells[2]:>16s} {cells[3]:>16s}")


if __name__ == "__main__":
    main()
