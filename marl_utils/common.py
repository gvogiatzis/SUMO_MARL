#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Tuple, List

# --------------------------- seed / tensor / env ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-n', type=int, default=3, help='Core grid size N (NxN traffic lights)')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--episode-steps', type=int, default=200)
    parser.add_argument('--sumo-steps-per-env-step', type=int, default=5)
    parser.add_argument('--regime', type=str, default=None,
                        choices=['corridor', 'cross', 'platoons', 'bursty', 'mixed', 'switching'],
                        help='Regime-conditioned training demand; "mixed" samples a regime per episode, '
                             '"switching" changes regime within the episode every --segment-steps '
                             '(default: original random flows)')
    parser.add_argument('--segment-steps', type=int, default=150,
                        help='Switching regime: segment length in env steps')
    parser.add_argument('--eval-trips-dir', type=str, default=None,
                        help='Directory of fixed eval trip files (default: eval_trips_grid_{N})')
    parser.add_argument('--gridlock-patience', type=int, default=0,
                        help='Truncate training episodes after N consecutive gridlocked env steps (0 = off)')
    parser.add_argument('--regime-intensity', type=float, default=1.0,
                        help='Global multiplier on regime flow rates')
    parser.add_argument('--regime-intensity-map', type=str, default=None,
                        help='Per-regime intensity overrides, e.g. "cross:2,platoons:8,bursty:6"')

    # Gated spatio-temporal model (AAMAS)
    parser.add_argument('--lambda-mem', type=float, default=0.01, help='Sparsity cost on memory gate')
    parser.add_argument('--lambda-com', type=float, default=0.01, help='Sparsity cost on communication gate')
    parser.add_argument('--gate-temp-start', type=float, default=2.0, help='Gumbel-sigmoid temperature at start')
    parser.add_argument('--gate-temp-end', type=float, default=0.5, help='Gumbel-sigmoid temperature at end')
    parser.add_argument('--gate-cost-warmup-eps', type=int, default=30,
                        help='Episodes before gate sparsity cost is enabled')
    parser.add_argument('--gate-budget', type=float, default=-1.0,
                        help='If >=0: gate activation budget rho (see --gate-budget-mode)')
    parser.add_argument('--gate-budget-mode', type=str, default='cap', choices=['cap', 'target'],
                        help='cap: penalise only usage above rho (gates may close freely); '
                             'target: two-sided (mean_gate - rho)^2 (forces ~rho usage)')
    parser.add_argument('--ckpt-every', type=int, default=20,
                        help='Save resumable training state every N episodes (0 = off)')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--gui-delay-ms', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    # model / opt
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--gnn-layers', type=int, default=2, help='number of gnn layers only for gnn+lstm structure')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for off-policy methods')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='learning rate for actor of on-policy methods')
    parser.add_argument('--lr-critic', type=float, default=1e-3, help='learning rate for critic of on-policy methods')
    parser.add_argument('--tau', type=float, default=0.005)

    # A2C/GAE specific
    parser.add_argument('--gae-lambda', type=float, default=0.9)
    parser.add_argument('--value-coef', type=float, default=0.7)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # exploration decay
    parser.add_argument('--eps-start', type=float, default=0.5)
    parser.add_argument('--eps-end', type=float, default=0.05)
    parser.add_argument('--eps-decay', type=float, default=0.998)

    # entropy decay
    parser.add_argument('--entropy-coef-start', type=float, default=0.05)
    parser.add_argument('--entropy-coef-end',   type=float, default=0.005)
    parser.add_argument('--entropy-coef-decay', type=float, default=0.995)

    # Stabilization knobs
    parser.add_argument('--normalize-rewards', action='store_true', default=True)
    parser.add_argument('--reward-scale', type=float, default=1.0)
    parser.add_argument('--normalize-adv', action='store_true', default=True)
    parser.add_argument('--huber-delta', type=float, default=1.0)
    parser.add_argument('--value-clip-eps', type=float, default=0.2)

    # replay / updates
    parser.add_argument('--replay-size', type=int, default=100000)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for none-sequence replay')
    parser.add_argument('--batch-size-seq', type=int, default=16, help='batch size for sequence replay')
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--burn-in', type=int, default=4)
    parser.add_argument('--updates-per-ep', type=int, default=32)
    parser.add_argument('--rb-seed', type=int, default=1234)

    # MA2C-PA specific knobs
    parser.add_argument('--advantage-mode', type=str, default='per_agent', choices=['per_agent','team'],
                        help="Use each agent's reward for GAE (per_agent) or a team reward (team).")
    parser.add_argument('--team-reward-reduce', type=str, default='mean', choices=['mean','sum'],
                        help="How to aggregate per-agent rewards when advantage_mode='team'.")
    parser.add_argument('--normalize-rewards-mode', type=str, default='per_agent', choices=['off','per_agent','global'],
                        help="Reward normalization: per-agent over time, or global over T*N.")
    # parser.add_argument('--reward-scale', type=float, default=1.0)

    # QMIX specific
    parser.add_argument("--mixing_embed_dim", type=int, default=32)
    parser.add_argument("--hypernet_embed_dim", type=int, default=64)
    # parser.add_argument("--grad_clip", type=float, default=1.0)

    return parser.parse_args()


def set_global_seed(random_seed: int):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

def to_tensor(device: torch.device, array_like, dtype=torch.float32):
    return torch.as_tensor(array_like, device=device, dtype=dtype)

def first_module_device(modules_dict: Dict[str, nn.Module]) -> torch.device:
    first = next(iter(modules_dict.values()))
    return next(first.parameters()).device

def parse_ij(tls_id: str) -> Tuple[int, int]:
    try:
        _, i, j = tls_id.split("_")
        return int(i), int(j)
    except Exception:
        return -10**9, -10**9

# --------------------------- evaluation info saver ---------------------------
class EvalHistory:
    def __init__(self, logdir: str, run_name: str, seed: int):
        """
        Saves KPI arrays under: <logdir>/seed<seed>/<run_name>_*.npy
        Example dir: logs_grid_3/seed42/
        """
        self.base_dir = Path(logdir)
        self.dir = self.base_dir / f"seed{int(seed)}"
        self.dir.mkdir(parents=True, exist_ok=True)

        self.run = run_name
        self.ret = self._load('avg_return.npy')
        self.thr = self._load('avg_throughput_veh_per_hour.npy')
        self.mtt = self._load('avg_mean_travel_time_s.npy')
        self.mwt = self._load('avg_mean_waiting_time_s.npy')

    def _path(self, fname: str) -> Path:
        return self.dir / f"{self.run}_{fname}"

    def _load(self, fname: str):
        p = self._path(fname)
        if p.exists():
            try:
                return np.load(p)
            except Exception:
                pass
        return None

    @staticmethod
    def _append(arr, val: float):
        return np.array([val], dtype=np.float64) if arr is None else np.concatenate([arr, [val]])

    def save(self, avg_return: float, avg_throughput: float, avg_travel_s: float, avg_wait_s: float):
        self.ret = self._append(self.ret, avg_return)
        self.thr = self._append(self.thr, avg_throughput)
        self.mtt = self._append(self.mtt, avg_travel_s)
        self.mwt = self._append(self.mwt, avg_wait_s)
        np.save(self._path('avg_return.npy'), self.ret)
        np.save(self._path('avg_throughput_veh_per_hour.npy'), self.thr)
        np.save(self._path('avg_mean_travel_time_s.npy'), self.mtt)
        np.save(self._path('avg_mean_waiting_time_s.npy'), self.mwt)


def clear_eval_history(logdir: str, run_name: str, seed: int):
    """
    Remove old KPI .npy files for this specific run prefix only,
    scoped to the seed subfolder: <logdir>/seed<seed>/.

    If `run_name` is falsy (None or ""), do nothing.

    Files removed (when run_name is provided):
      <logdir>/seed<seed>/{run_name}_avg_return.npy
      <logdir>/seed<seed>/{run_name}_avg_throughput_veh_per_hour.npy
      <logdir>/seed<seed>/{run_name}_avg_mean_travel_time_s.npy
      <logdir>/seed<seed>/{run_name}_avg_mean_waiting_time_s.npy
    """
    if not run_name:
        return

    seed_dir = Path(logdir) / f"seed{int(seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [
        "avg_return.npy",
        "avg_throughput_veh_per_hour.npy",
        "avg_mean_travel_time_s.npy",
        "avg_mean_waiting_time_s.npy",
    ]
    for suf in suffixes:
        (seed_dir / f"{run_name}_{suf}").unlink(missing_ok=True)

# =============================== Graph Helpers ===============================
def build_grid_edge_index(agent_ids: List[str]) -> torch.Tensor:
    id_to_idx = {aid: idx for idx, aid in enumerate(agent_ids)}
    coords = {aid: parse_ij(aid) for aid in agent_ids}
    edges: List[Tuple[int,int]] = []
    for aid, (i, j) in coords.items():
        if (i, j) == (-10**9, -10**9): continue
        for (ni, nj) in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
            nid = f"n_{ni}_{nj}"
            if nid in id_to_idx:
                edges.append((id_to_idx[aid], id_to_idx[nid]))
    if not edges:
        return torch.zeros((2,0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# build batched edge index
def build_batched_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
    """
    Replicates a single-graph edge_index for a batch of identical graphs.
    edge_index: [2, E] over nodes 0..N-1
    returns   : [2, B*E] over nodes 0..B*N-1, with per-batch offsets
    """
    Ei = edge_index
    device = Ei.device
    offsets = torch.arange(batch_size, device=device, dtype=Ei.dtype) * num_nodes  # [B]
    src = Ei[0].unsqueeze(0) + offsets.unsqueeze(1)                                 # [B,E]
    dst = Ei[1].unsqueeze(0) + offsets.unsqueeze(1)                                 # [B,E]
    return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)                   # [2,B*E]

# =============================== neighbor mapping helpers ===============================

# Direction order is fixed for feature concat
DIRS: List[Tuple[int, int]] = [(0, +1), (+1, 0), (0, -1), (-1, 0)]  # N, E, S, W
DIR_NAMES = ["N", "E", "S", "W"]

def build_neighbor_map(agent_ids: List[str]) -> Dict[str, List[str]]:
    """
    For each agent_id (assumed 'n_i_j'), find its 1-hop neighbors in
    the fixed order [N, E, S, W]. If a neighbor doesn't exist, store None.
    """
    id_set = set(agent_ids)
    neighbor_map: Dict[str, List[str | None]] = {}
    for aid in agent_ids:
        i, j = parse_ij(aid)
        neighs: List[str | None] = []
        for (di, dj) in DIRS:
            ni, nj = i + di, j + dj
            nid = f"n_{ni}_{nj}"
            neighs.append(nid if nid in id_set else None)
        neighbor_map[aid] = neighs
    return neighbor_map

def augment_obs_with_neighbors(
    obs_dict: Dict[str, np.ndarray],
    agent_ids: List[str],
    neighbor_map: Dict[str, List[str]],
) -> Dict[str, np.ndarray]:
    """
    For each agent's local obs (shape 13), append 1-hop neighbor summaries
    in [N,E,S,W] order. For each neighbor: [sum_queues_12, neighbor_phase].
    Missing neighbors -> [0.0, 0.0].
    Returns a dict with augmented vectors (shape 21).
    """
    aug: Dict[str, np.ndarray] = {}
    for aid in agent_ids:
        own = obs_dict[aid].astype(np.float32)
        feats: List[float] = []
        for neighbor_id in neighbor_map[aid]:
            if neighbor_id is None:
                feats.extend([0.0, 0.0])
            else:
                nobs = obs_dict[neighbor_id].astype(np.float32)
                total_q = float(np.sum(nobs[:12]))
                phase = float(nobs[12])
                feats.extend([total_q, phase])
        aug_vec = np.concatenate([own, np.array(feats, dtype=np.float32)], axis=0)
        aug[aid] = aug_vec
    return aug
