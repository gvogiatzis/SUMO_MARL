#!/usr/bin/env python3
"""Training-state checkpointing so preempted cloud runs resume instead of
restarting (Modal workers are preemptible; see research_log.md 2026-07-10).

Saves online/target nets, optimizer, replay buffer, epsilon, best eval return
and episode counter every `every` episodes; resume() restores them in place.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class TrainingCheckpoint:
    def __init__(self, logdir_prefix: str, grid_n: int, seed: int, run_name: str, every: int = 20):
        d = Path(f"{logdir_prefix}_grid_{grid_n}") / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        self.path = d / f"ckpt_{run_name}_seed{seed}.pt"
        self.every = int(every)

    def resume(self, online_q, target_q, optimizer, replay) -> dict:
        """If a checkpoint exists, restore state in place and return its
        counters; otherwise return fresh-start defaults."""
        fresh = {"start_episode": 1, "eps": None, "best": -np.inf, "resumed": False, "extra": {}}
        if self.every <= 0 or not self.path.exists():
            return fresh
        try:
            state = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ckpt] failed to load {self.path} ({e}); starting fresh")
            return fresh
        online_q.load_state_dict(state["online"])
        target_q.load_state_dict(state["target"])
        optimizer.load_state_dict(state["optimizer"])
        if replay is not None and state.get("replay") is not None:
            replay.__dict__.update(state["replay"].__dict__)
        print(f"[ckpt] resumed from episode {state['episode']} ({self.path.name})")
        return {
            "start_episode": int(state["episode"]) + 1,
            "eps": float(state["eps"]),
            "best": float(state["best"]),
            "resumed": True,
            "extra": state.get("extra") or {},
        }

    def maybe_save(self, episode: int, online_q, target_q, optimizer, replay,
                   eps: float, best: float, extra: Optional[dict] = None):
        if self.every <= 0 or episode % self.every != 0:
            return
        tmp = self.path.with_suffix(".tmp")
        torch.save({
            "episode": int(episode),
            "eps": float(eps),
            "best": float(best),
            "online": online_q.state_dict(),
            "target": target_q.state_dict(),
            "optimizer": optimizer.state_dict(),
            "replay": replay,
            "extra": extra or {},
        }, tmp)
        tmp.replace(self.path)
