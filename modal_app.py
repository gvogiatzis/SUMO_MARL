#!/usr/bin/env python3
"""
Modal orchestration: run training jobs in the cloud, one container per
(script, grid, regime, seed) combination, results persisted to a Modal volume.

Usage:
  # Single run (smoke test)
  modal run modal_app.py --script train_dqn_mlp.py --grid-n 3 --episodes 6 \
      --extra-args "--eval-every 2 --episode-steps 100"

  # Regime-conditioned run
  modal run modal_app.py --script train_drqn_lstm.py --grid-n 3 --regime platoons \
      --episodes 200 --extra-args "--batch-size-seq 16 --seq-len 8 --burn-in 4"

  # Fetch results (logs volume -> local ./results/)
  modal volume get sumo-marl-results / results/

SUMO rollouts are CPU-bound; default is CPU containers. Pass --gpu for the
recurrent/GNN variants when profiling shows the update step dominates.
"""
from __future__ import annotations
import shlex
import subprocess
import sys
from pathlib import Path

import modal

app = modal.App("sumo-marl")

REPO_LOCAL = Path(__file__).parent
REPO_REMOTE = "/root/SUMO_MARL"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # libsumo's native module links against X11/GL even when headless
    .apt_install(
        "libx11-6", "libxext6", "libxrender1", "libxft2", "libxt6", "libxi6",
        "libsm6", "libice6", "libgl1", "libglu1-mesa", "libfontconfig1", "libfreetype6",
    )
    .pip_install(
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "traci",
        "sumolib",
        "libsumo",
        "eclipse-sumo",
    )
    .env({"SUMO_MARL_BACKEND": "libsumo"})
    .add_local_dir(
        REPO_LOCAL,
        remote_path=REPO_REMOTE,
        ignore=[".venv", ".git", "__pycache__", "*.pyc", "logs*", "results"],
    )
)

results = modal.Volume.from_name("sumo-marl-results", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/results": results},
    timeout=60 * 60 * 12,
    cpu=4,
    memory=8192,
)
def train(script: str, args: str, run_tag: str, logdir: str = ""):
    """Run one training script; sync its logdir to the results volume."""
    logdir = logdir or f"/results/{run_tag}"
    cmd = [sys.executable, script, *shlex.split(args), "--logdir", logdir]
    print("[modal] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)
    results.commit()
    return run_tag


@app.function(
    image=image,
    volumes={"/results": results},
    timeout=60 * 60 * 12,
    cpu=4,
    memory=8192,
)
def run_script(script: str, args: str):
    """Run any repo script verbatim (evaluation, table generation, ...)."""
    cmd = [sys.executable, script, *shlex.split(args)]
    print("[modal] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_REMOTE)
    results.commit()


# Phase 1 baseline methods: script + method-specific args (README settings)
PHASE1_METHODS = {
    "dqn_mlp": ("train_dqn_mlp.py", ""),
    "drqn_lstm": ("train_drqn_lstm.py", "--batch-size-seq 16 --seq-len 8 --burn-in 4"),
    "dqn_gnn": ("train_dqn_gnn.py", ""),
    "drqn_gnn_lstm": ("train_drqn_gnn_lstm.py", "--batch-size-seq 16 --seq-len 8 --burn-in 4"),
    "colight": ("train_colight.py", ""),
}


@app.local_entrypoint()
def phase1(
    grids: str = "3",
    seeds: str = "42,43,44",
    episodes: int = 200,
    methods: str = "dqn_mlp,drqn_lstm,dqn_gnn,drqn_gnn_lstm",
):
    """Spawn the Phase 1 training matrix, detached. Run with `modal run --detach`.

    Checkpoints/KPIs land on the volume under phase1/logs_grid_{N}/seed{S}/,
    matching the layout evaluate_fixed_trips_all.py expects
    (--logs-base <fetched>/phase1).
    """
    calls = []
    for grid_n in [int(g) for g in grids.split(",")]:
        for seed in [int(s) for s in seeds.split(",")]:
            for m in methods.split(","):
                script, margs = PHASE1_METHODS[m.strip()]
                args = f"--grid-n {grid_n} --seed {seed} --episodes {episodes} --device cpu"
                if margs:
                    args += f" {margs}"
                # logdir is a prefix: scripts append _grid_{N}/seed{S}/
                handle = train.spawn(
                    script, args,
                    run_tag=f"phase1/{m}_grid{grid_n}_seed{seed}",
                    logdir="/results/phase1/logs",
                )
                calls.append((m, grid_n, seed, handle.object_id))
                print(f"spawned {m} grid={grid_n} seed={seed} -> {handle.object_id}")
    print(f"\n{len(calls)} runs spawned. Track: modal app list / modal volume ls sumo-marl-results phase1")


@app.local_entrypoint()
def main(
    script: str = "train_dqn_mlp.py",
    grid_n: int = 3,
    seed: int = 42,
    episodes: int = 200,
    regime: str = "",
    regime_intensity: float = 1.0,
    extra_args: str = "",
    device: str = "cpu",
):
    args = f"--grid-n {grid_n} --seed {seed} --episodes {episodes} --device {device}"
    tag_regime = regime if regime else "random"
    if regime:
        args += f" --regime {regime} --regime-intensity {regime_intensity}"
    if extra_args:
        args += f" {extra_args}"
    run_tag = f"{Path(script).stem}_grid{grid_n}_{tag_regime}_seed{seed}"
    print(f"[modal] launching {run_tag}")
    print(train.remote(script, args, run_tag))
