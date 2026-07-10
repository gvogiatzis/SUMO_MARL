#!/usr/bin/env python3
"""Spawn training/eval jobs against the DEPLOYED sumo-marl Modal app.

Unlike `modal run --detach modal_app.py::phase1` (ephemeral app per launch),
this targets the persistent deployment: jobs survive local exit, appear under
one app in the dashboard, and use the deployed image/retry config.

Deploy/update the app first:   modal deploy modal_app.py
Then e.g.:
  python launch.py train --methods gated --grids 3,5 --seeds 42,43,44 \
      --episodes 200 --regime mixed --tag phase3 --extra-args "--lambda-mem 0.003"
  python launch.py script --script analyze_gates.py \
      --args "--grid-n 3 --seed 42 --logs-base /results/phase3 --out-dir /results/phase3/gates"
"""
from __future__ import annotations
import argparse

import modal

from modal_app import PHASE1_METHODS

APP_NAME = "sumo-marl"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Spawn a training matrix")
    t.add_argument("--methods", default="gated")
    t.add_argument("--grids", default="3")
    t.add_argument("--seeds", default="42,43,44")
    t.add_argument("--episodes", type=int, default=200)
    t.add_argument("--regime", default="")
    t.add_argument("--extra-args", default="")
    t.add_argument("--tag", default="phase3")

    s = sub.add_parser("script", help="Spawn a single script job")
    s.add_argument("--script", required=True)
    s.add_argument("--args", default="")

    args = ap.parse_args()

    if args.cmd == "script":
        fn = modal.Function.from_name(APP_NAME, "run_script")
        call = fn.spawn(args.script, args.args)
        print(f"spawned {args.script} -> {call.object_id}")
        return

    fn = modal.Function.from_name(APP_NAME, "train")
    n = 0
    for grid_n in [int(g) for g in args.grids.split(",")]:
        for seed in [int(x) for x in args.seeds.split(",")]:
            for m in args.methods.split(","):
                script, margs = PHASE1_METHODS[m.strip()]
                a = f"--grid-n {grid_n} --seed {seed} --episodes {args.episodes} --device cpu"
                if margs:
                    a += f" {margs}"
                if args.regime:
                    a += f" --regime {args.regime}"
                if args.extra_args:
                    a += f" {args.extra_args}"
                call = fn.spawn(script, a, run_tag=f"{args.tag}/{m}_grid{grid_n}_seed{seed}",
                                logdir=f"/results/{args.tag}/logs")
                print(f"spawned {m} grid={grid_n} seed={seed} -> {call.object_id}")
                n += 1
    print(f"{n} jobs spawned against deployed app '{APP_NAME}'")


if __name__ == "__main__":
    main()
