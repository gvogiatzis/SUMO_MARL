#!/usr/bin/env python3
"""Generate frozen switching-schedule evaluation trip files.

Each case is one write_switching_trips draw with a fixed seed; the segment
schedule for every case is stored in schedules.json alongside the XMLs so
gate analyses can align activations with regime bands.

Example (matched to the phase2switch2 training config):
  python make_eval_trips_switching.py --grid-n 3 --episode-steps 450 \
      --sumo-steps-per-env-step 5 --segment-steps 150 --intensity 3.0 --n-cases 10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from marl_utils.demand_regimes import write_switching_trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-n", type=int, default=3)
    ap.add_argument("--episode-steps", type=int, default=450)
    ap.add_argument("--sumo-steps-per-env-step", type=int, default=5)
    ap.add_argument("--step-length", type=float, default=1.0)
    ap.add_argument("--segment-steps", type=int, default=150)
    ap.add_argument("--intensity", type=float, default=3.0)
    ap.add_argument("--intensity-map", type=str, default=None,
                    help='Per-regime overrides, e.g. "cross:1.5,platoons:9"')
    ap.add_argument("--n-cases", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=7000)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Default: eval_trips_switching_grid_{N}")
    args = ap.parse_args()

    out = Path(args.out_dir or f"eval_trips_switching_grid_{args.grid_n}")
    out.mkdir(parents=True, exist_ok=True)

    sim_end = args.episode_steps * args.sumo_steps_per_env_step * args.step_length
    seg_len = args.segment_steps * args.sumo_steps_per_env_step * args.step_length

    schedules = {}
    for i in range(args.n_cases):
        rng = np.random.default_rng(args.base_seed + i)
        fname = f"case_{i:02d}.xml"
        imap = None
        if args.intensity_map:
            imap = {k.strip(): float(v) for k, v in (kv.split(":") for kv in args.intensity_map.split(","))}
        schedule = write_switching_trips(
            str(out / fname),
            grid_n=args.grid_n,
            sim_end=sim_end,
            rng=rng,
            segment_len_s=seg_len,
            intensity=args.intensity,
            intensity_map=imap,
        )
        schedules[fname] = [(float(a), float(b), r) for a, b, r in schedule]
        print(f"{fname}: {[r for _, _, r in schedule]}")

    meta = {
        "grid_n": args.grid_n,
        "episode_steps": args.episode_steps,
        "sumo_steps_per_env_step": args.sumo_steps_per_env_step,
        "segment_steps": args.segment_steps,
        "intensity": args.intensity,
        "intensity_map": args.intensity_map,
        "base_seed": args.base_seed,
        "schedules": schedules,
    }
    (out / "schedules.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.n_cases} cases + schedules.json to {out}/")


if __name__ == "__main__":
    main()
