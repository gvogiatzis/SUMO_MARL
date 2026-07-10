#!/usr/bin/env python3
"""
Demand-regime flow generation, shared by evaluation trip files and
regime-conditioned *training* (env resets with a fresh seed each episode).

Wraps the case builders in make_eval_trips_fixed_demand_complex.py with the
per-grid presets applied, plus a global `intensity` multiplier on all flow
rates (used for intensity sweeps and held-out-demand generalisation).

Regime names: "corridor", "cross", "platoons", "bursty".
"""
from __future__ import annotations
from typing import Dict, List

import numpy as np

from make_eval_trips_fixed_demand_complex import (
    GRID_SPATIAL_PRESETS,
    GRID_TEMPORAL_PRESETS,
    case_bursty_on_off,
    case_cross_orthogonal_axes,
    case_heavy_corridor_ew_arterial,
    case_platoons_pulse_trains,
    get_nearest_preset,
    write_case,
)

REGIMES = ("corridor", "cross", "platoons", "bursty")


def _merged_preset(grid_n: int) -> Dict:
    preset = dict(get_nearest_preset(grid_n, GRID_TEMPORAL_PRESETS, "temporal"))
    preset.update(get_nearest_preset(grid_n, GRID_SPATIAL_PRESETS, "spatial"))
    return preset


def generate_regime_flows(
    regime: str,
    grid_n: int,
    sim_end: float,
    rng: np.random.Generator,
    intensity: float = 1.0,
) -> List[dict]:
    """Flow dicts (fid/begin/end/vph/src/dst) for one episode of `regime`."""
    p = _merged_preset(grid_n)

    if regime == "platoons":
        flows = case_platoons_pulse_trains(
            grid_n, sim_end, rng,
            heavy_pulses_per_train=p["platoon_heavy_pulses_per_train"],
            light_pulses_per_train=p["platoon_light_pulses_per_train"],
            precursor_width_min=p["platoon_precursor_width_min"],
            precursor_width_max=p["platoon_precursor_width_max"],
            precursor_vph_min=p["platoon_precursor_vph_min"],
            precursor_vph_max=p["platoon_precursor_vph_max"],
            precursor_to_main_gap_min=p["platoon_precursor_to_main_gap_min"],
            precursor_to_main_gap_max=p["platoon_precursor_to_main_gap_max"],
            heavy_pulse_gap_min=p["platoon_heavy_pulse_gap_min"],
            heavy_pulse_gap_max=p["platoon_heavy_pulse_gap_max"],
            light_pulse_gap_min=p["platoon_light_pulse_gap_min"],
            light_pulse_gap_max=p["platoon_light_pulse_gap_max"],
            heavy_width_min=p["platoon_heavy_width_min"],
            heavy_width_max=p["platoon_heavy_width_max"],
            light_width_min=p["platoon_light_width_min"],
            light_width_max=p["platoon_light_width_max"],
            heavy_vph_min=p["platoon_heavy_vph_min"],
            heavy_vph_max=p["platoon_heavy_vph_max"],
            light_vph_min=p["platoon_light_vph_min"],
            light_vph_max=p["platoon_light_vph_max"],
            train_gap_min=p["platoon_train_gap_min"],
            train_gap_max=p["platoon_train_gap_max"],
            jitter_frac=p["platoon_jitter_frac"],
        )
    elif regime == "bursty":
        flows = case_bursty_on_off(
            grid_n, sim_end, rng,
            window_min=p["bursty_win_min"],
            window_max=p["bursty_win_max"],
            vph_on_min=p["bursty_on_min"],
            vph_on_max=p["bursty_on_max"],
            vph_off_min=p["bursty_off_min"],
            vph_off_max=p["bursty_off_max"],
            route_multiplier=p["bursty_route_mult"],
            on_block_min=p["bursty_on_block_min"],
            on_block_max=p["bursty_on_block_max"],
            off_block_min=p["bursty_off_block_min"],
            off_block_max=p["bursty_off_block_max"],
            p_on_dropout=p["bursty_p_on_dropout"],
            p_off_false_burst=p["bursty_p_off_false_burst"],
        )
    elif regime == "corridor":
        flows = case_heavy_corridor_ew_arterial(
            grid_n, sim_end, rng,
            corridor_vph=p["corridor_vph"],
            corridor_pulse_width_min=p["corridor_pulse_width_min"],
            corridor_pulse_width_max=p["corridor_pulse_width_max"],
            corridor_pulse_gap_min=p["corridor_pulse_gap_min"],
            corridor_pulse_gap_max=p["corridor_pulse_gap_max"],
            corridor_jitter_frac=p["corridor_jitter_frac"],
            side_vph=p["corridor_side_vph"],
        )
    elif regime == "cross":
        flows = case_cross_orthogonal_axes(
            grid_n, sim_end, rng,
            cross_centre_ew_vph=p["cross_centre_ew_vph"],
            cross_centre_ns_vph=p["cross_centre_ns_vph"],
            cross_adjacent_ew_vph=p["cross_adjacent_ew_vph"],
            cross_adjacent_ns_vph=p["cross_adjacent_ns_vph"],
            cross_width=p["cross_width"],
        )
    else:
        raise ValueError(f"Unknown regime '{regime}'; expected one of {REGIMES}")

    if intensity != 1.0:
        for f in flows:
            f["vph"] = max(1, int(round(f["vph"] * float(intensity))))
    return flows


def write_regime_trips(
    path: str,
    regime: str,
    grid_n: int,
    sim_end: float,
    rng: np.random.Generator,
    intensity: float = 1.0,
    vehicle_sigma: float = 0.5,
) -> None:
    """Write one episode's regime flows as a SUMO trips/flows XML file."""
    flows = generate_regime_flows(regime, grid_n, sim_end, rng, intensity)
    write_case(path, flows, vehicle_sigma=vehicle_sigma)


def write_switching_trips(
    path: str,
    grid_n: int,
    sim_end: float,
    rng: np.random.Generator,
    segment_len_s: float,
    intensity: float = 1.0,
    vehicle_sigma: float = 0.5,
):
    """Within-episode regime switching: the episode is partitioned into
    consecutive segments of ~segment_len_s, each with demand from one regime
    (uniformly sampled, no immediate repeats). Returns the schedule as a list
    of (t_start_s, t_end_s, regime) for gate-analysis alignment."""
    flows: List[dict] = []
    schedule: List[tuple] = []
    t0 = 0.0
    prev = None
    while t0 < sim_end:
        t1 = min(sim_end, t0 + segment_len_s)
        choices = [r for r in REGIMES if r != prev]
        regime = choices[int(rng.integers(0, len(choices)))]
        seg_flows = generate_regime_flows(regime, grid_n, t1 - t0, rng, intensity)
        for f in seg_flows:
            f = dict(f)
            f["fid"] = f"seg{len(schedule)}_{f['fid']}"
            f["begin"] = float(f["begin"]) + t0
            f["end"] = min(float(f["end"]) + t0, t1)
            if f["end"] > f["begin"]:
                flows.append(f)
        schedule.append((t0, t1, regime))
        prev = regime
        t0 = t1
    write_case(path, flows, vehicle_sigma=vehicle_sigma)
    return schedule
