#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv

# Must match the env's backend (libsumo by default) — a plain `import traci`
# here would silently query a connectionless socket module.
from marl_utils.sumo_backend import get_backend
traci, SUMO_BACKEND = get_backend()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

NAMED_TRIP_TYPES = [
    "platoons",
    "bursty",
    "shifted",
    "ampm",
    "incident",
    "corridor",
    "cross",
]

TRIP_TYPE_CHOICES = ["all"] + NAMED_TRIP_TYPES


def parse_args():
    p = argparse.ArgumentParser(
        "Evaluate fixed-time, max-pressure, and longest-queue baselines on fixed-demand trip files."
    )

    # Fixed demand input/output
    p.add_argument(
        "--grid-n",
        type=str,
        nargs="+",
        default=["3"],
        help=(
            "Grid size(s) to evaluate. Examples: "
            "--grid-n 3, --grid-n 3 5 10, --grid-n 3,5,10, or --grid-n all."
        ),
    )
    p.add_argument(
        "--trip-type",
        type=str,
        default="all",
        choices=TRIP_TYPE_CHOICES,
        help=(
            "Which fixed trip type to evaluate. "
            "'all' evaluates all named trip types."
        ),
    )
    p.add_argument(
        "--trips-root",
        type=str,
        default=".",
        help=(
            "Root directory containing eval_trips_temporal_grid_N/ and "
            "eval_trips_spatial_grid_N/."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="eval_fixed_demand_type",
        help="Directory where aggregate CSV files are saved.",
    )

    # Environment settings
    p.add_argument("--steps", type=int, default=200, help="Episode steps in environment steps.")
    p.add_argument("--sumo-steps-per-env-step", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--duarouter-seed", type=int, default=2025)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--gui-delay-ms", type=int, default=0)
    p.add_argument("--verbose", action="store_true")

    # Controller settings
    p.add_argument(
        "--baseline-controller",
        type=str,
        default="all",
        choices=["fixed_time", "max_pressure", "longest_queue", "both", "all"],
        help="Controller to evaluate. 'both' and 'all' evaluate all three controllers.",
    )
    p.add_argument(
        "--fixed-time-hold-steps",
        type=int,
        default=4,
        help="Number of env steps to hold each selected phase. Used only by fixed_time.",
    )
    p.add_argument(
        "--fixed-time-cycle",
        type=str,
        default="0,1,2,3",
        help="Comma-separated fixed-time phase cycle. 0=NS-left, 1=NS-through/right, 2=EW-left, 3=EW-through/right.",
    )
    p.add_argument(
        "--tie-tolerance",
        type=float,
        default=1e-6,
        help="Near-tie tolerance for greedy controllers. If best score is not better than current score by this amount, keep current phase.",
    )

    return p.parse_args()


def parse_grid_ns(grid_args: List[str]) -> List[int]:
    """
    Parse --grid-n.

    Accepted forms:
        --grid-n 3
        --grid-n 3 5 10
        --grid-n 3,5,10
        --grid-n all
    """
    if not grid_args:
        return [3]

    tokens: List[str] = []
    for item in grid_args:
        tokens.extend([x.strip() for x in str(item).split(",") if x.strip()])

    if any(t.lower() == "all" for t in tokens):
        return [3, 5, 10]

    grid_ns: List[int] = []
    for t in tokens:
        try:
            n = int(t)
        except ValueError as e:
            raise ValueError(f"Invalid --grid-n value: {t}") from e

        if n <= 0:
            raise ValueError(f"--grid-n must be positive, got {n}")

        if n not in grid_ns:
            grid_ns.append(n)

    if not grid_ns:
        raise ValueError("No valid grid size specified.")

    return grid_ns


def parse_fixed_time_cycle(cycle_str: str, action_dim: int) -> List[int]:
    if cycle_str is None or str(cycle_str).strip() == "":
        return list(range(action_dim))

    cycle = [int(x.strip()) for x in str(cycle_str).split(",") if x.strip() != ""]
    if not cycle:
        raise ValueError("Fixed-time cycle cannot be empty.")

    for action in cycle:
        if action < 0 or action >= action_dim:
            raise ValueError(f"Invalid fixed-time action {action}. Valid range is [0, {action_dim - 1}].")

    return cycle


def controller_names(selection: str) -> List[str]:
    if selection in ("both", "all"):
        return ["fixed_time", "max_pressure", "longest_queue"]
    return [selection]


# -----------------------------------------------------------------------------
# Trip resolver by type
# -----------------------------------------------------------------------------

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p.resolve()
    return None


def resolve_trips_by_type(trips_root: Path, grid_n: int, trip_type: str) -> Path:
    """
    Resolve a named fixed-demand trip type to a single XML file.

    Expected folder layout:

        trips_root/
          eval_trips_temporal_grid_N/
            eval_trips_t01_platoons_pulse_trains_gridN<N>.xml
            eval_trips_t02_bursty_on_off_gridN<N>.xml
            eval_trips_t03_shifted_directional_peaks_gridN<N>.xml
            eval_trips_t04_ampm_global_gridN<N>.xml
            eval_trips_t05_incident_west_boundary_gridN<N>.xml

          eval_trips_spatial_grid_N/
            eval_trips_s01_heavy_corridor_ew_arterial_gridN<N>.xml
            eval_trips_s02_cross_orthogonal_axes_gridN<N>.xml
    """
    t = trip_type.lower()

    if t == "all":
        raise ValueError("resolve_trips_by_type() should not be called with trip_type='all'.")

    temporal_dir = trips_root / f"eval_trips_temporal_grid_{grid_n}"
    spatial_dir = trips_root / f"eval_trips_spatial_grid_{grid_n}"

    if t == "platoons":
        preferred = [
            temporal_dir / f"eval_trips_t01_platoons_pulse_trains_gridN{grid_n}.xml",
        ]
        glob_keys = ["platoons_pulse_trains", "platoons", "pulse_trains"]

    elif t == "bursty":
        preferred = [
            temporal_dir / f"eval_trips_t02_bursty_on_off_gridN{grid_n}.xml",
        ]
        glob_keys = ["bursty_on_off", "bursty"]

    elif t == "shifted":
        preferred = [
            temporal_dir / f"eval_trips_t03_shifted_directional_peaks_gridN{grid_n}.xml",
        ]
        glob_keys = ["shifted_directional_peaks", "shifted_directional", "shifted"]

    elif t == "ampm":
        preferred = [
            temporal_dir / f"eval_trips_t04_ampm_global_gridN{grid_n}.xml",
        ]
        glob_keys = ["ampm_global", "ampm"]

    elif t == "incident":
        preferred = [
            temporal_dir / f"eval_trips_t05_incident_west_boundary_gridN{grid_n}.xml",
        ]
        glob_keys = ["incident_west_boundary", "incident"]

    elif t == "corridor":
        preferred = [
            spatial_dir / f"eval_trips_s01_heavy_corridor_ew_arterial_gridN{grid_n}.xml",
        ]
        glob_keys = ["heavy_corridor_ew_arterial", "heavy_corridor", "corridor"]

    elif t == "cross":
        preferred = [
            spatial_dir / f"eval_trips_s02_cross_orthogonal_axes_gridN{grid_n}.xml",
        ]
        glob_keys = ["cross_orthogonal_axes", "orthogonal_axes", "cross"]

    else:
        raise ValueError(f"Unknown trip type: {trip_type}")

    p = _first_existing(preferred)
    if p is not None:
        return p

    matches: List[str] = []
    for key in glob_keys:
        pattern = str(trips_root / f"**/*{key}*gridN{grid_n}.xml")
        matches.extend(glob.glob(pattern, recursive=True))

    matches = sorted(set(matches))
    if matches:
        return Path(matches[0]).resolve()

    raise FileNotFoundError(
        f"Trips file not found for trip_type='{trip_type}', grid_n={grid_n}, trips_root={trips_root}"
    )


def collect_trip_specs_for_grid(args, grid_n: int) -> List[Tuple[str, Path]]:
    """
    Return a list of (trip_type, trip_path) for one grid size.

    If --trip-type all:
        evaluate all named trip types that can be found.

    If --trip-type <name>:
        evaluate only that trip type.
    """
    trips_root = Path(args.trips_root).expanduser().resolve()

    if args.trip_type != "all":
        t = str(args.trip_type).lower()
        p = resolve_trips_by_type(trips_root, grid_n, t)
        return [(t, p)]

    specs: List[Tuple[str, Path]] = []
    for t in NAMED_TRIP_TYPES:
        try:
            p = resolve_trips_by_type(trips_root, grid_n, t)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping grid {grid_n}, trip type '{t}': {e}")
            continue

        specs.append((t, p))

    return specs


def safe_filename_part(s: str) -> str:
    return (
        str(s)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


# -----------------------------------------------------------------------------
# Controllers
# -----------------------------------------------------------------------------

class FixedTimeController:
    """
    Fixed-time controller for the 4-phase traffic signal environment.

    Action mapping:
        0 = NS left
        1 = NS through + right
        2 = EW left
        3 = EW through + right
    """

    def __init__(self, agent_ids: List[str], action_dim: int, hold_steps: int, cycle: List[int]):
        self.agent_ids = list(agent_ids)
        self.action_dim = int(action_dim)
        self.hold_steps = max(1, int(hold_steps))
        self.cycle = [int(a) for a in cycle]
        self.t = 0

    def reset(self, env=None):
        self.t = 0

    def act(self, obs_dict: Dict[str, np.ndarray], env=None) -> Dict[str, int]:
        phase_pos = (self.t // self.hold_steps) % len(self.cycle)
        action = int(self.cycle[phase_pos])
        self.t += 1
        return {aid: action for aid in self.agent_ids}


class MaxPressureController:
    """
    Max-pressure controller using TraCI controlled links.

    Four phases:
        0 = NS left
        1 = NS through + right
        2 = EW left
        3 = EW through + right

    Score:
        pressure = sum(max(q_in - q_out, 0))

    Empty phases with no upstream queue are not selected.
    Phase-score normalisation is deliberately disabled.
    """

    def __init__(
        self,
        agent_ids: List[str],
        action_dim: int,
        hold_steps: int = 1,
        tie_tolerance: float = 1e-6,
    ):
        if action_dim != 4:
            raise ValueError(f"MaxPressureController expects action_dim=4, got {action_dim}.")

        self.agent_ids = list(agent_ids)
        self.action_dim = int(action_dim)
        self.hold_steps = max(1, int(hold_steps))
        self.tie_tolerance = float(tie_tolerance)

        # Normalisation is fully disabled.
        self.normalise_phase_score = False

        self.phase_movements: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}
        self.incoming_lane_to_obs_idx: Dict[str, Dict[str, int]] = {}

        self._built_for_env_id: Optional[int] = None
        self.t = 0
        self.current_actions: Optional[Dict[str, int]] = None

    def reset(self, env=None):
        self.t = 0
        self.current_actions = None
        if env is not None:
            self._build_phase_movements(env)

    @staticmethod
    def _parse_grid_node_id(node_id: str) -> Tuple[int, int]:
        try:
            _, i, j = node_id.split("_")
            return int(i), int(j)
        except Exception:
            return -1, -1

    @staticmethod
    def _parse_lane_index(lane_id: str) -> Optional[int]:
        """
        Parse final lane index from a SUMO lane id.

        Example:
            e_1_0_to_1_1_0 -> 0
            e_1_0_to_1_1_1 -> 1
            e_1_0_to_1_1_2 -> 2

        In this environment:
            lane *_0 and *_1 = through/right lanes
            lane *_2         = left-turn lane
        """
        try:
            return int(str(lane_id).rsplit("_", 1)[1])
        except Exception:
            return None

    @classmethod
    def _phase_lane_compatible(cls, phase_id: int, in_lane: str) -> bool:
        """
        Enforce which physical lane can contribute to each phase.

        Phase mapping:
            0 = NS left              -> only lane *_2
            1 = NS through + right   -> only lanes *_0 and *_1
            2 = EW left              -> only lane *_2
            3 = EW through + right   -> only lanes *_0 and *_1
        """
        lane_idx = cls._parse_lane_index(in_lane)

        if lane_idx is None:
            return False

        if phase_id in (0, 2):
            return lane_idx == 2

        if phase_id in (1, 3):
            return lane_idx in (0, 1)

        return False

    @staticmethod
    def _lane_queue(lane_id: str) -> float:
        if traci is None:
            return 0.0
        if not lane_id or lane_id.startswith(":"):
            return 0.0
        try:
            return float(traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            return 0.0

    def _build_obs_lane_maps(self, env):
        """
        Build lane -> observation index maps from env.tls_infos.

        Expected observation:
            obs[0:12] = queues for info.incoming_lanes
            obs[12]   = current phase
        """
        self.incoming_lane_to_obs_idx = {}
        for info in env.tls_infos:
            lane_map: Dict[str, int] = {}
            for idx, lane_id in enumerate(info.incoming_lanes):
                lane_map[lane_id] = idx
            self.incoming_lane_to_obs_idx[info.tls_id] = lane_map

    def _classify_link(self, env, tls_id: str, in_lane: str, out_lane: str) -> Tuple[str, str]:
        net = env._net
        grid_n = env.grid_n
        i, j = self._parse_grid_node_id(tls_id)

        north_id = f"n_{i}_{j + 1}" if j + 1 < grid_n else f"fn_{i}"
        south_id = f"n_{i}_{j - 1}" if j - 1 >= 0 else f"fs_{i}"
        east_id = f"n_{i + 1}_{j}" if i + 1 < grid_n else f"fe_{j}"
        west_id = f"n_{i - 1}_{j}" if i - 1 >= 0 else f"fw_{j}"

        try:
            in_edge_id = in_lane.rsplit("_", 1)[0]
            out_edge_id = out_lane.rsplit("_", 1)[0]
            in_edge = net.getEdge(in_edge_id)
            out_edge = net.getEdge(out_edge_id)
            in_from = in_edge.getFromNode().getID()
            out_to = out_edge.getToNode().getID()
        except Exception:
            return "?", "?"

        if in_from == north_id:
            approach = "N"
            left_target = east_id
            through_target = south_id
            right_target = west_id
        elif in_from == south_id:
            approach = "S"
            left_target = west_id
            through_target = north_id
            right_target = east_id
        elif in_from == west_id:
            approach = "W"
            left_target = north_id
            through_target = east_id
            right_target = south_id
        elif in_from == east_id:
            approach = "E"
            left_target = south_id
            through_target = west_id
            right_target = north_id
        else:
            return "?", "?"

        if out_to == left_target:
            movement = "L"
        elif out_to == through_target:
            movement = "T"
        elif out_to == right_target:
            movement = "R"
        else:
            movement = "?"

        return approach, movement

    @staticmethod
    def _phase_allows(phase_id: int, approach: str, movement: str) -> bool:
        if phase_id == 0:
            return approach in ("N", "S") and movement == "L"
        if phase_id == 1:
            return approach in ("N", "S") and movement in ("T", "R")
        if phase_id == 2:
            return approach in ("E", "W") and movement == "L"
        if phase_id == 3:
            return approach in ("E", "W") and movement in ("T", "R")
        return False

    def _build_phase_movements(self, env):
        if traci is None:
            raise RuntimeError("traci is not available. Cannot run max-pressure baseline.")

        env_id = id(env)
        if self._built_for_env_id == env_id and self.phase_movements:
            return

        self._build_obs_lane_maps(env)
        self.phase_movements = {}

        for tls_id in self.agent_ids:
            phase_map: Dict[int, List[Tuple[str, str]]] = {0: [], 1: [], 2: [], 3: []}
            seen = set()

            try:
                groups = traci.trafficlight.getControlledLinks(tls_id)
            except Exception:
                groups = []

            for group in groups:
                links = group if isinstance(group, (list, tuple)) else [group]
                for link in links:
                    if not link or len(link) < 2:
                        continue

                    in_lane = link[0]
                    out_lane = link[1]
                    if not in_lane or in_lane.startswith(":"):
                        continue
                    if not out_lane or out_lane.startswith(":"):
                        continue

                    key = (in_lane, out_lane)
                    if key in seen:
                        continue
                    seen.add(key)

                    approach, movement = self._classify_link(env, tls_id, in_lane, out_lane)

                    for phase_id in range(4):
                        if not self._phase_allows(phase_id, approach, movement):
                            continue

                        # Critical lane-scoring fix:
                        # Phase 1 and phase 3 are through/right phases and must
                        # not count left-turn lanes ending in *_2.
                        #
                        # Phase 0 and phase 2 are left-turn phases and should
                        # only count left-turn lanes ending in *_2.
                        if not self._phase_lane_compatible(phase_id, in_lane):
                            continue

                        phase_map[phase_id].append((in_lane, out_lane))

            self.phase_movements[tls_id] = phase_map

        self._built_for_env_id = env_id

    def _phase_pressure(self, tls_id: str, phase_id: int) -> float:
        movements = self.phase_movements.get(tls_id, {}).get(phase_id, [])
        if not movements:
            return -1e9

        pressure = 0.0
        total_in_queue = 0.0

        for in_lane, out_lane in movements:
            q_in = self._lane_queue(in_lane)
            q_out = self._lane_queue(out_lane)

            total_in_queue += q_in
            pressure += max(q_in - q_out, 0.0)

        # Empty phases should not be selected.
        if total_in_queue <= 0.0:
            return -1e9

        # If all movement pressures are clipped to zero but there is still demand,
        # fall back to a small queue-pressure term.
        if pressure <= 0.0:
            pressure = 1e-3 * total_in_queue

        return float(pressure)

    def _scores_for_tls(self, tls_id: str, obs_vec: Optional[np.ndarray] = None) -> List[float]:
        return [self._phase_pressure(tls_id, phase_id) for phase_id in range(4)]

    def _choose_with_current_phase_tie_break(self, tls_id: str, scores: List[float]) -> int:
        scores_np = np.asarray(scores, dtype=np.float32)
        best_action = int(np.argmax(scores_np))
        best_score = float(scores_np[best_action])

        if self.current_actions is not None and tls_id in self.current_actions:
            current_action = int(self.current_actions[tls_id])
            current_score = float(scores_np[current_action])
            if best_score <= current_score + self.tie_tolerance:
                return current_action

        return best_action

    def _select_actions(self, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, int]:
        actions: Dict[str, int] = {}
        for tls_id in self.agent_ids:
            obs_vec = obs_dict[tls_id] if obs_dict is not None and tls_id in obs_dict else None
            scores = self._scores_for_tls(tls_id, obs_vec)
            actions[tls_id] = self._choose_with_current_phase_tie_break(tls_id, scores)
        return actions

    def act(self, obs_dict: Dict[str, np.ndarray], env=None) -> Dict[str, int]:
        if env is not None:
            self._build_phase_movements(env)

        if self.current_actions is not None and (self.t % self.hold_steps) != 0:
            self.t += 1
            return dict(self.current_actions)

        actions = self._select_actions(obs_dict)
        self.current_actions = dict(actions)
        self.t += 1
        return actions


class LongestQueueController(MaxPressureController):
    """
    Longest-queue-first controller.

    Uses the same four phase groups as MaxPressureController:
        0 = NS left
        1 = NS through + right
        2 = EW left
        3 = EW through + right

    Score:
        score = sum(q_in)

    Phase-score normalisation is deliberately disabled.
    """

    def _phase_queue_score_from_obs(
        self,
        tls_id: str,
        phase_id: int,
        obs_vec: Optional[np.ndarray],
    ) -> float:
        movements = self.phase_movements.get(tls_id, {}).get(phase_id, [])
        if not movements:
            return -1e9

        incoming_lanes = sorted({in_lane for in_lane, _ in movements})

        if obs_vec is None:
            return float(sum(self._lane_queue(in_lane) for in_lane in incoming_lanes))

        lane_to_idx = self.incoming_lane_to_obs_idx.get(tls_id, {})
        total = 0.0
        used = 0

        for in_lane in incoming_lanes:
            idx = lane_to_idx.get(in_lane, None)
            if idx is None:
                continue
            if 0 <= idx < min(12, len(obs_vec)):
                total += float(obs_vec[idx])
                used += 1

        if used == 0:
            return -1e9

        return float(total)

    def _scores_for_tls(self, tls_id: str, obs_vec: Optional[np.ndarray] = None) -> List[float]:
        return [self._phase_queue_score_from_obs(tls_id, phase_id, obs_vec) for phase_id in range(4)]


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

class PhaseUseStats:
    def __init__(self, agent_ids: List[str], action_dim: int):
        self.agent_ids = list(agent_ids)
        self.action_dim = int(action_dim)
        self.phase_counts = {aid: np.zeros(action_dim, dtype=np.int64) for aid in self.agent_ids}
        self.switch_counts = {aid: 0 for aid in self.agent_ids}
        self.last_actions = {aid: None for aid in self.agent_ids}
        self.num_steps = 0

    def update(self, actions: Dict[str, int]):
        self.num_steps += 1
        for aid, action in actions.items():
            if aid not in self.phase_counts:
                continue
            a = int(action)
            if 0 <= a < self.action_dim:
                self.phase_counts[aid][a] += 1
            prev = self.last_actions.get(aid, None)
            if prev is not None and int(prev) != a:
                self.switch_counts[aid] += 1
            self.last_actions[aid] = a

    def average_phase_counts(self) -> np.ndarray:
        if not self.phase_counts:
            return np.zeros(self.action_dim, dtype=np.float32)
        return np.mean(np.stack(list(self.phase_counts.values()), axis=0), axis=0)

    def average_phase_fraction(self) -> np.ndarray:
        counts = self.average_phase_counts()
        denom = max(1.0, float(np.sum(counts)))
        return counts / denom

    def average_switches_per_tls(self) -> float:
        if not self.switch_counts:
            return 0.0
        return float(np.mean(list(self.switch_counts.values())))


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def make_env_for_trip(args, grid_n: int, trip_path: Path) -> SumoGridMARLFixedEnv:
    return SumoGridMARLFixedEnv(
        grid_n=grid_n,
        episode_steps=args.steps,
        sumo_steps_per_env_step=args.sumo_steps_per_env_step,
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        seed=args.seed,
        verbose=args.verbose,
        suppress_sumo_output=not args.verbose,
        fixed_trips_file=str(trip_path),
        duarouter_seed=args.duarouter_seed,
    )


def controller_hold_steps(name: str, args) -> int:
    """
    Only fixed_time uses --fixed-time-hold-steps.

    Greedy controllers, max_pressure and longest_queue, recompute their action
    every environment step.
    """
    if name == "fixed_time":
        return max(1, int(args.fixed_time_hold_steps))

    if name in ("max_pressure", "longest_queue"):
        return 1

    raise ValueError(f"Unknown controller: {name}")


def build_controller(
    name: str,
    args,
    agent_ids: List[str],
    action_dim: int,
    cycle: List[int],
):
    hold_steps = controller_hold_steps(name, args)

    if name == "fixed_time":
        return FixedTimeController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,
            cycle=cycle,
        )

    if name == "max_pressure":
        return MaxPressureController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,  # always 1
            tie_tolerance=args.tie_tolerance,
        )

    if name == "longest_queue":
        return LongestQueueController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,  # always 1
            tie_tolerance=args.tie_tolerance,
        )

    raise ValueError(f"Unknown controller: {name}")


def evaluate_one_trip_one_controller(
    args,
    grid_n: int,
    trip_type: str,
    trip_path: Path,
    controller_name: str,
) -> Dict[str, object]:
    env = make_env_for_trip(args, grid_n, trip_path)

    try:
        obs = env.reset()
        agent_ids = list(env.agent_ids)
        if not agent_ids:
            raise RuntimeError("No traffic lights discovered in SUMO. Check network/trips.")

        action_dim = int(env.action_spaces[agent_ids[0]].n)
        if action_dim != 4:
            raise RuntimeError(f"This script expects action_dim=4, got {action_dim}.")

        cycle = parse_fixed_time_cycle(args.fixed_time_cycle, action_dim)
        controller = build_controller(controller_name, args, agent_ids, action_dim, cycle)
        controller.reset(env=env)
        stats = PhaseUseStats(agent_ids=agent_ids, action_dim=action_dim)

        episode_return = 0.0
        final_kpis: Optional[Dict] = None
        steps_run = 0
        done = False

        while not done and steps_run < env.episode_steps:
            actions = controller.act(obs, env=env)
            stats.update(actions)

            obs, rewards, done, info = env.step(actions)
            episode_return += float(np.sum(list(rewards.values())))
            final_kpis = info.get("network_kpis", None)
            steps_run += 1

        if final_kpis is None:
            final_kpis = {}

        phase_frac = stats.average_phase_fraction()

        row: Dict[str, object] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "grid_n": int(grid_n),
            "trip_type": str(trip_type),
            "controller": controller_name,
            "trip_file": str(trip_path),
            "trip_name": trip_path.name,
            "seed": int(args.seed),
            "duarouter_seed": int(args.duarouter_seed),
            "episode_steps": int(args.steps),
            "steps_run": int(steps_run),
            "sumo_steps_per_env_step": int(args.sumo_steps_per_env_step),
            "fixed_time_hold_steps": int(args.fixed_time_hold_steps),
            "controller_hold_steps": int(controller_hold_steps(controller_name, args)),
            "fixed_time_cycle": str(args.fixed_time_cycle),
            "tie_tolerance": float(args.tie_tolerance),
            "phase_score_normalised": False,
            "episode_return": float(episode_return),
            "completed_vehicles": float(final_kpis.get("completed_vehicles", 0.0)),
            "active_vehicles": float(final_kpis.get("active_vehicles", 0.0)),
            "total_vehicles_seen_or_active": float(final_kpis.get("total_vehicles_seen_or_active", 0.0)),
            "throughput_veh_per_hour": float(final_kpis.get("throughput_veh_per_hour", 0.0)),
            "mean_travel_time_s": float(final_kpis.get("mean_travel_time_s", 0.0)),
            "mean_waiting_time_s": float(final_kpis.get("mean_waiting_time_s", 0.0)),
            "mean_travel_time_s_completed": float(final_kpis.get("mean_travel_time_s_completed", 0.0)),
            "mean_waiting_time_s_completed": float(final_kpis.get("mean_waiting_time_s_completed", 0.0)),
            "phase_frac_0": float(phase_frac[0]),
            "phase_frac_1": float(phase_frac[1]),
            "phase_frac_2": float(phase_frac[2]),
            "phase_frac_3": float(phase_frac[3]),
            "avg_switches_per_tls": float(stats.average_switches_per_tls()),
            "num_tls": int(len(agent_ids)),
        }

        print(
            f"[{datetime.now()}] grid={grid_n:<2d} | {trip_type:9s} | {controller_name:13s} | "
            f"return={row['episode_return']:.2f} | "
            f"completed={row['completed_vehicles']:.2f} | "
            f"active={row['active_vehicles']:.2f} | "
            f"throughput={row['throughput_veh_per_hour']:.2f} veh/h | "
            f"mean_travel={row['mean_travel_time_s']:.2f}s | "
            f"mean_wait={row['mean_waiting_time_s']:.2f}s | "
            f"phase_frac=[{row['phase_frac_0']:.3f}, {row['phase_frac_1']:.3f}, "
            f"{row['phase_frac_2']:.3f}, {row['phase_frac_3']:.3f}] | "
            f"avg_switches_per_tls={row['avg_switches_per_tls']:.2f}"
        )

        return row

    finally:
        try:
            env.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# CSV writer
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_evaluation(args):
    grid_ns = parse_grid_ns(args.grid_n)
    selected_controllers = controller_names(args.baseline_controller)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Fixed-demand baseline evaluation ===")
    print(f"Grid sizes            : {grid_ns}")
    print(f"Trip type             : {args.trip_type}")
    print(f"Trips root            : {Path(args.trips_root).expanduser().resolve()}")
    print(f"Output directory      : {out_dir}")
    print(f"Controllers           : {', '.join(selected_controllers)}")
    print(f"Episode steps         : {args.steps}")
    print(f"SUMO steps/env step   : {args.sumo_steps_per_env_step}")
    print(f"Fixed-time hold steps : {args.fixed_time_hold_steps}")
    print("Greedy hold steps     : 1")
    print("Normalise phase score : False")
    print("")

    rows_by_trip_type: Dict[str, List[Dict[str, object]]] = {}

    for grid_n in grid_ns:
        print(f"\n==============================")
        print(f"=== Grid {grid_n}x{grid_n}")
        print(f"==============================")

        trip_specs = collect_trip_specs_for_grid(args, grid_n)
        if not trip_specs:
            print(f"[WARN] No trip files found for grid {grid_n}. Skipping.")
            continue

        for trip_type, trip_path in trip_specs:
            print(f"\n=== Grid {grid_n} | Trip type: {trip_type} | File: {trip_path.name} ===")

            for controller_name in selected_controllers:
                try:
                    row = evaluate_one_trip_one_controller(
                        args=args,
                        grid_n=grid_n,
                        trip_type=trip_type,
                        trip_path=trip_path,
                        controller_name=controller_name,
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Failed: grid={grid_n}, trip_type={trip_type}, "
                        f"controller={controller_name}: {e}"
                    )
                    continue

                rows_by_trip_type.setdefault(trip_type, []).append(row)

    print("\n=== Saving aggregate files ===")

    for trip_type, rows in sorted(rows_by_trip_type.items()):
        if not rows:
            continue

        trip_type_part = safe_filename_part(trip_type)
        out_path = out_dir / f"all_baseline_results_{trip_type_part}.csv"
        write_csv(out_path, rows)
        print(f"Saved: {out_path}")

    print("\n=== Mean summary by trip type, grid, and controller ===")

    summary_groups: Dict[Tuple[str, int, str], List[Dict[str, object]]] = {}

    for rows in rows_by_trip_type.values():
        for row in rows:
            key = (
                str(row["trip_type"]),
                int(row["grid_n"]),
                str(row["controller"]),
            )
            summary_groups.setdefault(key, []).append(row)

    for (trip_type, grid_n, controller_name), rows in sorted(summary_groups.items()):
        def mean_of(key: str) -> float:
            return float(np.mean([float(r[key]) for r in rows]))

        print(
            f"{trip_type:9s} | grid={grid_n:<2d} | {controller_name:13s} | "
            f"return={mean_of('episode_return'):.2f} | "
            f"completed={mean_of('completed_vehicles'):.2f} | "
            f"active={mean_of('active_vehicles'):.2f} | "
            f"throughput={mean_of('throughput_veh_per_hour'):.2f} veh/h | "
            f"mean_travel={mean_of('mean_travel_time_s'):.2f}s | "
            f"mean_wait={mean_of('mean_waiting_time_s'):.2f}s | "
            f"avg_switches_per_tls={mean_of('avg_switches_per_tls'):.2f}"
        )

    print("\n=== Done ===")
    print(f"Only aggregate files were written to: {out_dir}")


if __name__ == "__main__":
    run_evaluation(parse_args())
