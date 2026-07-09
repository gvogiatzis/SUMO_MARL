#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from env_builder import get_or_create_eval_pool

# Must match the env's backend (libsumo by default) — a plain `import traci`
# here would silently query a connectionless socket module.
from marl_utils.sumo_backend import get_backend
traci, SUMO_BACKEND = get_backend()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-time, max-pressure, and longest-queue baselines on the fixed SUMO evaluation trips."
    )

    parser.add_argument("--grid_n", "--grid-n", dest="grid_n", type=int, default=3)
    parser.add_argument("--episode_steps", "--episode-steps", dest="episode_steps", type=int, default=200)
    parser.add_argument(
        "--sumo_steps_per_env_step",
        "--sumo-steps-per-env-step",
        dest="sumo_steps_per_env_step",
        type=int,
        default=5,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui_delay_ms", "--gui-delay-ms", dest="gui_delay_ms", type=int, default=0)

    parser.add_argument(
        "--baseline_controller",
        "--baseline-controller",
        dest="baseline_controller",
        type=str,
        default="both",
        choices=["fixed_time", "max_pressure", "longest_queue", "both"],
    )
    parser.add_argument(
        "--fixed_time_hold_steps",
        "--fixed-time-hold-steps",
        dest="fixed_time_hold_steps",
        type=int,
        default=4,
        help="Number of environment steps to hold each selected phase. Used only by fixed_time.",
    )
    parser.add_argument(
        "--fixed_time_cycle",
        "--fixed-time-cycle",
        dest="fixed_time_cycle",
        type=str,
        default="0,1,2,3",
        help="Comma-separated fixed-time phase cycle. 0=NS-left, 1=NS-through/right, 2=EW-left, 3=EW-through/right.",
    )
    parser.add_argument(
        "--tie_tolerance",
        "--tie-tolerance",
        dest="tie_tolerance",
        type=float,
        default=1e-6,
        help="Near-tie tolerance. Greedy controllers keep current phase if best score is not sufficiently better.",
    )

    # Longest-queue debug options. These print to terminal only.
    parser.add_argument(
        "--debug_longest_queue",
        "--debug-longest-queue",
        dest="debug_longest_queue",
        action="store_true",
        help="Print detailed longest-queue debug logs: lane q_in, phase scores, and selected actions.",
    )
    parser.add_argument(
        "--debug_lq_tls",
        "--debug-lq-tls",
        dest="debug_lq_tls",
        type=str,
        default="",
        help=(
            "Comma-separated TLS id filters for longest-queue debug output. "
            "Example: n_0_0,n_1_0. Empty means log all TLS."
        ),
    )
    parser.add_argument(
        "--debug_lq_first_steps",
        "--debug-lq-first-steps",
        dest="debug_lq_first_steps",
        type=int,
        default=-1,
        help="Only log the first K environment steps. Use -1 to log all steps.",
    )
    parser.add_argument(
        "--debug_lq_every",
        "--debug-lq-every",
        dest="debug_lq_every",
        type=int,
        default=1,
        help="Log every K environment steps for longest-queue debug output.",
    )

    return parser.parse_args()


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


# -----------------------------------------------------------------------------
# Controllers
# -----------------------------------------------------------------------------

class FixedTimeController:
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
    def __init__(
        self,
        agent_ids: List[str],
        action_dim: int,
        hold_steps: int = 1,
        tie_tolerance: float = 1e-6,
        normalise_phase_score: bool = False,
        debug: bool = False,
        debug_tls_filter: Optional[List[str]] = None,
        debug_first_steps: int = -1,
        debug_every: int = 1,
    ):
        if action_dim != 4:
            raise ValueError(f"MaxPressureController expects action_dim=4, got {action_dim}.")

        self.agent_ids = list(agent_ids)
        self.action_dim = int(action_dim)
        self.hold_steps = max(1, int(hold_steps))
        self.tie_tolerance = float(tie_tolerance)

        # Phase-score normalisation is intentionally disabled.
        # Scores should reflect total pressure/demand, not average pressure per lane.
        self.normalise_phase_score = False

        self.phase_movements: Dict[str, Dict[int, List[Tuple[str, str]]]] = {}
        self.incoming_lane_to_obs_idx: Dict[str, Dict[str, int]] = {}
        self._built_for_env_id: Optional[int] = None

        self.t = 0
        self.current_actions: Optional[Dict[str, int]] = None

        # Debug fields, mainly used by LongestQueueController.
        self.debug = bool(debug)
        self.debug_tls_filter = list(debug_tls_filter or [])
        self.debug_first_steps = int(debug_first_steps)
        self.debug_every = max(1, int(debug_every))

        self.debug_case_idx: Optional[int] = None
        self.debug_trip_path: Optional[str] = None

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
        Enforce which physical lane is allowed to contribute to each phase score.

        Phase mapping:
            0 = NS left              -> only lane *_2
            1 = NS through + right   -> only lanes *_0 and *_1
            2 = EW left              -> only lane *_2
            3 = EW through + right   -> only lanes *_0 and *_1
        """
        lane_idx = cls._parse_lane_index(in_lane)

        if lane_idx is None:
            return False

        # Left-turn phases.
        if phase_id in (0, 2):
            return lane_idx == 2

        # Through/right phases.
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

    def _debug_should_log(self, tls_id: str) -> bool:
        if not self.debug:
            return False

        if self.debug_first_steps >= 0 and self.t >= self.debug_first_steps:
            return False

        if self.debug_every > 1 and (self.t % self.debug_every) != 0:
            return False

        if self.debug_tls_filter:
            return any(pattern in tls_id for pattern in self.debug_tls_filter)

        return True

    def _write_debug_record(self, record: Dict):
        """
        Print a readable longest-queue debug record directly to terminal.
        """
        if not self.debug:
            return

        print("\n" + "=" * 100)
        print(
            f"[LONGEST_QUEUE DEBUG] "
            f"case={record.get('case_idx')} | "
            f"step={record.get('step')} | "
            f"tls={record.get('tls_id')}"
        )

        trip_path = record.get("trip_path", None)
        if trip_path is not None:
            print(f"trip={trip_path}")

        print("-" * 100)

        phase_names = record.get("phase_names", {})
        scores = record.get("scores", {})

        print("Scores:")
        for phase_id in sorted(scores.keys(), key=int):
            print(
                f"  phase {phase_id} "
                f"({phase_names.get(phase_id, 'unknown')}): "
                f"score={float(scores[phase_id]):.3f}"
            )

        greedy_best_action = record.get("greedy_best_action")
        selected_action = record.get("selected_action")

        print(
            f"\nGreedy best action : {greedy_best_action} "
            f"({phase_names.get(str(greedy_best_action), 'unknown')})"
        )
        print(f"Greedy best score  : {float(record.get('greedy_best_score', 0.0)):.3f}")
        print(f"Previous action    : {record.get('previous_action')}")
        previous_score = record.get("previous_score")
        if previous_score is not None:
            print(f"Previous score     : {float(previous_score):.3f}")
        else:
            print("Previous score     : None")
        print(f"Tie tolerance      : {float(record.get('tie_tolerance', 0.0)):.6g}")
        print(
            f"Selected action    : {selected_action} "
            f"({record.get('selected_phase_name')})"
        )

        print("\nAll incoming lane queues:")
        for lane_record in record.get("all_incoming_lane_queues", []):
            print(
                f"  lane={lane_record.get('lane')} | "
                f"obs_idx={lane_record.get('obs_idx')} | "
                f"q_in={float(lane_record.get('q_in', 0.0)):.3f} | "
                f"source={lane_record.get('q_in_source')}"
            )

        print("\nPhase details:")
        phase_details = record.get("phase_details", {})
        for phase_id in sorted(phase_details.keys(), key=int):
            detail = phase_details[phase_id]
            print(
                f"  phase {phase_id} ({detail.get('phase_name')}) | "
                f"score={float(detail.get('score', 0.0)):.3f} | "
                f"total_q_in_used={float(detail.get('total_q_in_used', 0.0)):.3f} | "
                f"num_lanes_used={detail.get('num_lanes_used')} | "
                f"normalised={detail.get('normalised')} | "
                f"reason={detail.get('reason')}"
            )

            for lane_record in detail.get("incoming_lanes", []):
                print(
                    f"    lane={lane_record.get('lane')} | "
                    f"obs_idx={lane_record.get('obs_idx')} | "
                    f"q_in={float(lane_record.get('q_in', 0.0)):.3f} | "
                    f"used_for_score={lane_record.get('used_for_score')} | "
                    f"source={lane_record.get('q_in_source')}"
                )

        print("=" * 100)

    def _build_obs_lane_maps(self, env):
        # env observation: obs[0:12] are queues for info.incoming_lanes; obs[12] is current phase.
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
            left_target, through_target, right_target = east_id, south_id, west_id
        elif in_from == south_id:
            approach = "S"
            left_target, through_target, right_target = west_id, north_id, east_id
        elif in_from == west_id:
            approach = "W"
            left_target, through_target, right_target = north_id, east_id, south_id
        elif in_from == east_id:
            approach = "E"
            left_target, through_target, right_target = south_id, west_id, north_id
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

                        # Critical bug fix:
                        # Phase 1 and phase 3 are through/right phases and must
                        # not count left-turn lanes ending with *_2.
                        #
                        # Phase 0 and phase 2 are left-turn phases and should
                        # only count left-turn lanes ending with *_2.
                        if not self._phase_lane_compatible(phase_id, in_lane):
                            continue

                        phase_map[phase_id].append((in_lane, out_lane))

            self.phase_movements[tls_id] = phase_map

        self._built_for_env_id = env_id

    def _phase_pressure(self, tls_id: str, phase_id: int) -> float:
        """
        Robust max-pressure score.

        Original score:
            sum(q_in - q_out)

        Problem:
            In heavy corridor cases, the useful corridor movement can get a negative
            score when downstream queues are large. Empty phases can then score 0
            and wrongly beat the congested useful phase.

        Fix:
            - Empty phases with no upstream demand are not selectable.
            - Negative movement pressures are clipped to zero.
            - If all clipped pressures are zero but there is upstream demand, fall
              back to a small queue-pressure term.
        """
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

        # If downstream queues make all movement pressures non-positive,
        # still allow the phase to compete if it has real upstream demand.
        if pressure <= 0.0:
            pressure = 1e-3 * total_in_queue

        # self.normalise_phase_score is always False, but keep the branch for clarity.
        if self.normalise_phase_score:
            pressure /= max(1, len(movements))

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
    PHASE_NAMES = {
        0: "NS_left",
        1: "NS_through_right",
        2: "EW_left",
        3: "EW_through_right",
    }

    def _lane_q_for_debug_and_score(
        self,
        tls_id: str,
        in_lane: str,
        obs_vec: Optional[np.ndarray],
    ) -> Dict[str, object]:
        """
        Return queue information for one incoming lane.

        The score follows the longest-queue logic:
          - If obs_vec is available, use obs[:12] through incoming_lane_to_obs_idx.
          - If obs index is missing, the lane is not used for score.
          - For debugging, still try to read the TraCI queue as fallback visibility.
        """
        lane_to_idx = self.incoming_lane_to_obs_idx.get(tls_id, {})
        idx = lane_to_idx.get(in_lane, None)

        if obs_vec is not None and idx is not None and 0 <= idx < min(12, len(obs_vec)):
            q_in = float(obs_vec[idx])
            return {
                "lane": in_lane,
                "obs_idx": int(idx),
                "q_in": q_in,
                "q_in_source": "obs",
                "used_for_score": True,
            }

        q_in_traci = float(self._lane_queue(in_lane))

        return {
            "lane": in_lane,
            "obs_idx": None if idx is None else int(idx),
            "q_in": q_in_traci,
            "q_in_source": "traci_debug_fallback",
            "used_for_score": obs_vec is None,
        }

    def _phase_queue_score_and_details(
        self,
        tls_id: str,
        phase_id: int,
        obs_vec: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, object]]:
        movements = self.phase_movements.get(tls_id, {}).get(phase_id, [])
        if not movements:
            return -1e9, {
                "phase": int(phase_id),
                "phase_name": self.PHASE_NAMES.get(phase_id, f"phase_{phase_id}"),
                "score": -1e9,
                "incoming_lanes": [],
                "total_q_in_used": 0.0,
                "num_lanes_used": 0,
                "normalised": bool(self.normalise_phase_score),
                "reason": "no_movements",
            }

        incoming_lanes = sorted({in_lane for in_lane, _ in movements})

        lane_records: List[Dict[str, object]] = []
        total = 0.0
        used = 0

        for in_lane in incoming_lanes:
            rec = self._lane_q_for_debug_and_score(tls_id, in_lane, obs_vec)
            lane_records.append(rec)

            if bool(rec["used_for_score"]):
                total += float(rec["q_in"])
                used += 1

        if used == 0:
            score = -1e9
            reason = "no_observation_lanes_used"
        else:
            # Normalisation is disabled, so this is effectively score = total.
            score = total / max(1, used) if self.normalise_phase_score else total
            reason = "ok"

        details = {
            "phase": int(phase_id),
            "phase_name": self.PHASE_NAMES.get(phase_id, f"phase_{phase_id}"),
            "score": float(score),
            "incoming_lanes": lane_records,
            "total_q_in_used": float(total),
            "num_lanes_used": int(used),
            "normalised": bool(self.normalise_phase_score),
            "reason": reason,
        }

        return float(score), details

    def _phase_queue_score_from_obs(
        self,
        tls_id: str,
        phase_id: int,
        obs_vec: Optional[np.ndarray],
    ) -> float:
        score, _ = self._phase_queue_score_and_details(tls_id, phase_id, obs_vec)
        return score

    def _scores_for_tls(self, tls_id: str, obs_vec: Optional[np.ndarray] = None) -> List[float]:
        return [
            self._phase_queue_score_from_obs(tls_id, phase_id, obs_vec)
            for phase_id in range(4)
        ]

    def _all_incoming_lane_queues_for_debug(
        self,
        tls_id: str,
        obs_vec: Optional[np.ndarray],
    ) -> List[Dict[str, object]]:
        """
        Output q_in for all incoming lanes in the observation map, not only lanes
        served by a particular phase.
        """
        lane_to_idx = self.incoming_lane_to_obs_idx.get(tls_id, {})
        items = sorted(lane_to_idx.items(), key=lambda kv: kv[1])

        records: List[Dict[str, object]] = []

        for lane_id, idx in items:
            if obs_vec is not None and 0 <= idx < min(12, len(obs_vec)):
                q_in = float(obs_vec[idx])
                source = "obs"
            else:
                q_in = float(self._lane_queue(lane_id))
                source = "traci"

            records.append(
                {
                    "lane": lane_id,
                    "obs_idx": int(idx),
                    "q_in": q_in,
                    "q_in_source": source,
                }
            )

        return records

    def _select_actions(self, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, int]:
        actions: Dict[str, int] = {}

        for tls_id in self.agent_ids:
            obs_vec = obs_dict[tls_id] if obs_dict is not None and tls_id in obs_dict else None

            scores: List[float] = []
            phase_details: Dict[str, Dict[str, object]] = {}

            for phase_id in range(4):
                score, details = self._phase_queue_score_and_details(tls_id, phase_id, obs_vec)
                scores.append(score)
                phase_details[str(phase_id)] = details

            scores_np = np.asarray(scores, dtype=np.float32)
            greedy_best_action = int(np.argmax(scores_np))
            greedy_best_score = float(scores_np[greedy_best_action])

            previous_action = None
            previous_score = None
            if self.current_actions is not None and tls_id in self.current_actions:
                previous_action = int(self.current_actions[tls_id])
                previous_score = float(scores_np[previous_action])

            selected_action = self._choose_with_current_phase_tie_break(tls_id, scores)
            actions[tls_id] = selected_action

            if self._debug_should_log(tls_id):
                record = {
                    "controller": "longest_queue",
                    "case_idx": self.debug_case_idx,
                    "trip_path": self.debug_trip_path,
                    "step": int(self.t),
                    "tls_id": tls_id,
                    "scores": {
                        str(i): float(scores[i])
                        for i in range(4)
                    },
                    "phase_names": {
                        str(i): self.PHASE_NAMES.get(i, f"phase_{i}")
                        for i in range(4)
                    },
                    "greedy_best_action": int(greedy_best_action),
                    "greedy_best_score": float(greedy_best_score),
                    "previous_action": previous_action,
                    "previous_score": previous_score,
                    "tie_tolerance": float(self.tie_tolerance),
                    "selected_action": int(selected_action),
                    "selected_phase_name": self.PHASE_NAMES.get(
                        int(selected_action),
                        f"phase_{selected_action}",
                    ),
                    "all_incoming_lane_queues": self._all_incoming_lane_queues_for_debug(
                        tls_id,
                        obs_vec,
                    ),
                    "phase_details": phase_details,
                }

                self._write_debug_record(record)

        return actions


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

def evaluate_controller(
    args,
    controller,
    run_name: str,
    agent_ids: List[str],
    action_dim: int,
) -> Dict[str, float]:
    eval_pool = get_or_create_eval_pool(args)

    returns_all = []
    completed_all = []
    active_all = []
    throughput_all = []
    mean_travel_all = []
    mean_wait_all = []
    phase_fraction_all = []
    switch_count_all = []

    for case_idx, env in enumerate(eval_pool.envs):
        trip_path = None
        if hasattr(eval_pool, "trip_paths") and case_idx < len(eval_pool.trip_paths):
            trip_path = eval_pool.trip_paths[case_idx]

        if hasattr(controller, "debug_case_idx"):
            controller.debug_case_idx = int(case_idx)

        if hasattr(controller, "debug_trip_path"):
            controller.debug_trip_path = str(trip_path) if trip_path is not None else None

        obs = env.reset()
        controller.reset(env=env)
        stats = PhaseUseStats(agent_ids=agent_ids, action_dim=action_dim)

        done = False
        episode_return = 0.0
        last_info: Dict = {}

        while not done:
            actions = controller.act(obs, env=env)
            stats.update(actions)
            obs, rewards, done, info = env.step(actions)
            episode_return += float(np.sum(list(rewards.values())))
            last_info = info

        kpis = last_info.get("network_kpis", {})
        completed = float(kpis.get("completed_vehicles", 0.0))
        active = float(kpis.get("active_vehicles", 0.0))
        throughput = float(kpis.get("throughput_veh_per_hour", 0.0))
        mean_travel = float(kpis.get("mean_travel_time_s", 0.0))
        mean_wait = float(kpis.get("mean_waiting_time_s", 0.0))

        phase_frac = stats.average_phase_fraction()
        avg_switches = stats.average_switches_per_tls()

        returns_all.append(episode_return)
        completed_all.append(completed)
        active_all.append(active)
        throughput_all.append(throughput)
        mean_travel_all.append(mean_travel)
        mean_wait_all.append(mean_wait)
        phase_fraction_all.append(phase_frac)
        switch_count_all.append(avg_switches)

        trip_msg = f" | trips={trip_path}" if trip_path is not None else ""
        print(
            f"[{datetime.now()}, {run_name}, case {case_idx + 1}/{len(eval_pool.envs)}] "
            f"return={episode_return:.2f} | "
            f"completed={completed:.2f} | "
            f"active={active:.2f} | "
            f"throughput={throughput:.2f} veh/h | "
            f"mean_travel_time={mean_travel:.2f}s | "
            f"mean_waiting_time={mean_wait:.2f}s | "
            f"phase_frac=[{phase_frac[0]:.3f}, {phase_frac[1]:.3f}, "
            f"{phase_frac[2]:.3f}, {phase_frac[3]:.3f}] | "
            f"avg_switches_per_tls={avg_switches:.2f}"
            f"{trip_msg}"
        )

    mean_phase_frac = (
        np.mean(np.stack(phase_fraction_all, axis=0), axis=0)
        if phase_fraction_all
        else np.zeros(action_dim, dtype=np.float32)
    )

    summary = {
        "return": float(np.mean(returns_all)) if returns_all else 0.0,
        "completed": float(np.mean(completed_all)) if completed_all else 0.0,
        "active": float(np.mean(active_all)) if active_all else 0.0,
        "throughput": float(np.mean(throughput_all)) if throughput_all else 0.0,
        "mean_travel_time": float(np.mean(mean_travel_all)) if mean_travel_all else 0.0,
        "mean_waiting_time": float(np.mean(mean_wait_all)) if mean_wait_all else 0.0,
        "avg_switches_per_tls": float(np.mean(switch_count_all)) if switch_count_all else 0.0,
        "phase_frac_0": float(mean_phase_frac[0]) if action_dim > 0 else 0.0,
        "phase_frac_1": float(mean_phase_frac[1]) if action_dim > 1 else 0.0,
        "phase_frac_2": float(mean_phase_frac[2]) if action_dim > 2 else 0.0,
        "phase_frac_3": float(mean_phase_frac[3]) if action_dim > 3 else 0.0,
        "cases": len(returns_all),
    }

    print(
        f"\n[{datetime.now()}, SUMMARY {run_name}] "
        f"cases={summary['cases']} | "
        f"return={summary['return']:.2f} | "
        f"completed={summary['completed']:.2f} | "
        f"active={summary['active']:.2f} | "
        f"throughput={summary['throughput']:.2f} veh/h | "
        f"mean_travel_time={summary['mean_travel_time']:.2f}s | "
        f"mean_waiting_time={summary['mean_waiting_time']:.2f}s | "
        f"phase_frac=[{summary['phase_frac_0']:.3f}, {summary['phase_frac_1']:.3f}, "
        f"{summary['phase_frac_2']:.3f}, {summary['phase_frac_3']:.3f}] | "
        f"avg_switches_per_tls={summary['avg_switches_per_tls']:.2f}\n"
    )

    return summary


def infer_eval_setup(args):
    eval_pool = get_or_create_eval_pool(args)
    if len(eval_pool.envs) == 0:
        raise RuntimeError("Evaluation pool is empty.")

    env = eval_pool.envs[0]
    obs = env.reset()
    agent_ids = list(env.agent_ids)

    if not agent_ids:
        raise RuntimeError("No traffic lights discovered in SUMO. Check network/trips.")

    action_dim = env.action_spaces[agent_ids[0]].n

    if action_dim != 4:
        raise RuntimeError(f"This script expects action_dim=4, got {action_dim}.")

    return eval_pool, agent_ids, action_dim


def run_evaluation(args):
    eval_pool, agent_ids, action_dim = infer_eval_setup(args)
    cycle = parse_fixed_time_cycle(args.fixed_time_cycle, action_dim)

    # Phase-score normalisation is now fully disabled for greedy baselines.
    normalise_phase_score = False

    print(
        f"Baseline evaluation setup | "
        f"controller={args.baseline_controller} | "
        f"N={len(agent_ids)} agents | "
        f"A={action_dim} actions | "
        f"eval_cases={len(eval_pool.envs)} | "
        f"episode_steps={args.episode_steps} | "
        f"sumo_steps_per_env_step={args.sumo_steps_per_env_step} | "
        f"fixed_time_hold_steps={args.fixed_time_hold_steps} | "
        f"greedy_hold_steps=1 | "
        f"tie_tolerance={args.tie_tolerance} | "
        f"normalise_phase_score={normalise_phase_score}"
    )

    if hasattr(eval_pool, "trip_paths"):
        print("\nEvaluation trip files:")
        for i, path in enumerate(eval_pool.trip_paths, start=1):
            print(f"  {i:02d}. {path}")
        print("")

    results: Dict[str, Dict[str, float]] = {}

    if args.baseline_controller in ("fixed_time", "both"):
        hold_steps = controller_hold_steps("fixed_time", args)
        controller = FixedTimeController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,
            cycle=cycle,
        )
        run_name = f"baseline_fixed_time_hold{hold_steps}_cycle{'-'.join(map(str, cycle))}"
        results["fixed_time"] = evaluate_controller(args, controller, run_name, agent_ids, action_dim)

    if args.baseline_controller in ("max_pressure", "both"):
        hold_steps = controller_hold_steps("max_pressure", args)
        controller = MaxPressureController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,
            tie_tolerance=args.tie_tolerance,
            normalise_phase_score=normalise_phase_score,
        )
        run_name = f"baseline_max_pressure_hold{hold_steps}"
        results["max_pressure"] = evaluate_controller(args, controller, run_name, agent_ids, action_dim)

    if args.baseline_controller in ("longest_queue", "both"):
        hold_steps = controller_hold_steps("longest_queue", args)

        debug_tls_filter = [
            x.strip()
            for x in str(args.debug_lq_tls).split(",")
            if x.strip()
        ]

        controller = LongestQueueController(
            agent_ids=agent_ids,
            action_dim=action_dim,
            hold_steps=hold_steps,
            tie_tolerance=args.tie_tolerance,
            normalise_phase_score=normalise_phase_score,
            debug=bool(args.debug_longest_queue),
            debug_tls_filter=debug_tls_filter,
            debug_first_steps=args.debug_lq_first_steps,
            debug_every=args.debug_lq_every,
        )

        run_name = f"baseline_longest_queue_hold{hold_steps}"

        if args.debug_longest_queue:
            print("[DEBUG] Longest-queue debug output: terminal")
            if debug_tls_filter:
                print(f"[DEBUG] Longest-queue TLS filter: {debug_tls_filter}")
            else:
                print("[DEBUG] Longest-queue TLS filter: all TLS")
            print(f"[DEBUG] Longest-queue first steps: {args.debug_lq_first_steps}")
            print(f"[DEBUG] Longest-queue log every: {args.debug_lq_every}")

        results["longest_queue"] = evaluate_controller(args, controller, run_name, agent_ids, action_dim)

    print("Final baseline comparison:")
    for name, metrics in results.items():
        print(
            f"  {name}: "
            f"return={metrics['return']:.2f} | "
            f"completed={metrics['completed']:.2f} | "
            f"active={metrics['active']:.2f} | "
            f"throughput={metrics['throughput']:.2f} veh/h | "
            f"mean_travel_time={metrics['mean_travel_time']:.2f}s | "
            f"mean_waiting_time={metrics['mean_waiting_time']:.2f}s | "
            f"phase_frac=[{metrics['phase_frac_0']:.3f}, {metrics['phase_frac_1']:.3f}, "
            f"{metrics['phase_frac_2']:.3f}, {metrics['phase_frac_3']:.3f}] | "
            f"avg_switches_per_tls={metrics['avg_switches_per_tls']:.2f}"
        )

    try:
        eval_pool.close_all()
    except Exception:
        pass


if __name__ == "__main__":
    run_evaluation(parse_args())
