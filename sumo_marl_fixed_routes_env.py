"""
SUMO MARL Evaluation Environment (fixed routes):
- Same 3x3 (configurable NxN) core grid + fringe stubs as the random-flow env.
- No random flow generation in reset().
- Accepts a fixed routes file (.rou.xml) OR a fixed trips/flows file (.xml) and
  (optionally) converts trips to routes via duarouter with a fixed seed.
- Online KPI tracking with TraCI (completed count, throughput, mean travel time,
  mean waiting time), identical to your random env so comparisons are fair.

Usage examples:
  # Use a fixed routes file for repeatable evaluation
  python sumo_marl_fixed_routes_env.py --routes path/to/eval.rou.xml --steps 150

  # Use a fixed trips/flows file; duarouter will produce a temp routes file
  python sumo_marl_fixed_routes_env.py --trips path/to/eval.trips.xml --duarouter-seed 123 --steps 150

Programmatic:
  env = SumoGridFixedRoutesEnv(...).with_fixed_routes("path/to/eval.rou.xml")
  obs = env.reset(); ...
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# ---------------- Gymnasium-lite spaces shim ----------------
try:
    from gymnasium import spaces  # type: ignore
except Exception:
    class _Discrete:
        def __init__(self, n: int):
            self.n = int(n)
        def sample(self) -> int:
            return int(np.random.randint(self.n))
    class _Box:
        def __init__(self, low, high, dtype=np.float32):
            self.low = np.array(low, dtype=dtype)
            self.high = np.array(high, dtype=dtype)
            self.shape = tuple(self.low.shape)
            self.dtype = dtype
    class spaces:
        Discrete = _Discrete
        Box = _Box

# ---------------- SUMO imports ----------------
from marl_utils.sumo_backend import get_backend
traci, SUMO_BACKEND = get_backend()  # with libsumo, sets SUMO_HOME to the pip eclipse-sumo package

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise RuntimeError("Please set SUMO_HOME to your SUMO installation directory.")
import sumolib  # type: ignore


@dataclass
class TLSInfo:
    tls_id: str
    incoming_lanes: List[str]  # expected 12 for 4 approaches × 3 lanes


class SumoGridMARLFixedEnv:
    def __init__(
        self,
        grid_n: int = 3,
        lanes_per_edge: int = 3,
        spacing: float = 150.0,             # meters between core junctions
        fringe_len: float = 150.0,          # meters for fringe stubs
        step_length_internal: float = 1.0,  # SUMO internal step (s)
        sumo_steps_per_env_step: int = 10,
        episode_steps: int = 500,
        # Fixed-input options (evaluation):
        fixed_routes_file: Optional[str] = None,   # .rou.xml (preferred)
        fixed_trips_file: Optional[str] = None,    # .trips.xml / flows xml (duarouted)
        duarouter_seed: int = 0,
        # General options:
        seed: int = 42,
        gui: bool = False,
        gui_delay_ms: int = 0,
        verbose: bool = False,
        suppress_sumo_output: bool = True,
        # Reward normalization
        reward_scale: float = 120.0,       # linear divisor S
        reward_clip_min: float = -1.0,    # clip lower bound after scaling
        reward_clip_max: float = 1.0,     # clip upper bound after scaling
        negate_if_positive: bool = False, # set True if your raw rewards are positive queues
        per_agent_reward_scale: dict | None = None,  # optional per-agent overrides
    ):
        self.grid_n = grid_n
        self.lanes_per_edge = lanes_per_edge
        self.spacing = spacing
        self.fringe_len = fringe_len
        self.step_length_internal = float(step_length_internal)
        self.sumo_steps_per_env_step = sumo_steps_per_env_step
        self.episode_steps = episode_steps

        self.seed = seed
        if gui and SUMO_BACKEND == "libsumo":
            raise RuntimeError("sumo-gui requires the traci backend: set SUMO_MARL_BACKEND=traci")
        self.gui = gui
        self.gui_delay_ms = gui_delay_ms
        self.verbose = verbose
        self.suppress_sumo_output = suppress_sumo_output

        self.reward_scale = float(reward_scale)
        self.reward_clip_min = float(reward_clip_min)
        self.reward_clip_max = float(reward_clip_max)
        self.negate_if_positive = bool(negate_if_positive)
        self.per_agent_reward_scale = dict(per_agent_reward_scale) if per_agent_reward_scale else None

        # Fixed inputs
        self.fixed_routes_file = fixed_routes_file
        self.fixed_trips_file = fixed_trips_file
        self.duarouter_seed = duarouter_seed

        self.tmpdir = tempfile.mkdtemp(prefix="sumo_grid_fixed_")
        self.net_file = os.path.join(self.tmpdir, "grid.net.xml")
        self.node_file = os.path.join(self.tmpdir, "nodes.nod.xml")
        self.edge_file = os.path.join(self.tmpdir, "edges.edg.xml")
        # We'll still create a default routes file for config, but will override with fixed one.
        self.default_route_file = os.path.join(self.tmpdir, "routes.placeholder.rou.xml")
        self.cfg_file = os.path.join(self.tmpdir, "sim.sumocfg")

        # Active route file used by SUMO (may be fixed or duarouted)
        self._active_route_file: Optional[str] = None

        # Build network
        self._write_nodes_and_edges()
        self._build_net(step_length_internal)
        self._write_placeholder_routes()  # contains vType definition
        self._set_active_route_file()     # determines self._active_route_file
        self._write_sumocfg(step_length_internal)

        # For topology lookup
        self._net = sumolib.net.readNet(self.net_file)

        # Populated after reset
        self.tls_infos: List[TLSInfo] = []
        self.agent_ids: List[str] = []
        self.num_agents: int = 0

        self.action_spaces: Dict[str, spaces.Discrete] = {}
        self.observation_spaces: Dict[str, spaces.Box] = {}

        self._connected = False
        self._step_count = 0

        # KPI accumulators (online TraCI)
        self._kpi_completed: int = 0
        self._kpi_total_travel_time: float = 0.0
        self._kpi_total_waiting_time: float = 0.0
        self._veh_depart_time: Dict[str, float] = {}
        self._veh_wait_time_last: Dict[str, float] = {}
        self._veh_wait_custom: Dict[str, float] = {}

    # ---------------------- Public helper ----------------------
    def with_fixed_routes(self, routes_file: str):
        """Chainable setter to swap in a fixed routes file before reset()."""
        self.fixed_routes_file = os.path.abspath(routes_file)
        self.fixed_trips_file = None  # ignore trips if routes provided
        self._set_active_route_file()
        self._rewrite_cfg_route_file()
        return self

    def with_fixed_trips(self, trips_file: str, duarouter_seed: int = 0):
        """Chainable setter to swap in a fixed trips/flows file; will duaroute to temp routes."""
        self.fixed_trips_file = os.path.abspath(trips_file)
        self.fixed_routes_file = None
        self.duarouter_seed = duarouter_seed
        self._set_active_route_file()  # triggers duarouter
        self._rewrite_cfg_route_file()
        return self

    # ---------------------- Network construction ----------------------
    def _write_nodes_and_edges(self):
        N = self.grid_n
        S = self.spacing
        F = self.fringe_len

        # Nodes
        nroot = ET.Element("nodes")
        for i in range(N):
            for j in range(N):
                nid = f"n_{i}_{j}"
                x = i * S
                y = j * S
                ET.SubElement(nroot, "node", id=nid, x=str(x), y=str(y), type="traffic_light")
        for j in range(N):
            ET.SubElement(nroot, "node", id=f"fw_{j}", x=str(-F), y=str(j*S), type="dead_end")
        for j in range(N):
            ET.SubElement(nroot, "node", id=f"fe_{j}", x=str((N-1)*S + F), y=str(j*S), type="dead_end")
        for i in range(N):
            ET.SubElement(nroot, "node", id=f"fs_{i}", x=str(i*S), y=str(-F), type="dead_end")
        for i in range(N):
            ET.SubElement(nroot, "node", id=f"fn_{i}", x=str(i*S), y=str((N-1)*S + F), type="dead_end")
        ET.ElementTree(nroot).write(self.node_file, encoding="utf-8", xml_declaration=True)

        # Edges
        eroot = ET.Element("edges")
        def add_edge(eid, fr, to):
            ET.SubElement(
                eroot, "edge",
                attrib={"id": eid, "from": fr, "to": to, "numLanes": str(self.lanes_per_edge), "speed": "16.7"}
            )
        for i in range(N):
            for j in range(N):
                if i < N - 1:
                    add_edge(f"e_{i}_{j}_to_{i+1}_{j}", f"n_{i}_{j}", f"n_{i+1}_{j}")
                    add_edge(f"e_{i+1}_{j}_to_{i}_{j}", f"n_{i+1}_{j}", f"n_{i}_{j}")
                if j < N - 1:
                    add_edge(f"e_{i}_{j}_to_{i}_{j+1}", f"n_{i}_{j}", f"n_{i}_{j+1}")
                    add_edge(f"e_{i}_{j+1}_to_{i}_{j}", f"n_{i}_{j+1}", f"n_{i}_{j}")
        for j in range(N):
            add_edge(f"e_fw{j}_to_n0_{j}", f"fw_{j}", f"n_0_{j}")
            add_edge(f"e_n0_{j}_to_fw{j}", f"n_0_{j}", f"fw_{j}")
        for j in range(N):
            add_edge(f"e_fe{j}_to_n{N-1}_{j}", f"fe_{j}", f"n_{N-1}_{j}")
            add_edge(f"e_n{N-1}_{j}_to_fe{j}", f"n_{N-1}_{j}", f"fe_{j}")
        for i in range(N):
            add_edge(f"e_fs{i}_to_n{i}_0", f"fs_{i}", f"n_{i}_0")
            add_edge(f"e_n{i}_0_to_fs{i}", f"n_{i}_0", f"fs_{i}")
        for i in range(N):
            add_edge(f"e_fn{i}_to_n{i}_{N-1}", f"fn_{i}", f"n_{i}_{N-1}")
            add_edge(f"e_n{i}_{N-1}_to_fn{i}", f"n_{i}_{N-1}", f"fn_{i}")
        ET.ElementTree(eroot).write(self.edge_file, encoding="utf-8", xml_declaration=True)

    def _build_net(self, step_length_internal: float):
        netconvert = os.path.join(os.environ["SUMO_HOME"], "bin", "netconvert")
        kwargs = {}
        if self.suppress_sumo_output:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        cmd = [
            netconvert, "--node-files", self.node_file, "--edge-files", self.edge_file,
            "-o", self.net_file, "--tls.guess", "--no-turnarounds", "true"
        ]
        subprocess.run(cmd, check=True, **kwargs)

    def _write_placeholder_routes(self):
        """A minimal routes file with vType; used if user forgets to pass fixed routes."""
        root = ET.Element("routes")
        ET.SubElement(
            root, "vType",
            id="car", accel="2.0", decel="4.5", sigma="0.5", length="3.5", width="1.0",
            maxSpeed="16.7", vClass="passenger", guiShape="passenger"
        )
        # No flows/routes by default.
        ET.ElementTree(root).write(self.default_route_file, encoding="utf-8", xml_declaration=True)

    def _set_active_route_file(self):
        """Decide which route file SUMO will use and prepare it if needed."""
        if self.fixed_routes_file:
            self._active_route_file = os.path.abspath(self.fixed_routes_file)
            return

        if self.fixed_trips_file:
            # Convert trips->routes with duarouter (deterministic via seed)
            duarouter = os.path.join(os.environ["SUMO_HOME"], "bin", "duarouter")
            out_rou = os.path.join(self.tmpdir, "eval.fixed.rou.xml")
            kwargs = {}
            if self.suppress_sumo_output:
                kwargs["stdout"] = subprocess.DEVNULL
                kwargs["stderr"] = subprocess.DEVNULL
            cmd = [
                duarouter,
                "-n", self.net_file,
                "-r", os.path.abspath(self.fixed_trips_file),
                "-o", out_rou,
                "--ignore-errors",
                "--seed", str(int(self.duarouter_seed)),
            ]
            subprocess.run(cmd, check=True, **kwargs)
            self._active_route_file = out_rou
            return

        # Fallback: still usable but will have zero demand
        self._active_route_file = self.default_route_file

    def _write_sumocfg(self, step_length_internal: float):
        cfg = ET.Element("configuration")
        input_el = ET.SubElement(cfg, "input")
        ET.SubElement(input_el, "net-file", value=self.net_file)  # absolute path OK
        ET.SubElement(input_el, "route-files", value=self._active_route_file or self.default_route_file)
        time_el = ET.SubElement(cfg, "time")
        ET.SubElement(time_el, "step-length", value=str(step_length_internal))
        ET.SubElement(time_el, "begin", value="0")
        ET.SubElement(time_el, "end", value=str(self.episode_steps * self.sumo_steps_per_env_step * step_length_internal))
        ET.ElementTree(cfg).write(self.cfg_file, encoding="utf-8", xml_declaration=True)

    def _rewrite_cfg_route_file(self):
        """If user changes fixed routes after construction, update the .sumocfg."""
        self._set_active_route_file()
        self._write_sumocfg(step_length_internal=float(self._read_cfg_steplength()))

    def _read_cfg_steplength(self) -> float:
        try:
            tree = ET.parse(self.cfg_file); root = tree.getroot()
            val = root.find("./time/step-length").get("value", "1.0")
            return float(val)
        except Exception:
            return 1.0

    # ---------------------- TLS discovery ----------------------
    def _discover_tls_infos_traci(self) -> List[TLSInfo]:
        tls_ids = list(traci.trafficlight.getIDList())
        tls_infos: List[TLSInfo] = []

        for tls_id in sorted(tls_ids):
            links = traci.trafficlight.getControlledLinks(tls_id)
            incoming: List[str] = []
            seen = set()

            # First pass: prefer lane indices within lanes_per_edge
            for group in links:
                for link in (group if isinstance(group, (list, tuple)) else [group]):
                    in_lane = link[0]
                    if not in_lane or in_lane.startswith(":"):
                        continue
                    try:
                        lane_idx = int(in_lane.split("_")[-1])
                        if lane_idx >= self.lanes_per_edge:
                            continue
                    except Exception:
                        pass
                    if in_lane not in seen:
                        incoming.append(in_lane)
                        seen.add(in_lane)

            # Fallback: fill up to target count
            if len(incoming) < 4 * self.lanes_per_edge:
                for group in links:
                    for link in (group if isinstance(group, (list, tuple)) else [group]):
                        in_lane = link[0]
                        if not in_lane or in_lane.startswith(":") or in_lane in seen:
                            continue
                        incoming.append(in_lane)
                        seen.add(in_lane)
                        if len(incoming) >= 4 * self.lanes_per_edge:
                            break
                    if len(incoming) >= 4 * self.lanes_per_edge:
                        break

            incoming = incoming[: 4 * self.lanes_per_edge]
            tls_infos.append(TLSInfo(tls_id=tls_id, incoming_lanes=incoming))

        return tls_infos

    # ---------------------- KPI helpers ----------------------
    def _kpi_reset(self):
        self._kpi_completed = 0
        self._kpi_total_travel_time = 0.0
        self._kpi_total_waiting_time = 0.0
        self._veh_depart_time.clear()
        self._veh_wait_time_last.clear()
        self._veh_wait_custom.clear()

    # ---------------------- Env API ----------------------
    def reset(self) -> Dict[str, np.ndarray]:
        # No random demand generation here. We use fixed routes prepared earlier.
        self._kpi_reset()

        if self._connected:
            self.close()
        elif SUMO_BACKEND == "libsumo":
            # libsumo allows one in-process simulation; another env instance may still own it.
            try:
                traci.close()
            except Exception:
                pass

        sumo_bin = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui" if self.gui else "sumo")
        cmd = [
            sumo_bin, "-c", self.cfg_file,
            "--seed", str(self.seed),
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--start", "--quit-on-end",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if self.suppress_sumo_output:
            devnull = os.devnull
            cmd += ["--log", devnull, "--error-log", devnull, "--message-log", devnull]
        if self.gui and self.gui_delay_ms and self.gui_delay_ms > 0:
            cmd += ["--delay", str(int(self.gui_delay_ms))]

        if self.verbose:
            print(f"[SUMO] Launching {'SUMO-GUI' if self.gui else 'SUMO'} with routes={self._active_route_file}")

        if self.suppress_sumo_output:
            import contextlib
            @contextlib.contextmanager
            def _silence():
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    with open(os.devnull, "w") as fnull:
                        sys.stdout = fnull; sys.stderr = fnull; yield
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
            with _silence():
                traci.start(cmd)
        else:
            traci.start(cmd)

        self._connected = True
        self._step_count = 0

        # TLS discovery and spaces
        self.tls_infos = self._discover_tls_infos_traci()
        self.agent_ids = [info.tls_id for info in self.tls_infos]
        self.num_agents = len(self.agent_ids)
        self.action_spaces = {aid: spaces.Discrete(4) for aid in self.agent_ids}
        self.observation_spaces = {
            aid: spaces.Box(
                low=np.zeros(13, dtype=np.float32),
                high=np.concatenate([np.full(12, np.inf, dtype=np.float32), np.array([3.0], dtype=np.float32)]),
                dtype=np.float32,
            )
            for aid in self.agent_ids
        }

        # Install the 4-phase protected-lefts program
        self._install_four_phase_programs()

        return self._observe_all()

    def step(self, actions: Dict[str, int]):
        """
        Same KPI logic as the updated SumoGridMARLRandomEnv:
        - depart timestamp at begin-of-step (t_before)
        - waiting time accumulated by us per internal step (speed-threshold based),
          so it remains available when a vehicle arrives (closer to "final")
        - report BOTH completed-only means and all-vehicles (completed+active) means
        """
        assert set(actions.keys()) == set(self.agent_ids)

        # Apply actions
        for tls_id, a in actions.items():
            traci.trafficlight.setPhase(tls_id, int(a) % 4)

        dt = float(self.step_length_internal)  # seconds per internal SUMO step

        # Advance SUMO internal steps with KPI updates
        for _ in range(self.sumo_steps_per_env_step):
            # --- begin-of-step time (more exact depart timestamps) ---
            t_before = float(traci.simulation.getTime())

            # --- update custom waiting accumulator for vehicles currently present ---
            # define "waiting" as nearly stopped: speed < 0.1 m/s
            try:
                for vid in traci.vehicle.getIDList():
                    try:
                        if float(traci.vehicle.getSpeed(vid)) < 0.1:
                            self._veh_wait_custom[vid] = self._veh_wait_custom.get(vid, 0.0) + dt
                        else:
                            # ensure key exists so active vehicles are included in "all vehicles" stats
                            self._veh_wait_custom.setdefault(vid, 0.0)
                    except Exception:
                        # vehicle may disappear between calls
                        pass
            except Exception:
                pass

            # --- advance simulation ---
            traci.simulationStep()
            t_after = float(traci.simulation.getTime())

            # --- departures (stamp at begin-of-step) ---
            try:
                for vid in traci.simulation.getDepartedIDList():
                    self._veh_depart_time[vid] = t_before
                    self._veh_wait_custom.setdefault(vid, 0.0)
            except Exception:
                pass

            # --- arrivals -> finalize KPIs using our own waiting accumulator ---
            try:
                for vid in traci.simulation.getArrivedIDList():
                    dep_t = self._veh_depart_time.pop(vid, t_before)  # fallback: t_before
                    travel = max(t_after - dep_t, 0.0)

                    wait = float(self._veh_wait_custom.pop(vid, 0.0))

                    self._kpi_total_travel_time += travel
                    self._kpi_total_waiting_time += wait
                    self._kpi_completed += 1
            except Exception:
                pass

        self._step_count += 1

        obs = self._observe_all()

        # --- rewards: raw -> normalized ---
        raw_rewards = self._compute_raw_rewards(obs)
        rewards = self._normalize_rewards(raw_rewards)

        # --- KPI snapshot ---
        t_now = float(traci.simulation.getTime())
        elapsed_sim_time = max(t_now, 1e-6)

        completed = int(self._kpi_completed)
        throughput_vph = (completed / elapsed_sim_time) * 3600.0

        # Completed-only (arrived vehicles) means
        mean_travel_completed = (self._kpi_total_travel_time / completed) if completed > 0 else 0.0
        mean_wait_completed = (self._kpi_total_waiting_time / completed) if completed > 0 else 0.0

        # Active vehicles currently in-system (departed but not arrived)
        active_ids = list(self._veh_depart_time.keys())
        active = len(active_ids)

        sum_active_travel = 0.0
        sum_active_wait = 0.0
        for vid in active_ids:
            dep_t = float(self._veh_depart_time.get(vid, t_now))
            sum_active_travel += max(t_now - dep_t, 0.0)
            sum_active_wait += float(self._veh_wait_custom.get(vid, 0.0))

        total_veh = completed + active

        # All vehicles (completed + active partial trips) means
        mean_travel_all = ((self._kpi_total_travel_time + sum_active_travel) / total_veh) if total_veh > 0 else 0.0
        mean_wait_all = ((self._kpi_total_waiting_time + sum_active_wait) / total_veh) if total_veh > 0 else 0.0

        done = self._step_count >= self.episode_steps
        info = {
            "network_kpis": {
                "completed_vehicles": int(completed),
                "active_vehicles": int(active),
                "total_vehicles_seen_or_active": int(total_veh),
                "throughput_veh_per_hour": float(throughput_vph),

                # Completed-only (old behaviour, but now with improved waiting/depart stamping)
                "mean_travel_time_s_completed": float(mean_travel_completed),
                "mean_waiting_time_s_completed": float(mean_wait_completed),

                # All vehicles currently present + completed (new behaviour)
                "mean_travel_time_s": float(mean_travel_all),
                "mean_waiting_time_s": float(mean_wait_all),
            },
            "raw_rewards": raw_rewards,
        }

        if done:
            self.close()

        return obs, rewards, done, info

    def close(self):
        if self._connected:
            try:
                traci.close()
            finally:
                self._connected = False

    # ---------------------- Helpers ----------------------
    def _install_four_phase_programs(self):
        """Install a true protected-lefts 4-phase program and select it."""
        net = self._net
        def parse_ij(nid: str):
            try:
                _, i, j = nid.split('_'); return int(i), int(j)
            except Exception:
                return -1, -1
        N = self.grid_n

        for info in self.tls_infos:
            tls_id = info.tls_id
            i, j = parse_ij(tls_id)
            north_id = f"n_{i}_{j+1}" if j+1 < N else f"fn_{i}"
            south_id = f"n_{i}_{j-1}" if j-1 >= 0 else f"fs_{i}"
            east_id  = f"n_{i+1}_{j}" if i+1 < N else f"fe_{j}"
            west_id  = f"n_{i-1}_{j}" if i-1 >= 0 else f"fw_{j}"

            groups = traci.trafficlight.getControlledLinks(tls_id)
            meta: List[Tuple[str, str]] = []
            for g in groups:
                links = g if isinstance(g, (list, tuple)) else [g]
                chosen = None
                for link in links:
                    if not link or len(link) < 2: continue
                    in_lane = link[0]
                    if in_lane and not in_lane.startswith(":"):
                        chosen = link; break
                if chosen is None and len(links) > 0:
                    chosen = links[0]
                if not chosen:
                    meta.append(('?', '?')); continue

                in_lane, out_lane = chosen[0], chosen[1]
                try:
                    in_edge_id = in_lane.rsplit('_', 1)[0]
                    out_edge_id = out_lane.rsplit('_', 1)[0]
                    in_edge = net.getEdge(in_edge_id)
                    out_edge = net.getEdge(out_edge_id)
                    in_from = in_edge.getFromNode().getID()
                    out_to  = out_edge.getToNode().getID()
                except Exception:
                    meta.append(('?', '?')); continue

                if in_from == north_id:
                    approach = 'N'; left_t, thru_t, right_t = east_id, south_id, west_id
                elif in_from == south_id:
                    approach = 'S'; left_t, thru_t, right_t = west_id, north_id, east_id
                elif in_from == west_id:
                    approach = 'W'; left_t, thru_t, right_t = north_id, east_id, south_id
                elif in_from == east_id:
                    approach = 'E'; left_t, thru_t, right_t = south_id, west_id, north_id
                else:
                    approach = '?'; left_t = thru_t = right_t = None

                if approach == '?':
                    movement = '?'
                elif out_to == left_t:
                    movement = 'L'
                elif out_to == thru_t:
                    movement = 'T'
                elif out_to == right_t:
                    movement = 'R'
                else:
                    movement = '?'
                meta.append((approach, movement))

            k_live = len(traci.trafficlight.getRedYellowGreenState(tls_id))
            def build_state(phase_idx: int) -> str:
                chars = []
                for appr, mov in meta:
                    allow = False
                    if phase_idx == 0:      # NS LEFTs only
                        allow = (appr in ('N','S')) and (mov == 'L')
                    elif phase_idx == 1:    # NS THROUGH+RIGHT
                        allow = (appr in ('N','S')) and (mov in ('T','R'))
                    elif phase_idx == 2:    # EW LEFTs only
                        allow = (appr in ('E','W')) and (mov == 'L')
                    elif phase_idx == 3:    # EW THROUGH+RIGHT
                        allow = (appr in ('E','W')) and (mov in ('T','R'))
                    chars.append('G' if allow else 'r')
                s = ''.join(chars)
                return s[:k_live].ljust(k_live, 'r')

            states = [build_state(p) for p in range(4)]
            TLLogic = traci.trafficlight.Logic
            TLPhase = traci.trafficlight.Phase
            phases = [
                TLPhase(30, states[0]),
                TLPhase(30, states[1]),
                TLPhase(30, states[2]),
                TLPhase(30, states[3]),
            ]
            logic = TLLogic("marl4", 0, 0, phases)
            traci.trafficlight.setProgramLogic(tls_id, logic)
            traci.trafficlight.setProgram(tls_id, "marl4")

    def _observe_all(self) -> Dict[str, np.ndarray]:
        observations: Dict[str, np.ndarray] = {}
        for info in self.tls_infos:
            queues = []
            for lane_id in info.incoming_lanes:
                qlen = float(traci.lane.getLastStepHaltingNumber(lane_id))
                queues.append(qlen)
            try:
                phase_idx = int(traci.trafficlight.getPhase(info.tls_id)) % 4
            except Exception:
                phase_idx = 0
            obs_vec = np.concatenate(
                [np.array(queues, dtype=np.float32), np.array([float(phase_idx)], dtype=np.float32)],
                axis=0,
            )
            observations[info.tls_id] = obs_vec
        return observations

    def _compute_raw_rewards(self, obs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Raw reward: negative sum of 12 incoming-lane queues (your original definition).
        r_raw(aid) = - sum(obs[:12])
        """
        return {aid: -float(np.sum(vec[:12])) for aid, vec in obs.items()}

    def _scale_for_agent(self, agent_id: str) -> float:
        if self.per_agent_reward_scale and agent_id in self.per_agent_reward_scale:
            return float(self.per_agent_reward_scale[agent_id])
        return self.reward_scale

    def _normalize_rewards(self, raw_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Static linear scale + clip:
            r_scaled = clip( sign(r_raw) * |r_raw| / S , clip_min, clip_max )
        If negate_if_positive=True and r_raw >= 0, we flip the sign first.
        (Useful if some code paths produce positive queue totals.)
        """
        scaled: Dict[str, float] = {}
        for aid, r_raw in raw_rewards.items():
            r = float(r_raw)
            if self.negate_if_positive and r >= 0.0:
                r = -r
            S = self._scale_for_agent(aid)
            if S <= 0:
                S = 1.0
            r_scaled = r / S
            # clip
            if r_scaled < self.reward_clip_min:
                r_scaled = self.reward_clip_min
            elif r_scaled > self.reward_clip_max:
                r_scaled = self.reward_clip_max
            scaled[aid] = float(r_scaled)
        return scaled

# ---------------------- CLI demo ----------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI if set")
    parser.add_argument("--gui-delay-ms", type=int, default=0, help="Delay per SUMO step in GUI")
    parser.add_argument("--grid-n", type=int, default=3, help="Grid size N")
    parser.add_argument("--steps", type=int, default=100, help="Episode steps")
    parser.add_argument("--rotate-every", type=int, default=1, help="Rotate action every N env steps")

    # New: how to feed the 10 evaluation cases
    parser.add_argument("--trips-dir", type=str, default=None,
                        help="Directory containing trips XML files (e.g., those generated by make_eval_trips_varied_grid.py)")
    parser.add_argument("--trips-files", type=str, nargs="*", default=None,
                        help="Explicit list of trips files to evaluate (overrides --trips-dir if given)")

    args = parser.parse_args()

    # --------- collect the 10 files ---------
    if args.trips_files:
        trips_files = [os.path.abspath(p) for p in args.trips_files]
    elif args.trips_dir:
        pattern = os.path.join(os.path.abspath(args.trips_dir), "eval_trips_random_case_*.xml")
        trips_files = sorted(glob.glob(pattern))
    else:
        raise SystemExit("Please provide --trips-dir (folder with eval_trips_random_case_*.xml) "
                         "or --trips-files <f1> <f2> ...")

    if not trips_files:
        raise SystemExit("No trips files found. Check --trips-dir or --trips-files input.")

    print("=== Fixed-routes evaluation over cases ===")
    for i, f in enumerate(trips_files, 1):
        print(f"  {i:02d}. {f}")

    # Storage for per-case KPIs
    all_completed = []
    all_throughput_vph = []
    all_mean_travel_s = []
    all_mean_wait_s = []
    all_episode_returns = []

    for case_idx, trips_path in enumerate(trips_files, 1):
        print(f"\n=== Evaluating case {case_idx}/{len(trips_files)} ===")
        print(f"Trips file: {trips_path}")

        env = SumoGridMARLFixedEnv(
            gui=args.gui,
            gui_delay_ms=args.gui_delay_ms,
            grid_n=args.grid_n,
            episode_steps=args.steps,
            verbose=False,
            suppress_sumo_output=True,
            fixed_trips_file=trips_path,
        )

        obs = env.reset()

        episode_return = 0.0
        final_kpis = None
        for t in range(env.episode_steps):
            # Timed action rotation: 0 -> 1 -> 2 -> 3 -> 0 -> ...
            actions = {aid: ((t // args.rotate_every) % 4) for aid in env.agent_ids}
            obs, rew, done, info = env.step(actions)
            episode_return += float(np.sum(list(rew.values())))
            final_kpis = info.get("network_kpis", None)
            if done:
                break

        env.close()

        # Safeguard: if nothing reported, fill zeros
        if final_kpis is None:
            final_kpis = {
                "completed_vehicles": 0.0,
                "throughput_veh_per_hour": 0.0,
                "mean_travel_time_s": 0.0,
                "mean_waiting_time_s": 0.0,
            }

        completed = float(final_kpis.get("completed_vehicles", 0.0))
        throughput_vph = float(final_kpis.get("throughput_veh_per_hour", 0.0))
        mean_travel_s = float(final_kpis.get("mean_travel_time_s", 0.0))
        mean_wait_s = float(final_kpis.get("mean_waiting_time_s", 0.0))

        all_completed.append(completed)
        all_throughput_vph.append(throughput_vph)
        all_mean_travel_s.append(mean_travel_s)
        all_mean_wait_s.append(mean_wait_s)
        all_episode_returns.append(episode_return)

        print(
            f"[Case {case_idx:02d}] "
            f"completed={completed:.2f} | throughput_vph={throughput_vph:.2f} | "
            f"mean_travel_s={mean_travel_s:.2f} | mean_wait_s={mean_wait_s:.2f} | "
            f"episode_return={episode_return:.2f}"
        )

    # --------- mean KPIs across all cases ---------
    mean_completed = float(np.mean(all_completed)) if all_completed else 0.0
    mean_throughput_vph = float(np.mean(all_throughput_vph)) if all_throughput_vph else 0.0
    mean_travel_s = float(np.mean(all_mean_travel_s)) if all_mean_travel_s else 0.0
    mean_wait_s = float(np.mean(all_mean_wait_s)) if all_mean_wait_s else 0.0
    mean_ep_return = float(np.mean(all_episode_returns)) if all_episode_returns else 0.0

    print("\n=== Mean KPIs across all evaluations ===")
    print(f"mean_completed={mean_completed:.2f}")
    print(f"mean_throughput_vph={mean_throughput_vph:.2f}")
    print(f"mean_travel_s={mean_travel_s:.2f}")
    print(f"mean_wait_s={mean_wait_s:.2f}")
    print(f"mean_episode_return={mean_ep_return:.2f}")
