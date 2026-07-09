"""
SUMO MARL Environment (v5): Explicit NxN core grid with 4-way junctions + FRINGE stubs.

- N×N core junctions (default 3×3), each with 4 directions.
- Boundary junctions get fringe (dead-end) stubs so every core junction has 4 approaches.
- Random flows are generated between fringe edges only (piecewise-stationary sections).
- Observation = 12 lane queues + 1 scalar current phase index (0..3) => shape (13,).
- Actions = Discrete(4) phase index; we install a protected-lefts 4-phase program.

KPI tracking (whole episode, cumulative so far; returned in info["network_kpis"] each step):
- completed_vehicles
- throughput_veh_per_hour
- mean_travel_time_s
- mean_waiting_time_s

Now uses TraCI online signals instead of parsing tripinfo.xml, so KPIs update during the run.
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    raise RuntimeError("Please set SUMO_HOME environment variable to your SUMO installation directory.")
import sumolib  # type: ignore


@dataclass
class TLSInfo:
    tls_id: str
    incoming_lanes: List[str]  # expected 12 for 4 approaches × 3 lanes


class SumoGridMARLRandomEnv:
    def __init__(
        self,
        grid_n: int = 3,
        lanes_per_edge: int = 3,
        spacing: float = 150.0,             # meters between core junctions
        fringe_len: float = 150.0,          # meters for fringe stubs
        step_length_internal: float = 1.0,  # SUMO internal time-step (seconds)
        sumo_steps_per_env_step: int = 5,
        episode_steps: int = 500,
        # Random traffic sections
        section_count: int = 50,
        # min_routes_per_section: int = 12,
        # max_routes_per_section: int = 60,
        min_flow_rate_vph: float = 100.0,
        max_flow_rate_vph: float = 2500.0,
        # Demand regime (None = original random fringe-to-fringe flows)
        regime: str | None = None,           # "corridor" | "cross" | "platoons" | "bursty"
        regime_intensity: float = 1.0,       # global multiplier on regime flow rates
        # General options:
        seed: int = 42,
        gui: bool = False,
        gui_delay_ms: int = 0,
        verbose: bool = False,                # controls our own prints
        suppress_sumo_output: bool = True,    # controls SUMO/TraCI console spam
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
        self.section_count = section_count
        self.min_routes_per_section = int(((grid_n*4)**2)*0.05)
        self.max_routes_per_section = int(((grid_n*4)**2)*0.2)
        self.min_flow_rate_vph = min_flow_rate_vph
        self.max_flow_rate_vph = max_flow_rate_vph
        if regime is not None:
            from marl_utils.demand_regimes import REGIMES
            if regime not in REGIMES:
                raise ValueError(f"Unknown regime '{regime}'; expected one of {REGIMES}")
        self.regime = regime
        self.regime_intensity = float(regime_intensity)
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

        self._rng = np.random.default_rng(seed)

        self.tmpdir = tempfile.mkdtemp(prefix="sumo_grid_fringe_")
        self.net_file = os.path.join(self.tmpdir, "grid.net.xml")
        self.node_file = os.path.join(self.tmpdir, "nodes.nod.xml")
        self.edge_file = os.path.join(self.tmpdir, "edges.edg.xml")
        self.route_file = os.path.join(self.tmpdir, "routes.rou.xml")
        self.trip_file = os.path.join(self.tmpdir, "trips.trips.xml")
        self.cfg_file = os.path.join(self.tmpdir, "sim.sumocfg")

        # Build network (nodes+edges -> netconvert)
        self._write_nodes_and_edges()
        self._build_net(step_length_internal)
        self._write_empty_routes()
        self._write_sumocfg(step_length_internal)
        # Keep a sumolib handle for topology lookups
        self._net = sumolib.net.readNet(self.net_file)

        # Populated after reset via TraCI
        self.tls_infos: List[TLSInfo] = []
        self.agent_ids: List[str] = []
        self.num_agents: int = 0

        self.action_spaces: Dict[str, spaces.Discrete] = {}
        self.observation_spaces: Dict[str, spaces.Box] = {}

        self._step_count = 0
        self._connected = False

        # Episode KPI accumulators (online via TraCI)
        self._kpi_completed: int = 0
        self._kpi_total_travel_time: float = 0.0
        self._kpi_total_waiting_time: float = 0.0
        # Per-vehicle bookkeeping
        self._veh_depart_time: Dict[str, float] = {}
        self._veh_wait_time_last: Dict[str, float] = {}
        self._veh_wait_custom: Dict[str, float] = {}

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
        # Fringe nodes
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
                eroot,
                "edge",
                attrib={
                    "id": eid,
                    "from": fr,
                    "to": to,
                    "numLanes": str(self.lanes_per_edge),
                    "speed": "16.7",
                },
            )

        # Core grid (bidirectional)
        for i in range(N):
            for j in range(N):
                if i < N - 1:
                    add_edge(f"e_{i}_{j}_to_{i+1}_{j}", f"n_{i}_{j}", f"n_{i+1}_{j}")
                    add_edge(f"e_{i+1}_{j}_to_{i}_{j}", f"n_{i+1}_{j}", f"n_{i}_{j}")
                if j < N - 1:
                    add_edge(f"e_{i}_{j}_to_{i}_{j+1}", f"n_{i}_{j}", f"n_{i}_{j+1}")
                    add_edge(f"e_{i}_{j+1}_to_{i}_{j}", f"n_{i}_{j+1}", f"n_{i}_{j}")

        # Fringe stubs (bidirectional)
        # West
        for j in range(N):
            add_edge(f"e_fw{j}_to_n0_{j}", f"fw_{j}", f"n_0_{j}")
            add_edge(f"e_n0_{j}_to_fw{j}", f"n_0_{j}", f"fw_{j}")
        # East
        for j in range(N):
            add_edge(f"e_fe{j}_to_n{N-1}_{j}", f"fe_{j}", f"n_{N-1}_{j}")
            add_edge(f"e_n{N-1}_{j}_to_fe{j}", f"n_{N-1}_{j}", f"fe_{j}")
        # South   (fixed stray brace)
        for i in range(N):
            add_edge(f"e_fs{i}_to_n{i}_0", f"fs_{i}", f"n_{i}_0")
            add_edge(f"e_n{i}_0_to_fs{i}", f"n_{i}_0", f"fs_{i}")
        # North
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
            netconvert,
            "--node-files", self.node_file,
            "--edge-files", self.edge_file,
            "-o", self.net_file,
            "--tls.guess",
            "--no-turnarounds", "true",
        ]
        subprocess.run(cmd, check=True, **kwargs)

    def _write_empty_routes(self):
        root = ET.Element("routes")
        ET.SubElement(root, "vType",
                      id="car", accel="2.0", decel="4.5", sigma="0.5",
                      length="3.5", width="1.0", maxSpeed="16.7",
                      vClass="passenger", guiShape="passenger")
        ET.ElementTree(root).write(self.route_file, encoding="utf-8", xml_declaration=True)

    def _write_sumocfg(self, step_length_internal: float):
        cfg = ET.Element("configuration")
        input_el = ET.SubElement(cfg, "input")
        ET.SubElement(input_el, "net-file", value=os.path.basename(self.net_file))
        ET.SubElement(input_el, "route-files", value=os.path.basename(self.route_file))
        time_el = ET.SubElement(cfg, "time")
        ET.SubElement(time_el, "step-length", value=str(step_length_internal))
        ET.SubElement(time_el, "begin", value="0")
        ET.SubElement(time_el, "end", value=str(self.episode_steps * self.sumo_steps_per_env_step * step_length_internal))
        out = os.path.join(self.tmpdir, "sim.sumocfg")
        ET.ElementTree(cfg).write(out, encoding="utf-8", xml_declaration=True)

    # ---------------------- TLS discovery via TraCI ----------------------
    def _discover_tls_infos_traci(self) -> List[TLSInfo]:
        tls_ids = list(traci.trafficlight.getIDList())
        tls_infos: List[TLSInfo] = []

        for tls_id in sorted(tls_ids):
            links = traci.trafficlight.getControlledLinks(tls_id)
            incoming: List[str] = []
            seen = set()

            # First pass: prefer lanes whose lane index is within lanes_per_edge
            for group in links:
                for link in (group if isinstance(group, (list, tuple)) else [group]):
                    in_lane = link[0]
                    if in_lane.startswith(":"):
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

            # Fallback: just take any remaining incoming lanes until we hit target count
            if len(incoming) < 4 * self.lanes_per_edge:
                for group in links:
                    for link in (group if isinstance(group, (list, tuple)) else [group]):
                        in_lane = link[0]
                        if in_lane.startswith(":") or in_lane in seen:
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

    # ---------------------- Random fringe-to-fringe flows ----------------------
    def _generate_random_routes_for_episode(self):
        net = sumolib.net.readNet(self.net_file)
        nodes_by_id = {n.getID(): n for n in net.getNodes()}
        def node_type(nid):
            n = nodes_by_id[nid]
            return n.getType() or ""
        in_edges = []
        out_edges = []
        for e in net.getEdges():
            fr = e.getFromNode().getID(); to = e.getToNode().getID()
            frt = node_type(fr); tot = node_type(to)
            if frt == "dead_end" and tot == "traffic_light":
                in_edges.append(e)
            if frt == "traffic_light" and tot == "dead_end":
                out_edges.append(e)
        ods: List[Tuple[str, str]] = []
        for src in in_edges:
            for dst in out_edges:
                if src.getToNode().getID() == dst.getFromNode().getID():
                    continue
                ods.append((src.getID(), dst.getID()))

        total_time = float(self.episode_steps * self.sumo_steps_per_env_step) * self.step_length_internal
        section_len = total_time / float(self.section_count)

        trips_root = ET.Element("routes")
        ET.SubElement(trips_root, "vType",
                      id="car", accel="2.0", decel="4.5", sigma="0.5",
                      length="3.5", width="1.0", maxSpeed="16.7",
                      vClass="passenger", guiShape="passenger")

        rng = self._rng
        flow_id = 0
        if len(ods) > 0:
            for s in range(self.section_count):
                begin = s * section_len
                end = (s + 1) * section_len
                avail = len(ods)
                if avail == 0:
                    break
                k_max = min(self.max_routes_per_section, avail)
                if k_max <= 0:
                    continue
                if s == 0:
                    k_min = min(max(1, self.min_routes_per_section), k_max)
                    k = int(rng.integers(k_min, k_max + 1))
                else:
                    k = int(rng.integers(0, k_max + 1))
                if k == 0:
                    continue
                idxs = rng.choice(avail, size=k, replace=False)
                for idx in idxs:
                    src, dst = ods[idx]
                    rate = float(rng.uniform(self.min_flow_rate_vph, self.max_flow_rate_vph))
                    period = 3600.0 / rate if rate > 1e-6 else 1e9
                    ET.SubElement(
                        trips_root, "flow",
                        id=f"sec{s}_f{flow_id}", type="car",
                        begin=f"{begin:.1f}", end=f"{end:.1f}",
                        period=f"{period:.6f}",
                        **{"from": src, "to": dst},
                    )
                    flow_id += 1
        ET.ElementTree(trips_root).write(self.trip_file, encoding="utf-8", xml_declaration=True)
        self._duaroute_trips_to_routes()

    def _generate_regime_routes_for_episode(self):
        """Regime-conditioned demand: fresh draw from the episode RNG each reset."""
        from marl_utils.demand_regimes import write_regime_trips
        total_time = float(self.episode_steps * self.sumo_steps_per_env_step) * self.step_length_internal
        write_regime_trips(
            self.trip_file,
            regime=self.regime,
            grid_n=self.grid_n,
            sim_end=total_time,
            rng=self._rng,
            intensity=self.regime_intensity,
        )
        self._duaroute_trips_to_routes()

    def _duaroute_trips_to_routes(self):
        duarouter = os.path.join(os.environ["SUMO_HOME"], "bin", "duarouter")
        kwargs = {}
        if self.suppress_sumo_output:
            kwargs["stdout"] = subprocess.DEVNULL
            kwargs["stderr"] = subprocess.DEVNULL
        cmd = [
            duarouter,
            "-n", self.net_file,
            "-r", self.trip_file,
            "-o", self.route_file,
            "--ignore-errors",
        ]
        subprocess.run(cmd, check=True, **kwargs)

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
        # New flows each episode: regime-conditioned if set, else random fringe-to-fringe
        if self.regime is not None:
            self._generate_regime_routes_for_episode()
        else:
            self._generate_random_routes_for_episode()
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
            sumo_bin,
            "-c", self.cfg_file,
            "--seed", str(self.seed),
            "--no-warnings", "true",
            "--time-to-teleport", "-1",
            "--start",
            "--quit-on-end",
            "--no-step-log", "true",
            "--duration-log.disable", "true",
        ]
        if self.suppress_sumo_output:
            devnull = os.devnull
            cmd += ["--log", devnull, "--error-log", devnull, "--message-log", devnull]
        if self.gui and self.gui_delay_ms and self.gui_delay_ms > 0:
            cmd += ["--delay", str(int(self.gui_delay_ms))]

        if self.verbose:
            print(f"[SUMO] Launching {'SUMO-GUI' if self.gui else 'SUMO'}")

        # Silence TraCI "Retrying..." etc. if requested
        if self.suppress_sumo_output:
            import contextlib, sys
            @contextlib.contextmanager
            def _silence_stdio():
                old_out, old_err = sys.stdout, sys.stderr
                try:
                    with open(os.devnull, "w") as fnull:
                        sys.stdout = fnull
                        sys.stderr = fnull
                        yield
                finally:
                    sys.stdout, sys.stderr = old_out, old_err
            with _silence_stdio():
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

        # Install a 4-phase protected-lefts program
        self._install_four_phase_programs()

        return self._observe_all()

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        Env step:
        - Apply per-agent discrete phase actions (0..3)
        - Advance SUMO by self.sumo_steps_per_env_step internal steps
        - Update KPIs online:
            * completed-only totals via arrived vehicles
            * custom per-vehicle waiting time accumulator (speed-threshold based)
            * exact(er) depart timestamping at begin-of-step
            * additional "all vehicles (active+completed)" mean travel/wait metrics
        - Return obs, rewards, done, info
        """
        assert set(actions.keys()) == set(self.agent_ids)

        # Apply actions (phase indices 0..3)
        for aid, a in actions.items():
            traci.trafficlight.setPhase(aid, int(a) % 4)

        dt = float(self.step_length_internal)  # seconds per internal SUMO step

        # Advance SUMO in internal steps; update KPIs online
        for _ in range(self.sumo_steps_per_env_step):
            # --- begin-of-step time (more exact depart timestamps) ---
            t_before = float(traci.simulation.getTime())

            # --- update custom waiting accumulator for vehicles currently present ---
            # We define "waiting" as nearly stopped: speed < 0.1 m/s
            try:
                for vid in traci.vehicle.getIDList():
                    try:
                        if float(traci.vehicle.getSpeed(vid)) < 0.1:
                            self._veh_wait_custom[vid] = self._veh_wait_custom.get(vid, 0.0) + dt
                        else:
                            # ensure key exists so active vehicles are included in "all vehicles" stats
                            self._veh_wait_custom.setdefault(vid, 0.0)
                    except Exception:
                        # If a vehicle disappears between getIDList and getSpeed, ignore safely
                        pass
            except Exception:
                pass

            # --- advance simulation ---
            traci.simulationStep()
            t_after = float(traci.simulation.getTime())

            # --- record depart times for vehicles that entered this internal step ---
            # Stamp departures at begin-of-step (t_before)
            try:
                for vid in traci.simulation.getDepartedIDList():
                    self._veh_depart_time[vid] = t_before
                    self._veh_wait_custom.setdefault(vid, 0.0)
            except Exception:
                pass

            # --- finalise KPIs for vehicles that arrived this internal step ---
            try:
                for vid in traci.simulation.getArrivedIDList():
                    dep_t = self._veh_depart_time.pop(vid, t_before)  # fallback: t_before
                    travel = max(t_after - dep_t, 0.0)

                    # Use our custom waiting accumulator (closer to "final" since it persists past arrival)
                    wait = float(self._veh_wait_custom.pop(vid, 0.0))

                    self._kpi_total_travel_time += travel
                    self._kpi_total_waiting_time += wait
                    self._kpi_completed += 1
            except Exception:
                pass

        self._step_count += 1

        observations = self._observe_all()

        # --- rewards: raw -> normalized ---
        raw_rewards = self._compute_raw_rewards(observations)
        rewards = self._normalize_rewards(raw_rewards)

        # --- KPI snapshot ---
        t_now = float(traci.simulation.getTime())
        elapsed_sim_time = max(t_now, 1e-6)  # seconds

        completed = int(self._kpi_completed)
        throughput_per_hour = (completed / elapsed_sim_time) * 3600.0

        # Completed-only means (trip averages for arrived vehicles)
        mean_travel_completed = (self._kpi_total_travel_time / completed) if completed > 0 else 0.0
        mean_wait_completed = (self._kpi_total_waiting_time / completed) if completed > 0 else 0.0

        # Active vehicles (present in the system): those that have departed but not arrived
        active_ids = list(self._veh_depart_time.keys())
        active = len(active_ids)

        sum_active_travel = 0.0
        sum_active_wait = 0.0
        for vid in active_ids:
            dep_t = float(self._veh_depart_time.get(vid, t_now))
            sum_active_travel += max(t_now - dep_t, 0.0)
            sum_active_wait += float(self._veh_wait_custom.get(vid, 0.0))

        total_veh = completed + active

        # "All vehicles" means (completed + active partial trips)
        mean_travel_all = ((self._kpi_total_travel_time + sum_active_travel) / total_veh) if total_veh > 0 else 0.0
        mean_wait_all = ((self._kpi_total_waiting_time + sum_active_wait) / total_veh) if total_veh > 0 else 0.0

        done = self._step_count >= self.episode_steps
        info = {
            "network_kpis": {
                "completed_vehicles": int(completed),
                "active_vehicles": int(active),
                "total_vehicles_seen_or_active": int(total_veh),
                "throughput_veh_per_hour": float(throughput_per_hour),

                # Completed-only (old behaviour, but now with improved waiting/depart stamping)
                "mean_travel_time_s_completed": float(mean_travel_completed),
                "mean_waiting_time_s_completed": float(mean_wait_completed),

                # All vehicles currently present + completed (new behaviour)
                "mean_travel_time_s": float(mean_travel_all),
                "mean_waiting_time_s": float(mean_wait_all),
            },
            "raw_rewards": raw_rewards,  # keep original (unscaled) for logging
        }

        if done:
            self.close()

        return observations, rewards, done, info

    def close(self):
        if self._connected:
            try:
                traci.close()
            finally:
                self._connected = False

    # ---------------------- Helpers ----------------------
    def _install_four_phase_programs(self):
        """Install a custom true protected-lefts 4-phase program and select it."""
        net = self._net

        def parse_ij(nid: str):
            try:
                _, i, j = nid.split('_')
                return int(i), int(j)
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
                    if not link or len(link) < 2:
                        continue
                    in_lane = link[0]
                    if in_lane and not in_lane.startswith(":"):
                        chosen = link
                        break
                if chosen is None and len(links) > 0:
                    chosen = links[0]
                if not chosen:
                    meta.append(('?', '?'))
                    continue

                in_lane, out_lane = chosen[0], chosen[1]
                try:
                    in_edge_id = in_lane.rsplit('_', 1)[0]
                    out_edge_id = out_lane.rsplit('_', 1)[0]
                    in_edge = net.getEdge(in_edge_id)
                    out_edge = net.getEdge(out_edge_id)
                    in_from = in_edge.getFromNode().getID()
                    out_to  = out_edge.getToNode().getID()
                except Exception:
                    meta.append(('?', '?'))
                    continue

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
                    if phase_idx == 0:  # NS LEFTs only
                        allow = (appr in ('N','S')) and (mov == 'L')
                    elif phase_idx == 1:  # NS THROUGH+RIGHT
                        allow = (appr in ('N','S')) and (mov in ('T','R'))
                    elif phase_idx == 2:  # EW LEFTs only
                        allow = (appr in ('E','W')) and (mov == 'L')
                    elif phase_idx == 3:  # EW THROUGH+RIGHT
                        allow = (appr in ('E','W')) and (mov in ('T','R'))
                    chars.append('G' if allow else 'r')
                s = ''.join(chars)
                return s[:k_live].ljust(k_live, 'r')

            states = [build_state(p) for p in range(4)]

            TLLogic = traci.trafficlight.Logic
            TLPhase = traci.trafficlight.Phase
            phases = [
                TLPhase(30, states[0]),  # 0: NS lefts
                TLPhase(30, states[1]),  # 1: NS thru+right
                TLPhase(30, states[2]),  # 2: EW lefts
                TLPhase(30, states[3]),  # 3: EW thru+right
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Use SUMO-GUI if set")
    parser.add_argument("--gui-delay-ms", type=int, default=0, help="Delay per SUMO step in GUI")
    parser.add_argument("--grid-n", type=int, default=3, help="Grid size N")
    parser.add_argument("--steps", type=int, default=100, help="Episode steps")
    parser.add_argument("--rotate-every", type=int, default=5, help="Rotate action every N env steps")
    args = parser.parse_args()

    env = SumoGridMARLRandomEnv(
        gui=args.gui,
        gui_delay_ms=args.gui_delay_ms,
        grid_n=args.grid_n,
        episode_steps=args.steps,
        verbose=True
    )
    obs = env.reset()
    print(f"Agents ({len(env.agent_ids)}): {env.agent_ids}")
    for t in range(env.episode_steps):
        # Timed action rotation: 0 -> 1 -> 2 -> 3 -> 0 -> ...
        actions = {aid: ((t // args.rotate_every) % 4) for aid in env.agent_ids}
        obs, rew, done, info = env.step(actions)
        if (t + 1) % 10 == 0:
                mean_q = np.mean([np.sum(o[:12]) for o in obs.values()])
                kpi = info.get("network_kpis", {})
                completed = float(kpi.get("completed_vehicles", 0.0))
                throughput_vph = float(kpi.get("throughput_veh_per_hour", 0.0))
                mean_travel_s = float(kpi.get("mean_travel_time_s", 0.0))
                mean_wait_s = float(kpi.get("mean_waiting_time_s", 0.0))
                print(
                    f"Step {t+1:>4} | mean_queue={mean_q:.2f} | "
                    f"completed={completed:.2f} | throughput_vph={throughput_vph:.2f} | "
                    f"mean_travel_s={mean_travel_s:.2f} | mean_wait_s={mean_wait_s:.2f}"
                )
        if done:
            break
    env.close()
