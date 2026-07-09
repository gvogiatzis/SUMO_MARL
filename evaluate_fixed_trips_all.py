#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

import numpy as np
import torch
import torch.nn as nn

# --- Your modules (assumed available on PYTHONPATH) ---
from sumo_marl_fixed_routes_env import SumoGridMARLFixedEnv
from marl_utils.models import (
    # DQN / DRQN families
    QNetMLP,
    RecurrentQNet,
    GNNPolicyQ,
    GNNLSTMPolicyQ,
    CoLightQ,
    # A2C actor families
    ActorMLP,
    ActorLSTM,
    ActorGNNAttn,
    ActorGNNLSTMAttn,
)
from marl_utils.common import build_grid_edge_index


# ----------------------------- method families -----------------------------
Q_METHODS = {
    "dqn_mlp", "drqn_lstm", "dqn_gnn", "drqn_gnn_lstm",
    "ctde_vdn_mlp", "ctde_vdn_lstm", "ctde_vdn_gnn", "ctde_vdn_gnn_lstm",
    "colight",
}
A2C_METHODS = {
    "ia2c_mlp", "ia2c_lstm", "ia2c_gnn", "ia2c_gnn_lstm",
    "ma2c_pa_mlp", "ma2c_pa_lstm", "ma2c_pa_gnn", "ma2c_pa_gnn_lstm",
}

DEFAULT_N_EVALS = 20


# ----------------------------- model + checkpoint helpers -----------------------------
def build_model(method: str, obs_dim: int, act_dim: int, hidden_q: int, hidden_a2c: int, gnn_layers: int) -> nn.Module:
    """Pick the right net + hidden size based on method family (Q vs A2C)."""
    m = method.lower()
    if m in Q_METHODS:
        hidden = hidden_q
        if m == "dqn_mlp" or m == "ctde_vdn_mlp":
            return QNetMLP(observation_dim=obs_dim, action_dim=act_dim, hidden_units=hidden)
        elif m == "drqn_lstm" or m == "ctde_vdn_lstm":
            return RecurrentQNet(observation_dim=obs_dim, action_dim=act_dim, hidden_units=hidden)
        elif m == "dqn_gnn" or m == "ctde_vdn_gnn":
            return GNNPolicyQ(node_dim=obs_dim, actions=act_dim, hidden=hidden, layers=gnn_layers)
        elif m == "drqn_gnn_lstm" or m == "ctde_vdn_gnn_lstm":
            return GNNLSTMPolicyQ(node_dim=obs_dim, actions=act_dim, hidden=hidden, gnn_layers=gnn_layers)
        elif m == "colight":
            return CoLightQ(node_dim=obs_dim, actions=act_dim, hidden=hidden)
    elif m in A2C_METHODS:
        hidden = hidden_a2c
        if m == "ia2c_mlp" or m == "ma2c_pa_mlp":
            return ActorMLP(obs_dim=obs_dim, action_dim=act_dim, hidden=hidden)
        elif m == "ia2c_lstm" or m == "ma2c_pa_lstm":
            return ActorLSTM(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden)
        elif m == "ia2c_gnn" or m == "ma2c_pa_gnn":
            return ActorGNNAttn(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, layers=gnn_layers, edge_dim=0)
        elif m == "ia2c_gnn_lstm" or m == "ma2c_pa_gnn_lstm":
            return ActorGNNLSTMAttn(obs_dim=obs_dim, act_dim=act_dim, hidden=hidden, layers=gnn_layers, edge_dim=0)

    raise ValueError(f"Unknown method: {method}")


def try_paths_in_order(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def checkpoint_path_for(method: str, grid_n: int, seed: int, logs_base: Path) -> Optional[Path]:
    """
    Uses your provided filenames for all Q and A2C variants.
    """
    d = logs_base / f"logs_grid_{grid_n}" / f"seed{seed}"
    m = method.lower()

    names_primary = {
        # ----- Q/VDN -----
        "dqn_mlp"           : f"model_best_dqn_mlp_shared_seed{seed}.pt",
        "drqn_lstm"         : f"model_best_drqn_lstm_shared_seqlen8_seed{seed}.pt",
        "dqn_gnn"           : f"model_best_dqn_gnn_shared_seed{seed}.pt",
        "colight"           : f"model_best_colight_shared_seed{seed}.pt",
        "drqn_gnn_lstm"     : f"model_best_drqn_gnn_lstm_shared_seqlen8_seed{seed}.pt",
        "ctde_vdn_mlp"      : f"model_best_vdn_ctde_mlp_shared_seed{seed}.pt",
        "ctde_vdn_lstm"     : f"model_best_vdn_ctde_lstm_shared_seqlen8_seed{seed}.pt",
        "ctde_vdn_gnn"      : f"model_best_vdn_ctde_gnn_shared_seed{seed}.pt",
        "ctde_vdn_gnn_lstm" : f"model_best_vdn_ctde_gnn_lstm_shared_seqlen8_seed{seed}.pt",
        # ----- A2C actors -----
        "ia2c_mlp"          : f"model_best_ia2c_mlp_shared_seed{seed}.pt",
        "ia2c_lstm"         : f"model_best_ia2c_lstm_shared_seed{seed}.pt",
        "ia2c_gnn"          : f"model_best_ia2c_gnn_shared_seed{seed}.pt",
        "ia2c_gnn_lstm"     : f"model_best_ia2c_gnn_lstm_shared_seed{seed}.pt",
        "ma2c_pa_mlp"       : f"model_best_ma2c_pa_mlp_seed{seed}.pt",
        "ma2c_pa_lstm"      : f"model_best_ma2c_pa_lstm_seed{seed}.pt",
        "ma2c_pa_gnn"       : f"model_best_ma2c_pa_gnn_seed{seed}.pt",
        "ma2c_pa_gnn_lstm"  : f"model_best_ma2c_pa_gnn_lstm_seed{seed}.pt",
    }

    candidates = [d / names_primary[m]]
    return try_paths_in_order(candidates)


def load_checkpoint(model: nn.Module, ckpt_path: Path, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)


# ----------------------------- action helpers -----------------------------
@torch.no_grad()
def act_mlp_q(model: QNetMLP, obs_mat: np.ndarray, device: torch.device) -> List[int]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    q = model(X)
    return torch.argmax(q, dim=-1).tolist()


@torch.no_grad()
def act_lstm_q(
    model: RecurrentQNet,
    obs_mat: np.ndarray,
    device: torch.device,
    rnn: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    x = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(1)
    q_seq, rnn = model(x, rnn)
    return torch.argmax(q_seq[:, -1, :], dim=-1).tolist(), rnn


@torch.no_grad()
def act_gnn_q(model: GNNPolicyQ, obs_mat: np.ndarray, edge_index: torch.Tensor, device: torch.device) -> List[int]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    q = model(X, edge_index.to(device))
    return torch.argmax(q, dim=-1).tolist()


@torch.no_grad()
def act_gnn_lstm_q(
    model: GNNLSTMPolicyQ,
    obs_mat: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    rnn: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device).unsqueeze(0)
    q_t, rnn = model.step(X, edge_index.to(device), rnn)
    return torch.argmax(q_t[0], dim=-1).tolist(), rnn


@torch.no_grad()
def act_mlp_actor(model: ActorMLP, obs_mat: np.ndarray, device: torch.device) -> List[int]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    logits = model(X)
    return torch.argmax(logits, dim=-1).tolist()


@torch.no_grad()
def act_lstm_actor(
    model: ActorLSTM,
    obs_mat: np.ndarray,
    device: torch.device,
    rnn: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    logits, rnn = model.step(X, rnn)
    return torch.argmax(logits, dim=-1).tolist(), rnn


@torch.no_grad()
def act_gnn_actor(model: ActorGNNAttn, obs_mat: np.ndarray, edge_index: torch.Tensor, device: torch.device) -> List[int]:
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    logits = model(X, edge_index.to(device))
    return torch.argmax(logits, dim=-1).tolist()


@torch.no_grad()
def act_gnn_lstm_actor(
    model: ActorGNNLSTMAttn,
    obs_mat: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    rnn: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    X = torch.as_tensor(obs_mat, dtype=torch.float32, device=device)
    logits, rnn = model.step(X, edge_index.to(device), None, rnn)
    return torch.argmax(logits, dim=-1).tolist(), rnn


# ----------------------------- one-episode runner -----------------------------
def run_single_episode(
    env: SumoGridMARLFixedEnv,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Assumes env.seed is already set.
    Runs one episode and returns KPI dict for that episode.
    """
    obs = env.reset()
    agent_ids = list(env.agent_ids)
    edge_index = build_grid_edge_index(agent_ids)

    is_q_mlp = isinstance(model, QNetMLP)
    is_q_lstm = isinstance(model, RecurrentQNet)
    is_q_gnn = isinstance(model, (GNNPolicyQ, CoLightQ))  # same call signature
    is_q_gnnl = isinstance(model, GNNLSTMPolicyQ)

    is_pi_mlp = isinstance(model, ActorMLP)
    is_pi_lstm = isinstance(model, ActorLSTM)
    is_pi_gnn = isinstance(model, ActorGNNAttn)
    is_pi_gnnl = isinstance(model, ActorGNNLSTMAttn)

    rnn_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    episode_return = 0.0
    final_kpis = None

    for _ in range(env.episode_steps):
        obs_mat = np.stack([obs[a] for a in agent_ids], axis=0).astype(np.float32)

        if is_q_mlp:
            acts = act_mlp_q(model, obs_mat, device)
        elif is_q_lstm:
            acts, rnn_state = act_lstm_q(model, obs_mat, device, rnn_state)
        elif is_q_gnn:
            acts = act_gnn_q(model, obs_mat, edge_index, device)
        elif is_q_gnnl:
            acts, rnn_state = act_gnn_lstm_q(model, obs_mat, edge_index, device, rnn_state)
        elif is_pi_mlp:
            acts = act_mlp_actor(model, obs_mat, device)
        elif is_pi_lstm:
            acts, rnn_state = act_lstm_actor(model, obs_mat, device, rnn_state)
        elif is_pi_gnn:
            acts = act_gnn_actor(model, obs_mat, edge_index, device)
        elif is_pi_gnnl:
            acts, rnn_state = act_gnn_lstm_actor(model, obs_mat, edge_index, device, rnn_state)
        else:
            raise RuntimeError("Unsupported model type for evaluation.")

        action_dict = {aid: int(acts[i]) for i, aid in enumerate(agent_ids)}
        obs, rew, done, info = env.step(action_dict)

        episode_return += float(np.sum(list(rew.values())))
        final_kpis = info.get("network_kpis", None)

        if done:
            break

    if final_kpis is None:
        final_kpis = {
            "throughput_veh_per_hour": 0.0,
            "mean_travel_time_s": 0.0,
            "mean_waiting_time_s": 0.0,
        }

    return {
        "throughput_vph": float(final_kpis.get("throughput_veh_per_hour", 0.0)),
        "mean_travel_s": float(final_kpis.get("mean_travel_time_s", 0.0)),
        "mean_wait_s": float(final_kpis.get("mean_waiting_time_s", 0.0)),
        "episode_return": float(episode_return),
    }


# ----------------------------- trip resolver by type -----------------------------
def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_trips_by_type(trips_root: Path, grid_n: int, trip_type: str) -> Path:
    """
    Preferred folder pattern:
      - Temporal: trips_root / eval_trips_temporal_grid_{N} / <temporal_filename>.xml
      - Spatial : trips_root / eval_trips_spatial_grid_{N}  / <spatial_filename>.xml
    """
    t = trip_type.lower()
    temporal_dir = trips_root / f"eval_trips_temporal_grid_{grid_n}"
    spatial_dir = trips_root / f"eval_trips_spatial_grid_{grid_n}"

    if t == "platoons":
        preferred = [temporal_dir / f"eval_trips_t01_platoons_pulse_trains_gridN{grid_n}.xml"]
        glob_keys = ["platoons", "pulse_trains"]
    elif t == "bursty":
        preferred = [temporal_dir / f"eval_trips_t02_bursty_on_off_gridN{grid_n}.xml"]
        glob_keys = ["bursty_on_off", "bursty"]
    elif t == "shifted":
        preferred = [
            temporal_dir / f"eval_trips_t03_shifted_directional_peaks_gridN{grid_n}.xml",
            temporal_dir / f"eval_trips_t03_shifted_peaks_dir_staggered_gridN{grid_n}.xml",
        ]
        glob_keys = ["shifted_directional", "shifted_peaks", "shifted"]
    elif t == "ampm":
        preferred = [temporal_dir / f"eval_trips_t04_ampm_global_gridN{grid_n}.xml"]
        glob_keys = ["ampm_global", "ampm"]
    elif t == "incident":
        preferred = [temporal_dir / f"eval_trips_t05_incident_west_boundary_gridN{grid_n}.xml"]
        glob_keys = ["incident_west_boundary", "incident"]
    elif t == "corridor":
        preferred = [
            spatial_dir / f"eval_trips_s01_heavy_corridor_ew_arterial_gridN{grid_n}.xml",
            spatial_dir / f"eval_trips_t06_heavy_corridor_ew_arterial_gridN{grid_n}.xml",
        ]
        glob_keys = ["heavy_corridor_ew_arterial", "heavy_corridor", "corridor"]
    elif t == "cross":
        preferred = [
            spatial_dir / f"eval_trips_s02_cross_orthogonal_axes_gridN{grid_n}.xml",
            spatial_dir / f"eval_trips_t07_cross_orthogonal_axes_gridN{grid_n}.xml",
        ]
        glob_keys = ["cross_orthogonal_axes", "orthogonal_axes", "cross"]
    else:
        raise ValueError(f"Unknown trip type: {trip_type}")

    p = _first_existing(preferred)
    if p is not None:
        return p

    patterns = []
    for key in glob_keys:
        patterns.append(str(trips_root / f"**/*{key}*gridN{grid_n}.xml"))

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))

    if matches:
        matches.sort()
        return Path(matches[0]).resolve()

    raise FileNotFoundError(
        f"Trips file not found for type='{trip_type}' grid_n={grid_n} under {trips_root}"
    )


# ----------------------------- CSV writer -----------------------------
def write_csv(rows: List[Dict[str, object]], out_path: Path):
    import csv

    fieldnames = [
        "grid_n", "seed", "method",
        "throughput_vph_mean", "throughput_vph_std",
        "mean_travel_s_mean", "mean_travel_s_std",
        "mean_wait_s_mean", "mean_wait_s_std",
        "episode_return_mean", "episode_return_std",
        "steps", "sumo_steps_per_env_step", "n_evals",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nSaved CSV to: {out_path}")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser("Series evaluation with repeated SUMO seeds; report mean±std.")

    ap.add_argument(
        "--trip-type",
        type=str,
        default="platoons",
        choices=["platoons", "bursty", "shifted", "ampm", "incident", "corridor", "cross"],
        help="Which fixed trip pattern to evaluate.",
    )

    ap.add_argument(
        "--grids",
        type=int,
        nargs="*",
        default=[3, 5, 10],
        help="Grid sizes to evaluate",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for model folders/checkpoint naming",
    )

    ap.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[
            # Q/VDN
            "dqn_mlp", "drqn_lstm", "dqn_gnn", "drqn_gnn_lstm",
            "ctde_vdn_mlp", "ctde_vdn_lstm", "ctde_vdn_gnn", "ctde_vdn_gnn_lstm",
            # A2C
            "ia2c_mlp", "ia2c_lstm", "ia2c_gnn", "ia2c_gnn_lstm",
            "ma2c_pa_mlp", "ma2c_pa_lstm", "ma2c_pa_gnn", "ma2c_pa_gnn_lstm",
        ],
        help="Methods to evaluate",
    )

    ap.add_argument(
        "--n-evals",
        type=int,
        default=DEFAULT_N_EVALS,
        help="Number of repeated evaluations using shared SUMO seeds.",
    )

    ap.add_argument("--logs-base", type=str, default=".", help="Base dir containing logs_grid_N/seedSEED/")
    ap.add_argument("--trips-root", type=str, default=".", help="Root dir containing eval_trips_*_grid_N/")
    ap.add_argument("--steps", type=int, default=200, help="Episode steps")
    ap.add_argument("--sumo-steps-per-env-step", type=int, default=5, help="SUMO internal steps per env step")

    # --- Separate hidden sizes ---
    ap.add_argument("--hidden-q", type=int, default=128, help="Hidden width for Q/VDN models")
    ap.add_argument("--hidden-a2c", type=int, default=128, help="Hidden width for A2C actor models")
    ap.add_argument("--gnn-layers", type=int, default=2, help="GNN layers for GNN(-LSTM) models")

    ap.add_argument("--duarouter-seed", type=int, default=0, help="Seed for duarouter")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--gui", action="store_true", help="Show SUMO GUI")
    ap.add_argument("--gui-delay-ms", type=int, default=0, help="Delay per SUMO step in GUI")
    ap.add_argument("--verbose", action="store_true", help="Verbose SUMO logs")

    # ------------------------------------------------------------------
    # Minimal additions for reproducible/debuggable evaluation
    # ------------------------------------------------------------------
    ap.add_argument(
        "--eval-seed",
        type=int,
        default=12345,
        help="Seed used to generate the shared list of SUMO evaluation seeds.",
    )

    ap.add_argument(
        "--deterministic-sumo-seed",
        type=int,
        default=None,
        help="If set, use this same SUMO seed for all repeated evaluations.",
    )

    ap.add_argument(
        "--print-each-eval",
        action="store_true",
        help="Print KPI values for every individual evaluation episode.",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    logs_base = Path(args.logs_base).resolve()
    trips_root = Path(args.trips_root).resolve()

    out_path = Path(
        f"eval_fixed_demand_type/eval_results_fixed_trip_{args.trip_type}_seed{args.seed}.csv"
    ).resolve()

    # ------------------------------------------------------------------
    # Updated SUMO seed generation
    # ------------------------------------------------------------------
    n_evals = int(args.n_evals)

    if args.deterministic_sumo_seed is not None:
        sumo_seeds = [int(args.deterministic_sumo_seed)] * n_evals
    else:
        rng = np.random.default_rng(args.eval_seed)
        sumo_seeds = rng.integers(
            0,
            2**31 - 1,
            size=n_evals,
            dtype=np.int64,
        ).tolist()

    print(f"[INFO] Using the same {n_evals} SUMO seeds for all grids/methods:")
    print(f"       {sumo_seeds}")

    rows: List[Dict[str, object]] = []

    for grid_n in args.grids:
        try:
            trips_file = resolve_trips_by_type(trips_root, grid_n, args.trip_type)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n=== Grid {grid_n} — {args.trip_type}: {e}")
            print("    Skipping this grid.")
            continue

        print(f"\n=== Grid {grid_n} — trips: {trips_file} — model seed folder: {args.seed} ===")

        env = SumoGridMARLFixedEnv(
            grid_n=grid_n,
            episode_steps=args.steps,
            sumo_steps_per_env_step=args.sumo_steps_per_env_step,
            fixed_trips_file=str(trips_file),
            duarouter_seed=args.duarouter_seed,
            gui=args.gui,
            gui_delay_ms=args.gui_delay_ms,
            seed=int(sumo_seeds[0]),
            verbose=args.verbose,
            suppress_sumo_output=not args.verbose,
        )

        # Probe obs/action dims
        env.seed = int(sumo_seeds[0])
        obs_probe = env.reset()
        agent_ids_probe = list(env.agent_ids)
        O = len(obs_probe[agent_ids_probe[0]])
        A = env.action_spaces[agent_ids_probe[0]].n
        env.close()

        for method in args.methods:
            ckpt = checkpoint_path_for(method, grid_n, args.seed, logs_base)

            if ckpt is None:
                print(f"    [SKIP] {method}: checkpoint not found under logs_grid_{grid_n}/seed{args.seed}/")
                continue

            model = build_model(
                method,
                O,
                A,
                args.hidden_q,
                args.hidden_a2c,
                args.gnn_layers,
            ).to(device)

            load_checkpoint(model, ckpt, device)
            model.eval()

            tp_vals, mtt_vals, awt_vals, ret_vals = [], [], [], []

            print(f"    Eval {method:>18s}  |  {ckpt.name}")

            for i, s in enumerate(sumo_seeds, start=1):
                env.seed = int(s)

                try:
                    metrics = run_single_episode(env, model, device)
                except Exception as e:
                    print(f"      !! Eval {i:02d}/{n_evals} failed (seed={s}): {e}")
                    continue

                if args.print_each_eval:
                    print(
                        f"      Eval {i:02d}/{n_evals} seed={s} "
                        f"TP={metrics['throughput_vph']:.2f} "
                        f"MTT={metrics['mean_travel_s']:.2f} "
                        f"AWT={metrics['mean_wait_s']:.2f} "
                        f"Ret={metrics['episode_return']:.2f}"
                    )

                tp_vals.append(metrics["throughput_vph"])
                mtt_vals.append(metrics["mean_travel_s"])
                awt_vals.append(metrics["mean_wait_s"])
                ret_vals.append(metrics["episode_return"])

            tp_mean, tp_std = (
                float(np.mean(tp_vals)) if tp_vals else 0.0,
                float(np.std(tp_vals, ddof=0)) if tp_vals else 0.0,
            )

            mtt_mean, mtt_std = (
                float(np.mean(mtt_vals)) if mtt_vals else 0.0,
                float(np.std(mtt_vals, ddof=0)) if mtt_vals else 0.0,
            )

            awt_mean, awt_std = (
                float(np.mean(awt_vals)) if awt_vals else 0.0,
                float(np.std(awt_vals, ddof=0)) if awt_vals else 0.0,
            )

            ret_mean, ret_std = (
                float(np.mean(ret_vals)) if ret_vals else 0.0,
                float(np.std(ret_vals, ddof=0)) if ret_vals else 0.0,
            )

            print(
                f"      KPIs (mean ± std over {len(tp_vals)}/{n_evals} evals)"
                f" -> TP: {tp_mean:.2f} ± {tp_std:.2f} | "
                f"MTT: {mtt_mean:.2f} ± {mtt_std:.2f} | "
                f"AWT: {awt_mean:.2f} ± {awt_std:.2f} | "
                f"Return: {ret_mean:.2f} ± {ret_std:.2f}"
            )

            rows.append({
                "grid_n": grid_n,
                "seed": args.seed,
                "method": method,
                "throughput_vph_mean": tp_mean,
                "throughput_vph_std": tp_std,
                "mean_travel_s_mean": mtt_mean,
                "mean_travel_s_std": mtt_std,
                "mean_wait_s_mean": awt_mean,
                "mean_wait_s_std": awt_std,
                "episode_return_mean": ret_mean,
                "episode_return_std": ret_std,
                "steps": args.steps,
                "sumo_steps_per_env_step": args.sumo_steps_per_env_step,
                "n_evals": len(tp_vals),
            })

        env.close()

    write_csv(rows, out_path)


if __name__ == "__main__":
    main()
