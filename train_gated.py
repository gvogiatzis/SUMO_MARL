#!/usr/bin/env python3
"""Train the gated adaptive spatio-temporal model (AAMAS 2027 contribution).

Mirrors train_drqn_gnn_lstm.py's protocol (sequence replay, burn-in, double
DQN, shared network) with three additions: Gumbel-sigmoid gate temperature
annealing, module-use sparsity costs (enabled after a warmup), and per-eval
gate-activation logging.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path

from marl_utils.models import GatedSpatioTemporalQ
from marl_utils.replay_buffers import GlobalSequenceReplay
from marl_utils.checkpointing import TrainingCheckpoint
from marl_utils.network_update import dqn_update_shared_gated, soft_update
from marl_utils.common import (
    parse_args,
    set_global_seed,
    EvalHistory,
    clear_eval_history,
    build_grid_edge_index,
)
from env_builder import build_train_env, get_or_create_eval_pool


@torch.no_grad()
def evaluate_gated_shared(
    args,
    online_q: GatedSpatioTemporalQ,
    agent_id_list: List[str],
    edge_index: torch.Tensor,
    run_name: str,
) -> float:
    """Greedy evaluation; also records mean hard gate activation rates."""
    original_mode = online_q.training
    online_q.eval()
    device = next(online_q.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []
        gmem_all, gcom_all = [], []

        for eval_env in eval_env_pool.envs:
            obs_dict = eval_env.reset()
            done = False
            episode_return = 0.0
            last_info: Dict = {}
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

            while not done:
                X_t = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
                X_t = torch.as_tensor(X_t, dtype=torch.float32, device=device).unsqueeze(0)

                q_t, hidden, g_t = online_q.step(X_t, edge_index.to(device), hidden)
                gmem_all.append(float((g_t[..., 0] > 0.5).float().mean()))
                gcom_all.append(float((g_t[..., 1] > 0.5).float().mean()))

                greedy_actions = torch.argmax(q_t[0], dim=-1).tolist()
                action_dict = {aid: int(greedy_actions[i]) for i, aid in enumerate(agent_id_list)}
                next_obs, reward_dict, done, info = eval_env.step(action_dict)
                episode_return += float(np.sum(list(reward_dict.values())))
                obs_dict = next_obs
                last_info = info

            kpis = last_info.get("network_kpis", {})
            returns_all.append(episode_return)
            throughput_all.append(float(kpis.get("throughput_veh_per_hour", 0.0)))
            mean_travel_all.append(float(kpis.get("mean_travel_time_s", 0.0)))
            mean_wait_all.append(float(kpis.get("mean_waiting_time_s", 0.0)))

        avg_return = float(np.mean(returns_all))
        avg_throughput = float(np.mean(throughput_all))
        avg_travel_s = float(np.mean(mean_travel_all))
        avg_wait_s = float(np.mean(mean_wait_all))
        avg_gmem = float(np.mean(gmem_all))
        avg_gcom = float(np.mean(gcom_all))

        print(
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | avg waiting time={avg_wait_s:.2f}s | "
            f"g_mem={avg_gmem:.3f} | g_com={avg_gcom:.3f}"
        )
        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)

        # append gate activation history alongside the KPI arrays
        gdir = Path(args.logdir + '_grid_' + str(args.grid_n)) / f"seed{args.seed}"
        for name, val in (("g_mem", avg_gmem), ("g_com", avg_gcom)):
            p = gdir / f"{run_name}_{name}_activation.npy"
            hist = np.load(p).tolist() if p.exists() else []
            hist.append(val)
            np.save(p, np.array(hist, dtype=np.float32))

        return avg_return
    finally:
        online_q.train(original_mode)


def run_training(args):
    device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    train_env = build_train_env(args)
    obs_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(obs_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n
    edge_index = build_grid_edge_index(agent_id_list)

    online_q = GatedSpatioTemporalQ(node_dim=O, actions=A_dim, hidden=args.hidden,
                                    gnn_layers=args.gnn_layers, gate_temp=args.gate_temp_start,
                                    gate_mem_mode=args.gate_mem_mode, gate_com_mode=args.gate_com_mode).to(device)
    target_q = GatedSpatioTemporalQ(node_dim=O, actions=A_dim, hidden=args.hidden,
                                    gnn_layers=args.gnn_layers, gate_temp=args.gate_temp_start,
                                    gate_mem_mode=args.gate_mem_mode, gate_com_mode=args.gate_com_mode).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optim_q = optim.Adam(online_q.parameters(), lr=args.lr)
    seq_replay = GlobalSequenceReplay(capacity_steps=args.replay_size, num_agents=N, obs_dim=O, seed=args.rb_seed)

    eps = args.eps_start
    total_steps = 0
    run_name = "gated_shared_seqlen" + str(args.seq_len)
    if args.gate_mem_mode != "learned" or args.gate_com_mode != "learned":
        run_name += f"_m{args.gate_mem_mode[0]}c{args.gate_com_mode[0]}"  # ablation variants
    ckpt = TrainingCheckpoint(args.logdir, args.grid_n, args.seed, run_name, every=args.ckpt_every)
    _resume = ckpt.resume(online_q, target_q, optim_q, seq_replay)
    if _resume["resumed"]:
        eps = _resume["eps"]
        best_eval_return = _resume["best"]
        total_steps = _resume["extra"].get("total_steps", 0)
    else:
        clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)
        # gate histories are appended per eval; clear them too (restarts would pollute)
        gdir = Path(args.logdir + '_grid_' + str(args.grid_n)) / f"seed{args.seed}"
        for name in ("g_mem", "g_com"):
            p = gdir / f"{run_name}_{name}_activation.npy"
            if p.exists():
                p.unlink()

    for ep_idx in range(_resume["start_episode"], args.episodes + 1):
        # anneal gate temperature linearly over training
        frac = (ep_idx - 1) / max(args.episodes - 1, 1)
        gate_temp = args.gate_temp_start + frac * (args.gate_temp_end - args.gate_temp_start)
        online_q.gate_temp = gate_temp
        target_q.gate_temp = gate_temp

        obs_dict = train_env.reset()
        done = False
        steps_this_ep = 0
        states_seq, next_states_seq, actions_seq, rewards_seq, dones_seq = [], [], [], [], []
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        while not done and steps_this_ep < args.episode_steps:
            X_t = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
            X_tt = torch.as_tensor(X_t, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_t, hidden, _ = online_q.step(X_tt, edge_index.to(device), hidden)

            greedy_actions = torch.argmax(q_t[0], dim=-1).cpu().numpy()
            chosen_actions = greedy_actions.copy()
            for i in range(N):
                if np.random.rand() < eps:
                    chosen_actions[i] = np.random.randint(A_dim)

            action_dict = {aid: int(chosen_actions[i]) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _info = train_env.step(action_dict)

            X_next = np.stack([next_obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
            R_t = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)

            states_seq.append(X_t)
            next_states_seq.append(X_next)
            actions_seq.append(chosen_actions.astype(np.int64))
            rewards_seq.append(R_t.astype(np.float32))
            dones_seq.append(float(done))

            obs_dict = next_obs_dict
            total_steps += 1
            steps_this_ep += 1

        seq_replay.push_episode(
            states=np.asarray(states_seq, dtype=np.float32),
            actions=np.asarray(actions_seq, dtype=np.int64),
            rewards=np.asarray(rewards_seq, dtype=np.float32),
            next_states=np.asarray(next_states_seq, dtype=np.float32),
            dones=np.asarray(dones_seq, dtype=np.float32),
        )

        if total_steps >= args.warmup_steps:
            # gate cost disabled during warmup so pathways can become useful first
            cost_on = ep_idx > args.gate_cost_warmup_eps
            lam_mem = args.lambda_mem if cost_on else 0.0
            lam_com = args.lambda_com if cost_on else 0.0

            losses, gmems, gcoms = [], [], []
            for _ in range(args.updates_per_ep):
                batch = seq_replay.sample(batch_size=args.batch_size_seq, seq_len=args.seq_len)
                loss_val, td_val, g_mem, g_com = dqn_update_shared_gated(
                    device=device,
                    online_q=online_q,
                    target_q=target_q,
                    optimizer_q=optim_q,
                    batch_tuple=batch,
                    edge_index=edge_index,
                    gamma=args.gamma,
                    burn_in=args.burn_in,
                    double_dqn=True,
                    grad_clip=1.0,
                    lambda_mem=lam_mem,
                    lambda_com=lam_com,
                    gate_budget=args.gate_budget,
                    gate_budget_onesided=(args.gate_budget_mode == "cap"),
                )
                losses.append(td_val)
                gmems.append(g_mem)
                gcoms.append(g_com)
                soft_update(target_q, online_q, args.tau)

            print(f"[ep {ep_idx}] td_loss={np.mean(losses):.3f} g_mem={np.mean(gmems):.3f} "
                  f"g_com={np.mean(gcoms):.3f} temp={gate_temp:.2f} cost={'on' if cost_on else 'off'} eps={eps:.3f}")

            if ep_idx % args.eval_every == 0:
                eval_return = evaluate_gated_shared(args, online_q, agent_id_list, edge_index, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(online_q.state_dict(), save_path)

            eps = max(args.eps_end, eps * args.eps_decay)
        else:
            print(f"[ep {ep_idx}] total steps={total_steps} (warming up) eps={eps:.3f}")

        ckpt.maybe_save(ep_idx, online_q, target_q, optim_q, seq_replay, eps, best_eval_return, extra={'total_steps': total_steps})

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
