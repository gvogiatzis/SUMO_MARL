#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import GNNLSTMPolicyQ
from marl_utils.replay_buffers import GlobalSequenceReplay
from marl_utils.network_update import dqn_update_shared_gnn_lstm, soft_update
from marl_utils.common import (
    parse_args,
    set_global_seed,
    EvalHistory,
    clear_eval_history,
    build_grid_edge_index,
)
from marl_utils.checkpointing import TrainingCheckpoint
from env_builder import build_train_env, get_or_create_eval_pool

# ------------------------- Evaluation (greedy) -------------------------

@torch.no_grad()
def evaluate_idqn_gnnlstm_shared(
    args,
    online_q: GNNLSTMPolicyQ,
    agent_id_list: List[str],
    edge_index: torch.Tensor,
    run_name: str,
) -> float:
    """Greedy evaluation over cached fixed-route envs with per-episode hidden reset."""
    original_mode = online_q.training
    online_q.eval()
    device = next(online_q.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for eval_env in eval_env_pool.envs:
            obs_dict = eval_env.reset()
            done = False
            episode_return = 0.0
            last_info: Dict = {}

            # Reset recurrent state at episode start
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

            while not done:
                X_t = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
                X_t = torch.as_tensor(X_t, dtype=torch.float32, device=device).unsqueeze(0)          # [1,N,O]

                q_t, hidden = online_q.step(X_t, edge_index.to(device), hidden)  # q_t: [1,N,A]
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

        print(
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_throughput:.2f} veh/h | "
            f"mean travel time={avg_travel_s:.2f}s | "
            f"avg waiting time={avg_wait_s:.2f}s"
        )
        recorder.save(avg_return, avg_throughput, avg_travel_s, avg_wait_s)
        # model selection on the median case return: robust to single-case
        # congestion tip-overs that dominate the mean
        return float(np.median(returns_all))
    finally:
        online_q.train(original_mode)

# ------------------------- Training -------------------------

def run_training(args):
    # device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # --- Env & topology
    train_env = build_train_env(args)
    obs_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    N = len(agent_id_list)
    O = len(next(iter(obs_dict.values())))
    A_dim = train_env.action_spaces[agent_id_list[0]].n
    edge_index = build_grid_edge_index(agent_id_list)  # [2,E]

    # --- Q networks (shared) + optimiser
    online_q = GNNLSTMPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, gnn_layers=args.gnn_layers).to(device)
    target_q = GNNLSTMPolicyQ(node_dim=O, actions=A_dim, hidden=args.hidden, gnn_layers=args.gnn_layers).to(device)
    target_q.load_state_dict(online_q.state_dict())
    target_q.eval()

    optim_q = optim.Adam(online_q.parameters(), lr=args.lr)

    # --- Sequence replay (episodes of full graph)
    seq_replay = GlobalSequenceReplay(capacity_steps=args.replay_size, num_agents=N, obs_dim=O, seed=args.rb_seed)

    eps = args.eps_start
    total_steps = 0
    run_name = "drqn_gnn_lstm_shared_seqlen" + str(args.seq_len)

    ckpt = TrainingCheckpoint(args.logdir, args.grid_n, args.seed, run_name, every=args.ckpt_every)
    _resume = ckpt.resume(online_q, target_q, optim_q, seq_replay)
    if _resume["resumed"]:
        eps = _resume["eps"]
        best_eval_return = _resume["best"]
        total_steps = _resume["extra"].get("total_steps", 0)
    else:
        clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for ep_idx in range(_resume["start_episode"], args.episodes + 1):
        obs_dict = train_env.reset()
        done = False
        steps_this_ep = 0

        # Episode buffers (T varies)
        states_seq: List[np.ndarray] = []
        next_states_seq: List[np.ndarray] = []
        actions_seq: List[np.ndarray] = []
        rewards_seq: List[np.ndarray] = []
        dones_seq: List[float] = []

        # Recurrent hidden state across the rollout
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        while not done and steps_this_ep < args.episode_steps:
            X_t = np.stack([obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
            X_tt = torch.as_tensor(X_t, dtype=torch.float32, device=device).unsqueeze(0)         # [1,N,O]

            with torch.no_grad():
                q_t, hidden = online_q.step(X_tt, edge_index.to(device), hidden)                 # [1,N,A], (h,c)

            greedy_actions = torch.argmax(q_t[0], dim=-1).cpu().numpy()  # [N]
            chosen_actions = greedy_actions.copy()
            for i in range(N):
                if np.random.rand() < eps:
                    chosen_actions[i] = np.random.randint(A_dim)

            action_dict = {aid: int(chosen_actions[i]) for i, aid in enumerate(agent_id_list)}
            next_obs_dict, reward_dict, done, _info = train_env.step(action_dict)
            # truncation is a time limit, not a terminal state: bootstrap through it
            done_stored = 0.0 if (_info.get("truncated_gridlock") or args.bootstrap_episode_end) else float(done)

            X_next = np.stack([next_obs_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)  # [N,O]
            R_t = np.array([float(reward_dict[aid]) for aid in agent_id_list], dtype=np.float32)         # [N]

            # Record step
            states_seq.append(X_t)
            next_states_seq.append(X_next)
            actions_seq.append(chosen_actions.astype(np.int64))
            rewards_seq.append(R_t.astype(np.float32))
            dones_seq.append(done_stored)

            obs_dict = next_obs_dict
            total_steps += 1
            steps_this_ep += 1

        # Push full joint episode
        seq_replay.push_episode(
            states=np.asarray(states_seq, dtype=np.float32),           # [T,N,O]
            actions=np.asarray(actions_seq, dtype=np.int64),           # [T,N]
            rewards=np.asarray(rewards_seq, dtype=np.float32),         # [T,N]
            next_states=np.asarray(next_states_seq, dtype=np.float32), # [T,N,O]
            dones=np.asarray(dones_seq, dtype=np.float32),             # [T]
        )

        # -------- Parameter updates --------
        mean_loss = 0.0
        # if total_steps >= args.warmup_steps and len(seq_replay) > 0:
        if total_steps >= args.warmup_steps:
            losses = []
            for _ in range(args.updates_per_ep):
                batch = seq_replay.sample(batch_size=args.batch_size_seq, seq_len=args.seq_len)  # tuple of arrays
                loss_val = dqn_update_shared_gnn_lstm(
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
                )
                losses.append(loss_val)
                # <-- use your soft_update after each grad step
                soft_update(target_q, online_q, args.tau)

            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(f"[ep {ep_idx}] train_loss={mean_loss:.3f} eps={eps:.3f}")

            if ep_idx % args.eval_every == 0:
                # Evaluate
                eval_return = evaluate_idqn_gnnlstm_shared(args, online_q, agent_id_list, edge_index, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(online_q.state_dict(), save_path)

            # Epsilon decay
            eps = max(args.eps_end, eps * args.eps_decay)
        else:
            print(f"[ep {ep_idx}] total steps={total_steps} (warming up) eps={eps:.3f}")

        ckpt.maybe_save(ep_idx, online_q, target_q, optim_q, seq_replay, eps, best_eval_return, extra={'total_steps': total_steps})

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
