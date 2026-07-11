#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, List

import numpy as np
from pathlib import Path
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import QNetMLP
from marl_utils.replay_buffers import Transition, ReplayBuffer
from marl_utils.network_update import dqn_update, soft_update
from marl_utils.common import (
    parse_args,
    set_global_seed,
    EvalHistory,
    clear_eval_history
)
from marl_utils.checkpointing import TrainingCheckpoint
from env_builder import build_train_env, get_or_create_eval_pool

# --------- evaluation (greedy over cached fixed-route envs) ----------
@torch.no_grad()
def evaluate_idqn_mlp_shared(args, shared_q_network: QNetMLP, agent_id_list: List[str], run_name: str) -> float:

    # switch nets to eval; restore later
    original_mode = shared_q_network.training
    shared_q_network.eval()
    device = next(shared_q_network.parameters()).device

    # Build (or reuse) all fixed-route envs for this grid/settings
    eval_env_pool = get_or_create_eval_pool(args)
    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    try:
        returns_all, throughput_all, mean_travel_all, mean_wait_all = [], [], [], []

        for eval_env in eval_env_pool.envs:
            episode_return = 0.0
            last_info_dict = {}
            observation_dict = eval_env.reset()
            done_flag = False

            while not done_flag:
                # Batch all TLS observations through the single shared model
                obs_matrix = np.stack([observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
                obs_tensor = torch.as_tensor(obs_matrix, dtype=torch.float32, device=device)  # [N, obs]
                q_values = shared_q_network(obs_tensor)                                      # [N, A]
                greedy_actions = torch.argmax(q_values, dim=-1).tolist()
                action_dict = {aid: int(greedy_actions[i]) for i, aid in enumerate(agent_id_list)}

                next_obs, reward_dict, done_flag, info = eval_env.step(action_dict)
                episode_return += float(np.sum(list(reward_dict.values())))
                observation_dict = next_obs
                last_info_dict = info

            kpis = last_info_dict.get("network_kpis", {})
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
        # record the median case return per eval (means are tip-over dominated)
        med = float(np.median(returns_all))
        mdir = Path(args.logdir + '_grid_' + str(args.grid_n)) / f"seed{args.seed}"
        mp = mdir / f"{run_name}_median_return.npy"
        hist = np.load(mp).tolist() if mp.exists() else []
        hist.append(med)
        np.save(mp, np.array(hist, dtype=np.float32))
        # model selection on the median case return: robust to single-case
        # congestion tip-overs that dominate the mean
        return med
    finally:
        shared_q_network.train(original_mode)


# ------------------------------------------------------------

def run_training(args):
    # compute_device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    compute_device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Build training environment (random flows; respects args.grid_n)
    train_env = build_train_env(args)
    observation_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    observation_dim = len(next(iter(observation_dict.values())))
    action_dim = train_env.action_spaces[agent_id_list[0]].n

    # Shared (tied) Q-network and target network across all junctions
    shared_q_network = QNetMLP(observation_dim, action_dim, hidden_units=args.hidden).to(compute_device)
    target_q_network = QNetMLP(observation_dim, action_dim, hidden_units=args.hidden).to(compute_device)
    target_q_network.load_state_dict(shared_q_network.state_dict())
    target_q_network.eval()

    optimizer_q = optim.Adam(shared_q_network.parameters(), lr=args.lr)

    # One shared replay buffer that mixes transitions from all junctions
    shared_replay_buffer = ReplayBuffer(
        capacity=args.replay_size,
        observation_dim=observation_dim,
        seed=args.rb_seed
    )

    exploration_epsilon = args.eps_start
    total_env_steps_collected = 0
    run_name = "dqn_mlp_shared"

    # delete the previously left eval log files
    ckpt = TrainingCheckpoint(args.logdir, args.grid_n, args.seed, run_name, every=args.ckpt_every)
    _resume = ckpt.resume(shared_q_network, target_q_network, optimizer_q, shared_replay_buffer)
    if _resume["resumed"]:
        exploration_epsilon = _resume["eps"]
        best_eval_return = _resume["best"]
        total_env_steps_collected = _resume["extra"].get("total_steps", 0)
    else:
        clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for episode_idx in range(_resume["start_episode"], args.episodes + 1):
        observation_dict = train_env.reset()
        episode_done_flag = False
        steps_this_episode = 0

        while not episode_done_flag and steps_this_episode < args.episode_steps:
            # Greedy + epsilon actions for all agents in one forward pass
            obs_matrix = np.stack([observation_dict[aid] for aid in agent_id_list], axis=0).astype(np.float32)
            obs_tensor = torch.as_tensor(obs_matrix, dtype=torch.float32, device=compute_device)  # [N, obs]
            with torch.no_grad():
                q_values_all = shared_q_network(obs_tensor)                                       # [N, A]
                greedy_actions = torch.argmax(q_values_all, dim=-1).cpu().numpy()                # [N]

            chosen_actions_np = greedy_actions.copy()
            for i in range(len(agent_id_list)):
                if np.random.rand() < exploration_epsilon:
                    chosen_actions_np[i] = np.random.randint(action_dim)

            action_dict = {aid: int(chosen_actions_np[i]) for i, aid in enumerate(agent_id_list)}
            next_observation_dict, reward_dict, episode_done_flag, _info = train_env.step(action_dict)
            # truncation is a time limit, not a terminal state: bootstrap through it
            done_stored = 0.0 if (_info.get("truncated_gridlock") or args.bootstrap_episode_end) else float(episode_done_flag)

            # Push each junction's transition into the shared replay buffer
            for i, agent_id in enumerate(agent_id_list):
                transition = Transition(
                    state=observation_dict[agent_id],
                    action=int(chosen_actions_np[i]),
                    reward=float(reward_dict[agent_id]),
                    next_state=next_observation_dict[agent_id],
                    done=done_stored,
                )
                shared_replay_buffer.push(transition)

            observation_dict = next_observation_dict
            total_env_steps_collected += 1
            steps_this_episode += 1

        # Parameter updates (shared)
        mean_training_loss = 0.0
        if total_env_steps_collected >= args.warmup_steps:
            loss_values_for_logging: List[float] = []
            for _ in range(args.updates_per_ep):
                if len(shared_replay_buffer) < args.batch_size:
                    break
                batch_tuple = shared_replay_buffer.sample(args.batch_size)
                loss_val = dqn_update(
                    compute_device,
                    shared_q_network,
                    target_q_network,
                    optimizer_q,
                    batch_tuple,
                    gamma=args.gamma
                )
                loss_values_for_logging.append(loss_val)

                # soft target update every grad step
                soft_update(target_q_network, shared_q_network, args.tau)

            mean_training_loss = float(np.mean(loss_values_for_logging)) if loss_values_for_logging else 0.0
            print(f"[ep {episode_idx}] train_loss={mean_training_loss:.3f} eps={exploration_epsilon:.3f}")

            if episode_idx % args.eval_every == 0:
                # Evaluate across ALL fixed cases (via cached env pool)
                eval_return = evaluate_idqn_mlp_shared(args, shared_q_network, agent_id_list, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(shared_q_network.state_dict(), save_path)

            # epsilon decay
            exploration_epsilon = max(args.eps_end, exploration_epsilon * args.eps_decay)
        else:
            print(f"[ep {episode_idx}] total steps={total_env_steps_collected} (warming up) eps={exploration_epsilon:.3f}")

        # # Target sync
        # if episode_idx % args.target_sync_ep == 0:
        #     target_q_network.load_state_dict(shared_q_network.state_dict())


        ckpt.maybe_save(episode_idx, shared_q_network, target_q_network, optimizer_q, shared_replay_buffer, exploration_epsilon, best_eval_return, extra={'total_steps': total_env_steps_collected})

    train_env.close()

if __name__ == "__main__":
    run_training(parse_args())
