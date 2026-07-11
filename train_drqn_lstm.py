#!/usr/bin/env python3
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime

from marl_utils.models import RecurrentQNet
from marl_utils.replay_buffers import AgentEpisode, SequenceReplay
from marl_utils.network_update import drqn_update, soft_update
from marl_utils.common import (
    parse_args,
    set_global_seed,
    EvalHistory,
    clear_eval_history
)
from marl_utils.checkpointing import TrainingCheckpoint
from env_builder import build_train_env, get_or_create_eval_pool

# --------------------------- Evaluation (shared LSTM) ---------------------------
@torch.no_grad()
def evaluate_idrqn_lstm_shared(args,
                               shared_q_net: RecurrentQNet,
                               agent_id_list: List[str],
                               run_name: str) -> float:
    """
    Evaluate with greedy policy on all cached fixed-route environments (one per trips file).
    Maintains one LSTM hidden state per agent during each evaluation episode.
    """
    original_mode = shared_q_net.training
    shared_q_net.eval()
    device = next(shared_q_net.parameters()).device

    eval_env_pool = get_or_create_eval_pool(args)
    if not eval_env_pool.envs:
        print(f"[EVAL {run_name}] No trips found for grid_n={args.grid_n}. Skipping.")
        return 0.0

    recorder = EvalHistory(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    returns_all, thr_all, mtt_all, mwt_all = [], [], [], []

    try:
        for eval_env in eval_env_pool.envs:
            obs_dict = eval_env.reset()
            done_flag = False
            episode_return = 0.0
            last_info = {}

            # one hidden state per agent for the shared LSTM
            hidden_states: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {
                aid: None for aid in agent_id_list
            }

            while not done_flag:
                action_dict: Dict[str, int] = {}
                for aid in agent_id_list:
                    obs_t = torch.as_tensor(
                        obs_dict[aid], dtype=torch.float32, device=device
                    ).view(1, 1, -1)  # [B=1, T=1, obs]
                    q_seq, hidden_states[aid] = shared_q_net(obs_t, hidden_states[aid])
                    q_last = q_seq[:, -1, :]  # [1, A]
                    action_dict[aid] = int(torch.argmax(q_last, dim=-1).item())

                next_obs, rew_dict, done_flag, info = eval_env.step(action_dict)
                episode_return += float(np.sum(list(rew_dict.values())))
                obs_dict = next_obs
                last_info = info

            kpis = last_info.get("network_kpis", {})
            thr_all.append(float(kpis.get("throughput_veh_per_hour", 0.0)))
            mtt_all.append(float(kpis.get("mean_travel_time_s", 0.0)))
            mwt_all.append(float(kpis.get("mean_waiting_time_s", 0.0)))
            returns_all.append(episode_return)

        avg_return = float(np.mean(returns_all))
        avg_thr = float(np.mean(thr_all))
        avg_mtt = float(np.mean(mtt_all))
        avg_mwt = float(np.mean(mwt_all))

        print(
            f"[{datetime.now()}, EVAL {run_name}] MEAN over {len(eval_env_pool.envs)} cases | "
            f"return={avg_return:.2f} | throughput={avg_thr:.2f} veh/h | "
            f"mean travel time={avg_mtt:.2f}s | avg waiting time={avg_mwt:.2f}s"
        )
        recorder.save(avg_return, avg_thr, avg_mtt, avg_mwt)
        return avg_return
    finally:
        shared_q_net.train(original_mode)


# --------------------------- Trainer (shared LSTM) ---------------------------

def run_training(args):
    # compute_device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    compute_device = torch.device(args.device)
    set_global_seed(args.seed)
    best_eval_return = -np.inf

    # Training env (random flows; respects args.grid_n via build_train_env)
    train_env = build_train_env(args)
    obs_dict = train_env.reset()
    agent_id_list = list(train_env.agent_ids)
    obs_dim = len(next(iter(obs_dict.values())))
    act_dim = train_env.action_spaces[agent_id_list[0]].n

    # Shared DRQN (LSTM) across all junctions + target
    shared_q_net = RecurrentQNet(observation_dim=obs_dim, action_dim=act_dim, hidden_units=args.hidden).to(compute_device)
    target_q_net = RecurrentQNet(observation_dim=obs_dim, action_dim=act_dim, hidden_units=args.hidden).to(compute_device)
    target_q_net.load_state_dict(shared_q_net.state_dict())
    target_q_net.eval()

    optimizer_q = optim.Adam(shared_q_net.parameters(), lr=args.lr, weight_decay=1e-5)

    # One shared sequence replay (stores per-agent trajectories as episodes)
    seq_replay = SequenceReplay(capacity_steps=args.replay_size, observation_dim=obs_dim, seed=args.rb_seed)

    exploration_epsilon = args.eps_start
    total_steps_collected = 0
    run_name = "drqn_lstm_shared_seqlen" + str(args.seq_len)

    # delete the previously left eval log files
    ckpt = TrainingCheckpoint(args.logdir, args.grid_n, args.seed, run_name, every=args.ckpt_every)
    _resume = ckpt.resume(shared_q_net, target_q_net, optimizer_q, seq_replay)
    if _resume["resumed"]:
        exploration_epsilon = _resume["eps"]
        best_eval_return = _resume["best"]
        total_steps_collected = _resume["extra"].get("total_steps", 0)
    else:
        clear_eval_history(args.logdir + '_grid_' + str(args.grid_n), run_name, args.seed)

    for episode_idx in range(_resume["start_episode"], args.episodes + 1):
        obs_dict = train_env.reset()
        done_flag = False
        steps_this_ep = 0

        # LSTM hidden state per agent for this episode
        hidden_states: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {aid: None for aid in agent_id_list}

        # Per-agent episode buffers to push into SequenceReplay at episode end
        episode_buffers = {
            aid: {"obs": [], "next_obs": [], "actions": [], "rewards": [], "dones": []}
            for aid in agent_id_list
        }

        while not done_flag and steps_this_ep < args.episode_steps:
            action_dict: Dict[str, int] = {}

            # act with shared LSTM (one step per agent) + epsilon exploration
            with torch.no_grad():
                for aid in agent_id_list:
                    obs_t = torch.as_tensor(
                        obs_dict[aid], dtype=torch.float32, device=compute_device
                    ).view(1, 1, -1)  # [1,1,obs]
                    q_seq, hidden_states[aid] = shared_q_net(obs_t, hidden_states[aid])
                    q_last = q_seq[:, -1, :]  # [1, A]
                    greedy_action = int(torch.argmax(q_last, dim=-1).item())
                    chosen = greedy_action if np.random.rand() >= exploration_epsilon else int(np.random.randint(act_dim))
                    action_dict[aid] = chosen

            next_obs, rew_dict, done_flag, _info = train_env.step(action_dict)
            # truncation is a time limit, not a terminal state: bootstrap through it
            done_stored = 0.0 if _info.get("truncated_gridlock") else float(done_flag)

            # record per-agent transition into episode buffers
            for aid in agent_id_list:
                episode_buffers[aid]["obs"].append(obs_dict[aid])
                episode_buffers[aid]["next_obs"].append(next_obs[aid])
                episode_buffers[aid]["actions"].append(action_dict[aid])
                episode_buffers[aid]["rewards"].append(rew_dict[aid])
                episode_buffers[aid]["dones"].append(float(done_stored))

            obs_dict = next_obs
            steps_this_ep += 1
            total_steps_collected += 1

        # push each agent's episode into the shared sequence buffer
        for aid in agent_id_list:
            ep_obs = np.asarray(episode_buffers[aid]["obs"], dtype=np.float32)
            ep_next = np.asarray(episode_buffers[aid]["next_obs"], dtype=np.float32)
            ep_act = np.asarray(episode_buffers[aid]["actions"], dtype=np.int64)
            ep_rew = np.asarray(episode_buffers[aid]["rewards"], dtype=np.float32)
            ep_done = np.asarray(episode_buffers[aid]["dones"], dtype=np.float32)
            seq_replay.push_episode(AgentEpisode(
                observations=ep_obs,
                actions=ep_act,
                rewards=ep_rew,
                dones=ep_done,
                next_observations=ep_next,
            ))

        # learning updates
        mean_train_loss = 0.0
        # if seq_replay.total_steps >= args.warmup_steps:
        if total_steps_collected >= args.warmup_steps:
            losses = []
            for _ in range(args.updates_per_ep):
                if len(seq_replay) == 0:
                    break
                batch = seq_replay.sample(batch_size=args.batch_size_seq, seq_len=args.seq_len)
                loss_val = drqn_update(
                    device=compute_device,
                    batch=batch,
                    online_qnet=shared_q_net,
                    target_qnet=target_q_net,
                    optimizer_q=optimizer_q,
                    gamma=args.gamma,
                    burn_in=args.burn_in,
                    double_dqn=True,
                )
                # soft target update every grad step
                soft_update(target_q_net, shared_q_net, args.tau)

                losses.append(loss_val)
            mean_train_loss = float(np.mean(losses)) if losses else 0.0
            print(f"[ep {episode_idx}] train_loss={mean_train_loss:.3f} eps={exploration_epsilon:.3f}")

            if episode_idx % args.eval_every == 0:
                # eval on all fixed cases (cached envs)
                eval_return = evaluate_idrqn_lstm_shared(args, shared_q_net, agent_id_list, run_name=run_name)
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    save_path = f"{args.logdir}_grid_{args.grid_n}/seed{args.seed}/model_best_{run_name}_seed{args.seed}.pt"
                    torch.save(shared_q_net.state_dict(), save_path)

            # epsilon decay once per episode (after warmup it will matter for learning)
            exploration_epsilon = max(args.eps_end, exploration_epsilon * args.eps_decay)
        else:
            print(f"[ep {episode_idx}] total steps={total_steps_collected} (warming up) eps={exploration_epsilon:.3f}")

        # # (Optional) hard sync every few episodes as a safety net
        # if ep % args.target_sync_ep == 0:
        #     target_q_net.load_state_dict(shared_q_net.state_dict())

        ckpt.maybe_save(episode_idx, shared_q_net, target_q_net, optimizer_q, seq_replay, exploration_epsilon, best_eval_return, extra={'total_steps': total_steps_collected})

    train_env.close()


if __name__ == "__main__":
    run_training(parse_args())
