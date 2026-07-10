#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple, List, Optional

from marl_utils.common import to_tensor, build_batched_edge_index
from marl_utils.models import QNetMLP, RecurrentQNet, GNNPolicyQ, GNNLSTMPolicyQ

# =================================== DQN Update ====================================

def dqn_update(device: torch.device, online_q: QNetMLP, target_q: QNetMLP,
               optimizer_q: optim.Optimizer, batch_tuple, gamma: float = 0.99,
               double_dqn: bool = True) -> float:
    states, actions, rewards, next_states, dones = batch_tuple
    state_t = to_tensor(device, states); next_state_t = to_tensor(device, next_states)
    action_t = torch.as_tensor(actions, device=device, dtype=torch.long)
    reward_t = to_tensor(device, rewards); done_t = to_tensor(device, dones)
    q_vals = online_q(state_t); q_taken = q_vals.gather(1, action_t.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_actions = (online_q if double_dqn else target_q)(next_state_t).argmax(dim=1)
        next_q = target_q(next_state_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = reward_t + (1.0 - done_t) * gamma * next_q
    loss = nn.functional.smooth_l1_loss(q_taken, target)
    optimizer_q.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(online_q.parameters(), 1.0); optimizer_q.step()
    return float(loss.item())

# =============================== DRQN Update for LSTM ===============================
def drqn_update(
    device: torch.device,
    batch: Dict[str, np.ndarray],
    online_qnet: RecurrentQNet,
    target_qnet: RecurrentQNet,
    optimizer_q: optim.Optimizer,
    gamma: float = 0.99,
    burn_in: int = 8,
    double_dqn: bool = True,
) -> float:
    """
    DRQN/LSTM DQN update on a batch of sequences with:
      - burn-in warmup of hidden states, and
      - hidden-state reset (masking) INSIDE sequences when terminals occur.

    Expects `batch` with keys:
      - 'obs':       [B, T, obs_dim]
      - 'next_obs':  [B, T, obs_dim]
      - 'actions':   [B, T] (int)
      - 'rewards':   [B, T] (float)
      - 'dones':     [B, T] (float; 1.0 means terminal / padding)
    """
    # --- Move batch to device ---
    observations_seq      = to_tensor(device, batch['obs'])            # [B, T, O]
    next_observations_seq = to_tensor(device, batch['next_obs'])       # [B, T, O]
    action_indices        = torch.as_tensor(batch['actions'], device=device, dtype=torch.long)   # [B, T]
    rewards_seq           = to_tensor(device, batch['rewards'])        # [B, T]
    dones_seq             = to_tensor(device, batch['dones'])          # [B, T]

    B, T, _ = observations_seq.shape

    # --- Clamp burn-in so at least one learning step remains ---
    burn = int(burn_in)
    if burn >= T:
        burn = max(0, T - 1)

    # --- Hidden-state warmup (burn-in) ---
    with torch.no_grad():
        h_online = None
        h_target = None
        if burn > 0:
            _q_bi, h_online = online_qnet(observations_seq[:, :burn, :], None)
            _tq_bi, h_target = target_qnet(next_observations_seq[:, :burn, :], None)

    # --- Slice learning window ---
    S_w   = observations_seq[:, burn:, :]        # [B, Tw, O]
    NS_w  = next_observations_seq[:, burn:, :]   # [B, Tw, O]
    A_w   = action_indices[:, burn:]             # [B, Tw]
    R_w   = rewards_seq[:, burn:]                # [B, Tw]
    D_w   = dones_seq[:, burn:]                  # [B, Tw]
    Tw = S_w.size(1)

    q_taken_list: List[torch.Tensor] = []
    target_list:  List[torch.Tensor] = []

    # --- Unroll step-by-step to apply hidden masking on terminals ---
    for t in range(Tw):
        # Current step slices
        s_t   = S_w[:, t:t+1, :]          # [B,1,O]
        ns_t  = NS_w[:, t:t+1, :]         # [B,1,O]
        a_t   = A_w[:, t]                 # [B]
        r_t   = R_w[:, t]                 # [B]
        d_t   = D_w[:, t]                 # [B]  (1.0 = terminal/pad)

        # Online Q(s_t, ·) and pick Q_taken
        q_t, h_online = online_qnet(s_t, h_online)              # q_t: [B,1,A], h_online updated
        q_t_s = q_t.squeeze(1)                                  # [B,A]
        q_taken_t = q_t_s.gather(1, a_t.unsqueeze(1)).squeeze(1)  # [B]
        q_taken_list.append(q_taken_t)

        # ---- Targets (no grad) ----
        with torch.no_grad():
            # Argmax from online on next_obs (do not advance online hidden we keep for learning path)
            qn_online_t, _ = online_qnet(ns_t, h_online)        # [B,1,A], temp hidden discarded
            qn_target_t, h_target = target_qnet(ns_t, h_target) # [B,1,A], advance target hidden

            if double_dqn:
                next_a_t = torch.argmax(qn_online_t.squeeze(1), dim=-1)   # [B]
            else:
                next_a_t = torch.argmax(qn_target_t.squeeze(1), dim=-1)   # [B]

            next_q_t = qn_target_t.squeeze(1).gather(1, next_a_t.unsqueeze(1)).squeeze(1)  # [B]
            target_t = r_t + (1.0 - d_t) * gamma * next_q_t
            target_list.append(target_t)

        # ---- Hidden reset on terminals at this step (mask where done==1) ----
        keep_mask = (1.0 - d_t).view(1, B, 1)   # [1,B,1]
        if h_online is not None:
            h_online = (h_online[0] * keep_mask, h_online[1] * keep_mask)  # each [1,B,H]
        if h_target is not None:
            h_target = (h_target[0] * keep_mask, h_target[1] * keep_mask)

    # Stack over Tw
    Q_taken = torch.stack(q_taken_list, dim=1)   # [B, Tw]
    Target  = torch.stack(target_list,  dim=1)   # [B, Tw]

    # --- Loss & optimization ---
    loss_value = nn.functional.smooth_l1_loss(Q_taken, Target)  # Huber

    optimizer_q.zero_grad()
    loss_value.backward()
    nn.utils.clip_grad_norm_(online_qnet.parameters(), 1.0)
    optimizer_q.step()

    return float(loss_value.item())


# =============================== DQN Update (shared-GNN IDQN) ===============================

def dqn_update_shared_gnn(
    device: torch.device,
    gnn_online_q_network: GNNPolicyQ,
    gnn_target_q_network: GNNPolicyQ,
    optimizer_q: optim.Optimizer,
    batch_tuple,
    graph_edge_index: torch.Tensor,
    graph_edge_features: Optional[torch.Tensor] = None,
    discount_gamma: float = 0.99,
    double_dqn: bool = True,
) -> float:
    """
    Perform a Double-DQN update with a shared GNN Q-network.

    Args:
        batch_tuple:
            Tuple (states, actions, rewards, next_states, dones) with shapes
              states       : [batch_size, num_nodes, obs_dim]
              actions      : [batch_size, num_nodes]        (int64)
              rewards      : [batch_size, num_nodes]        (float32)
              next_states  : [batch_size, num_nodes, obs_dim]
              dones        : [batch_size]                   (float32 scalar per transition)
        graph_edge_index: [2, num_edges] directed edges (same graph for all batch samples)
        graph_edge_features : [num_edges, edge_feat_dim] or None
        discount_gamma: discount factor for TD target
        double_dqn: if True, use online net for argmax and target net for evaluation

    Returns:
        float: scalar MSE loss value.
    """
    states_batch, actions_batch, rewards_batch, next_states_batch, terminal_flags_batch = batch_tuple
    batch_size, num_nodes, obs_dim = states_batch.shape

    edge_index_device = graph_edge_index.to(device)
    edge_features_device = graph_edge_features.to(device) if graph_edge_features is not None else None

    total_mse_loss = torch.zeros((), device=device)

    for batch_idx in range(batch_size):
        # Move a single transition (all nodes) to device
        state_nodes = torch.as_tensor(states_batch[batch_idx], dtype=torch.float32, device=device)       # [N, O]
        next_state_nodes = torch.as_tensor(next_states_batch[batch_idx], dtype=torch.float32, device=device)  # [N, O]
        action_nodes = torch.as_tensor(actions_batch[batch_idx], dtype=torch.long, device=device)        # [N]
        reward_nodes = torch.as_tensor(rewards_batch[batch_idx], dtype=torch.float32, device=device)     # [N]
        done_flag = torch.as_tensor(terminal_flags_batch[batch_idx], dtype=torch.float32, device=device) # scalar

        # Q(s, ·) and Q(s, a)
        q_values_current = gnn_online_q_network(state_nodes, edge_index_device, edge_features_device)     # [N, A]
        q_values_taken = q_values_current.gather(1, action_nodes.view(-1, 1)).squeeze(1)                # [N]

        with torch.no_grad():
            q_values_next_online = gnn_online_q_network(next_state_nodes, edge_index_device, edge_features_device)  # [N, A]
            q_values_next_target = gnn_target_q_network(next_state_nodes, edge_index_device, edge_features_device)  # [N, A]

            if double_dqn:
                next_actions = torch.argmax(q_values_next_online, dim=1)                                 # [N]
            else:
                next_actions = torch.argmax(q_values_next_target, dim=1)                                 # [N]

            next_q_values = q_values_next_target.gather(1, next_actions.view(-1, 1)).squeeze(1)         # [N]
            target_q_values = reward_nodes + (1.0 - done_flag) * discount_gamma * next_q_values         # [N]; done_flag broadcasts

        total_mse_loss = total_mse_loss + nn.functional.smooth_l1_loss(q_values_taken, target_q_values)

    total_mse_loss = total_mse_loss / batch_size

    optimizer_q.zero_grad()
    total_mse_loss.backward()
    nn.utils.clip_grad_norm_(gnn_online_q_network.parameters(), 1.0)
    optimizer_q.step()

    return float(total_mse_loss.item())

# =============================== DQN Update (shared-GNN+LSTM IDQN) ===============================

def dqn_update_shared_gnn_lstm(
    device: torch.device,
    online_q: "GNNLSTMPolicyQ",
    target_q: "GNNLSTMPolicyQ",
    optimizer_q: torch.optim.Optimizer,
    batch_tuple,
    edge_index: torch.Tensor,
    gamma: float = 0.99,
    burn_in: int = 8,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    S, A, R, NS, D = batch_tuple
    S  = torch.as_tensor(S,  device=device, dtype=torch.float32)  # [B,T,N,O]
    NS = torch.as_tensor(NS, device=device, dtype=torch.float32)  # [B,T,N,O]
    A  = torch.as_tensor(A,  device=device, dtype=torch.long)     # [B,T,N]
    R  = torch.as_tensor(R,  device=device, dtype=torch.float32)  # [B,T,N]
    D  = torch.as_tensor(D,  device=device, dtype=torch.float32)  # [B,T] or [B,T,N]

    if D.dim() == 2:  # -> [B,T,N]
        D = D.unsqueeze(-1).expand(-1, -1, S.size(2))

    B, T, N, O = S.shape
    burn = min(max(int(burn_in), 0), max(T - 1, 0))
    Ei = edge_index.to(device)

    # ---- burn-in to warm hidden ----
    with torch.no_grad():
        h_online = h_target = None
        if burn > 0:
            _, h_online = online_q(S[:, :burn], Ei, None)
            _, h_target = target_q(NS[:, :burn], Ei, None)

    # ---- learning window ----
    S_w, NS_w = S[:, burn:], NS[:, burn:]           # [B,Tw,N,O]
    A_w, R_w, D_w = A[:, burn:], R[:, burn:], D[:, burn:]  # [B,Tw,N]
    Tw = S_w.size(1)

    h_o, h_t = h_online, h_target
    q_list, qn_online_list, qn_target_list = [], [], []

    for t in range(Tw):
        # ----- ONLINE: forward with grad to produce Q_taken -----
        # Use full forward with T=1, not .step (which is @torch.no_grad in your class)
        q_t, h_o = online_q(S_w[:, t].unsqueeze(1), Ei, h_o)  # q_t: [B,1,N,A]
        q_list.append(q_t)  # keep [B,1,N,A]

        # ----- Argmax and Target: no grad -----
        with torch.no_grad():
            # Online for argmax (do not backprop through this path)
            qn_o_t, _ = online_q(NS_w[:, t].unsqueeze(1), Ei, h_o)    # [B,1,N,A]
            # Target for evaluation
            qn_t_t, h_t = target_q(NS_w[:, t].unsqueeze(1), Ei, h_t)  # [B,1,N,A]

        qn_online_list.append(qn_o_t)
        qn_target_list.append(qn_t_t)

        # ----- hidden reset on dones at this step -----
        if h_o is not None:
            done_mask = (1.0 - D_w[:, t]).view(1, B, N, 1)        # [1,B,N,1]
            h0 = h_o[0].view(1, B, N, -1); c0 = h_o[1].view(1, B, N, -1)
            h0.mul_(done_mask); c0.mul_(done_mask)
            h_o = (h0.view(1, B * N, -1), c0.view(1, B * N, -1))
        if h_t is not None:
            done_mask = (1.0 - D_w[:, t]).view(1, B, N, 1)
            h1 = h_t[0].view(1, B, N, -1); c1 = h_t[1].view(1, B, N, -1)
            h1.mul_(done_mask); c1.mul_(done_mask)
            h_t = (h1.view(1, B * N, -1), c1.view(1, B * N, -1))

    Q_seq = torch.cat(q_list, dim=1)                 # [B,Tw,1,N,A] → actually [B,Tw,N,A]
    Q_seq = Q_seq.squeeze(2)                         # remove the size-1 dim if present -> [B,Tw,N,A]
    Qn_o  = torch.cat(qn_online_list, dim=1).squeeze(2)  # [B,Tw,N,A]
    Qn_t  = torch.cat(qn_target_list, dim=1).squeeze(2)  # [B,Tw,N,A]

    # Gather Q(s,a)
    Q_taken = Q_seq.gather(-1, A_w.unsqueeze(-1)).squeeze(-1)  # [B,Tw,N]

    # Targets (no grad)
    with torch.no_grad():
        next_idx = torch.argmax(Qn_o, dim=-1) if double_dqn else torch.argmax(Qn_t, dim=-1)  # [B,Tw,N]
        Qn_star = Qn_t.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)                         # [B,Tw,N]
        target = R_w + (1.0 - D_w) * gamma * Qn_star

    loss = nn.functional.smooth_l1_loss(Q_taken, target)

    optimizer_q.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer_q.step()

    return float(loss.item())



# ============================= CTDE-specific: VDN update (Double-DQN) =============================
def vdn_update_mlp(
    device: torch.device,
    online_q: QNetMLP,          # shared per-agent Q_i(o_i, a_i)
    target_q: QNetMLP,          # target copy
    optimizer: optim.Optimizer,
    batch_tuple,
    gamma: float = 0.99,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    VDN: Q_tot = sum_i Q_i. Double-DQN for targets, with **team reward**.
    Inputs (from JointReplayBuffer.sample):
      states      : [B, N, O]
      actions     : [B, N]         (int64)
      rewards     : [B, N]         (float32)  -- we sum over N to get team reward
      next_states : [B, N, O]
      dones       : [B]            (float32)
    """
    S, A, R, NS, D = batch_tuple
    B, N, O = S.shape

    S_t  = torch.as_tensor(S,  device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    NS_t = torch.as_tensor(NS, device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    A_t  = torch.as_tensor(A,  device=device, dtype=torch.long).view(B * N)        # [B*N]
    R_t  = torch.as_tensor(R,  device=device, dtype=torch.float32)                 # [B,N]
    D_t  = torch.as_tensor(D,  device=device, dtype=torch.float32)                 # [B]

    # Per-agent Q(s,·) and gather chosen actions
    q_all = online_q(S_t)                                      # [B*N, A]
    q_taken = q_all.gather(1, A_t.view(-1, 1)).squeeze(1)      # [B*N]
    q_taken = q_taken.view(B, N)                                # [B,N]
    q_tot = q_taken.sum(dim=1)                                  # [B]  (VDN sum)

    with torch.no_grad():
        # Next actions via online (Double-DQN), per agent
        q_next_online = online_q(NS_t).view(B, N, -1)           # [B,N,A]
        next_actions = q_next_online.argmax(dim=-1)             # [B,N]

        # Evaluate those actions on target net
        q_next_target = target_q(NS_t).view(B, N, -1)           # [B,N,A]
        q_next_a = q_next_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)  # [B,N]
        q_next_tot = q_next_a.sum(dim=1)                        # [B]

        # Team reward (sum of per-agent rewards)
        R_team = R_t.sum(dim=1)                                 # [B]
        target = R_team + (1.0 - D_t) * gamma * q_next_tot      # [B]

    loss = nn.functional.smooth_l1_loss(q_tot, target)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())

# ============================= CTDE-specific: QMIX update (Double-DQN) =============================
def qmix_update_mlp(
    device: torch.device,
    online_q: QNetMLP,
    target_q: QNetMLP,
    online_mixer: QMIXMixer,
    target_mixer: QMIXMixer,
    optimizer: optim.Optimizer,
    batch_tuple,
    gamma: float = 0.99,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    QMIX update with Double-DQN target.

    Inputs from JointReplayBuffer.sample:
        states      : [B, N, O]
        actions     : [B, N]
        rewards     : [B, N]
        next_states : [B, N, O]
        dones       : [B]

    This implementation uses flattened joint observations as the QMIX global state:
        global_state = concat(o_1, ..., o_N)
    """

    S, A, R, NS, D = batch_tuple
    B, N, O = S.shape

    S_t = torch.as_tensor(S, device=device, dtype=torch.float32)       # [B, N, O]
    NS_t = torch.as_tensor(NS, device=device, dtype=torch.float32)     # [B, N, O]
    A_t = torch.as_tensor(A, device=device, dtype=torch.long)          # [B, N]
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)       # [B, N]
    D_t = torch.as_tensor(D, device=device, dtype=torch.float32)       # [B]

    # Global state for QMIX mixer.
    # If your env has a true global state, use that instead.
    global_state = S_t.view(B, N * O)          # [B, N*O]
    next_global_state = NS_t.view(B, N * O)    # [B, N*O]

    # Flatten per-agent observations for shared Q network
    S_flat = S_t.view(B * N, O)                # [B*N, O]
    NS_flat = NS_t.view(B * N, O)              # [B*N, O]
    A_flat = A_t.view(B * N)                   # [B*N]

    # Current per-agent selected Q-values
    q_all = online_q(S_flat)                   # [B*N, A_dim]
    q_taken = q_all.gather(1, A_flat.view(-1, 1)).squeeze(1)
    q_taken = q_taken.view(B, N)               # [B, N]

    # QMIX centralised mixing
    q_tot = online_mixer(q_taken, global_state)  # [B]

    with torch.no_grad():
        q_next_online = online_q(NS_flat).view(B, N, -1)  # [B, N, A_dim]
        q_next_target = target_q(NS_flat).view(B, N, -1)  # [B, N, A_dim]

        if double_dqn:
            # Action selection using online network
            next_actions = q_next_online.argmax(dim=-1)  # [B, N]

            # Action evaluation using target network
            q_next_taken = q_next_target.gather(
                -1,
                next_actions.unsqueeze(-1)
            ).squeeze(-1)  # [B, N]
        else:
            q_next_taken = q_next_target.max(dim=-1).values  # [B, N]

        # Target QMIX mixing
        q_next_tot = target_mixer(q_next_taken, next_global_state)  # [B]

        # Team reward.
        # Keep this as sum if your reward_dict contains local rewards.
        # If each agent already receives the same global reward, use mean or just R_t[:, 0].
        R_team = R_t.sum(dim=1)  # [B]

        target = R_team + (1.0 - D_t) * gamma * q_next_tot  # [B]

    loss = nn.functional.smooth_l1_loss(q_tot, target)

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(
        list(online_q.parameters()) + list(online_mixer.parameters()),
        grad_clip,
    )

    optimizer.step()

    return float(loss.item())

# ================= CTDE-specific: VDN update for LSTM (Double-DQN) ===================

def vdn_update_lstm(
    device: torch.device,
    online_q: "RecurrentQNet",    # forward([B,T,O], hidden) -> ([B,T,A], new_hidden)
    target_q: "RecurrentQNet",
    optimizer: torch.optim.Optimizer,
    batch_tuple,                   # (S, A, R, NS, D, M) from JointSequenceReplay
    gamma: float = 0.99,
    burn_in: int = 6,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    S, A, R, NS, D, M = batch_tuple
    S  = torch.as_tensor(S,  device=device, dtype=torch.float32)  # [B,L,N,O]
    A  = torch.as_tensor(A,  device=device, dtype=torch.long)     # [B,L,N]
    R  = torch.as_tensor(R,  device=device, dtype=torch.float32)  # [B,L,N]
    NS = torch.as_tensor(NS, device=device, dtype=torch.float32)  # [B,L,N,O]
    D  = torch.as_tensor(D,  device=device, dtype=torch.float32)  # [B,L] (team)
    M  = torch.as_tensor(M,  device=device, dtype=torch.float32)  # [B,L] (mask)

    B, L, N, O = S.shape
    burn = int(burn_in)
    if burn >= L:
        burn = max(0, L - 1)

    # warm hidden with burn-in (vectorized across agents: B*N sequences)
    h_o = h_t = None
    if burn > 0:
        with torch.no_grad():
            _, h_o = online_q(S[:, :burn].reshape(B * N, burn, O), None)
            _, h_t = target_q(NS[:, :burn].reshape(B * N, burn, O), None)

    Tw = L - burn
    Q_team_list, Target_team_list = [], []

    for t in range(Tw):
        tt = burn + t

        s_t  = S[:, tt].reshape(B * N, 1, O)
        ns_t = NS[:, tt].reshape(B * N, 1, O)
        a_t  = A[:, tt].reshape(B * N)
        r_t  = R[:, tt]                 # [B,N]
        d_ts = D[:, tt]                 # [B] team done

        # online forward (with grad)
        q_now, h_o = online_q(s_t, h_o)            # [B*N,1,A]
        q_now = q_now.squeeze(1)                   # [B*N,A]
        q_taken_nodes = q_now.gather(1, a_t.unsqueeze(1)).squeeze(1)  # [B*N]
        q_taken_team  = q_taken_nodes.view(B, N).sum(dim=1)           # [B]
        Q_team_list.append(q_taken_team)

        with torch.no_grad():
            qn_o, _ = online_q(ns_t, h_o)         # argmax path (disposable)
            qn_t, h_t = target_q(ns_t, h_t)       # target path (stateful)
            qn_o = qn_o.squeeze(1); qn_t = qn_t.squeeze(1)           # [B*N,A]

            next_a = torch.argmax(qn_o if double_dqn else qn_t, dim=-1)     # [B*N]
            next_q_team = qn_t.gather(1, next_a.unsqueeze(1)).squeeze(1)    # [B*N]
            next_q_team = next_q_team.view(B, N).sum(dim=1)                  # [B]

            r_team = r_t.sum(dim=1)                                         # [B]
            target_team = r_team + (1.0 - d_ts) * gamma * next_q_team       # [B]
            Target_team_list.append(target_team)

        # reset hidden where team done
        if h_o is not None:
            keep = (1.0 - d_ts).view(1, B, 1, 1)
            h0 = h_o[0].view(1, B, N, -1); c0 = h_o[1].view(1, B, N, -1)
            h0.mul_(keep); c0.mul_(keep)
            h_o = (h0.view(1, B * N, -1), c0.view(1, B * N, -1))
        if h_t is not None:
            keep = (1.0 - d_ts).view(1, B, 1, 1)
            h1 = h_t[0].view(1, B, N, -1); c1 = h_t[1].view(1, B, N, -1)
            h1.mul_(keep); c1.mul_(keep)
            h_t = (h1.view(1, B * N, -1), c1.view(1, B * N, -1))

    Q_team  = torch.stack(Q_team_list,      dim=1)   # [B, Tw]
    TargetT = torch.stack(Target_team_list, dim=1)   # [B, Tw]
    M_w = M[:, burn:]                                  # [B, Tw]

    td = nn.functional.smooth_l1_loss(Q_team, TargetT, reduction='none') * M_w
    denom = M_w.sum().clamp_min(1.0)
    loss = td.sum() / denom

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())

# ================= CTDE-specific: QMIX update for LSTM (Double-DQN) ===================
def qmix_update_lstm(
    device: torch.device,
    online_q: "RecurrentQNet",
    target_q: "RecurrentQNet",
    online_mixer: nn.Module,
    target_mixer: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_tuple,
    gamma: float = 0.99,
    burn_in: int = 6,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    QMIX update for DRQN/LSTM per-agent Q_i.

    Inputs from JointSequenceReplay.sample:

        S  : [B, L, N, O]
        A  : [B, L, N]
        R  : [B, L, N]
        NS : [B, L, N, O]
        D  : [B, L]
        M  : [B, L]

    where:
        B = sequence batch size
        L = sampled sequence length
        N = number of agents
        O = local observation dimension

    The recurrent Q-network produces per-agent Q_i values.

    QMIX then mixes:

        [Q_1, Q_2, ..., Q_N]

    into:

        Q_tot

    using a centralised state:

        s_t = concat(o_1, ..., o_N)
    """

    S, A, R, NS, D, M = batch_tuple

    S = torch.as_tensor(
        S,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N, O]

    A = torch.as_tensor(
        A,
        device=device,
        dtype=torch.long,
    )                                                               # [B, L, N]

    R = torch.as_tensor(
        R,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N]

    NS = torch.as_tensor(
        NS,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N, O]

    D = torch.as_tensor(
        D,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L]

    M = torch.as_tensor(
        M,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L]

    B, L, N, O = S.shape

    burn = int(burn_in)

    if burn >= L:
        burn = max(0, L - 1)

    # ------------------------------------------------------------------
    # Burn-in recurrent hidden states
    # ------------------------------------------------------------------

    h_online = None
    h_target = None

    if burn > 0:
        with torch.no_grad():
            # Warm online hidden using S.
            _, h_online = online_q(
                S[:, :burn].reshape(B * N, burn, O),
                None,
            )

            # Warm target hidden using NS.
            _, h_target = target_q(
                NS[:, :burn].reshape(B * N, burn, O),
                None,
            )

    Tw = L - burn

    q_tot_list = []
    target_tot_list = []

    # ------------------------------------------------------------------
    # Unroll from burn-in point
    # ------------------------------------------------------------------

    for t in range(Tw):
        tt = burn + t

        s_t = S[:, tt]                                               # [B, N, O]
        ns_t_full = NS[:, tt]                                        # [B, N, O]

        s_t_flat = s_t.reshape(B * N, 1, O)                          # [B*N, 1, O]
        ns_t_flat = ns_t_full.reshape(B * N, 1, O)                   # [B*N, 1, O]

        a_t = A[:, tt].reshape(B * N)                                # [B*N]
        r_t = R[:, tt]                                               # [B, N]
        d_t = D[:, tt]                                               # [B]

        # Centralised states for QMIX mixer.
        global_state_t = s_t.reshape(B, N * O)                       # [B, N*O]
        next_global_state_t = ns_t_full.reshape(B, N * O)            # [B, N*O]

        # --------------------------------------------------------------
        # Current Q_tot
        # --------------------------------------------------------------

        q_now_seq, h_online = online_q(s_t_flat, h_online)           # [B*N, 1, A]
        q_now = q_now_seq.squeeze(1)                                 # [B*N, A]

        q_taken_nodes = q_now.gather(
            1,
            a_t.unsqueeze(1),
        ).squeeze(1)                                                 # [B*N]

        q_taken_nodes = q_taken_nodes.view(B, N)                     # [B, N]

        q_tot = online_mixer(
            q_taken_nodes,
            global_state_t,
        )                                                           # [B]

        q_tot_list.append(q_tot)

        # --------------------------------------------------------------
        # Target Q_tot
        # --------------------------------------------------------------

        with torch.no_grad():
            q_next_online_seq, _ = online_q(
                ns_t_flat,
                h_online,
            )                                                       # [B*N, 1, A]

            q_next_target_seq, h_target = target_q(
                ns_t_flat,
                h_target,
            )                                                       # [B*N, 1, A]

            q_next_online = q_next_online_seq.squeeze(1)             # [B*N, A]
            q_next_target = q_next_target_seq.squeeze(1)             # [B*N, A]

            if double_dqn:
                # Action selection using online recurrent Q-network.
                next_actions = torch.argmax(q_next_online, dim=-1)   # [B*N]
            else:
                next_actions = torch.argmax(q_next_target, dim=-1)   # [B*N]

            # Action evaluation using target recurrent Q-network.
            q_next_taken_nodes = q_next_target.gather(
                1,
                next_actions.unsqueeze(1),
            ).squeeze(1)                                             # [B*N]

            q_next_taken_nodes = q_next_taken_nodes.view(B, N)       # [B, N]

            q_next_tot = target_mixer(
                q_next_taken_nodes,
                next_global_state_t,
            )                                                       # [B]

            # Team reward = sum over agents.
            r_team = r_t.sum(dim=1)                                  # [B]

            target_tot = r_team + (1.0 - d_t) * gamma * q_next_tot   # [B]

            target_tot_list.append(target_tot)

        # --------------------------------------------------------------
        # Reset recurrent hidden states where the team episode is done
        # --------------------------------------------------------------

        if h_online is not None:
            keep = (1.0 - d_t).view(1, B, 1, 1)                      # [1, B, 1, 1]

            h0 = h_online[0].view(1, B, N, -1)
            c0 = h_online[1].view(1, B, N, -1)

            h0 = h0 * keep
            c0 = c0 * keep

            h_online = (
                h0.reshape(1, B * N, -1),
                c0.reshape(1, B * N, -1),
            )

        if h_target is not None:
            keep = (1.0 - d_t).view(1, B, 1, 1)                      # [1, B, 1, 1]

            h1 = h_target[0].view(1, B, N, -1)
            c1 = h_target[1].view(1, B, N, -1)

            h1 = h1 * keep
            c1 = c1 * keep

            h_target = (
                h1.reshape(1, B * N, -1),
                c1.reshape(1, B * N, -1),
            )

    # ------------------------------------------------------------------
    # Masked TD loss over post-burn-in timesteps
    # ------------------------------------------------------------------

    Q_tot = torch.stack(q_tot_list, dim=1)                           # [B, Tw]
    Target_tot = torch.stack(target_tot_list, dim=1)                 # [B, Tw]

    M_w = M[:, burn:]                                                # [B, Tw]

    td_loss = nn.functional.smooth_l1_loss(
        Q_tot,
        Target_tot,
        reduction="none",
    )                                                               # [B, Tw]

    td_loss = td_loss * M_w

    denom = M_w.sum().clamp_min(1.0)
    loss = td_loss.sum() / denom

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(online_q.parameters()) + list(online_mixer.parameters()),
        grad_clip,
    )

    optimizer.step()

    return float(loss.item())

# =================== CTDE-specific: VDN update (Double-DQN) — GNN ======================
def vdn_update_gnn(
    device: torch.device,
    gnn_online_q: nn.Module,          # GNNPolicyQ: per-agent Q_i with message-passing
    gnn_target_q: nn.Module,          # target copy
    optimizer: optim.Optimizer,
    batch_tuple,                      # from JointReplayBuffer.sample -> (S, A, R, NS, D)
    graph_edge_index: torch.Tensor,   # [2,E] directed, same for all items in batch
    gamma: float = 0.99,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    VDN with GNN Q_i:
      - Q_tot = sum_i Q_i
      - Double-DQN targets per agent
      - team reward = sum over per-agent rewards

    Shapes from JointReplayBuffer:
      S  : [B, N, O]
      A  : [B, N]         (long)
      R  : [B, N]         (float)
      NS : [B, N, O]
      D  : [B]            (float in {0,1})
    """
    import torch.nn as nn
    S, A, R, NS, D = batch_tuple
    B, N, O = S.shape

    S_t  = torch.as_tensor(S,  device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    NS_t = torch.as_tensor(NS, device=device, dtype=torch.float32).view(B * N, O)  # [B*N,O]
    A_t  = torch.as_tensor(A,  device=device, dtype=torch.long).view(B * N)        # [B*N]
    R_t  = torch.as_tensor(R,  device=device, dtype=torch.float32)                 # [B,N]
    D_t  = torch.as_tensor(D,  device=device, dtype=torch.float32)                 # [B]

    # Build batched edges once
    Ei_batched = build_batched_edge_index(graph_edge_index.to(device), B, N)       # [2,B*E]

    # Per-agent Q(s,a) (flattened across batch and nodes)
    q_all = gnn_online_q(S_t, Ei_batched)                         # [B*N, A]
    q_taken = q_all.gather(1, A_t.view(-1, 1)).squeeze(1)         # [B*N]
    q_taken = q_taken.view(B, N)                                   # [B,N]
    q_tot = q_taken.sum(dim=1)                                     # [B]

    with torch.no_grad():
        # Next actions via online (Double-DQN), per agent
        q_next_online = gnn_online_q(NS_t, Ei_batched).view(B, N, -1)   # [B,N,A]
        next_actions = q_next_online.argmax(dim=-1)                     # [B,N]

        # Evaluate those actions on target net
        q_next_target = gnn_target_q(NS_t, Ei_batched).view(B, N, -1)   # [B,N,A]
        q_next_a = q_next_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)  # [B,N]
        q_next_tot = q_next_a.sum(dim=1)                                 # [B]

        # Team reward (sum over agents)
        R_team = R_t.sum(dim=1)                                          # [B]
        target = R_team + (1.0 - D_t) * gamma * q_next_tot               # [B]

    loss = nn.functional.smooth_l1_loss(q_tot, target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gnn_online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())


# =================== CTDE-specific: QMIX update (Double-DQN) — GNN ======================
def qmix_update_gnn(
    device: torch.device,
    gnn_online_q: nn.Module,
    gnn_target_q: nn.Module,
    online_mixer: nn.Module,
    target_mixer: nn.Module,
    optimizer: optim.Optimizer,
    batch_tuple,
    graph_edge_index: torch.Tensor,
    gamma: float = 0.99,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
) -> float:
    """
    QMIX update with GNN per-agent Q_i and Double-DQN targets.

    Shapes from JointReplayBuffer.sample:

        S  : [B, N, O]
        A  : [B, N]
        R  : [B, N]
        NS : [B, N, O]
        D  : [B]

    The GNN processes a batched disconnected graph with B copies of the same
    traffic-signal topology.

    The QMIX mixer receives:

        q_taken:     [B, N]
        global_state [B, N * O]

    where global_state is the flattened joint observation.
    """

    S, A, R, NS, D = batch_tuple
    B, N, O = S.shape

    S_t = torch.as_tensor(
        S,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, N, O]

    NS_t = torch.as_tensor(
        NS,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, N, O]

    A_t = torch.as_tensor(
        A,
        device=device,
        dtype=torch.long,
    )                                                               # [B, N]

    R_t = torch.as_tensor(
        R,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, N]

    D_t = torch.as_tensor(
        D,
        device=device,
        dtype=torch.float32,
    )                                                               # [B]

    # Centralised states for QMIX.
    global_state = S_t.reshape(B, N * O)                            # [B, N*O]
    next_global_state = NS_t.reshape(B, N * O)                      # [B, N*O]

    # Flatten batch and node dimensions for GNN.
    S_flat = S_t.reshape(B * N, O)                                  # [B*N, O]
    NS_flat = NS_t.reshape(B * N, O)                                # [B*N, O]
    A_flat = A_t.reshape(B * N)                                     # [B*N]

    # Build B disconnected copies of the same traffic-signal graph.
    batched_edge_index = build_batched_edge_index(
        graph_edge_index.to(device),
        batch_size=B,
        num_nodes=N,
    )                                                               # [2, B*E]

    # ------------------------------------------------------------------
    # Current Q_tot
    # ------------------------------------------------------------------

    q_all = gnn_online_q(S_flat, batched_edge_index)                 # [B*N, A_dim]

    q_taken = q_all.gather(
        1,
        A_flat.view(-1, 1),
    ).squeeze(1)                                                     # [B*N]

    q_taken = q_taken.view(B, N)                                     # [B, N]

    q_tot = online_mixer(q_taken, global_state)                      # [B]

    # ------------------------------------------------------------------
    # Target Q_tot
    # ------------------------------------------------------------------

    with torch.no_grad():
        q_next_online = gnn_online_q(
            NS_flat,
            batched_edge_index,
        ).view(B, N, -1)                                             # [B, N, A_dim]

        q_next_target = gnn_target_q(
            NS_flat,
            batched_edge_index,
        ).view(B, N, -1)                                             # [B, N, A_dim]

        if double_dqn:
            # Select next actions using the online GNN Q-network.
            next_actions = q_next_online.argmax(dim=-1)              # [B, N]

            # Evaluate selected actions using the target GNN Q-network.
            q_next_taken = q_next_target.gather(
                -1,
                next_actions.unsqueeze(-1),
            ).squeeze(-1)                                            # [B, N]
        else:
            q_next_taken = q_next_target.max(dim=-1).values          # [B, N]

        q_next_tot = target_mixer(
            q_next_taken,
            next_global_state,
        )                                                           # [B]

        # Team reward: sum over per-agent rewards.
        R_team = R_t.sum(dim=1)                                      # [B]

        target = R_team + (1.0 - D_t) * gamma * q_next_tot           # [B]

    # ------------------------------------------------------------------
    # Loss and update
    # ------------------------------------------------------------------

    loss = nn.functional.smooth_l1_loss(q_tot, target)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(gnn_online_q.parameters()) + list(online_mixer.parameters()),
        grad_clip,
    )

    optimizer.step()

    return float(loss.item())


# ============= CTDE-specific: VDN update (Double-DQN) — GNN+LSTM ===============
def _shift_next_actions_double_dqn(
    online_q, S_next: torch.Tensor, edge_index: torch.Tensor
) -> torch.Tensor:
    """
    Argmax_a Q_online(S_next, a) per node across the sequence.
    Input:  S_next [B, T, N, O]
    Return: a_next [B, T, N] (long)
    """
    q_next, _ = online_q(S_next, edge_index, hidden=None)      # [B,T,N,A]
    return q_next.argmax(dim=-1).long()                        # [B,T,N]


def vdn_update_gnn_lstm(
    device: torch.device,
    gnnlstm_online_q,                # GNNLSTMPolicyQ
    gnnlstm_target_q,                # GNNLSTMPolicyQ (target)
    optimizer: optim.Optimizer,
    seq_batch,                       # (S_seq, A_seq, R_seq, NS_seq, D_seq)
    graph_edge_index: torch.Tensor,  # [2,E]
    gamma: float = 0.99,
    double_dqn: bool = True,
    burn_in: int = 8,
    grad_clip: float = 1.0,
) -> float:
    """
    CTDE-VDN with spatio-temporal Q:
      - Inputs are **sequence windows**:
          S_seq   : [B, L, N, O]
          A_seq   : [B, L, N]
          R_seq   : [B, L, N]
          NS_seq  : [B, L, N, O]   (next-state aligned)
          D_seq   : [B, L]
        where L = burn_in + unroll_len.
      - We compute losses only on the last `unroll_len` steps (mask out burn-in).
      - VDN: Q_tot = sum_i Q_i
      - Double-DQN targets across the sequence.
    """
    S, A, R, NS, D = seq_batch
    B, L, N, O = S.shape

    S_t  = torch.as_tensor(S,  device=device, dtype=torch.float32)
    A_t  = torch.as_tensor(A,  device=device, dtype=torch.long)
    R_t  = torch.as_tensor(R,  device=device, dtype=torch.float32)
    NS_t = torch.as_tensor(NS, device=device, dtype=torch.float32)
    D_t  = torch.as_tensor(D,  device=device, dtype=torch.float32)

    # Forward over full window to warm hidden; mask out burn-in for the loss
    q_seq, _ = gnnlstm_online_q(S_t, graph_edge_index.to(device), hidden=None)   # [B,L,N,A]
    q_taken = q_seq.gather(-1, A_t.unsqueeze(-1)).squeeze(-1)                    # [B,L,N]
    q_tot   = q_taken.sum(dim=-1)                                                # [B,L]

    with torch.no_grad():
        if double_dqn:
            next_actions = _shift_next_actions_double_dqn(gnnlstm_online_q, NS_t, graph_edge_index.to(device))  # [B,L,N]
        else:
            tmp_q, _ = gnnlstm_target_q(NS_t, graph_edge_index.to(device), hidden=None)
            next_actions = tmp_q.argmax(dim=-1).long()  # [B,L,N]

        q_next_targ, _ = gnnlstm_target_q(NS_t, graph_edge_index.to(device), hidden=None)          # [B,L,N,A]
        q_next_a = q_next_targ.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)                  # [B,L,N]
        q_next_tot = q_next_a.sum(dim=-1)                                                          # [B,L]

        R_team = R_t.sum(dim=-1)                                                                   # [B,L]
        target = R_team + (1.0 - D_t) * gamma * q_next_tot                                         # [B,L]

    # Mask out burn-in steps
    if burn_in > 0:
        mask = torch.zeros((B, L), device=device)
        mask[:, burn_in:] = 1.0
    else:
        mask = torch.ones((B, L), device=device)

    td = nn.functional.smooth_l1_loss(q_tot, target, reduction='none')   # [B,L]
    loss = (td * mask).sum() / (mask.sum().clamp_min(1.0))

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gnnlstm_online_q.parameters(), grad_clip)
    optimizer.step()

    return float(loss.item())

# ============= CTDE-specific: QMIX update (Double-DQN) — GNN+LSTM ===============
def qmix_update_gnn_lstm(
    device: torch.device,
    gnnlstm_online_q,
    gnnlstm_target_q,
    online_mixer: nn.Module,
    target_mixer: nn.Module,
    optimizer: optim.Optimizer,
    seq_batch,
    graph_edge_index: torch.Tensor,
    gamma: float = 0.99,
    double_dqn: bool = True,
    burn_in: int = 8,
    grad_clip: float = 1.0,
) -> float:
    """
    CTDE-QMIX with spatio-temporal GNN-LSTM Q_i.

    Inputs are sequence windows:

        S_seq  : [B, L, N, O]
        A_seq  : [B, L, N]
        R_seq  : [B, L, N]
        NS_seq : [B, L, N, O]
        D_seq  : [B, L]

    where:

        B = sampled sequence batch size
        L = sequence length
        N = number of agents / junctions
        O = local observation dimension

    The per-agent Q-network is recurrent and graph-based:

        Q_i = GNNLSTMPolicyQ(o_i, graph, h_i)

    QMIX replaces VDN's sum:

        VDN:
            Q_tot = sum_i Q_i

        QMIX:
            Q_tot = mixer([Q_1, ..., Q_N], global_state)

    Here global_state is the flattened joint observation:

        global_state_t = concat(o_1, ..., o_N)
    """

    S, A, R, NS, D = seq_batch

    B, L, N, O = S.shape

    S_t = torch.as_tensor(
        S,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N, O]

    A_t = torch.as_tensor(
        A,
        device=device,
        dtype=torch.long,
    )                                                               # [B, L, N]

    R_t = torch.as_tensor(
        R,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N]

    NS_t = torch.as_tensor(
        NS,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L, N, O]

    D_t = torch.as_tensor(
        D,
        device=device,
        dtype=torch.float32,
    )                                                               # [B, L]

    edge_index = graph_edge_index.to(device)

    burn = int(burn_in)

    if burn >= L:
        burn = max(0, L - 1)

    # ------------------------------------------------------------------
    # Online Q over full sequence
    # ------------------------------------------------------------------

    q_seq, _ = gnnlstm_online_q(
        S_t,
        edge_index,
        hidden=None,
    )                                                               # [B, L, N, A]

    q_taken = q_seq.gather(
        -1,
        A_t.unsqueeze(-1),
    ).squeeze(-1)                                                    # [B, L, N]

    # Centralised state for QMIX at each timestep.
    global_states = S_t.reshape(B, L, N * O)                         # [B, L, N*O]

    # Flatten B and L so we can call the standard QMIXMixer.
    q_taken_flat = q_taken.reshape(B * L, N)                         # [B*L, N]
    global_states_flat = global_states.reshape(B * L, N * O)         # [B*L, N*O]

    q_tot_flat = online_mixer(
        q_taken_flat,
        global_states_flat,
    )                                                               # [B*L]

    q_tot = q_tot_flat.view(B, L)                                    # [B, L]

    # ------------------------------------------------------------------
    # Target Q_tot
    # ------------------------------------------------------------------

    with torch.no_grad():
        if double_dqn:
            next_actions = _shift_next_actions_double_dqn(
                gnnlstm_online_q,
                NS_t,
                edge_index,
            )                                                      # [B, L, N]
        else:
            tmp_q, _ = gnnlstm_target_q(
                NS_t,
                edge_index,
                hidden=None,
            )                                                       # [B, L, N, A]

            next_actions = tmp_q.argmax(dim=-1).long()              # [B, L, N]

        q_next_target_seq, _ = gnnlstm_target_q(
            NS_t,
            edge_index,
            hidden=None,
        )                                                           # [B, L, N, A]

        q_next_taken = q_next_target_seq.gather(
            -1,
            next_actions.unsqueeze(-1),
        ).squeeze(-1)                                                # [B, L, N]

        next_global_states = NS_t.reshape(B, L, N * O)               # [B, L, N*O]

        q_next_taken_flat = q_next_taken.reshape(B * L, N)           # [B*L, N]
        next_global_states_flat = next_global_states.reshape(
            B * L,
            N * O,
        )                                                           # [B*L, N*O]

        q_next_tot_flat = target_mixer(
            q_next_taken_flat,
            next_global_states_flat,
        )                                                           # [B*L]

        q_next_tot = q_next_tot_flat.view(B, L)                      # [B, L]

        # Team reward = sum of local rewards.
        R_team = R_t.sum(dim=-1)                                     # [B, L]

        target = R_team + (1.0 - D_t) * gamma * q_next_tot           # [B, L]

    # ------------------------------------------------------------------
    # Mask out burn-in steps
    # ------------------------------------------------------------------

    if burn > 0:
        mask = torch.zeros(
            (B, L),
            device=device,
            dtype=torch.float32,
        )

        mask[:, burn:] = 1.0
    else:
        mask = torch.ones(
            (B, L),
            device=device,
            dtype=torch.float32,
        )

    td = nn.functional.smooth_l1_loss(
        q_tot,
        target,
        reduction="none",
    )                                                               # [B, L]

    loss = (td * mask).sum() / mask.sum().clamp_min(1.0)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(gnnlstm_online_q.parameters()) + list(online_mixer.parameters()),
        grad_clip,
    )

    optimizer.step()

    return float(loss.item())

# =============================== A2C update (MLP) =================================
# GAE utilities
def _compute_gae(
    rewards: torch.Tensor,      # [T,N]
    values: torch.Tensor,       # [T,N]
    dones: torch.Tensor,        # [T]  (1.0 at terminal step, else 0.0)
    next_value: torch.Tensor,   # [N]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (advantages, returns) each shape [T,N].
    dones[t]=1 means terminal at step t (stop bootstrapping beyond t).
    """
    T, N = rewards.shape
    adv = torch.zeros((T, N), device=rewards.device)
    gae = torch.zeros((N,), device=rewards.device)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]           # scalar broadcast over N
        v_next = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nonterm * v_next - values[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    ret = adv + values
    return adv, ret

def ia2c_update_mlp(
    actor: nn.Module,                 # ActorMLP: obs -> logits[A]
    critic: nn.Module,                # CriticMLP: obs -> value
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,            # [T,N,O]
    act_seq: torch.Tensor,            # [T,N] (long)
    rew_seq: torch.Tensor,            # [T,N] (float)
    done_seq: torch.Tensor,           # [T]   (float; episode terminal flags)
    last_obs: torch.Tensor,           # [N,O] (for bootstrap if not terminal)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # stability knobs (no diagnostics, no repeats)
    normalize_rewards: bool = True,
    reward_scale: float = 1.0,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,      # set 0 to disable clipping
) -> dict:
    """
    One IA2C update with GAE(λ) for a single rollout (length T).
    Adds: reward normalization, advantage normalization, Huber value loss,
    and optional value clipping (A2C-style).
    """
    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # Flatten (T*N, O) once
    obs_flat = obs_seq.reshape(T * N, O).to(device)
    actions  = act_seq.to(device)
    dones    = done_seq.to(device).float()
    last_obs = last_obs.to(device)

    # Forward passes
    logits_flat = actor(obs_flat)                 # [T*N, A]
    values_flat = critic(obs_flat)                # [T*N]
    A_dim = logits_flat.size(-1)

    logits = logits_flat.view(T, N, A_dim)       # [T,N,A]
    values = values_flat.view(T, N)              # [T,N]

    # Policy log-prob & entropy
    dist = Categorical(logits=logits)            # broadcast over [T,N]
    logp_taken = dist.log_prob(actions.long())   # [T,N]
    entropy = dist.entropy().mean()

    # Bootstrap value
    with torch.no_grad():
        if dones[-1] > 0.5:
            next_value = torch.zeros((N,), device=device)
        else:
            next_value = critic(last_obs)        # [N]

    # Reward normalization (across T*N), optional
    rewards = rew_seq.to(device).float()
    if normalize_rewards:
        r_mean = rewards.mean()
        r_std  = rewards.std(unbiased=False).clamp_min(1e-6)
        rewards = reward_scale * (rewards - r_mean) / r_std

    # GAE (on-device)
    adv, ret = _compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        gamma=gamma,
        lam=gae_lambda,
    )  # [T,N] each

    # Advantage normalization (over T*N), optional
    if normalize_adv:
        adv_flat = adv.reshape(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # Losses
    policy_loss = -(logp_taken * adv.detach()).mean()

    # Value loss: Huber + optional value clipping (A2C-style)
    v_pred   = values
    v_target = ret.detach()
    if value_clip_eps and value_clip_eps > 0.0:
        v_clipped = v_pred + (v_target - v_pred).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,   v_target, reduction='mean', beta=huber_delta)
        v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_target, reduction='mean', beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_target, reduction='mean', beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # Optimize
    optim_actor.zero_grad()
    optim_critic.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
    }

# =============================== A2C update (LSTM) =================================
def ia2c_update_lstm(
    actor: ActorLSTM,
    critic: CriticLSTM,
    optim_actor: optim.Optimizer,
    optim_critic: optim.Optimizer,
    obs_seq: torch.Tensor,     # [T,N,O]
    act_seq: torch.Tensor,     # [T,N] (long)
    rew_seq: torch.Tensor,     # [T,N]
    done_seq: torch.Tensor,    # [T]  (float; 1.0 terminal)
    last_obs: torch.Tensor,    # [N,O] bootstrap
    gamma: float,
    gae_lambda: float,
    entropy_coef: float,
    value_coef: float,
    grad_clip: float,
    # optional stabilizers (match your MLP version)
    normalize_rewards: bool = True,
    reward_scale: float = 0.1,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,
) -> Dict[str, float]:

    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape
    obs_seq  = obs_seq.to(device)
    act_seq  = act_seq.to(device).long()
    rew_seq  = rew_seq.to(device)
    done_seq = done_seq.to(device)
    last_obs = last_obs.to(device)

    # -------- TRAIN MODE for the forwards we backprop through --------
    actor.train()
    critic.train()

    # -------- critic forward over the whole sequence (builds graph) --------
    # CriticLSTM(...).forward should return (values_seq_batched, (h,c)) where
    # values_seq_batched has shape [1, T, N] or [T, N] depending on your impl.
    values_batched, _ = critic(obs_seq)     # no torch.no_grad() here
    if values_batched.dim() == 3:
        values_seq = values_batched[0]      # [T, N]
    else:
        values_seq = values_batched         # already [T, N]

    # -------- bootstrap next value (no grad, eval ok) --------
    with torch.no_grad():
        critic.eval()
        next_value, _ = critic.step(last_obs)     # [N]
    critic.train()  # switch back for the losses/opt step

    # -------- reward normalization / scaling (optional) --------
    if normalize_rewards:
        mean_r = rew_seq.mean()
        std_r  = rew_seq.std(unbiased=False).clamp_min(1e-6)
        rew_use = (rew_seq - mean_r) / std_r
    else:
        rew_use = rew_seq
    if reward_scale is not None and reward_scale != 1.0:
        rew_use = rew_use * reward_scale

    # -------- GAE (no grad) --------
    with torch.no_grad():
        adv, ret = _compute_gae(
            rewards=rew_use, values=values_seq, dones=done_seq,
            next_value=next_value, gamma=gamma, lam=gae_lambda
        )  # [T,N] each

    # -------- actor forward for logits over the sequence (builds graph) --------
    logits_batched, _ = actor(obs_seq)    # no torch.no_grad()
    if logits_batched.dim() == 4:
        logits_seq = logits_batched[0]    # [T, N, A]
    else:
        logits_seq = logits_batched       # already [T, N, A]

    dist = Categorical(logits=logits_seq)
    logp_taken = dist.log_prob(act_seq)         # [T,N]
    entropy = dist.entropy().mean()

    # -------- advantage normalization (over T*N) --------
    if normalize_adv:
        adv_flat = adv.view(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # -------- losses --------
    policy_loss = -(logp_taken * adv.detach()).mean()

    # value loss with Huber + value clipping
    with torch.no_grad():
        old_values = values_seq.clone()
    # clipped target around old values to stabilize critic
    values_clipped = old_values + (values_seq - old_values).clamp(-value_clip_eps, value_clip_eps)
    # Huber (smooth L1) on unclipped and clipped, take max like PPO value clipping
    v_err_unclipped = ret.detach() - values_seq
    v_err_clipped   = ret.detach() - values_clipped
    value_loss_unclipped = torch.nn.functional.smooth_l1_loss(
        values_seq, ret.detach(), beta=huber_delta
    )
    value_loss_clipped = torch.nn.functional.smooth_l1_loss(
        values_clipped, ret.detach(), beta=huber_delta
    )
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

    loss_total = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # -------- optimize --------
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    loss_total.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(loss_total.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
    }

# ================= A2C update (LSTM for actor, MLP for critic) ====================
def ia2c_update_lstm_actor_mlp(
    actor: nn.Module,                 # ActorLSTM: local seq [T,N,O] -> logits [T,N,A]
    critic: nn.Module,                # CriticMLP: local obs [O]    -> value  [1]
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,            # [T, N, O]
    act_seq: torch.Tensor,            # [T, N] (long)
    rew_seq: torch.Tensor,            # [T, N] (float)
    done_seq: torch.Tensor,           # [T]   (float; 1.0 if terminal at t)
    last_obs: torch.Tensor,           # [N, O] local obs for bootstrap
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # stabilizers
    normalize_rewards: bool = True,
    reward_scale: float = 1.0,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,      # 0 to disable clipping
) -> dict:
    """
    IA2C update for one rollout using an LSTM actor (shared, decentralized)
    and a per-agent MLP critic (shared). Computes *per-agent* GAE on local values.
    """
    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # Move to device
    obs_seq  = obs_seq.to(device).float()     # [T,N,O]
    actions  = act_seq.to(device).long()      # [T,N]
    rewards  = rew_seq.to(device).float()     # [T,N]
    dones    = done_seq.to(device).float()    # [T]
    last_obs = last_obs.to(device).float()    # [N,O]

    # ---------- Actor forward over the sequence ----------
    # ActorLSTM.forward accepts [T,N,O] and returns [1,T,N,A]
    logits_seq, _ = actor(obs_seq)            # [1,T,N,A]
    logits = logits_seq[0]                    # [T,N,A]

    dist       = Categorical(logits=logits)   # broadcast over [T,N]
    logp_taken = dist.log_prob(actions)       # [T,N]
    entropy    = dist.entropy().mean()

    # ---------- Per-agent critic on LOCAL obs ----------
    obs_flat     = obs_seq.reshape(T * N, O)     # [T*N,O]
    values_flat  = critic(obs_flat)              # [T*N]
    values       = values_flat.view(T, N)        # [T,N]

    with torch.no_grad():
        if dones[-1] > 0.5:
            next_value = torch.zeros((N,), device=device)    # [N]
        else:
            next_value = critic(last_obs)                    # [N]

    # ---------- Reward normalization (optional) ----------
    if normalize_rewards:
        r_mean = rewards.mean()
        r_std  = rewards.std(unbiased=False).clamp_min(1e-6)
        rewards = reward_scale * (rewards - r_mean) / r_std
    elif reward_scale != 1.0:
        rewards = reward_scale * rewards

    # ---------- Per-agent GAE ----------
    with torch.no_grad():
        adv = torch.zeros_like(values)          # [T,N]
        gae = torch.zeros((N,), device=device)  # [N]
        for t in reversed(range(T)):
            nonterm = (1.0 - dones[t])          # scalar
            v_next  = next_value if t == T - 1 else values[t + 1]  # [N]
            delta   = rewards[t] + gamma * nonterm * v_next - values[t]
            gae     = delta + gamma * gae_lambda * nonterm * gae
            adv[t]  = gae
        ret = adv + values                       # [T,N]

    # Advantage normalization (over all T*N), optional
    if normalize_adv:
        adv_flat = adv.reshape(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # ---------- Losses ----------
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = values
    v_target = ret.detach()
    if value_clip_eps and value_clip_eps > 0.0:
        v_clipped        = v_pred + (v_target - v_pred).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,    v_target, reduction='mean', beta=huber_delta)
        v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_target, reduction='mean', beta=huber_delta)
        value_loss       = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss       = torch.nn.functional.smooth_l1_loss(v_pred, v_target, reduction='mean', beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # ---------- Optimize ----------
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
        # (optional) some diagnostics:
        # "value_mean":  float(values.mean().item()),
        # "adv_mean":    float(adv.mean().item()),
    }

# ================= A2C update (MLP for actor, LSTM for critic) ====================
def ia2c_update_mlp_actor_lstm_critic(
    actor: nn.Module,                 # ActorMLP: local obs [O] -> logits [A]
    critic: nn.Module,                # CriticLSTM: local seq [T,N,O] -> values [T,N]
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,            # [T, N, O]
    act_seq: torch.Tensor,            # [T, N] (long)
    rew_seq: torch.Tensor,            # [T, N] (float)
    done_seq: torch.Tensor,           # [T]     (float; episode terminal flags)
    last_obs: torch.Tensor,           # [N, O]  (for bootstrap if not terminal)
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # stability knobs
    normalize_rewards: bool = True,
    reward_scale: float = 1.0,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,      # 0 to disable clipping
) -> dict:
    """
    IA2C update for a single rollout using an MLP actor (shared, decentralized)
    and an LSTM critic (shared, decentralized). Computes *per-agent* GAE on local values.
    """
    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # to device
    obs_seq  = obs_seq.to(device).float()     # [T,N,O]
    actions  = act_seq.to(device).long()      # [T,N]
    rewards  = rew_seq.to(device).float()     # [T,N]
    dones    = done_seq.to(device).float()    # [T]
    last_obs = last_obs.to(device).float()    # [N,O]

    # -------- Policy (MLP actor) --------
    obs_flat    = obs_seq.reshape(T * N, O)   # [T*N,O]
    logits_flat = actor(obs_flat)             # [T*N,A]
    A_dim       = logits_flat.size(-1)
    logits      = logits_flat.view(T, N, A_dim)  # [T,N,A]

    dist       = Categorical(logits=logits)
    logp_taken = dist.log_prob(actions)       # [T,N]
    entropy    = dist.entropy().mean()

    # -------- Value (LSTM critic) --------
    # CriticLSTM.forward accepts [T,N,O] and returns [1,T,N]
    values_bt, _ = critic(obs_seq)            # [1,T,N]
    values = values_bt[0]                     # [T,N]

    with torch.no_grad():
        if dones[-1] > 0.5:
            next_value = torch.zeros((N,), device=device)   # [N]
        else:
            last_vals_bt, _ = critic(last_obs)              # input [N,O] -> [1,1,N]
            next_value = last_vals_bt[0, -1]                # [N]

    # -------- Reward normalization / scaling --------
    if normalize_rewards:
        r_mean = rewards.mean()
        r_std  = rewards.std(unbiased=False).clamp_min(1e-6)
        rewards = reward_scale * (rewards - r_mean) / r_std
    elif reward_scale != 1.0:
        rewards = reward_scale * rewards

    # -------- Per-agent GAE --------
    with torch.no_grad():
        adv = torch.zeros_like(values)          # [T,N]
        gae = torch.zeros((N,), device=device)  # [N]
        for t in reversed(range(T)):
            nonterm = (1.0 - dones[t])          # scalar
            v_next  = next_value if t == T - 1 else values[t + 1]  # [N]
            delta   = rewards[t] + gamma * nonterm * v_next - values[t]
            gae     = delta + gamma * gae_lambda * nonterm * gae
            adv[t]  = gae
        ret = adv + values                       # [T,N]

    # Advantage normalization (over T*N), optional
    if normalize_adv:
        adv_flat = adv.reshape(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # -------- Losses --------
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = values
    v_target = ret.detach()
    if value_clip_eps and value_clip_eps > 0.0:
        v_clipped        = v_pred + (v_target - v_pred).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,    v_target, reduction='mean', beta=huber_delta)
        v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_target, reduction='mean', beta=huber_delta)
        value_loss       = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss       = torch.nn.functional.smooth_l1_loss(v_pred, v_target, reduction='mean', beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # -------- Optimize --------
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
    }

# =============================== A2C update (GNN) =================================
def ia2c_update_gnn_attn(
    actor: nn.Module,                      # ActorGNNAttn: [N,O] -> [N,A] (with graph)
    critic: nn.Module,                     # CriticGNNPerAgentAttn: [N,O] -> [N] (with graph)
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,                 # [T, N, O]
    act_seq: torch.Tensor,                 # [T, N]  (long)
    rew_seq: torch.Tensor,                 # [T, N]  (float)
    done_seq: torch.Tensor,                # [T]     (float 0/1)
    last_obs: torch.Tensor,                # [N, O]  (for bootstrap if not terminal)
    edge_index: torch.Tensor,              # [2, E]  (torch.long)
    edge_attr: Optional[torch.Tensor] = None,  # [E, D_e] or None
    gamma: float = 0.97,
    gae_lambda: float = 0.90,
    entropy_coef: float = 0.02,
    value_coef: float = 0.25,
    grad_clip: float = 1.0,
    # stabilizers
    normalize_rewards: bool = True,
    reward_scale: float = 1.0,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,           # set 0 to disable clipping
) -> dict:
    """
    IA2C with attention-GNN actor (per-agent policy) and per-agent attention-GNN critic.
    - Computes per-agent GAE on critic values V_i(o_i, neighbors)
    - No dropout anywhere
    """
    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    obs_seq  = obs_seq.to(device).float()     # [T,N,O]
    acts     = act_seq.to(device).long()      # [T,N]
    rewards  = rew_seq.to(device).float()     # [T,N]
    dones    = done_seq.to(device).float()    # [T]
    last_obs = last_obs.to(device).float()    # [N,O]

    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # -------- Forward through actor & critic per time-step (keeps code simple/readable) --------
    logits_list = []
    v_list      = []

    for t in range(T):
        X_t = obs_seq[t]                                      # [N,O]
        logits_t = actor(X_t, edge_index, edge_attr)          # [N,A]
        v_t      = critic(X_t, edge_index, edge_attr)         # [N]
        logits_list.append(logits_t)
        v_list.append(v_t)

    logits_seq = torch.stack(logits_list, dim=0)              # [T,N,A]
    v_seq      = torch.stack(v_list, dim=0)                   # [T,N]

    dist       = Categorical(logits=logits_seq)
    logp_taken = dist.log_prob(acts)                          # [T,N]
    entropy    = dist.entropy().mean()

    # Bootstrap per-agent values at the end
    with torch.no_grad():
        if dones[-1] > 0.5:
            next_value = torch.zeros((N,), device=device)
        else:
            next_value = critic(last_obs, edge_index, edge_attr)  # [N]

    # -------- Reward normalization (over T*N), optional --------
    r = rewards
    if normalize_rewards:
        r_mean = r.mean()
        r_std  = r.std(unbiased=False).clamp_min(1e-6)
        r = reward_scale * (r - r_mean) / r_std
    elif reward_scale != 1.0:
        r = reward_scale * r

    # -------- Per-agent GAE (vectorized over agents) --------
    # v_seq: [T,N], r: [T,N], dones: [T], next_value: [N]
    with torch.no_grad():
        adv = torch.zeros_like(v_seq)          # [T,N]
        gae = torch.zeros((N,), device=device) # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            v_next  = next_value if t == T - 1 else v_seq[t + 1]        # [N]
            delta   = r[t] + gamma * nonterm * v_next - v_seq[t]        # [N]
            gae     = delta + gamma * gae_lambda * nonterm * gae        # [N]
            adv[t]  = gae
        ret = adv + v_seq                                               # [T,N]

    # Advantage normalization (global over T*N), optional
    if normalize_adv:
        adv_flat = adv.reshape(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # -------- Losses --------
    policy_loss = -(logp_taken * adv.detach()).mean()

    # Value loss: Huber + optional value clipping (A2C-style), per-agent
    v_pred   = v_seq
    v_target = ret.detach()

    if value_clip_eps and value_clip_eps > 0.0:
        v_old = v_pred.detach()
        v_clipped = v_old + (v_pred - v_old).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,   v_target, reduction='mean', beta=huber_delta)
        v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_target, reduction='mean', beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_target, reduction='mean', beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # -------- Optimize --------
    optim_actor.zero_grad()
    optim_critic.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
    }

# =============================== A2C update (GNN+LSTM) =================================
def ia2c_update_gnn_lstm_attn(
    actor: nn.Module,                      # ActorGNNLSTMAttn
    critic: nn.Module,                     # CriticGNNPerAgentLSTMAttn
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,                 # [T,N,O]
    act_seq: torch.Tensor,                 # [T,N] (long)
    rew_seq: torch.Tensor,                 # [T,N] (float)
    done_seq: torch.Tensor,                # [T]   (float 0/1)
    last_obs: torch.Tensor,                # [N,O] (for bootstrap if not terminal)
    edge_index: torch.Tensor,              # [2,E] (long)
    edge_attr: Optional[torch.Tensor] = None,
    gamma: float = 0.97,
    gae_lambda: float = 0.90,
    entropy_coef: float = 0.02,
    value_coef: float = 0.25,
    grad_clip: float = 1.0,
    # stabilisers
    normalize_rewards: bool = True,
    reward_scale: float = 1.0,
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,           # set 0 to disable clipping
) -> Dict[str, float]:

    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    obs_seq  = obs_seq.to(device).float()      # [T,N,O]
    acts     = act_seq.to(device).long()       # [T,N]
    rewards  = rew_seq.to(device).float()      # [T,N]
    dones    = done_seq.to(device).float()     # [T]
    last_obs = last_obs.to(device).float()     # [N,O]

    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # -------- Forward pass with LSTM memory across time --------
    actor.train(); critic.train()
    logits_seq, _ = actor(obs_seq, edge_index, edge_attr, hidden=None)   # [T,N,A]
    v_seq,     _  = critic(obs_seq, edge_index, edge_attr, hidden=None)  # [T,N]

    dist       = Categorical(logits=logits_seq)
    logp_taken = dist.log_prob(acts)                                     # [T,N]
    entropy    = dist.entropy().mean()

    # -------- Bootstrap using critic at last_obs (respect terminal) --------
    with torch.no_grad():
        if dones[-1] > 0.5:
            next_value = torch.zeros((N,), device=device)
        else:
            v_last, _ = critic(last_obs, edge_index, edge_attr, hidden=None)  # [N]
            next_value = v_last

    # -------- Reward normalisation (global over T*N), optional --------
    r = rewards
    if normalize_rewards:
        r_mean = r.mean()
        r_std  = r.std(unbiased=False).clamp_min(1e-6)
        r = reward_scale * (r - r_mean) / r_std
    elif reward_scale != 1.0:
        r = reward_scale * r

    # -------- Inlined per-agent GAE and returns --------
    with torch.no_grad():
        adv = torch.zeros_like(v_seq)                      # [T,N]
        gae = torch.zeros((N,), device=device)             # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]                       # []
            v_next  = next_value if t == T - 1 else v_seq[t + 1]  # [N]
            delta   = r[t] + gamma * nonterm * v_next - v_seq[t]  # [N]
            gae     = delta + gamma * gae_lambda * nonterm * gae  # [N]
            adv[t]  = gae
        ret = adv + v_seq                                   # [T,N]

    # Advantage normalisation (global)
    if normalize_adv:
        adv_flat = adv.view(-1)
        adv = (adv - adv_flat.mean()) / (adv_flat.std(unbiased=False).clamp_min(1e-6))

    # -------- Losses --------
    policy_loss = -(logp_taken * adv.detach()).mean()

    if value_clip_eps and value_clip_eps > 0.0:
        v_old = v_seq.detach()
        v_clipped = v_old + (v_seq - v_old).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = nn.functional.smooth_l1_loss(v_seq,     ret.detach(), reduction='mean', beta=huber_delta)
        v_loss_clipped   = nn.functional.smooth_l1_loss(v_clipped, ret.detach(), reduction='mean', beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = nn.functional.smooth_l1_loss(v_seq, ret.detach(), reduction='mean', beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # -------- Optimise --------
    optim_actor.zero_grad()
    optim_critic.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
    nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
    }

# ============ MA2C-PA updater (MLP, vectorized GAE over agents)=================
def ma2c_pa_update_mlp(
    actor: nn.Module,                  # ActorMLP: [O] -> logits [A]
    critic_pa: nn.Module,              # CriticPAMLP: [N*O] -> values [N]
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,             # [T, N, O]  (local)
    act_seq: torch.Tensor,             # [T, N]     (long)
    rew_seq: torch.Tensor,             # [T, N]     (per-agent rewards)
    done_seq: torch.Tensor,            # [T]        (float; 1 if terminal at t)
    last_obs: torch.Tensor,            # [N, O]
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # Reward handling
    advantage_mode: str = "per_agent",         # {"per_agent","team"}
    team_reward_reduce: str = "mean",          # {"mean","sum"} (used when advantage_mode="team")
    normalize_rewards: str = "per_agent",      # {"off","per_agent","global"}
    reward_scale: float = 1.0,                 # applied only if normalize_rewards != "off"
    # Advantage/value stabilizers
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,
) -> dict:
    """
    MA2C-PA: shared actor; centralized critic outputs a value per agent (N heads).
    - If advantage_mode=="per_agent": GAE uses each agent's own reward r_i and V_i(s).
    - If advantage_mode=="team": build r_team (mean/sum) and broadcast it to all agents,
      but still use per-agent baselines V_i(s) (lower variance than scalar baseline).
    - Rewards can be normalized per-agent across time or globally across T*N.
    """
    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # ---- To device / shapes ----
    obs_seq  = obs_seq.to(device).float()      # [T,N,O]
    acts     = act_seq.to(device).long()       # [T,N]
    dones    = done_seq.to(device).float()     # [T]
    rews     = rew_seq.to(device).float()      # [T,N]
    last_obs = last_obs.to(device).float()     # [N,O]

    # ---- Actor forward on local obs (for log-probs) ----
    obs_flat    = obs_seq.reshape(T * N, O)            # [T*N,O]
    logits_flat = actor(obs_flat)                      # [T*N,A]
    A_dim       = logits_flat.size(-1)
    logits      = logits_flat.view(T, N, A_dim)        # [T,N,A]
    dist        = Categorical(logits=logits)
    logp_taken  = dist.log_prob(acts)                  # [T,N]
    entropy     = dist.entropy().mean()

    # ---- Centralized per-agent values on joint obs ----
    joint_seq = obs_seq.reshape(T, N * O)              # [T, N*O]
    v_seq     = critic_pa(joint_seq)                   # [T, N]
    with torch.no_grad():
        v_last = critic_pa(last_obs.reshape(1, -1))[0] # [N]
        # If the last step was terminal due to time-limit we already have done_seq[-1]=1 in the trainer.
        # Still, for generality:
        if dones[-1] > 0.5:
            v_last = torch.zeros_like(v_last)

    # ---- Choose reward matrix used for GAE ----
    if advantage_mode == "per_agent":
        r_mat = rews.clone()                           # [T,N]
    elif advantage_mode == "team":
        if team_reward_reduce == "mean":
            r_team = rews.mean(dim=-1, keepdim=True)   # [T,1]
        elif team_reward_reduce == "sum":
            r_team = rews.sum(dim=-1, keepdim=True)    # [T,1]
        else:
            raise ValueError("team_reward_reduce must be 'mean' or 'sum'")
        r_mat = r_team.expand(T, N)                    # [T,N]
    else:
        raise ValueError("advantage_mode must be 'per_agent' or 'team'")

    # ---- Reward normalization (optional) ----
    if normalize_rewards != "off":
        if normalize_rewards == "per_agent":
            mean = r_mat.mean(dim=0, keepdim=True)                     # [1,N]
            std  = r_mat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        elif normalize_rewards == "global":
            mean = r_mat.mean()
            std  = r_mat.std(unbiased=False).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        else:
            raise ValueError("normalize_rewards must be {'off','per_agent','global'}")

    # ---- Vectorized GAE over agents ----
    with torch.no_grad():
        adv = torch.zeros_like(v_seq)                  # [T,N]
        gae = torch.zeros((N,), device=device)         # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]                  # scalar
            v_next  = v_last if t == T - 1 else v_seq[t + 1]  # [N]
            delta   = r_mat[t] + gamma * nonterm * v_next - v_seq[t]  # [N]
            gae     = delta + gamma * gae_lambda * nonterm * gae      # [N]
            adv[t]  = gae                                             # [N]
        ret = adv + v_seq                                             # [T,N]

    # ---- Advantage normalization (global over T*N), optional ----
    if normalize_adv:
        flat = adv.reshape(-1)
        adv = (adv - flat.mean()) / (flat.std(unbiased=False).clamp_min(1e-6))

    # ---- Losses ----
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = v_seq
    v_target = ret.detach()

    if value_clip_eps and value_clip_eps > 0.0:
        v_clipped = v_pred + (v_target - v_pred).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = nn.functional.smooth_l1_loss(
            v_pred, v_target, reduction="mean", beta=huber_delta
        )
        v_loss_clipped   = nn.functional.smooth_l1_loss(
            v_clipped, v_target, reduction="mean", beta=huber_delta
        )
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = nn.functional.smooth_l1_loss(
            v_pred, v_target, reduction="mean", beta=huber_delta
        )

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # ---- Optimize ----
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
    nn.utils.clip_grad_norm_(critic_pa.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
        "adv_mean":    float(adv.mean().item()),
        "v_mean":      float(v_seq.mean().item()),
    }

# ============================ MA2C-PA updater (LSTM)=======================
def ma2c_pa_update_lstm(
    actor: nn.Module,                  # ActorLSTM: [T,N,O] -> logits [1,T,N,A]
    critic_pa: nn.Module,              # CriticPALSTM: [T,N,O] -> values [1,T,N]
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,             # [T, N, O]  (local)
    act_seq: torch.Tensor,             # [T, N]     (long)
    rew_seq: torch.Tensor,             # [T, N]     (per-agent rewards)
    done_seq: torch.Tensor,            # [T]        (float; 1 if terminal at t)
    last_obs: torch.Tensor,            # [N, O]
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # Reward handling
    advantage_mode: str = "per_agent",         # {"per_agent","team"}
    team_reward_reduce: str = "mean",          # {"mean","sum"} (used when advantage_mode="team")
    normalize_rewards: str = "per_agent",      # {"off","per_agent","global"}
    reward_scale: float = 1.0,
    # Advantage/value stabilisers
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,
) -> Dict[str, float]:

    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # ---- To device ----
    obs_seq  = obs_seq.to(device).float()      # [T,N,O]
    acts     = act_seq.to(device).long()       # [T,N]
    dones    = done_seq.to(device).float()     # [T]
    rews     = rew_seq.to(device).float()      # [T,N]
    last_obs = last_obs.to(device).float()     # [N,O]

    # ---- Actor forward on local obs (sequence) ----
    logits_btna, _ = actor(obs_seq, hidden=None)          # [1,T,N,A]
    logits_tna = logits_btna[0]                           # [T,N,A]
    dist       = Categorical(logits=logits_tna)
    logp_taken = dist.log_prob(acts)                      # [T,N]
    entropy    = dist.entropy().mean()

    # ---- Centralised per-agent values on joint obs (sequence + last) ----
    v_btn, _ = critic_pa(obs_seq, hidden=None)            # [1,T,N]
    v_seq = v_btn[0]                                      # [T,N]
    with torch.no_grad():
        v_last, _ = critic_pa.step(last_obs, hidden=None) # [N]
        if dones[-1] > 0.5:
            v_last = torch.zeros_like(v_last)

    # ---- Choose reward matrix for GAE ----
    if advantage_mode == "per_agent":
        r_mat = rews.clone()                               # [T,N]
    elif advantage_mode == "team":
        if team_reward_reduce == "mean":
            r_team = rews.mean(dim=-1, keepdim=True)       # [T,1]
        elif team_reward_reduce == "sum":
            r_team = rews.sum(dim=-1, keepdim=True)        # [T,1]
        else:
            raise ValueError("team_reward_reduce must be 'mean' or 'sum'")
        r_mat = r_team.expand(T, N)                        # [T,N]
    else:
        raise ValueError("advantage_mode must be 'per_agent' or 'team'")

    # ---- Reward normalisation (optional) ----
    if normalize_rewards != "off":
        if normalize_rewards == "per_agent":
            mean = r_mat.mean(dim=0, keepdim=True)         # [1,N]
            std  = r_mat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        elif normalize_rewards == "global":
            mean = r_mat.mean()
            std  = r_mat.std(unbiased=False).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        else:
            raise ValueError("normalize_rewards ∈ {'off','per_agent','global'}")

    # ---- Vectorised per-agent GAE over time ----
    with torch.no_grad():
        adv = torch.zeros_like(v_seq)                      # [T,N]
        gae = torch.zeros((N,), device=device)             # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            v_next  = v_last if t == T - 1 else v_seq[t + 1]
            delta   = r_mat[t] + gamma * nonterm * v_next - v_seq[t]
            gae     = delta + gamma * gae_lambda * nonterm * gae
            adv[t]  = gae
        ret = adv + v_seq                                  # [T,N]

    # ---- Advantage normalisation (global over T*N) ----
    if normalize_adv:
        flat = adv.reshape(-1)
        adv = (adv - flat.mean()) / (flat.std(unbiased=False).clamp_min(1e-6))

    # ---- Losses ----
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = v_seq
    v_target = ret.detach()

    if value_clip_eps and value_clip_eps > 0.0:
        v_old = v_pred.detach()
        v_clipped = v_old + (v_pred - v_old).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = nn.functional.smooth_l1_loss(v_pred,     v_target, reduction="mean", beta=huber_delta)
        v_loss_clipped   = nn.functional.smooth_l1_loss(v_clipped,  v_target, reduction="mean", beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = nn.functional.smooth_l1_loss(v_pred, v_target, reduction="mean", beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # ---- Optimise ----
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),     grad_clip)
    nn.utils.clip_grad_norm_(critic_pa.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
        "adv_mean":    float(adv.mean().item()),
        "v_mean":      float(v_seq.mean().item()),
    }

# ============================ MA2C-PA updater (GNN)=======================
def ma2c_pa_update_gnn_attn(
    actor: nn.Module,                      # ActorGNNAttn: (X_t, edge) -> [N,A]
    critic_pa: nn.Module,                  # CriticGNNPerAgentAttn: (X_t, edge) -> [N]
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,                 # [T, N, O]
    act_seq: torch.Tensor,                 # [T, N]  (long)
    rew_seq: torch.Tensor,                 # [T, N]  (float)
    done_seq: torch.Tensor,                # [T]     (float 0/1)
    last_obs: torch.Tensor,                # [N, O]
    edge_index: torch.Tensor,              # [2, E] (long)
    edge_attr: Optional[torch.Tensor] = None,  # [E, D_e] or None
    # A2C/GAE
    gamma: float = 0.97,
    gae_lambda: float = 0.90,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # Reward handling
    advantage_mode: str = "per_agent",         # {"per_agent","team"}
    team_reward_reduce: str = "mean",          # {"mean","sum"} (only when advantage_mode="team")
    normalize_rewards: str = "per_agent",      # {"off","per_agent","global"}
    reward_scale: float = 1.0,
    # Advantage/value stabilisers
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,
) -> Dict[str, float]:

    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # ---- To device / types ----
    obs_seq  = obs_seq.to(device).float()    # [T,N,O]
    acts     = act_seq.to(device).long()     # [T,N]
    dones    = done_seq.to(device).float()   # [T]
    rews_in  = rew_seq.to(device).float()    # [T,N]
    last_obs = last_obs.to(device).float()   # [N,O]

    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # ---- Forward through actor/critic per time step ----
    logits_list, v_list = [], []
    for t in range(T):
        X_t = obs_seq[t]                                       # [N,O]
        logits_t = actor(X_t, edge_index, edge_attr)           # [N,A]
        v_t      = critic_pa(X_t, edge_index, edge_attr)       # [N]
        logits_list.append(logits_t)
        v_list.append(v_t)

    logits = torch.stack(logits_list, dim=0)                   # [T,N,A]
    v_seq  = torch.stack(v_list,   dim=0)                      # [T,N]

    dist       = Categorical(logits=logits)
    logp_taken = dist.log_prob(acts)                           # [T,N]
    entropy    = dist.entropy().mean()

    # ---- Bootstrap value at last_obs unless terminal ----
    with torch.no_grad():
        if dones[-1] > 0.5:
            v_last = torch.zeros((N,), device=device)
        else:
            v_last = critic_pa(last_obs, edge_index, edge_attr)  # [N]

    # ---- Inlined reward-matrix construction (no helper) ----
    if advantage_mode == "per_agent":
        r_mat = rews_in.clone()                                 # [T,N]
    elif advantage_mode == "team":
        if team_reward_reduce == "mean":
            r_team = rews_in.mean(dim=-1, keepdim=True)         # [T,1]
        elif team_reward_reduce == "sum":
            r_team = rews_in.sum(dim=-1, keepdim=True)          # [T,1]
        else:
            raise ValueError("team_reward_reduce must be {'mean','sum'}")
        r_mat = r_team.expand(T, N)                              # [T,N]
    else:
        raise ValueError("advantage_mode must be {'per_agent','team'}")

    # ---- Reward normalisation (optional) ----
    if normalize_rewards != "off":
        if normalize_rewards == "per_agent":
            mean = r_mat.mean(dim=0, keepdim=True)                                   # [1,N]
            std  = r_mat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        elif normalize_rewards == "global":
            mean = r_mat.mean()
            std  = r_mat.std(unbiased=False).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        else:
            raise ValueError("normalize_rewards must be {'off','per_agent','global'}")

    # ---- Vectorised per-agent GAE over time ----
    with torch.no_grad():
        adv = torch.zeros_like(v_seq)                        # [T,N]
        gae = torch.zeros((N,), device=device)               # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            v_next  = v_last if t == T - 1 else v_seq[t + 1]
            delta   = r_mat[t] + gamma * nonterm * v_next - v_seq[t]
            gae     = delta + gamma * gae_lambda * nonterm * gae
            adv[t]  = gae
        ret = adv + v_seq                                    # [T,N]

    # ---- Advantage normalisation (global over T*N) ----
    if normalize_adv:
        flat = adv.reshape(-1)
        adv = (adv - flat.mean()) / (flat.std(unbiased=False).clamp_min(1e-6))

    # ---- Losses ----
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = v_seq
    v_target = ret.detach()

    if value_clip_eps and value_clip_eps > 0.0:
        v_old = v_pred.detach()
        v_clipped = v_old + (v_pred - v_old).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = nn.functional.smooth_l1_loss(v_pred,     v_target, reduction="mean", beta=huber_delta)
        v_loss_clipped   = nn.functional.smooth_l1_loss(v_clipped,  v_target, reduction="mean", beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = nn.functional.smooth_l1_loss(v_pred, v_target, reduction="mean", beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # ---- Optimise ----
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),     grad_clip)
    nn.utils.clip_grad_norm_(critic_pa.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
        "adv_mean":    float(adv.mean().item()),
        "v_mean":      float(v_seq.mean().item()),
    }

# ============================ MA2C-PA updater (GNN+LSTM)=======================
def ma2c_pa_update_gnn_lstm_attn(
    actor: nn.Module,                      # ActorGNNLSTMAttn: ([T,N,O], edge) -> ([T,N,A], hidden)
    critic_pa: nn.Module,                  # CriticGNNPerAgentLSTMAttn: ([T,N,O], edge) -> ([T,N], hidden)
    optim_actor: torch.optim.Optimizer,
    optim_critic: torch.optim.Optimizer,
    obs_seq: torch.Tensor,                 # [T, N, O]
    act_seq: torch.Tensor,                 # [T, N]  (long)
    rew_seq: torch.Tensor,                 # [T, N]  (float)
    done_seq: torch.Tensor,                # [T]     (float 0/1)
    last_obs: torch.Tensor,                # [N, O]
    edge_index: torch.Tensor,              # [2, E]  (long)
    edge_attr: Optional[torch.Tensor] = None,  # [E, D_e] or None
    # A2C/GAE
    gamma: float = 0.97,
    gae_lambda: float = 0.90,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    grad_clip: float = 1.0,
    # Reward handling
    advantage_mode: str = "per_agent",         # {"per_agent","team"}
    team_reward_reduce: str = "mean",          # {"mean","sum"} (when advantage_mode == "team")
    normalize_rewards: str = "per_agent",      # {"off","per_agent","global"}
    reward_scale: float = 1.0,
    # Advantage/value stabilisers
    normalize_adv: bool = True,
    huber_delta: float = 1.0,
    value_clip_eps: float = 0.2,
) -> Dict[str, float]:

    device = next(actor.parameters()).device
    T, N, O = obs_seq.shape

    # ---- To device ----
    obs_seq  = obs_seq.to(device).float()     # [T,N,O]
    acts     = act_seq.to(device).long()      # [T,N]
    dones    = done_seq.to(device).float()    # [T]
    rews_in  = rew_seq.to(device).float()     # [T,N]
    last_obs = last_obs.to(device).float()    # [N,O]

    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # ---- Sequence forward (fresh hidden) ----
    logits_tna, _ = actor(obs_seq, edge_index, edge_attr, hidden=None)  # [T,N,A]
    v_tn, _       = critic_pa(obs_seq, edge_index, edge_attr, hidden=None)  # [T,N]

    dist       = Categorical(logits=logits_tna)
    logp_taken = dist.log_prob(acts)                 # [T,N]
    entropy    = dist.entropy().mean()

    # ---- Bootstrap at last_obs unless terminal ----
    with torch.no_grad():
        if dones[-1] > 0.5:
            v_last = torch.zeros((N,), device=device)
        else:
            v_last, _ = critic_pa.step(last_obs, edge_index, edge_attr, hidden=None)  # [N]

    # ---- Build reward matrix (inlined helper) ----
    if advantage_mode == "per_agent":
        r_mat = rews_in.clone()                                  # [T,N]
    elif advantage_mode == "team":
        if team_reward_reduce == "mean":
            r_team = rews_in.mean(dim=-1, keepdim=True)          # [T,1]
        elif team_reward_reduce == "sum":
            r_team = rews_in.sum(dim=-1, keepdim=True)           # [T,1]
        else:
            raise ValueError("team_reward_reduce must be {'mean','sum'}")
        r_mat = r_team.expand(T, N)                               # [T,N]
    else:
        raise ValueError("advantage_mode must be {'per_agent','team'}")

    # ---- Reward normalisation ----
    if normalize_rewards != "off":
        if normalize_rewards == "per_agent":
            mean = r_mat.mean(dim=0, keepdim=True)                                   # [1,N]
            std  = r_mat.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        elif normalize_rewards == "global":
            mean = r_mat.mean()
            std  = r_mat.std(unbiased=False).clamp_min(1e-6)
            r_mat = reward_scale * (r_mat - mean) / std
        else:
            raise ValueError("normalize_rewards must be {'off','per_agent','global'}")

    # ---- Vectorised per-agent GAE over time ----
    with torch.no_grad():
        adv = torch.zeros_like(v_tn)                         # [T,N]
        gae = torch.zeros((N,), device=device)               # [N]
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            v_next  = v_last if t == T - 1 else v_tn[t + 1]
            delta   = r_mat[t] + gamma * nonterm * v_next - v_tn[t]
            gae     = delta + gamma * gae_lambda * nonterm * gae
            adv[t]  = gae
        ret = adv + v_tn                                     # [T,N]

    # ---- Advantage normalisation ----
    if normalize_adv:
        flat = adv.reshape(-1)
        adv = (adv - flat.mean()) / (flat.std(unbiased=False).clamp_min(1e-6))

    # ---- Losses ----
    policy_loss = -(logp_taken * adv.detach()).mean()

    v_pred   = v_tn
    v_target = ret.detach()

    if value_clip_eps and value_clip_eps > 0.0:
        v_old = v_pred.detach()
        v_clipped = v_old + (v_pred - v_old).clamp(-value_clip_eps, value_clip_eps)
        v_loss_unclipped = nn.functional.smooth_l1_loss(v_pred,     v_target, reduction="mean", beta=huber_delta)
        v_loss_clipped   = nn.functional.smooth_l1_loss(v_clipped,  v_target, reduction="mean", beta=huber_delta)
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
    else:
        value_loss = nn.functional.smooth_l1_loss(v_pred, v_target, reduction="mean", beta=huber_delta)

    total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

    # ---- Optimise ----
    optim_actor.zero_grad(set_to_none=True)
    optim_critic.zero_grad(set_to_none=True)
    total_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),     grad_clip)
    nn.utils.clip_grad_norm_(critic_pa.parameters(), grad_clip)
    optim_actor.step()
    optim_critic.step()

    return {
        "loss_total":  float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_value":  float(value_loss.item()),
        "entropy":     float(entropy.item()),
        "adv_mean":    float(adv.mean().item()),
        "v_mean":      float(v_tn.mean().item()),
    }


# # =============================== MA2C update (MLP) ================================
# def ma2c_update_mlp(
#     actor: nn.Module,                 # ActorMLP: local obs [O] -> logits [A]
#     critic: nn.Module,                # CriticMLP: joint obs [N*O] -> scalar value
#     optim_actor: torch.optim.Optimizer,
#     optim_critic: torch.optim.Optimizer,
#     obs_seq: torch.Tensor,            # [T,N,O]
#     act_seq: torch.Tensor,            # [T,N] (long)
#     rew_seq: torch.Tensor,            # [T,N] (float; per-agent rewards)
#     done_seq: torch.Tensor,           # [T]   (float; episode terminal flags)
#     last_obs: torch.Tensor,           # [N,O] (for bootstrap if not terminal)
#     gamma: float = 0.99,
#     gae_lambda: float = 0.95,
#     entropy_coef: float = 0.01,
#     value_coef: float = 0.5,
#     grad_clip: float = 1.0,
#     # stability knobs (match IA2C style)
#     normalize_rewards: bool = True,
#     reward_scale: float = 1.0,
#     normalize_adv: bool = True,
#     huber_delta: float = 1.0,
#     value_clip_eps: float = 0.2,      # set 0 to disable clipping
#     # team reward aggregation
#     team_reward_reduce: str = "mean", # "mean" | "sum"
# ) -> dict:
#     """
#     One MA2C update with GAE(λ) for a single rollout (length T).
#     - Shared decentralized actor uses local obs; centralized critic uses joint obs.
#     - Per-step team reward is aggregated from per-agent rewards (mean/sum).
#     - Scalar GAE is computed on the team signal and broadcast to all agents.
#     - Adds: reward normalization, advantage normalization, Huber value loss,
#       and optional value clipping (A2C-style), mirroring your IA2C updater.
#     """
#     device = next(actor.parameters()).device
#     T, N, O = obs_seq.shape
#
#     # Tensors to device
#     obs_seq  = obs_seq.to(device).float()
#     actions  = act_seq.to(device).long()
#     dones    = done_seq.to(device).float()
#     last_obs = last_obs.to(device).float()
#     rewards  = rew_seq.to(device).float()  # [T,N]
#
#     # -------- Actor forward on local obs --------
#     obs_flat    = obs_seq.reshape(T * N, O)          # [T*N, O]
#     logits_flat = actor(obs_flat)                    # [T*N, A]
#     A_dim       = logits_flat.size(-1)
#     logits      = logits_flat.view(T, N, A_dim)      # [T,N,A]
#
#     dist        = Categorical(logits=logits)         # over [T,N]
#     logp_taken  = dist.log_prob(actions)             # [T,N]
#     entropy     = dist.entropy().mean()
#
#     # -------- Centralized critic on JOINT obs --------
#     joint_seq = obs_seq.reshape(T, N * O)            # [T, N*O]
#     v_seq     = critic(joint_seq).reshape(-1)        # [T]
#     with torch.no_grad():
#         if dones[-1] > 0.5:
#             next_value = torch.zeros((), device=device)
#         else:
#             next_value = critic(last_obs.reshape(1, -1)).reshape(-1)[0]  # []
#
#     # -------- Team reward aggregation --------
#     if team_reward_reduce == "mean":
#         r_team = rewards.mean(dim=-1)                # [T]
#     elif team_reward_reduce == "sum":
#         r_team = rewards.sum(dim=-1)                 # [T]
#     else:
#         raise ValueError("team_reward_reduce must be 'mean' or 'sum'")
#
#     # print(f"[update] r_team mean={r_team.mean().item():.3f} std={r_team.std(unbiased=False).item():.3f}")
#
#     # Reward normalization (over T), optional — matches IA2C style (includes reward_scale here)
#     if normalize_rewards:
#         r_mean = r_team.mean()
#         r_std  = r_team.std(unbiased=False).clamp_min(1e-6)
#         r_team = reward_scale * (r_team - r_mean) / r_std
#
#     # -------- Scalar GAE on team signal --------
#     with torch.no_grad():
#         adv = torch.zeros_like(v_seq)                # [T]
#         gae = torch.zeros((), device=device)
#         for t in reversed(range(T)):
#             nonterm = 1.0 - dones[t]
#             v_next  = next_value if t == T - 1 else v_seq[t + 1]
#             delta   = r_team[t] + gamma * nonterm * v_next - v_seq[t]
#             gae     = delta + gamma * gae_lambda * nonterm * gae
#             adv[t]  = gae
#         ret = adv + v_seq                            # [T]
#
#     # Advantage normalization (over T), optional
#     if normalize_adv:
#         adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp_min(1e-6))
#
#     # -------- Losses --------
#     # Broadcast scalar adv to all agents at each t
#     adv_broadcast = adv.unsqueeze(-1).expand(T, N)   # [T,N]
#     policy_loss   = -(logp_taken * adv_broadcast.detach()).mean()
#
#     # Value loss: Huber + optional value clipping (A2C-style)
#     v_pred   = v_seq
#     v_target = ret.detach()
#     if value_clip_eps and value_clip_eps > 0.0:
#         v_clipped = v_pred + (v_target - v_pred).clamp(-value_clip_eps, value_clip_eps)
#         v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,   v_target, reduction='mean', beta=huber_delta)
#         v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_target, reduction='mean', beta=huber_delta)
#         value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
#     else:
#         value_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_target, reduction='mean', beta=huber_delta)
#
#     total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss
#
#     # -------- Optimize --------
#     optim_actor.zero_grad()
#     optim_critic.zero_grad()
#     total_loss.backward()
#     nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
#     nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
#     optim_actor.step()
#     optim_critic.step()
#
#     return {
#         "loss_total":  float(total_loss.item()),
#         "loss_policy": float(policy_loss.item()),
#         "loss_value":  float(value_loss.item()),
#         "entropy":     float(entropy.item()),
#     }

# # =============================== MA2C update (LSTM) ================================
# def ma2c_update_lstm(
#     actor: nn.Module,                 # ActorLSTM: local obs [T,N,O] -> logits [T,N,A]
#     critic: nn.Module,                # CriticMLP: joint obs [N*O]  -> scalar V
#     optim_actor: torch.optim.Optimizer,
#     optim_critic: torch.optim.Optimizer,
#     obs_seq: torch.Tensor,            # [T, N, O]  local observations
#     act_seq: torch.Tensor,            # [T, N]     actions (long)
#     rew_seq: torch.Tensor,            # [T, N]     per-agent rewards
#     done_seq: torch.Tensor,           # [T]        1.0 if episode terminal at step t, else 0.0
#     last_obs: torch.Tensor,           # [N, O]     final joint obs for bootstrap (critic)
#     gamma: float = 0.97,
#     gae_lambda: float = 0.90,
#     entropy_coef: float = 0.02,
#     value_coef: float = 0.25,
#     grad_clip: float = 0.5,
#     normalize_adv: bool = True,
#     normalize_rewards: bool = True,
#     reward_scale: float = 0.1,
#     huber_delta: float = 1.0,
#     value_clip_eps: float = 0.2,
#     team_reward_reduce: str = "mean",  # "mean" | "sum"
# ) -> dict:
#     """
#     MA2C update (centralized critic, decentralized LSTM actor) for one rollout.
#     - Team reward can be MEAN or SUM over agents (configurable).
#     - Rewards/advantages can be normalized.
#     - Value loss uses Huber + optional PPO-style value clipping.
#     """
#     device = next(actor.parameters()).device
#     obs_seq  = obs_seq.to(device)
#     act_seq  = act_seq.to(device).long()
#     rew_seq  = rew_seq.to(device).float()
#     done_seq = done_seq.to(device).float()
#     last_obs = last_obs.to(device).float()
#
#     T, N, O = obs_seq.shape
#
#     # ---------- Centralized critic on JOINT obs ----------
#     joint_seq = obs_seq.reshape(T, N * O)                # [T, N*O]
#     v_seq = critic(joint_seq).reshape(-1)                # [T]
#     with torch.no_grad():
#         # Bootstrap with last joint obs unless terminal
#         last_val = critic(last_obs.reshape(1, -1)).reshape(-1)[0]
#         last_val = torch.where(done_seq[-1] > 0.5, torch.zeros_like(last_val), last_val)
#
#     # ---------- Team reward aggregation (MEAN or SUM) ----------
#     if team_reward_reduce == "mean":
#         r_team = rew_seq.mean(dim=-1)                    # [T]
#     elif team_reward_reduce == "sum":
#         r_team = rew_seq.sum(dim=-1)                     # [T]
#     else:
#         raise ValueError("team_reward_reduce must be 'mean' or 'sum'")
#
#     # Optional reward normalization + global scaling
#     if normalize_rewards:
#         mean_r = r_team.mean()
#         std_r  = r_team.std(unbiased=False).clamp_min(1e-6)
#         r_team = (r_team - mean_r) / std_r
#     if reward_scale != 1.0:
#         r_team = r_team * reward_scale
#
#     # ---------- GAE on centralized value ----------
#     with torch.no_grad():
#         adv = torch.zeros_like(v_seq)                    # [T]
#         gae = torch.zeros((), device=device)
#         for t in reversed(range(T)):
#             nonterm = 1.0 - done_seq[t]
#             v_next  = last_val if t == T - 1 else v_seq[t + 1]
#             delta   = r_team[t] + gamma * nonterm * v_next - v_seq[t]
#             gae     = delta + gamma * gae_lambda * nonterm * gae
#             adv[t]  = gae
#         ret = adv + v_seq                                # [T]
#
#     # Advantage normalization over time
#     if normalize_adv:
#         adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp_min(1e-6))
#
#     # ---------- Policy loss via LSTM actor over the sequence ----------
#     # ActorLSTM forward accepts [T,N,O] and returns [1,T,N,A]
#     logits_seq, _ = actor(obs_seq)                       # [1,T,N,A]
#     logits_seq = logits_seq[0]                           # [T,N,A]
#
#     dist      = Categorical(logits=logits_seq)           # batched over [T,N]
#     logp      = dist.log_prob(act_seq)                   # [T,N]
#     entropy   = dist.entropy().mean()
#
#     adv_expanded = adv.unsqueeze(-1).expand(T, N)        # [T,N]
#     policy_loss  = -(logp * adv_expanded.detach()).mean()
#
#     # ---------- Value loss (Huber + value clipping) ----------
#     with torch.no_grad():
#         old_values = v_seq.clone()                       # treat current eval as "old" target for clipping baseline
#     values_clipped = old_values + (v_seq - old_values).clamp(-value_clip_eps, value_clip_eps)
#
#     v_loss_unclipped = torch.nn.functional.smooth_l1_loss(
#         v_seq, ret.detach(), beta=huber_delta, reduction='mean'
#     )
#     v_loss_clipped = torch.nn.functional.smooth_l1_loss(
#         values_clipped, ret.detach(), beta=huber_delta, reduction='mean'
#     )
#     value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
#
#     # ---------- Optimize ----------
#     total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss
#     optim_actor.zero_grad(set_to_none=True)
#     optim_critic.zero_grad(set_to_none=True)
#     total_loss.backward()
#     nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
#     nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
#     optim_actor.step()
#     optim_critic.step()
#
#     return {
#         "loss_total":  float(total_loss.item()),
#         "loss_policy": float(policy_loss.item()),
#         "loss_value":  float(value_loss.item()),
#         "entropy":     float(entropy.item()),
#     }


# # ======================= neighbor-critic MA2C update (MLP) =========================
# def ma2c_update_neighbor_mlp(
#     actor: nn.Module,                 # ActorMLP: local obs [O] -> logits [A]
#     critic: nn.Module,                # CriticMLP: neighbor-aug obs [C] -> value scalar
#     optim_actor: torch.optim.Optimizer,
#     optim_critic: torch.optim.Optimizer,
#     obs_seq: torch.Tensor,            # [T, N, O]  local observations (for actor)
#     critic_obs_seq: torch.Tensor,     # [T, N, C]  neighbor-aug observations (for critic)
#     act_seq: torch.Tensor,            # [T, N]     actions (long)
#     rew_seq: torch.Tensor,            # [T, N]     per-agent rewards
#     done_seq: torch.Tensor,           # [T]        1.0 if episode terminal at step t
#     last_obs: torch.Tensor,           # [N, O]     final local obs (not used by critic)
#     last_critic_obs: torch.Tensor,    # [N, C]     final neighbor-aug obs for bootstrap
#     gamma: float = 0.99,
#     gae_lambda: float = 0.95,
#     entropy_coef: float = 0.01,
#     value_coef: float = 0.5,
#     grad_clip: float = 1.0,
#     normalize_adv: bool = True,
#     # Stabilization knobs (match trainer call)
#     normalize_rewards: bool = True,
#     reward_scale: float = 1.0,
#     huber_delta: float = 1.0,
#     value_clip_eps: float = 0.2,      # set 0 to disable clipping
#     # How to derive team signals
#     team_value_reduce: str = "mean",  # "mean" | "sum"
#     team_reward_reduce: str = "mean",  # "sum" | "mean"
# ) -> dict:
#     """
#     Neighbor-critic MA2C with inlined team GAE, optional reward normalization/scaling,
#     advantage normalization, and Huber + value clipping for the critic.
#     """
#     device = next(actor.parameters()).device
#
#     obs_seq         = obs_seq.to(device).float()              # [T,N,O]
#     critic_obs_seq  = critic_obs_seq.to(device).float()       # [T,N,C]
#     act_seq         = act_seq.to(device).long()               # [T,N]
#     rew_seq         = rew_seq.to(device).float()              # [T,N]
#     done_seq        = done_seq.to(device).float()             # [T]
#     last_critic_obs = last_critic_obs.to(device).float()      # [N,C]
#
#     T, N, O = obs_seq.shape
#     C = critic_obs_seq.size(-1)
#
#     # -------- Critic: per-agent values on neighbor-aug inputs --------
#     crit_flat = critic_obs_seq.reshape(T * N, C)              # [T*N,C]
#     v_flat    = critic(crit_flat)                             # [T*N] or [T*N,1]
#     v_agents  = v_flat.reshape(T, N)                          # [T,N]
#
#     last_v_agents = critic(last_critic_obs).reshape(-1)       # [N]
#
#     # -------- Team reward aggregation --------
#     if team_reward_reduce == "sum":
#         r_team = rew_seq.sum(dim=-1)                          # [T]
#     elif team_reward_reduce == "mean":
#         r_team = rew_seq.mean(dim=-1)                         # [T]
#     else:
#         raise ValueError("team_reward_reduce must be 'sum' or 'mean'")
#
#     # Reward normalization + global scaling (optional)
#     if normalize_rewards:
#         mean_r = r_team.mean()
#         std_r  = r_team.std(unbiased=False).clamp_min(1e-6)
#         r_team = (r_team - mean_r) / std_r
#     if reward_scale != 1.0:
#         r_team = r_team * reward_scale
#
#     # -------- Team value aggregation --------
#     if team_value_reduce == "mean":
#         v_team = v_agents.mean(dim=-1)                        # [T]
#         last_v = last_v_agents.mean()                         # []
#     elif team_value_reduce == "sum":
#         v_team = v_agents.sum(dim=-1)                         # [T]
#         last_v = last_v_agents.sum()                          # []
#     else:
#         raise ValueError("team_value_reduce must be 'mean' or 'sum'")
#
#     # -------- Inlined GAE on team signal --------
#     with torch.no_grad():
#         adv_team = torch.zeros_like(v_team)                   # [T]
#         gae = torch.zeros((), device=device)
#         for t in reversed(range(T)):
#             nonterm = 1.0 - done_seq[t]
#             v_next  = last_v if t == T - 1 else v_team[t + 1]
#             delta   = r_team[t] + gamma * nonterm * v_next - v_team[t]
#             gae     = delta + gamma * gae_lambda * nonterm * gae
#             adv_team[t] = gae
#         ret_team = adv_team + v_team                          # [T]
#
#     if normalize_adv:
#         adv_team = (adv_team - adv_team.mean()) / (adv_team.std(unbiased=False).clamp_min(1e-6))
#
#     # -------- Policy loss (shared decentralized actor) --------
#     obs_flat  = obs_seq.reshape(T * N, O)                     # [T*N,O]
#     logits    = actor(obs_flat)                               # [T*N,A]
#     dist      = torch.distributions.Categorical(logits=logits)
#     acts_flat = act_seq.reshape(T * N)                        # [T*N]
#     logp      = dist.log_prob(acts_flat)                      # [T*N]
#     entropy   = dist.entropy().mean()
#
#     adv_broadcast = adv_team.unsqueeze(-1).expand(T, N).reshape(T * N)  # [T*N]
#     policy_loss = -(logp * adv_broadcast.detach()).mean()
#
#     # -------- Value loss (Huber + optional value clipping) --------
#     ret_broadcast = ret_team.unsqueeze(-1).expand(T, N)       # [T,N]
#     v_pred = v_agents                                         # [T,N]
#     v_targ = ret_broadcast.detach()                           # [T,N]
#
#     if value_clip_eps and value_clip_eps > 0.0:
#         with torch.no_grad():
#             v_old = v_pred.clone()
#         v_clipped = v_old + (v_pred - v_old).clamp(-value_clip_eps, value_clip_eps)
#         v_loss_unclipped = torch.nn.functional.smooth_l1_loss(v_pred,   v_targ, beta=huber_delta)
#         v_loss_clipped   = torch.nn.functional.smooth_l1_loss(v_clipped, v_targ, beta=huber_delta)
#         value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
#     else:
#         value_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_targ, beta=huber_delta)
#
#     # -------- Optimize --------
#     total_loss = policy_loss - entropy_coef * entropy + value_coef * value_loss
#     optim_actor.zero_grad(set_to_none=True)
#     optim_critic.zero_grad(set_to_none=True)
#     total_loss.backward()
#     nn.utils.clip_grad_norm_(actor.parameters(),  grad_clip)
#     nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
#     optim_actor.step()
#     optim_critic.step()
#
#     return {
#         "loss_total":  float(total_loss.item()),
#         "loss_policy": float(policy_loss.item()),
#         "loss_value":  float(value_loss.item()),
#         "entropy":     float(entropy.item()),
#     }


# =============================== MAPPO update (MLP) ================================
# def mappo_update_mlp(
#     actor: nn.Module,
#     critic: nn.Module,
#     optim_actor: torch.optim.Optimizer,
#     optim_critic: torch.optim.Optimizer,
#     # Backward-compatible names:
#     obs_seq=None,                 # [T,N,O] list/np/tensor
#     last_obs=None,                # [N,O]   list/np/tensor
#     # Preferred names (take precedence if provided):
#     obs_local_seq=None,           # [T,N,O]
#     obs_joint_seq=None,           # [T,N*O]
#     act_seq=None,                 # [T,N]   int
#     rew_seq=None,                 # [T,N]   float
#     done_seq=None,                # [T]     float; 1.0 terminal
#     last_obs_joint=None,          # [N*O]
#     gamma: float = 0.99,
#     gae_lambda: float = 0.95,
#     entropy_coef: float = 0.01,
#     value_coef: float = 0.5,
#     max_grad_norm: float = 1.0,
#     normalize_adv: bool = True,
#     # --- New stabilizers (optional) ---
#     reward_scale: float = 1.0,            # e.g., 0.02~0.1 for TSC
#     team_reduce: str = "sum",             # "sum" (default) or "mean"
#     value_clip_eps: float = 0.2,          # PPO-style value clipping (0 disables if <=0)
#     use_huber_value: bool = True,         # robust critic loss
#     huber_delta: float = 1.0,
# ):
#     """
#     Centralized-critic A2C update (no PPO policy ratio clip).
#     Adds reward scaling, value clipping, and robust value loss.
#
#     Backward-compatible: keeps (obs_seq, last_obs) names and shapes you use today.
#     """
#
#     # ------------ small helpers ------------
#     def _as_tensor(x, device=None, dtype=None):
#         if isinstance(x, torch.Tensor):
#             t = x
#         elif isinstance(x, (list, tuple)):
#             t = torch.stack([_as_tensor(xx) for xx in x], dim=0)
#         else:
#             t = torch.as_tensor(x)
#         if dtype is not None:
#             t = t.to(dtype)
#         if device is not None:
#             t = t.to(device)
#         return t
#
#     def _ensure_TNO(x):  # [T,N,O]
#         if x.dim() != 3:
#             raise ValueError(f"Expected [T,N,O], got {tuple(x.shape)}")
#         return x
#
#     def _ensure_TN(x):   # [T,N]
#         if x.dim() != 2:
#             raise ValueError(f"Expected [T,N], got {tuple(x.shape)}")
#         return x
#
#     def _ensure_T(x):    # [T]
#         if x.dim() != 1:
#             raise ValueError(f"Expected [T], got {tuple(x.shape)}")
#         return x
#
#     device = next(actor.parameters()).device
#
#     # ------------ reconcile names & coerce ------------
#     if obs_local_seq is None:
#         if obs_seq is None:
#             raise ValueError("Provide obs_local_seq=[T,N,O] or obs_seq=[T,N,O].")
#         obs_local_seq = obs_seq
#     obs_local_seq = _ensure_TNO(_as_tensor(obs_local_seq))           # [T,N,O] (CPU for now)
#     T, N, O = obs_local_seq.shape
#
#     if obs_joint_seq is None:
#         obs_joint_seq = obs_local_seq.reshape(T, N * O)
#     else:
#         obs_joint_seq = _as_tensor(obs_joint_seq)
#         if obs_joint_seq.dim() != 2 or obs_joint_seq.shape[0] != T or obs_joint_seq.shape[1] != N * O:
#             raise ValueError(f"obs_joint_seq must be [T,{N*O}], got {tuple(obs_joint_seq.shape)}")
#
#     if last_obs_joint is None:
#         if last_obs is None:
#             raise ValueError("Provide last_obs_joint=[N*O] or last_obs=[N,O].")
#         last_obs = _as_tensor(last_obs)
#         if last_obs.dim() != 2 or last_obs.shape != (N, O):
#             raise ValueError(f"last_obs must be [N,O], got {tuple(last_obs.shape)}")
#         last_obs_joint = last_obs.reshape(N * O)                      # [N*O]
#     else:
#         last_obs_joint = _as_tensor(last_obs_joint)
#         if last_obs_joint.dim() != 1 or last_obs_joint.numel() != N * O:
#             raise ValueError(f"last_obs_joint must be [N*O], got {tuple(last_obs_joint.shape)}")
#
#     if act_seq is None or rew_seq is None or done_seq is None:
#         raise ValueError("act_seq, rew_seq, and done_seq must be provided.")
#
#     act_seq  = _ensure_TN(_as_tensor(act_seq)).long()                # [T,N]
#     rew_seq  = _ensure_TN(_as_tensor(rew_seq)).float()               # [T,N]
#     done_seq = _ensure_T(_as_tensor(done_seq)).float()               # [T]
#
#     # Move to device
#     obs_local_seq  = obs_local_seq.to(device)
#     obs_joint_seq  = obs_joint_seq.to(device)
#     last_obs_joint = last_obs_joint.to(device)
#     act_seq        = act_seq.to(device)
#     rew_seq        = (rew_seq * reward_scale).to(device)             # <<< reward scaling
#     done_seq       = done_seq.to(device)
#
#     # ------------ centralized critic values ------------
#     v_seq = critic(obs_joint_seq).reshape(-1)                        # [T]
#     last_val = critic(last_obs_joint.view(1, -1)).reshape(-1)[0]     # scalar
#
#     # team reward reducer
#     if team_reduce == "sum":
#         r_team = rew_seq.sum(dim=-1)                                 # [T]
#     elif team_reduce == "mean":
#         r_team = rew_seq.mean(dim=-1)                                # [T]
#     else:
#         raise ValueError("team_reduce must be 'sum' or 'mean'.")
#
#     # ------------ GAE (team) ------------
#     with torch.no_grad():
#         adv = torch.zeros_like(v_seq)                                 # [T]
#         gae = torch.zeros((), device=device)
#         for t in reversed(range(T)):
#             nonterm = 1.0 - done_seq[t]
#             v_next  = last_val if t == T - 1 else v_seq[t + 1]
#             delta   = r_team[t] + gamma * nonterm * v_next - v_seq[t]
#             gae     = delta + gamma * gae_lambda * nonterm * gae
#             adv[t]  = gae
#         ret = adv + v_seq                                             # [T]
#         old_v = v_seq.detach()                                        # for value clipping
#
#     # Advantage normalization (over time only; shared across agents)
#     if normalize_adv:
#         adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
#
#     # ------------ policy loss (shared decentralized actors) ------------
#     obs_flat = obs_local_seq.reshape(T * N, O)                        # [T*N, O]
#     logits   = actor(obs_flat)                                        # [T*N, A]
#     dist     = torch.distributions.Categorical(logits=logits)
#
#     acts_flat = act_seq.reshape(T * N)                                # [T*N]
#     logp      = dist.log_prob(acts_flat)                              # [T*N]
#     entropy   = dist.entropy().mean()
#
#     # Broadcast adv from team to all agents at each t
#     adv_expanded = adv.unsqueeze(-1).expand(T, N).reshape(T * N)      # [T*N]
#     policy_loss  = -(logp * adv_expanded.detach()).mean()
#
#     # ------------ value loss (clipped & robust) ------------
#     if value_clip_eps is not None and value_clip_eps > 0.0:
#         v_clipped = old_v + (v_seq - old_v).clamp(-value_clip_eps, value_clip_eps)
#         if use_huber_value:
#             vloss1 = nn.functional.smooth_l1_loss(v_seq,    ret.detach(), reduction='none')
#             vloss2 = nn.functional.smooth_l1_loss(v_clipped, ret.detach(), reduction='none')
#         else:
#             vloss1 = 0.5 * (ret.detach() - v_seq).pow(2)
#             vloss2 = 0.5 * (ret.detach() - v_clipped).pow(2)
#         value_loss = torch.max(vloss1, vloss2).mean()
#     else:
#         if use_huber_value:
#             value_loss = nn.functional.smooth_l1_loss(v_seq, ret.detach())
#         else:
#             value_loss = 0.5 * (ret.detach() - v_seq).pow(2).mean()
#
#     # ------------ optimize ------------
#     loss_total = policy_loss + value_coef * value_loss - entropy_coef * entropy
#
#     optim_actor.zero_grad(set_to_none=True)
#     optim_critic.zero_grad(set_to_none=True)
#     loss_total.backward()
#     torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
#     torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
#     optim_actor.step()
#     optim_critic.step()
#
#     return {
#         "loss_total":  float(loss_total.item()),
#         "loss_policy": float(policy_loss.item()),
#         "loss_value":  float(value_loss.item()),
#         "entropy":     float(entropy.item()),
#         "value_mean":  float(v_seq.mean().item()),
#         "adv_mean":    float(adv.mean().item()),
#     }

# ================================ Soft Update ==================================

@torch.no_grad()
def soft_update(target: torch.nn.Module, online: torch.nn.Module, tau: float = 0.01):
    """Polyak averaging: target <- tau * online + (1 - tau) * target"""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * op.data)


# ========================= Gated spatio-temporal update (AAMAS model) =========================
def dqn_update_shared_gated(
    device: torch.device,
    online_q,
    target_q,
    optimizer_q: torch.optim.Optimizer,
    batch_tuple,
    edge_index: torch.Tensor,
    gamma: float = 0.99,
    burn_in: int = 8,
    double_dqn: bool = True,
    grad_clip: float = 1.0,
    lambda_mem: float = 0.0,
    lambda_com: float = 0.0,
):
    """Double-DQN sequence update for GatedSpatioTemporalQ, with module-use
    sparsity penalties on the soft gate probabilities.

    Returns (total_loss, td_loss, mean_g_mem, mean_g_com).
    """
    S, A, R, NS, D = batch_tuple
    S  = torch.as_tensor(S,  device=device, dtype=torch.float32)
    NS = torch.as_tensor(NS, device=device, dtype=torch.float32)
    A  = torch.as_tensor(A,  device=device, dtype=torch.long)
    R  = torch.as_tensor(R,  device=device, dtype=torch.float32)
    D  = torch.as_tensor(D,  device=device, dtype=torch.float32)
    if D.dim() == 2:
        D = D.unsqueeze(-1).expand(-1, -1, S.size(2))

    B, T, N, O = S.shape
    burn = min(max(int(burn_in), 0), max(T - 1, 0))
    Ei = edge_index.to(device)

    with torch.no_grad():
        h_online = h_target = None
        if burn > 0:
            _, h_online, _ = online_q(S[:, :burn], Ei, None)
            _, h_target, _ = target_q(NS[:, :burn], Ei, None)

    S_w, NS_w = S[:, burn:], NS[:, burn:]
    A_w, R_w, D_w = A[:, burn:], R[:, burn:], D[:, burn:]
    Tw = S_w.size(1)

    h_o, h_t = h_online, h_target
    q_list, qn_online_list, qn_target_list, g_list = [], [], [], []

    for t in range(Tw):
        q_t, h_o, g_t = online_q(S_w[:, t].unsqueeze(1), Ei, h_o)
        q_list.append(q_t)
        g_list.append(g_t)

        with torch.no_grad():
            qn_o_t, _, _ = online_q(NS_w[:, t].unsqueeze(1), Ei, h_o)
            qn_t_t, h_t, _ = target_q(NS_w[:, t].unsqueeze(1), Ei, h_t)
        qn_online_list.append(qn_o_t)
        qn_target_list.append(qn_t_t)

        for hpair in (h_o, h_t):
            if hpair is not None:
                done_mask = (1.0 - D_w[:, t]).view(1, B, N, 1)
                h0 = hpair[0].view(1, B, N, -1); c0 = hpair[1].view(1, B, N, -1)
                h0.mul_(done_mask); c0.mul_(done_mask)
        # views share storage; tuples already updated in place

    Q_seq = torch.cat(q_list, dim=1)
    Qn_o = torch.cat(qn_online_list, dim=1)
    Qn_t = torch.cat(qn_target_list, dim=1)
    G_seq = torch.cat(g_list, dim=1)                                  # [B,Tw,N,2] soft probs

    Q_taken = Q_seq.gather(-1, A_w.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_idx = torch.argmax(Qn_o, dim=-1) if double_dqn else torch.argmax(Qn_t, dim=-1)
        Qn_star = Qn_t.gather(-1, next_idx.unsqueeze(-1)).squeeze(-1)
        target = R_w + (1.0 - D_w) * gamma * Qn_star

    td_loss = nn.functional.smooth_l1_loss(Q_taken, target)
    mean_g_mem = G_seq[..., 0].mean()
    mean_g_com = G_seq[..., 1].mean()
    loss = td_loss + lambda_mem * mean_g_mem + lambda_com * mean_g_com

    optimizer_q.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_q.parameters(), grad_clip)
    optimizer_q.step()

    return float(loss.item()), float(td_loss.item()), float(mean_g_mem.item()), float(mean_g_com.item())
