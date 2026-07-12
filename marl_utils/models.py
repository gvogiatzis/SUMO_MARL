#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# =================================== MLP Model ===================================
class QNetMLP(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, action_dim),
        )
    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(state_tensor)

# ========================== LSTM (Recurrent) Model ===============================
class RecurrentQNet(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_units: int = 128):
        super().__init__()
        # Encoder to hidden size H
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_units),
            nn.ReLU(),
        )
        # NEW: LayerNorm over the last dim (H) before the LSTM
        self.pre_lstm_norm = nn.LayerNorm(hidden_units)

        self.lstm = nn.LSTM(hidden_units, hidden_units, batch_first=True)

        # OPTIONAL: LayerNorm after LSTM (helps if LSTM outputs drift)
        self.post_lstm_norm = nn.LayerNorm(hidden_units)

        self.q_head = nn.Linear(hidden_units, action_dim)

    def forward(
        self,
        observation_seq: torch.Tensor,  # [B, T, O]
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        # Encode per step to H
        z = self.encoder(observation_seq)           # [B, T, H]
        z = self.pre_lstm_norm(z)                   # LN over H

        # Recurrent pass
        z, hidden_state = self.lstm(z, hidden_state)  # [B, T, H]

        # (Optional) stabilize head input
        z = self.post_lstm_norm(z)                  # LN over H

        # Q-values per step
        q_values = self.q_head(z)                   # [B, T, A]
        return q_values, hidden_state

# ====================================== GNN models =========================================
# Attention block
class SimpleAttnMP(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.phi = nn.Linear(node_dim, hidden)
        self.psi = nn.Linear(edge_dim, hidden) if edge_dim > 0 else None
        self.attn = nn.Linear(2 * hidden, 1)
        self.upd = nn.GRUCell(hidden, hidden)
    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                comm_gate: Optional[torch.Tensor] = None):
        # comm_gate: optional [N] or [N,1] per-(receiving-)node scale on the aggregated
        # neighbour message. comm_gate=0 -> node receives no messages (self-only update).
        N = node_x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h = torch.relu(self.phi(node_x))
        m_src = h[src]; m_dst = h[dst]
        m_e = torch.relu(self.psi(edge_attr)) if (self.psi is not None and edge_attr is not None) else torch.zeros_like(m_src)
        scores = self.attn(torch.cat([m_src + m_e, m_dst], dim=-1)).squeeze(-1)
        exp_scores = torch.exp(scores - scores.max())
        denom = torch.zeros(N, device=node_x.device); denom.index_add_(0, dst, exp_scores)
        alpha = exp_scores / (denom[dst] + 1e-9)
        aggregated = torch.zeros_like(h); aggregated.index_add_(0, dst, alpha.unsqueeze(-1) * (m_src + m_e))
        if comm_gate is not None:
            aggregated = aggregated * comm_gate.view(N, 1)
        return self.upd(aggregated, h)

class GNNPolicyQ(nn.Module):
    def __init__(self, node_dim: int, actions: int = 4, edge_dim: int = 0, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleAttnMP(node_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))
    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        z = node_feats
        for mp in self.layers:
            z = mp(z, edge_index, edge_attr)
        return self.head(z)

# ====================================== GNN+LSTM models =========================================
class GNNLSTMPolicyQ(nn.Module):
    """
    Spatio-temporal Q-network:
      - Per-timestep GNN message passing (vectorized over batch) → node embeddings z_t ∈ R^{B×N×H}
      - LayerNorm over embeddings (stabilizes scale/drift across nodes & time)
      - Per-node LSTM over time on z_{1:T} (nodes flattened into batch: B*N sequences)
      - Q-head maps LSTM outputs to per-node Q-values

    Inputs:
      X_seq    : [B, T, N, O]
      edge_index : [2, E]   (same topology for all items in the batch)
    Outputs:
      Q_seq    : [B, T, N, A]
      new_hidden: (h, c) with shape [1, B*N, H]
    """
    def __init__(self, node_dim: int, actions: int, hidden: int = 128, gnn_layers: int = 2):
        super().__init__()
        self.hidden = hidden
        self.actions = actions

        self.gnn_layers = nn.ModuleList([
            SimpleAttnMP(node_dim if i == 0 else hidden, edge_dim=0, hidden=hidden)
            for i in range(gnn_layers)
        ])

        # NEW: LayerNorm before the LSTM (applied over last dim H)
        self.pre_lstm_norm = nn.LayerNorm(hidden)

        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions)
        )

    def _build_batched_edges(self, edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
        Ei = edge_index
        device = Ei.device
        offsets = torch.arange(B, device=device, dtype=Ei.dtype) * N  # [B]
        src = Ei[0].unsqueeze(0) + offsets.unsqueeze(1)               # [B,E]
        dst = Ei[1].unsqueeze(0) + offsets.unsqueeze(1)               # [B,E]
        return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0) # [2,B*E]

    def forward(
        self,
        X_seq: torch.Tensor,                 # [B, T, N, O]
        edge_index: torch.Tensor,            # [2, E]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        B, T, N, O = X_seq.shape
        device = X_seq.device
        Ei = edge_index.to(device)

        # ----- GNN per timestep (vectorized over batch) -----
        Ei_batched = self._build_batched_edges(Ei, B, N)              # [2,B*E]
        z_time = []
        for t in range(T):
            x_t = X_seq[:, t, :, :]                                   # [B,N,O]
            x_flat = x_t.reshape(B * N, O).contiguous()               # [B*N,O]
            z = x_flat
            for mp in self.gnn_layers:
                z = mp(z, Ei_batched, edge_attr=None)                 # [B*N,H]
            z_time.append(z.reshape(B, N, self.hidden).unsqueeze(1))  # [B,1,N,H]
        z_seq = torch.cat(z_time, dim=1)                              # [B,T,N,H]

        # ----- NEW: LayerNorm over H before LSTM -----
        # LayerNorm expects the last dimension (H); works on [B,T,N,H] directly
        z_seq = self.pre_lstm_norm(z_seq)

        # ----- LSTM over time, per node -----
        z_seq_bn = z_seq.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.hidden)  # [B*N,T,H]
        if hidden is not None:
            h0, c0 = hidden
            if h0.size(1) != B * N or c0.size(1) != B * N:
                raise RuntimeError(f"LSTM hidden batch mismatch: got {h0.size(1)}, expected {B*N}")
            if h0.size(2) != self.hidden or c0.size(2) != self.hidden:
                raise RuntimeError(f"LSTM hidden width mismatch: got {h0.size(2)}, expected {self.hidden}")

        q_h_seq, new_hidden = self.lstm(z_seq_bn, hidden)             # [B*N,T,H]

        # ----- Q head and reshape back to [B,T,N,A] -----
        q_seq_bn = self.head(q_h_seq)                                 # [B*N,T,A]
        BN, T_out, A = q_seq_bn.shape
        if A != self.actions:
            raise RuntimeError(f"Q-head actions mismatch: got {A}, expected {self.actions}")
        if BN != B * N:
            raise RuntimeError(f"Internal size mismatch: BN={BN} vs B*N={B*N}")

        q_seq = (
            q_seq_bn.reshape(B, N, T_out, A)   # [B,N,T,A]
                    .permute(0, 2, 1, 3)       # [B,T,N,A]
                    .contiguous()
        )
        return q_seq, new_hidden

    @torch.no_grad()
    def step(
        self,
        X_t: torch.Tensor,               # [B, N, O]
        edge_index: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        Q_seq, new_hidden = self.forward(X_t.unsqueeze(1), edge_index, hidden)  # -> [B,1,N,A]
        return Q_seq[:, 0], new_hidden

# ====================================== QMIX mixer for CTDE =========================================
class QMIXMixer(nn.Module):
    """
    QMIX mixing network.

    Inputs:
        agent_qs: [B, N]
            Selected per-agent Q-values Q_i(o_i, a_i)

        states: [B, state_dim]
            Centralised training state, e.g. flattened [o_1, ..., o_N]

    Output:
        q_tot: [B]
            Mixed joint action-value Q_tot
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        mixing_embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
    ):
        super().__init__()

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        # First-layer hypernetwork: produces weights from global state
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, num_agents * mixing_embed_dim),
        )

        self.hyper_b_1 = nn.Linear(state_dim, mixing_embed_dim)

        # Final-layer hypernetwork
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, mixing_embed_dim),
        )

        # State-dependent bias V(s)
        self.V = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: [B, N]
        states:   [B, state_dim]
        """
        B = agent_qs.shape[0]

        # Enforce monotonicity by making mixer weights non-negative
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        w1 = w1.view(B, self.num_agents, self.mixing_embed_dim)
        b1 = b1.view(B, 1, self.mixing_embed_dim)

        agent_qs = agent_qs.view(B, 1, self.num_agents)

        hidden = nn.functional.elu(torch.bmm(agent_qs, w1) + b1)  # [B, 1, mixing_embed_dim]

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(B, self.mixing_embed_dim, 1)

        v = self.V(states).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + v  # [B, 1, 1]

        return q_tot.view(B)

# =================================== A2C MLP Models ==================================

class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # value

# ===== Per-Agent centralized critic =====
class CriticPAMLP(nn.Module):
    """
    Joint-observation critic with N outputs (one per agent).
    Forward accepts [..., N*O] and returns [..., N].
    """
    def __init__(self, joint_obs_dim: int, n_agents: int, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_agents)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N*O] or [T, N*O] or [N*O]
        y = self.net(x)
        return y  # [..., N]

# =================================== A2C LSTM Models  ==================================
class ActorLSTM(nn.Module):
    """
    Shared decentralized actor. Accepts [B,T,N,O], [T,N,O], or [N,O].
    Outputs logits [B,T,N,A] (or reduced dims accordingly).
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.enc = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.pi = nn.Linear(hidden, act_dim)

    def _normalize_btno(self, X: torch.Tensor) -> torch.Tensor:
        # Accept [N,O], [T,N,O], or [B,T,N,O] -> return [B,T,N,O]
        if X.dim() == 2:      # [N,O]
            X = X.unsqueeze(0).unsqueeze(0)  # [1,1,N,O]
        elif X.dim() == 3:    # [T,N,O]
            X = X.unsqueeze(0)               # [1,T,N,O]
        elif X.dim() != 4:
            raise AssertionError(f"ActorLSTM expects [B,T,N,O]/[T,N,O]/[N,O], got {tuple(X.shape)}")
        return X

    def forward(self, X: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]|None=None):
        X = self._normalize_btno(X)  # [B,T,N,O]
        B, T, N, O = X.shape
        x = X.permute(0, 2, 1, 3).reshape(B*N, T, O)           # [B*N,T,O]
        z = torch.relu(self.enc(x))                             # [B*N,T,H]
        z, new_hidden = self.lstm(z, hidden)                    # hidden: (1, B*N, H)
        logits = self.pi(z)                                     # [B*N,T,A]
        logits = logits.view(B, N, T, self.act_dim).permute(0, 2, 1, 3).contiguous()  # [B,T,N,A]
        return logits, new_hidden

    @torch.no_grad()
    def step(self, X_t: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]|None=None):
        """
        Single-step policy: X_t is [N,O] or [1,1,N,O]. Returns logits [N,A], new hidden.
        """
        logits, new_hidden = self.forward(X_t, hidden)  # [1,1,N,A]
        return logits[0, -1], new_hidden                # [N,A]

class CriticLSTM(nn.Module):
    """
    Shared decentralized critic. Same input protocol as the actor.
    Outputs values per agent: [B,T,N].
    """
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.enc = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.v = nn.Linear(hidden, 1)

    def _normalize_btno(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 2:
            X = X.unsqueeze(0).unsqueeze(0)
        elif X.dim() == 3:
            X = X.unsqueeze(0)
        elif X.dim() != 4:
            raise AssertionError(f"CriticLSTM expects [B,T,N,O]/[T,N,O]/[N,O], got {tuple(X.shape)}")
        return X

    def forward(self, X: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]|None=None):
        X = self._normalize_btno(X)  # [B,T,N,O]
        B, T, N, O = X.shape
        x = X.permute(0, 2, 1, 3).reshape(B*N, T, O)     # [B*N,T,O]
        z = torch.relu(self.enc(x))
        z, new_hidden = self.lstm(z, hidden)
        vals = self.v(z).squeeze(-1)                     # [B*N,T]
        vals = vals.view(B, N, T).permute(0, 2, 1).contiguous()  # [B,T,N]
        return vals, new_hidden

    @torch.no_grad()
    def step(self, X_t: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]|None=None):
        vals, new_hidden = self.forward(X_t, hidden)   # [1,1,N]
        return vals[0, -1], new_hidden                 # [N]

class CriticPALSTM(nn.Module):
    """
    Centralised per-agent critic with temporal memory.
    Accepts joint observations and outputs N values (one per agent) at each time step.

    Input forms:
      - [N, O]           -> returns (values [N], (h,c))
      - [T, N, O]        -> returns (values [1, T, N], (h,c))  (B=1 implicit)
      - [B, T, N, O]     -> returns (values [B, T, N], (h,c))

    Notes:
      - n_agents must match N at runtime.
      - Hidden state shape: (num_layers=1, batch=B, hidden_size=hidden)
        where batch is the first dim fed to the LSTM (B).
    """
    def __init__(self, joint_obs_dim: int, n_agents: int, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.hidden = hidden
        self.enc = nn.Linear(joint_obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)  # expects [B, T, H]
        self.v_head = nn.Linear(hidden, n_agents)

    def _normalize_btno(self, X: torch.Tensor) -> torch.Tensor:
        # Map [N,O]/[T,N,O]/[B,T,N,O] -> [B,T,N,O]
        if X.dim() == 2:      # [N,O]
            X = X.unsqueeze(0).unsqueeze(0)  # [1,1,N,O]
        elif X.dim() == 3:    # [T,N,O]
            X = X.unsqueeze(0)               # [1,T,N,O]
        elif X.dim() != 4:    # else must already be [B,T,N,O]
            raise AssertionError(f"CriticPALSTM expects [B,T,N,O]/[T,N,O]/[N,O], got {tuple(X.shape)}")
        return X

    def forward(
        self,
        X: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # X_btno: [B,T,N,O] -> flatten joint obs to J=N*O then LSTM over time
        X = self._normalize_btno(X)
        B, T, N, O = X.shape
        assert N == self.n_agents, f"CriticPALSTM: n_agents mismatch (got N={N}, expected {self.n_agents})"
        J = N * O
        x_btj = X.reshape(B, T, J)                 # [B,T,J]
        z = torch.relu(self.enc(x_btj))            # [B,T,H]
        z_out, new_hidden = self.lstm(z, hidden)   # [B,T,H]
        vals_btn = self.v_head(z_out)              # [B,T,N]
        return vals_btn, new_hidden

    @torch.no_grad()
    def step(
        self,
        X_t: torch.Tensor,  # [N,O]
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Single-step evaluation:
          returns (values [N], new_hidden)
        """
        vals_btn, new_hidden = self.forward(X_t, hidden)  # [1,1,N]
        return vals_btn[0, -1], new_hidden

# =================================== A2C GNN Models ==================================
class ActorGNNAttn(nn.Module):
    """
    Shared decentralized actor over the traffic-light graph.
    node_feats: [N, O], edge_index: [2, E] (src->dst), edge_attr: [E, D_e] or None
    returns logits per node: [N, A]
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, layers: int = 2, edge_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden
        self.layers = nn.ModuleList([
            SimpleAttnMP(obs_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        self.pi_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = node_feats
        for mp in self.layers:
            z = mp(z, edge_index, edge_attr)  # [N, H]
        return self.pi_head(z)                # [N, A]


class CriticGNNPerAgentAttn(nn.Module):
    """
    Per-agent value critic over the same graph.
    node_feats: [N, O], edge_index: [2, E], edge_attr: [E, D_e] or None
    returns per-node values: [N]
    """
    def __init__(self, obs_dim: int, hidden: int = 128, layers: int = 2, edge_dim: int = 0):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleAttnMP(obs_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        self.v_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = node_feats
        for mp in self.layers:
            z = mp(z, edge_index, edge_attr)  # [N, H]
        return self.v_head(z).squeeze(-1)     # [N]

# ============================ A2C GNN + LSTM Models ===========================
class ActorGNNLSTMAttn(nn.Module):
    """
    Shared decentralised actor over a graph with temporal memory.

    Per-step pipeline:
      node_feats -> GNN (attention message passing) -> per-node emb H
      Over time (per node): LSTM(H_t) -> logits

    Inputs:
      node_feats: [N,O]  (single step)  or  [T,N,O] (sequence)
      edge_index: [2,E]  (long, src->dst)
      edge_attr:  [E,De] or None

    Returns:
      (logits, new_hidden)
        - logits: [N,A] for single step, or [T,N,A] for sequence
        - new_hidden: (h, c) with shape [1, N, H]
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, layers: int = 2, edge_dim: int = 0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        self.gnn_layers = nn.ModuleList([
            SimpleAttnMP(obs_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        # batch_first=True -> input to LSTM is [N, T, H]
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.pi_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def _mp_once(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]):
        z = node_feats
        for mp in self.gnn_layers:
            z = mp(z, edge_index, edge_attr)  # [N,H]
        return z

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        If node_feats is [N,O]: returns (logits [N,A], new_hidden)
        If node_feats is [T,N,O]: returns (logits [T,N,A], new_hidden)
        """
        if node_feats.dim() == 2:
            # ---------- single step ----------
            # node_feats: [N,O]
            z = self._mp_once(node_feats, edge_index, edge_attr)  # [N,H]
            z_in = z.unsqueeze(1)                                  # [N,1,H] (time=1)
            z_out, new_hidden = self.lstm(z_in, hidden)            # [N,1,H]
            logits = self.pi_head(z_out.squeeze(1))                # [N,A]
            return logits, new_hidden

        elif node_feats.dim() == 3:
            # ---------- sequence ----------
            # node_feats: [T,N,O]
            T, N, _ = node_feats.shape
            z_seq = []
            for t in range(T):
                z_t = self._mp_once(node_feats[t], edge_index, edge_attr)  # [N,H]
                z_seq.append(z_t)
            z_seq = torch.stack(z_seq, dim=0)                               # [T,N,H]
            z_seq_bn = z_seq.permute(1, 0, 2).contiguous()                  # [N,T,H]
            z_out_bn, new_hidden = self.lstm(z_seq_bn, hidden)              # [N,T,H]
            z_out = z_out_bn.permute(1, 0, 2).contiguous()                  # [T,N,H]
            logits = self.pi_head(z_out)                                     # [T,N,A]
            return logits, new_hidden

        else:
            raise AssertionError(f"ActorGNNLSTMAttn expects [N,O] or [T,N,O], got {tuple(node_feats.shape)}")

    @torch.no_grad()
    def step(
        self,
        node_feats_t: torch.Tensor,           # [N,O]
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single environment step:
          returns (logits [N,A], new_hidden  [1,N,H])
        """
        logits, new_hidden = self.forward(node_feats_t, edge_index, edge_attr, hidden)
        return logits, new_hidden


class CriticGNNPerAgentLSTMAttn(nn.Module):
    """
    Per-agent value critic with GNN + LSTM temporal memory.

    Inputs:
      node_feats: [N,O] (single step) or [T,N,O] (sequence)
      edge_index: [2,E], edge_attr: [E,De] or None

    Returns:
      (values, new_hidden)
        - values: [N] for single step, or [T,N] for sequence
        - new_hidden: (h, c) with shape [1, N, H]
    """
    def __init__(self, obs_dim: int, hidden: int = 128, layers: int = 2, edge_dim: int = 0):
        super().__init__()
        self.hidden = hidden
        self.gnn_layers = nn.ModuleList([
            SimpleAttnMP(obs_dim if i == 0 else hidden, edge_dim=edge_dim, hidden=hidden)
            for i in range(layers)
        ])
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)   # input [N,T,H]
        self.v_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def _mp_once(self, node_feats: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]):
        z = node_feats
        for mp in self.gnn_layers:
            z = mp(z, edge_index, edge_attr)  # [N,H]
        return z

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if node_feats.dim() == 2:
            # ---------- single step ----------
            # node_feats: [N,O]
            z = self._mp_once(node_feats, edge_index, edge_attr)  # [N,H]
            z_in = z.unsqueeze(1)                                  # [N,1,H]
            z_out, new_hidden = self.lstm(z_in, hidden)            # [N,1,H]
            v = self.v_head(z_out.squeeze(1)).squeeze(-1)          # [N]
            return v, new_hidden

        elif node_feats.dim() == 3:
            # ---------- sequence ----------
            # node_feats: [T,N,O]
            T, N, _ = node_feats.shape
            z_seq = []
            for t in range(T):
                z_t = self._mp_once(node_feats[t], edge_index, edge_attr)  # [N,H]
                z_seq.append(z_t)
            z_seq = torch.stack(z_seq, dim=0)                               # [T,N,H]
            z_seq_bn = z_seq.permute(1, 0, 2).contiguous()                  # [N,T,H]
            z_out_bn, new_hidden = self.lstm(z_seq_bn, hidden)              # [N,T,H]
            z_out = z_out_bn.permute(1, 0, 2).contiguous()                  # [T,N,H]
            v = self.v_head(z_out).squeeze(-1)                              # [T,N]
            return v, new_hidden

        else:
            raise AssertionError(f"CriticGNNPerAgentLSTMAttn expects [N,O] or [T,N,O], got {tuple(node_feats.shape)}")

    @torch.no_grad()
    def step(
        self,
        node_feats_t: torch.Tensor,           # [N,O]
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single environment step:
          returns (values [N], new_hidden [1,N,H])
        """
        v, new_hidden = self.forward(node_feats_t, edge_index, edge_attr, hidden)
        return v, new_hidden

# ====================================== CoLight =========================================
class CoLightMultiHeadAtt(nn.Module):
    """
    CoLight graph attention (Wei et al., CIKM 2019), ported from the
    LibSignal implementation to plain PyTorch (no torch_geometric).

    Per head: e_ij = <ReLU(W_t h_i), ReLU(W_s h_j)> for j in N(i) ∪ {i},
    alpha = softmax_j(e_ij), out_i = mean over heads of sum_j alpha_ij ReLU(W_h h_j),
    followed by a shared dv -> d_out projection and ReLU.

    Last attention weights are kept (detached) in self.last_alpha as
    ([E_with_self_loops, heads], (src, dst)) for communication analysis.
    """
    def __init__(self, d: int = 128, dv: int = 16, d_out: int = 128, nv: int = 5):
        super().__init__()
        self.d, self.dv, self.d_out, self.nv = d, dv, d_out, nv
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.last_alpha = None

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = h.size(0)
        loops = torch.arange(N, device=h.device)
        src = torch.cat([edge_index[0].to(h.device), loops])   # [E]
        dst = torch.cat([edge_index[1].to(h.device), loops])   # [E]

        h_t = F.relu(self.W_target(h)).view(N, self.nv, self.dv)
        h_s = F.relu(self.W_source(h)).view(N, self.nv, self.dv)
        m = F.relu(self.hidden_embedding(h)).view(N, self.nv, self.dv)

        e = (h_t[dst] * h_s[src]).sum(-1)                       # [E, nv]
        # numerically stable softmax over incoming edges per target node
        e_max = torch.full((N, self.nv), -torch.inf, device=h.device)
        e_max = e_max.index_reduce(0, dst, e, "amax", include_self=False)
        e_exp = torch.exp(e - e_max[dst])
        denom = torch.zeros(N, self.nv, device=h.device).index_add_(0, dst, e_exp)
        alpha = e_exp / (denom[dst] + 1e-12)                    # [E, nv]
        self.last_alpha = (alpha.detach(), (src, dst))

        weighted = alpha.unsqueeze(-1) * m[src]                 # [E, nv, dv]
        agg = torch.zeros(N, self.nv, self.dv, device=h.device).index_add_(0, dst, weighted)
        return F.relu(self.out(agg.mean(dim=1)))                # [N, d_out]


class CoLightQ(nn.Module):
    """CoLight Q-network: embedding MLP + graph-attention layer(s) + Q head.

    Same call signature as GNNPolicyQ so it reuses dqn_update_shared_gnn.
    Defaults follow LibSignal's colight.yml (128-128 embedding, 1 att layer,
    5 heads x 16 dims, 128 out).
    """
    def __init__(self, node_dim: int, actions: int = 4, hidden: int = 128,
                 att_layers: int = 1, heads: int = 5, dv: int = 16):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.att = nn.ModuleList([
            CoLightMultiHeadAtt(d=hidden, dv=dv, d_out=hidden, nv=heads)
            for _ in range(att_layers)
        ])
        self.q_head = nn.Linear(hidden, actions)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.embed(node_feats)
        for blk in self.att:
            z = blk(z, edge_index)
        return self.q_head(z)

# ============================== Gated spatio-temporal model (AAMAS) ==============================
class GatedSpatioTemporalQ(nn.Module):
    """
    Adaptive gated spatio-temporal Q-network (the AAMAS 2027 contribution).

    Per agent i and step t:
      e_t = local encoder(o_t)
      m_t = graph-attention aggregation of neighbour embeddings   (spatial branch)
      h_t = LSTM over local embedding history                     (temporal branch)
      (g_mem, g_com) = gate MLP on [e_t, h_{t-1}]                 (per-agent, per-step)
      z_t = LayerNorm(e_t + g_mem * W_m h_t + g_com * W_c m_t) -> Q head

    Gates are receive-side, Gumbel-sigmoid with straight-through hard samples
    during training (temperature self.gate_temp, annealed by the trainer) and
    deterministic hard thresholds in eval. forward() returns soft gate
    probabilities [B,T,N,2] (mem, com) for sparsity penalties and logging.

    Gate configuration recovers the fixed baselines: (0,0)=MLP-like local,
    (1,0)=memory-only, (0,1)=GNN-like, (1,1)=fixed spatio-temporal.
    """
    def __init__(self, node_dim: int, actions: int, hidden: int = 128,
                 gnn_layers: int = 2, gate_temp: float = 1.0, gate_bias_init: float = 1.0,
                 gate_mem_mode: str = "learned", gate_com_mode: str = "learned"):
        super().__init__()
        self.hidden = hidden
        self.actions = actions
        self.gate_temp = gate_temp  # set by the trainer (annealed)
        # ablation modes: "learned" | "open" (forced 1) | "closed" (forced 0)
        assert gate_mem_mode in ("learned", "open", "closed")
        assert gate_com_mode in ("learned", "open", "closed")
        self.gate_mem_mode = gate_mem_mode
        self.gate_com_mode = gate_com_mode

        self.encoder = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.spatial = nn.ModuleList([
            SimpleAttnMP(hidden, edge_dim=0, hidden=hidden) for _ in range(gnn_layers)
        ])
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

        # separate gate networks: no shared trunk, so mem/com gates can differentiate
        def _gate_head():
            head = nn.Sequential(
                nn.Linear(2 * hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            # start open so the pathway can learn before the cost bites
            nn.init.constant_(head[-1].bias, gate_bias_init)
            return head
        self.gate_mem_mlp = _gate_head()
        self.gate_com_mlp = _gate_head()

        self.W_mem = nn.Linear(hidden, hidden)
        self.W_com = nn.Linear(hidden, hidden)
        self.fuse_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))

    def _build_batched_edges(self, edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
        Ei = edge_index
        offsets = torch.arange(B, device=Ei.device, dtype=Ei.dtype) * N
        src = Ei[0].unsqueeze(0) + offsets.unsqueeze(1)
        dst = Ei[1].unsqueeze(0) + offsets.unsqueeze(1)
        return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

    def _gate(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (g_used, g_soft). Training: Gumbel-sigmoid + straight-through."""
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            noisy = (logits + torch.log(u) - torch.log1p(-u)) / max(self.gate_temp, 1e-3)
            g_soft = torch.sigmoid(noisy)
            g_hard = (g_soft > 0.5).float()
            return g_hard + g_soft - g_soft.detach(), torch.sigmoid(logits)
        g_soft = torch.sigmoid(logits)
        return (g_soft > 0.5).float(), g_soft

    def forward(
        self,
        X_seq: torch.Tensor,                 # [B,T,N,O]
        edge_index: torch.Tensor,            # [2,E]
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        B, T, N, O = X_seq.shape
        Ei = edge_index.to(X_seq.device)
        Ei_b = self._build_batched_edges(Ei, B, N)

        e_seq = self.encoder(X_seq)                                    # [B,T,N,H]

        # spatial branch per timestep on local embeddings
        m_time = []
        for t in range(T):
            z = e_seq[:, t].reshape(B * N, self.hidden)
            for mp in self.spatial:
                z = mp(z, Ei_b, edge_attr=None)
            m_time.append(z.reshape(B, N, self.hidden).unsqueeze(1))
        m_seq = torch.cat(m_time, dim=1)                               # [B,T,N,H]

        # temporal branch: LSTM over local embeddings, per node
        e_bn = e_seq.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden)
        h_seq_bn, new_hidden = self.lstm(e_bn, hidden)                 # [B*N,T,H]
        h_seq = h_seq_bn.reshape(B, N, T, self.hidden).permute(0, 2, 1, 3)  # [B,T,N,H]

        # h_{t-1} per step for the gate input (initial from provided hidden or zeros)
        if hidden is not None:
            h_init = hidden[0].reshape(B, N, self.hidden).unsqueeze(1)  # [B,1,N,H]
        else:
            h_init = torch.zeros(B, 1, N, self.hidden, device=X_seq.device)
        h_prev = torch.cat([h_init, h_seq[:, :-1]], dim=1)             # [B,T,N,H]

        gate_in = torch.cat([e_seq, h_prev], dim=-1)
        logits = torch.cat([self.gate_mem_mlp(gate_in), self.gate_com_mlp(gate_in)], dim=-1)  # [B,T,N,2]
        g_used, g_soft = self._gate(logits)
        # ablation overrides (also reflected in g_soft so logging/penalties see them)
        for idx, mode in ((0, self.gate_mem_mode), (1, self.gate_com_mode)):
            if mode != "learned":
                const = 1.0 if mode == "open" else 0.0
                g_used = g_used.clone(); g_soft = g_soft.clone()
                g_used[..., idx] = const
                g_soft[..., idx] = const
        g_mem, g_com = g_used[..., 0:1], g_used[..., 1:2]

        z = self.fuse_norm(e_seq + g_mem * self.W_mem(h_seq) + g_com * self.W_com(m_seq))
        return self.head(z), new_hidden, g_soft                        # [B,T,N,A], (h,c), [B,T,N,2]

    @torch.no_grad()
    def step(
        self,
        X_t: torch.Tensor,                   # [B,N,O]
        edge_index: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        q_seq, new_hidden, g_soft = self.forward(X_t.unsqueeze(1), edge_index, hidden)
        return q_seq[:, 0], new_hidden, g_soft[:, 0]                   # [B,N,A], (h,c), [B,N,2]

# ============== Communication-gated sequential GNN->LSTM (AAMAS, scalable) ==============
class CommGatedGNNLSTMQ(nn.Module):
    """Sequential GNN->LSTM backbone (composes spatial then temporal, so memory
    operates on spatially-aggregated features) with a per-agent, per-step
    COMMUNICATION gate that masks neighbour aggregation, and a memory-read gate.

    At g_com = g_mem = 1 this reduces exactly to GNNLSTMPolicyQ, so removing the
    usage penalty recovers the strong spatio-temporal baseline (the property the
    parallel-fusion GatedSpatioTemporalQ lacked at scale).

    Communication is the costed coordination resource (message rate). The memory
    gate is retained as a learned/forceable robustness--interpretability probe.

    forward() returns Q [B,T,N,A], hidden, and soft gates [B,T,N,2] with
    index 0 = memory, index 1 = communication (matches dqn_update_shared_gated).
    """
    def __init__(self, node_dim: int, actions: int, hidden: int = 128,
                 gnn_layers: int = 2, gate_temp: float = 1.0, gate_bias_init: float = 1.0,
                 gate_mem_mode: str = "learned", gate_com_mode: str = "learned"):
        super().__init__()
        self.hidden = hidden
        self.actions = actions
        self.gate_temp = gate_temp
        assert gate_mem_mode in ("learned", "open", "closed")
        assert gate_com_mode in ("learned", "open", "closed")
        self.gate_mem_mode = gate_mem_mode
        self.gate_com_mode = gate_com_mode

        self.gnn_layers = nn.ModuleList([
            SimpleAttnMP(node_dim if i == 0 else hidden, edge_dim=0, hidden=hidden)
            for i in range(gnn_layers)
        ])
        self.pre_lstm_norm = nn.LayerNorm(hidden)
        self.lstm = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, actions))

        # communication gate from local observation (batchable, no recurrent coupling)
        self.comm_gate_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        nn.init.constant_(self.comm_gate_mlp[-1].bias, gate_bias_init)
        # memory-read gate from [spatial features, lstm output]
        self.mem_gate_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))
        nn.init.constant_(self.mem_gate_mlp[-1].bias, gate_bias_init)

    def _build_batched_edges(self, edge_index, B, N):
        Ei = edge_index
        offsets = torch.arange(B, device=Ei.device, dtype=Ei.dtype) * N
        src = Ei[0].unsqueeze(0) + offsets.unsqueeze(1)
        dst = Ei[1].unsqueeze(0) + offsets.unsqueeze(1)
        return torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

    def _gate(self, logits, mode):
        if mode == "open":
            g = torch.ones_like(logits)
            return g, g
        if mode == "closed":
            g = torch.zeros_like(logits)
            return g, g
        if self.training:
            u = torch.rand_like(logits).clamp(1e-6, 1 - 1e-6)
            noisy = (logits + torch.log(u) - torch.log1p(-u)) / max(self.gate_temp, 1e-3)
            g_soft = torch.sigmoid(noisy)
            g_hard = (g_soft > 0.5).float()
            return g_hard + g_soft - g_soft.detach(), torch.sigmoid(logits)
        g_soft = torch.sigmoid(logits)
        return (g_soft > 0.5).float(), g_soft

    def forward(self, X_seq, edge_index, hidden=None):
        B, T, N, O = X_seq.shape
        Ei_b = self._build_batched_edges(edge_index.to(X_seq.device), B, N)

        com_logits = self.comm_gate_mlp(X_seq)                       # [B,T,N,1]
        g_com_used, g_com_soft = self._gate(com_logits, self.gate_com_mode)

        z_time = []
        for t in range(T):
            x_t = X_seq[:, t].reshape(B * N, O)
            gcom_t = g_com_used[:, t].reshape(B * N)                 # [B*N]
            z = x_t
            for mp in self.gnn_layers:
                z = mp(z, Ei_b, edge_attr=None, comm_gate=gcom_t)
            z_time.append(z.reshape(B, N, self.hidden).unsqueeze(1))
        z_seq = self.pre_lstm_norm(torch.cat(z_time, dim=1))          # [B,T,N,H]

        z_bn = z_seq.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden)
        lstm_bn, new_hidden = self.lstm(z_bn, hidden)
        lstm_out = lstm_bn.reshape(B, N, T, self.hidden).permute(0, 2, 1, 3)  # [B,T,N,H]

        mem_logits = self.mem_gate_mlp(torch.cat([z_seq, lstm_out], dim=-1))  # [B,T,N,1]
        g_mem_used, g_mem_soft = self._gate(mem_logits, self.gate_mem_mode)

        y = (1.0 - g_mem_used) * z_seq + g_mem_used * lstm_out
        q = self.head(y)                                             # [B,T,N,A]
        g_soft = torch.cat([g_mem_soft, g_com_soft], dim=-1)         # [B,T,N,2]
        return q, new_hidden, g_soft

    @torch.no_grad()
    def step(self, X_t, edge_index, hidden=None):
        q, new_hidden, g_soft = self.forward(X_t.unsqueeze(1), edge_index, hidden)
        return q[:, 0], new_hidden, g_soft[:, 0]
