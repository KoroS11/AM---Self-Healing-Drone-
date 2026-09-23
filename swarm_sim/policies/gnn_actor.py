from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorGNN(nn.Module):
    """Centralized Training Decentralized Execution (CTDE) Local k-hop Actor GNN.
    
    Predicts continuous 2D control force acceleration vectors for active mobile UAVs
    using k-hop local graph message passing (k=1 default). Supports both deterministic
    point estimation (for BC/DAgger and evaluation) and stochastic Gaussian distribution
    sampling with log-probability and entropy computation (for MAPPO RL).
    """
    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 5,
        hidden_dim: int = 64,
        k_hops: int = 1,
        max_accel: float = 5.0,
        init_log_std: float = -0.5
    ):
        super().__init__()
        self.k_hops = k_hops
        self.hidden_dim = hidden_dim
        self.max_accel = max_accel
        
        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # Message passing layers
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(k_hops)
        ])
        
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(k_hops)
        ])
        
        # Action head predicting 2D acceleration mean (ax, ay)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Trainable log standard deviation for continuous Gaussian exploration in MAPPO
        self.log_std = nn.Parameter(torch.ones(2) * init_log_std)

    def _compute_mean(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        mobile_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute deterministic mean action vectors mu(s) for all nodes."""
        N = node_feats.shape[0]
        device = node_feats.device
        
        h = self.node_encoder(node_feats)  # (N, hidden_dim)
        
        if edge_index.shape[1] > 0:
            e = self.edge_encoder(edge_feats)  # (E, hidden_dim // 2)
            src, dst = edge_index[0], edge_index[1]
            
            for hop in range(self.k_hops):
                msg_input = torch.cat([h[src], h[dst], e], dim=-1)
                msgs = self.msg_mlps[hop](msg_input)
                
                # Aggregate messages via scatter add
                agg_msgs = torch.zeros(N, self.hidden_dim, device=device)
                agg_msgs.index_add_(0, dst, msgs)
                
                # Update node representations
                h = self.update_mlps[hop](torch.cat([h, agg_msgs], dim=-1))
                
        raw_actions = self.action_head(h)  # (N, 2)
        clamped_actions = torch.tanh(raw_actions) * self.max_accel
        actions = clamped_actions * mobile_mask.unsqueeze(-1).float()
        return actions

    def forward(
        self,
        node_feats: torch.Tensor,   # Shape (N, node_in_dim)
        edge_feats: torch.Tensor,   # Shape (E, edge_in_dim)
        edge_index: torch.Tensor,   # Shape (2, E)
        mobile_mask: torch.Tensor   # Shape (N,) boolean mask
    ) -> torch.Tensor:
        """Deterministic forward pass returning point estimate mean actions mu(s) clamped to max_accel."""
        return self._compute_mean(node_feats, edge_feats, edge_index, mobile_mask)

    def get_action(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        mobile_mask: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, compute joint log-probability on pre-clamp sample, and return executed & raw actions.
        
        Returns:
            actions: (N, 2) executed continuous action tensor (clamped to [-max_accel, max_accel])
            raw_sampled: (N, 2) pre-clamp sampled action from Normal(mu, std)
            step_log_prob: Scalar joint log-probability evaluated on pre-clamp sample
            step_entropy: Scalar joint entropy of active mobile agent action distributions
        """
        mu = self._compute_mean(node_feats, edge_feats, edge_index, mobile_mask)
        std = torch.exp(self.log_std).to(mu.device)
        dist = torch.distributions.Normal(mu, std)
        
        if deterministic:
            raw_sampled = mu
            actions = mu
        else:
            raw_sampled = dist.sample()
            actions = torch.clamp(raw_sampled, -self.max_accel, self.max_accel)
            actions = actions * mobile_mask.unsqueeze(-1).float()
            
        # Compute log-probability on pre-clamp raw sample across mobile active nodes
        log_probs_all = dist.log_prob(raw_sampled)  # (N, 2) evaluated at pre-clamp sample!
        entropy_all = dist.entropy()                # (N, 2)
        
        # Sum over action dims (2D: ax, ay) and active mobile agents (N)
        active_bool = mobile_mask.bool()
        if active_bool.sum() > 0:
            step_log_prob = log_probs_all[active_bool].sum()
            step_entropy = entropy_all[active_bool].sum()
        else:
            step_log_prob = torch.tensor(0.0, device=mu.device)
            step_entropy = torch.tensor(0.0, device=mu.device)
            
        return actions, raw_sampled, step_log_prob, step_entropy

    def evaluate_actions(
        self,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        mobile_mask: torch.Tensor,
        raw_actions: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log-probability and entropy for given pre-clamp actions under current policy theta.
        
        Returns:
            step_log_prob: Scalar or (B,) joint log-probability of active mobile agents
            step_entropy: Scalar or (B,) joint entropy of active mobile agent action distributions
        """
        mu = self._compute_mean(node_feats, edge_feats, edge_index, mobile_mask)
        std = torch.exp(self.log_std).to(mu.device)
        dist = torch.distributions.Normal(mu, std)
        
        log_probs_all = dist.log_prob(raw_actions)  # (N, 2) evaluated at pre-clamp sample
        entropy_all = dist.entropy()                # (N, 2)
        
        active_bool = mobile_mask.bool()
        if batch_idx is None:
            if active_bool.sum() > 0:
                step_log_prob = log_probs_all[active_bool].sum()
                step_entropy = entropy_all[active_bool].sum()
            else:
                step_log_prob = torch.tensor(0.0, device=mu.device)
                step_entropy = torch.tensor(0.0, device=mu.device)
            return step_log_prob, step_entropy
        else:
            B = int(batch_idx.max().item()) + 1
            step_log_prob = torch.zeros(B, device=mu.device)
            step_entropy = torch.zeros(B, device=mu.device)
            if active_bool.sum() > 0:
                active_b = batch_idx[active_bool]
                step_log_prob.index_add_(0, active_b, log_probs_all[active_bool].sum(dim=-1))
                step_entropy.index_add_(0, active_b, entropy_all[active_bool].sum(dim=-1))
            return step_log_prob, step_entropy
