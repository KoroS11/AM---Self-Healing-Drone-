from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class CriticGNN(nn.Module):
    """Centralized Training Decentralized Execution (CTDE) Global Critic GNN.
    
    Processes full graph (X, E) with global pooling to output scalar team value estimate V(s).
    Preserves scalar output dimension regardless of node failure or variable active agent count.
    """
    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
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
        
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # Global pooling and value head
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # (N, node_in_dim)
        edge_feats: torch.Tensor,   # (E, edge_in_dim)
        edge_index: torch.Tensor,   # (2, E)
        batch_idx: Optional[torch.Tensor] = None # (N,) graph index for batched minibatches
    ) -> torch.Tensor:
        """Forward pass outputting scalar team value estimate V(s) or (B,) tensor for batched graphs."""
        N = node_feats.shape[0]
        device = node_feats.device
        
        h = self.node_encoder(node_feats)
        
        if edge_index.shape[1] > 0:
            e = self.edge_encoder(edge_feats)
            src, dst = edge_index[0], edge_index[1]
            
            for layer in range(self.num_layers):
                msg_input = torch.cat([h[src], h[dst], e], dim=-1)
                msgs = self.msg_mlps[layer](msg_input)
                
                agg_msgs = torch.zeros(N, self.hidden_dim, device=device)
                agg_msgs.index_add_(0, dst, msgs)
                
                h = self.update_mlps[layer](torch.cat([h, agg_msgs], dim=-1))
                
        # Global pooling: combine sum and mean pooling
        if batch_idx is None:
            h_sum = torch.sum(h, dim=0, keepdim=True)
            h_mean = torch.mean(h, dim=0, keepdim=True)
            graph_rep = torch.cat([h_sum, h_mean], dim=-1)  # (1, 2 * hidden_dim)
            value = self.value_head(graph_rep)  # (1, 1)
            return value.squeeze(-1)  # (1,)
        else:
            B = int(batch_idx.max().item()) + 1
            h_sum = torch.zeros(B, self.hidden_dim, device=device)
            h_sum.index_add_(0, batch_idx, h)
            ones = torch.ones(N, 1, device=device)
            node_counts = torch.zeros(B, 1, device=device).index_add_(0, batch_idx, ones).clamp(min=1.0)
            h_mean = h_sum / node_counts
            graph_rep = torch.cat([h_sum, h_mean], dim=-1)  # (B, 2 * hidden_dim)
            value = self.value_head(graph_rep)  # (B, 1)
            return value.squeeze(-1)  # (B,)
