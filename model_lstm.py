"""
LSTM-based End-of-Turn classifier.

Architecture:
  Input: (batch, SEQ_LEN, N_FRAME_FEATS) frame sequence
  → Bidirectional LSTM (2 layers)
  → Attention pooling over time
  → Concat with handcrafted scalar features (pause_index, pause_start, etc.)
  → MLP head → sigmoid → p_eot
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    """Soft attention over LSTM hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h, mask):
        # h: (B, T, H), mask: (B, T) bool
        scores = self.attn(h).squeeze(-1)          # (B, T)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)    # (B, T)
        return (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, H)


class EOTClassifier(nn.Module):
    def __init__(
        self,
        n_frame_feats=19,
        lstm_hidden=64,
        lstm_layers=2,
        n_scalar_feats=8,
        mlp_hidden=64,
        dropout=0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_frame_feats,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_pool = AttentionPool(lstm_hidden * 2)

        # project frame features before LSTM
        self.input_proj = nn.Sequential(
            nn.Linear(n_frame_feats, n_frame_feats),
            nn.LayerNorm(n_frame_feats),
            nn.ReLU(),
        )

        lstm_out_dim = lstm_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim + n_scalar_feats, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, seq, mask, scalar):
        """
        seq    : (B, T, F) float32
        mask   : (B, T) bool — True for real frames
        scalar : (B, S) float32
        Returns: (B,) logits
        """
        x = self.input_proj(seq)
        h, _ = self.lstm(x)                         # (B, T, 2*H)
        pooled = self.attn_pool(h, mask)             # (B, 2*H)
        combined = torch.cat([pooled, scalar], dim=-1)
        return self.head(combined).squeeze(-1)       # (B,)
