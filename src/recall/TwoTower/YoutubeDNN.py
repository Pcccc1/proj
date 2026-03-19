from __future__ import annotations

from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        layers = []
        prev = input_dim
        for hid in hidden_dims:
            layers.append(nn.Linear(prev, hid))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = hid
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class YoutubeDNN(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 256,
        user_hidden_dims: Iterable[int] = (512, 256),
        item_hidden_dims: Iterable[int] = (512, 256),
        output_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.05,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if num_items <= 1:
            raise ValueError("`num_items` must be > 1 (include pad idx 0 and at least 1 real item).")
        
        self.temperature = temperature
        self.pad_idx = pad_idx
        self.item_embedding = nn.Embedding(num_items, embedding_dim, padding_idx=pad_idx)
        self.embedding_dim = embedding_dim
        self.user_tower = MLP(
            input_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
        self.item_tower = MLP(
            input_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )


    def _init_weights(self):
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.pad_idx].fill_(0.0)

    def init_item_embedding_from_content(
        self,
        item2idx: dict[int, int],
        item_content_dict: dict[int, np.ndarray],
    ) -> None:
        with torch.no_grad():
            hit_cnt = 0
            for item_id, idx in item2idx.items():
                if idx == self.pad_idx:
                    continue
                if item_id not in item_content_dict:
                    continue
                vec = item_content_dict[item_id]
                if isinstance(vec, np.ndarray):
                    vec = torch.from_numpy(vec)
                elif isinstance(vec, list):
                    vec = torch.tensor(vec, dtype=torch.float32)
                vec = vec.float().view(-1)

                if vec.numel() != self.embedding_dim:
                    raise ValueError(f"Content vector dimension {vec.numel()} does not match embedding dimension {self.embedding_dim}.")
                self.item_embedding.weight[idx].copy_(vec)
                hit_cnt += 1

            self.item_embedding.weight[self.pad_idx].fill_(0.0)
        print(f"Initialized item embedding from content for {hit_cnt}/{len(item2idx)} items.")



    def encode_user(self, hist_item_ids: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        item_vec = self.item_embedding(hist_item_ids) # (B, L, E)
        mask = hist_mask.unsqueeze(-1) # (B, L, 1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (item_vec * mask).sum(dim=1) / denom # (B, E)
        user_vec = self.user_tower(pooled) # (B, D)
        return F.normalize(user_vec, p=2, dim=-1)
    
    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_vec = self.item_embedding(item_ids) # (N, E)
        item_vec = self.item_tower(item_vec) # (N, D)
        return F.normalize(item_vec, p=2, dim=-1)
    
    def forward(self, 
        hist_item_ids: torch.Tensor,
        hist_mask: torch.Tensor,
        pos_item_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.encode_user(hist_item_ids, hist_mask) # (B, D)
        item_vec = self.encode_item(pos_item_ids) # (B, D)
        logits = torch.matmul(user_vec, item_vec.transpose(0, 1)) / self.temperature # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        return logits, labels
    