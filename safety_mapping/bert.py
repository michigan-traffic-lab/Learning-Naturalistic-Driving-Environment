""" Transformer Model Classes & Config Class """

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)



class Config():
    "Configuration for BERT model"
    dim: int = 128  # Dimension of Hidden Layer in Transformer Encoder
    n_layers: int = 8  # Numher of Hidden Layers
    n_heads: int = 8  # Numher of Heads in Multi-Headed Attention Layers
    dim_ff: int = 512  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    p_drop_hidden: float = 0  # Probability of Dropout of various Hidden Layers
    p_drop_attn: float = 0  # Probability of Dropout of Attention Layers
    max_len: int = 128  # Maximum Length for Positional Embeddings
    max_norm: float = 1

    def __str__(self):
        string = ""
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                string += attr + " = " + str(getattr(self, attr)) + "\n"
        return string



def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GeLU(nn.Module):
    def __init__(self):
        super(GeLU, self).__init__()

    def forward(self, x):
        return gelu(x)

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim, max_norm=cfg.max_norm)

        self.norm = LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        seq_len = x.size(1)

        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), -1)
        pos = self.pos_embed(pos)
        x += pos
    
        return self.drop(self.norm(x))


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg.dim)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


