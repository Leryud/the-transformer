import math

import torch
import torch.nn as nn


def clones(module, N):
    """Produces N identical copies of the input layer"""


class LayerNorm(nn.Module):
    """Constructs a layernorm module"""

    def __init__(self, features: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdims=True)
        return (self.a_2 * (x - mean) / (std + self.eps)) + self.b_2


class SkipConnection(nn.Module):
    """A skip connection followed by a LayerNorm"""

    def __init__(self, size, dropout):
        super(SkipConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, skiplayer):
        """Apply residual connection to any skiplayer with similar size"""
        return x + self.dropout(skiplayer(self.norm(x)))


class Encoder(nn.Module):
    """Core encoder is a stack of N Layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.sizes)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attention and feed-forward"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.skiplayer = clones(SkipConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.skiplayer[0](
            x, lambda x: self.self_attn(x, x, x, mask)
        )  # key, query, value
        return self.skiplayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """Generic decoder with N layers and masking"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self attention, masked multihead attention and feed forward"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.skiplayer = clones(SkipConnection(size, dropout))

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions"""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)

    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Takes in model size and number of heads"""
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # Assuming d_v is always d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # The same mask is applied on all heads
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # 1. Apply the linear projections in the batch:
        # d_model => h x d_k
        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ] # `view` returns a new tensor with same data but different shape

        # 2. Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. Concatenate outputs using a view and apply a final linear
        x = (
            x.transpose(1, 2)
            .contiguous() # returns a tensor with same data as self, or self if the tensor is in the correct memory format
            .view(n_batches, -1, self.h * self.d_k)
        )

        # Free memory
        del query
        del key
        del value

        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """FFN equation : FFN(x) = max(0, xW_1 + b1)W_2 + b_2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # xW_1 + b_1
        self.w_2 = nn.Linear(d_ff, d_model) # xW_2 + b_2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # relu is max(0, w_1)
        return self.w_2(self.dropout(self.w_1(x).relu()))

