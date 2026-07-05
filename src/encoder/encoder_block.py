"""
Encoder Block :

This module implements a single Encoder Block from the original Transformer
architecture.

Each Encoder Block consists of

    Multi-Head Self Attention
            ↓
    Residual Connection
            ↓
      Layer Normalization
            ↓
    Position-wise Feed Forward Network
            ↓
    Residual Connection
            ↓
      Layer Normalization

Mathematical approach :

Given an input sequence X,

Self Attention

    A = MultiHeadAttention(X)

First Residual

    R₁ = X + A

First LayerNorm

    N₁ = LayerNorm(R₁)

Feed Forward Network

    F = W₂(ReLU(W₁(N₁)))

Second Residual

    R₂ = N₁ + F

Second LayerNorm

    Output = LayerNorm(R₂)

Backward:

Gradients propagate in the exact reverse order

LayerNorm₂
      ↓
Residual₂
      ↓
Feed Forward
      ↓
LayerNorm₁
      ↓
Residual₁
      ↓
Multi-Head Attention
      ↓
Input

------
Input

    X : (B, L, D)

Output

    Y : (B, L, D)

where

B : Batch Size
L : Sequence Length
D : Model Dimension

Applications : 

Multiple Encoder Blocks are stacked to build the Transformer Encoder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from attention.multi_head_attention import MultiHeadAttention
from layers.LayerNorm import LayerNorm
from layers.linear import Linear
from layers.ReLU import ReLU

Array = NDArray[np.float64]


class EncoderBlock:
    """
    Single Transformer Encoder Block.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
    ) -> None:
        if model_dim <= 0:
            raise ValueError("model_dim must be greater than zero.")

        if ff_dim <= 0:
            raise ValueError("ff_dim must be greater than zero.")

        self.self_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )

        self.norm1 = LayerNorm(model_dim)

        self.ffn1 = Linear(model_dim, ff_dim)
        self.relu = ReLU()
        self.ffn2 = Linear(ff_dim, model_dim)

        self.norm2 = LayerNorm(model_dim)

        self._residual1: Array | None = None
        self._residual2: Array | None = None



    def forward(self, x: Array) -> Array:
        
        attention = self.self_attention(x, x, x)

        self._residual1 = x + attention

        x = self.norm1(self._residual1)

        ffn = self.ffn1(x)
        ffn = self.relu(ffn)
        ffn = self.ffn2(ffn)

        self._residual2 = x + ffn

        return self.norm2(self._residual2)
    


    def backward(self, grad_output: Array) -> Array:
        
        grad = self.norm2.backward(grad_output)

        grad_ffn = grad
        grad_skip = grad

        grad_ffn = self.ffn2.backward(grad_ffn)
        grad_ffn = self.relu.backward(grad_ffn)
        grad_ffn = self.ffn1.backward(grad_ffn)

        grad = grad_skip + grad_ffn

        grad = self.norm1.backward(grad)

        grad_attention = grad
        grad_skip = grad

        d_q, d_k, d_v = self.self_attention.backward(grad_attention)

        grad = grad_skip + d_q + d_k + d_v

        return grad
    


    def zero_grad(self) -> None:
        """
        Reset all parameter gradients.
        """
        self.self_attention.zero_grad()
        self.norm1.zero_grad()
        self.ffn1.zero_grad()
        self.ffn2.zero_grad()
        self.norm2.zero_grad()

        

    def __call__(self, x: Array) -> Array:
        return self.forward(x)