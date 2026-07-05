"""
Self Attention: 

This module implements Multi-Head Self-Attention for the Transformer.

Unlike the Scaled Dot-Product Attention module, this layer is responsible for
learning the Query (Q), Key (K), and Value (V) projections from a single input
sequence before computing attention.

Mathematical approach:

Given an input sequence

    X ∈ R^(B × L × D)

Learnable projections

    Q = XW_Q + b_Q
    K = XW_K + b_K
    V = XW_V + b_V

where

    W_Q, W_K, W_V ∈ R^(D × D)

The projected tensors are reshaped into multiple attention heads

    Q, K, V
        ↓
(B, L, D)
        ↓
(B, H, L, d_k)

Scaled Dot-Product Attention is then applied independently to each head

                QKᵀ
S = --------------------------
          √d_k

A = Softmax(S)

Head = AV

Finally,

MultiHead = Concat(Head₁, ..., Headₕ)

Output Projection

    Y = MultiHead W_O + b_O

Backward
--------

Gradients flow in the reverse order

Output Projection
        ↓
Head Concatenation
        ↓
Scaled Dot-Product Attention
        ↓
Q, K, V Linear Projections
        ↓
Input



------

Input

X : (B, L, D)

Output

Y : (B, L, D)

Applications in Transformer:

Self Attention is used inside every Encoder block.

Encoder Block

Input
  ↓
Self Attention
  ↓
Residual
  ↓
LayerNorm
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from attention.Scaled_Dot_Product import ScaledDotProductAttention
from layers.linear import Linear

Array = NDArray[np.float64]

'''
 Constants we will use in the Attention Module :
 B = Batch Size
S = Sequence Length
D = Model Dimension
I = Input Dimension
O = Output Dimension
H = Number of Heads
d_k = Head Dimension
V = Vocabulary Size

'''


class SelfAttention:
    """
    Multi-Head Self Attention.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
    ) -> None:
        if model_dim <= 0:
            raise ValueError("model_dim must be greater than zero.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")

        if model_dim % num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads."
            )

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = Linear(model_dim, model_dim)
        self.k_proj = Linear(model_dim, model_dim)
        self.v_proj = Linear(model_dim, model_dim)
        self.out_proj = Linear(model_dim, model_dim)

        self.attention = ScaledDotProductAttention()

        self.batch_size: int | None = None
        self.sequence_length: int | None = None

    def _split_heads(self, x: Array) -> Array:
        """
        (B, L, D) -> (B, H, L, d_k)
        """
        batch, seq_len, _ = x.shape

        x = x.reshape(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: Array) -> Array:
        """
        (B, H, L, d_k) -> (B, L, D)
        """
        batch, _, seq_len, _ = x.shape

        x = np.transpose(x, (0, 2, 1, 3))

        return x.reshape(
            batch,
            seq_len,
            self.model_dim,
        )
    


    def forward(self, x: Array) -> Array:
        """
        Compute multi-head self-attention.

        """
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape (batch, sequence, model_dim)."
            )

        if x.shape[-1] != self.model_dim:
            raise ValueError(
                f"Expected model_dim={self.model_dim}, "
                f"received {x.shape[-1]}."
            )

        self.batch_size = x.shape[0]
        self.sequence_length = x.shape[1]

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        context = self.attention.forward(q, k, v)

        context = self._combine_heads(context)

        output = self.out_proj.forward(context)

        return output
    


    def backward(self, grad_output: Array) -> Array:
        
        grad_context = self.out_proj.backward(grad_output)

        grad_context = self._split_heads(grad_context)

        grad_q, grad_k, grad_v = self.attention.backward(grad_context)

        grad_q = self._combine_heads(grad_q)
        grad_k = self._combine_heads(grad_k)
        grad_v = self._combine_heads(grad_v)

        grad_input = (
            self.q_proj.backward(grad_q)
            + self.k_proj.backward(grad_k)
            + self.v_proj.backward(grad_v)
        )

        return grad_input
    


    def zero_grad(self) -> None:
        """
        Reset parameter gradients.
        """
        self.q_proj.zero_grad()
        self.k_proj.zero_grad()
        self.v_proj.zero_grad()
        self.out_proj.zero_grad()
        self.attention.zero_grad()

    

    def __call__(self, x: Array) -> Array:
        return self.forward(x)