"""
Multi-Head Attention:

This module implements the Multi-Head Attention mechanism introduced in the
original Transformer paper.

Instead of performing a single attention operation, the input is projected
into multiple subspaces (heads). Each head independently performs Scaled
Dot-Product Attention, after which all heads are concatenated and projected
back to the model dimension.

Mathematical approach : 

For each attention head i,

    head_i = Attention(Q_i, K_i, V_i)

where

    Q_i = XW_i^Q
    K_i = XW_i^K
    V_i = XW_i^V

The outputs from all heads are concatenated

    MultiHead = Concat(head₁, head₂, ..., head_h)

Finally,

    Output = MultiHead W^O + b^O


    

------

Input
    Q : (B, L_q, D)
    K : (B, L_k, D)
    V : (B, L_k, D)

Output
    O : (B, L_q, D)

where

    B : Batch size
    L : Sequence length
    D : Model dimension
    H : Number of heads
    d_k = D / H

Backward:

Gradients propagate through

Output Projection
        ↓
Head Concatenation
        ↓
Scaled Dot-Product Attention
        ↓
Linear Projections
        ↓
Input



Used in

- Encoder Self-Attention
- Decoder Masked Self-Attention
- Encoder-Decoder Cross Attention
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from attention.Scaled_Dot_Product import ScaledDotProductAttention
from layers.linear import Linear

Array = NDArray[np.float64]


class MultiHeadAttention:
    """
    Multi-Head Attention.
    """

    def __init__(self, model_dim: int, num_heads: int) -> None:
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

    def _split_heads(self, x: Array) -> Array:
        batch_size, seq_len, _ = x.shape

        x = x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return np.transpose(x, (0, 2, 1, 3))

    def _combine_heads(self, x: Array) -> Array:
        batch_size, _, seq_len, _ = x.shape

        x = np.transpose(x, (0, 2, 1, 3))

        return x.reshape(
            batch_size,
            seq_len,
            self.model_dim,
        )

    def forward(
        self,
        q: Array,
        k: Array,
        v: Array,
    ) -> Array:
        """
        Multi-Head Attention.

        We use three params : q -> ndarray , k -> ndarray , v -> ndarray
        q : ndarray
            Query tensor of shape (B, L_q, D)

        k : ndarray
            Key tensor of shape (B, L_k, D)

        v : ndarray
            Value tensor of shape (B, L_k, D)

        it will return the output tensor of shape (B, L_q, D)
            
        """
        if q.shape[-1] != self.model_dim:
            raise ValueError("Invalid query dimension.")

        if k.shape[-1] != self.model_dim:
            raise ValueError("Invalid key dimension.")

        if v.shape[-1] != self.model_dim:
            raise ValueError("Invalid value dimension.")

        q = self.q_proj.forward(q)
        k = self.k_proj.forward(k)
        v = self.v_proj.forward(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        context = self.attention.forward(q, k, v)

        context = self._combine_heads(context)

        return self.out_proj.forward(context)
    

    def backward(
        self,
        grad_output: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Returns a tuple : (d_q, d_k, d_v) where each is the gradient of the loss with respect to q, k, and v respectively.
        """
        grad_context = self.out_proj.backward(grad_output)

        grad_context = self._split_heads(grad_context)

        grad_q, grad_k, grad_v = self.attention.backward(
            grad_context
        )

        grad_q = self._combine_heads(grad_q)
        grad_k = self._combine_heads(grad_k)
        grad_v = self._combine_heads(grad_v)

        d_q = self.q_proj.backward(grad_q)
        d_k = self.k_proj.backward(grad_k)
        d_v = self.v_proj.backward(grad_v)

        return d_q, d_k, d_v
    

    def zero_grad(self) -> None:
        """
        Reset parameter gradients.
        """
        self.q_proj.zero_grad()
        self.k_proj.zero_grad()
        self.v_proj.zero_grad()
        self.out_proj.zero_grad()
        self.attention.zero_grad()
        

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
    ) -> Array:
        return self.forward(q, k, v)