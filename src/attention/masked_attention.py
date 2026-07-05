"""
Masked Self-Attention : 

This module implements Masked Multi-Head Self-Attention used in the Transformer
decoder.

Unlike encoder self-attention, decoder self-attention must prevent each token
from attending to future tokens during training. This is achieved by applying
a causal mask before the Softmax operation.

Mathematical approach : 

Given

    Scores = QKᵀ / √d_k

Apply the causal mask

                  {
MaskedScores(i,j) {
                  { Scores(i,j),    if j ≤ i
                  {
                  { -∞,             otherwise

Attention weights are then computed as

    A = Softmax(MaskedScores)

Finally,

    Output = AV

Backward : 

Gradients do not propagate through masked positions since their probability
after Softmax is effectively zero.


------

Input

Q : (B, L, D)
K : (B, L, D)
V : (B, L, D)

Mask

M : (L, L)

Output

O : (B, L, D)

Applications in Transformer : 

Used only inside the decoder.

Decoder Input
      ↓
Masked Multi-Head Attention
      ↓
Residual Connection
      ↓
Layer Normalization
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from layers.linear import Linear
from layers.softmax import Softmax

Array = NDArray[np.float64]


class MaskedAttention:
    """
    Masked Multi-Head Self-Attention.
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
        self.scale = np.sqrt(self.head_dim)

        self.q_proj = Linear(model_dim, model_dim)
        self.k_proj = Linear(model_dim, model_dim)
        self.v_proj = Linear(model_dim, model_dim)
        self.out_proj = Linear(model_dim, model_dim)

        self.softmax = Softmax(axis=-1)

        self.q: Optional[Array] = None
        self.k: Optional[Array] = None
        self.v: Optional[Array] = None
        self.attention: Optional[Array] = None

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

    @staticmethod
    def _causal_mask(length: int) -> Array:
        """
        Create a lower-triangular causal mask.
        """
        return np.tril(
            np.ones((length, length), dtype=np.float64)
        )

    def forward(
        self,
        q: Array,
        k: Array,
        v: Array,
    ) -> Array:
        """
        Compute masked multi-head self-attention.
        """
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(
                "Q, K and V must have identical shapes."
            )

        q = self.q_proj.forward(q)
        k = self.k_proj.forward(k)
        v = self.v_proj.forward(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = (
            np.matmul(q, np.swapaxes(k, -2, -1))
            / self.scale
        )

        seq_len = scores.shape[-1]

        mask = self._causal_mask(seq_len)

        scores = np.where(
            mask == 1,
            scores,
            -1e9,
        )

        attention = self.softmax.forward(scores)

        context = np.matmul(attention, v)

        self.q = q
        self.k = k
        self.v = v
        self.attention = attention

        context = self._combine_heads(context)

        return self.out_proj.forward(context)
    

    def backward(
        self,
        grad_output: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Backpropagate through masked attention.
        """
        if (
            self.q is None
            or self.k is None
            or self.v is None
            or self.attention is None
        ):
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        grad_context = self.out_proj.backward(grad_output)

        grad_context = self._split_heads(grad_context)

        d_attention = np.matmul(
            grad_context,
            np.swapaxes(self.v, -2, -1),
        )

        d_v = np.matmul(
            np.swapaxes(self.attention, -2, -1),
            grad_context,
        )

        d_scores = self.softmax.backward(d_attention)

        d_q = (
            np.matmul(d_scores, self.k)
            / self.scale
        )

        d_k = (
            np.matmul(
                np.swapaxes(d_scores, -2, -1),
                self.q,
            )
            / self.scale
        )

        d_q = self._combine_heads(d_q)
        d_k = self._combine_heads(d_k)
        d_v = self._combine_heads(d_v)

        d_q = self.q_proj.backward(d_q)
        d_k = self.k_proj.backward(d_k)
        d_v = self.v_proj.backward(d_v)

        return d_q, d_k, d_v
    

    def zero_grad(self) -> None:
        """
        Reset parameter gradients.
        """
        self.q_proj.zero_grad()
        self.k_proj.zero_grad()
        self.v_proj.zero_grad()
        self.out_proj.zero_grad()
        self.softmax.zero_grad()

        

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
    ) -> Array:
        return self.forward(q, k, v)