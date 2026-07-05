"""
Scaled Dot-Product Attention : 

This module implements the Scaled Dot-Product Attention mechanism introduced
in the original Transformer paper:

    "Attention Is All You Need"
    Vaswani et al., 2017

Scaled Dot-Product Attention computes attention weights by comparing Query (Q)
vectors against Key (K) vectors, normalizing the similarity scores using the
Softmax function, and using the resulting attention distribution to compute a
weighted combination of the Value (V) vectors.

Mathematical approach : 

Forward

              QKᵀ
S = -----------------
        √d_k

A = Softmax(S)

Output = AV

where

Q -> Query Matrix
K -> Key Matrix
V -> Value Matrix

Backward

Given

dO = ∂L/∂Output

Gradients propagate as

dA = dO Vᵀ

dV = Aᵀ dO

dS = SoftmaxBackward(dA)

dQ = (dS K) / √d_k

dK = (dSᵀ Q) / √d_k


------
Input

Q : (B, H, L_q, d_k)
K : (B, H, L_k, d_k)
V : (B, H, L_k, d_v)

Scores

S : (B, H, L_q, L_k)

Attention

A : (B, H, L_q, L_k)

Output

O : (B, H, L_q, d_v)

Applications in Transformer : 

This module is the mathematical core of

- Self Attention
- Masked Self Attention
- Cross Attention
- Multi Head Attention

It performs only the attention computation itself. Linear projections,
head splitting, concatenation, masking and output projections are handled
by higher-level modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from layers.softmax import Softmax

Array = NDArray[np.float64]


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention.
    """

    def __init__(self) -> None:
        self.softmax = Softmax(axis=-1)

        self.q: Optional[Array] = None
        self.k: Optional[Array] = None
        self.v: Optional[Array] = None
        self.attention: Optional[Array] = None

        self.scale: Optional[float] = None


    def forward(self,q: Array,k: Array,v: Array,) -> Array:
        """
        Compute scaled dot-product attention.

        We use Three params : q -> ndarray , k -> ndarray , v -> ndarray
        q : ndarray
            Query tensor of shape
            (batch, heads, query_length, head_dim)

        k : ndarray
            Key tensor of shape
            (batch, heads, key_length, head_dim)

        v : ndarray
            Value tensor of shape
            (batch, heads, key_length, value_dim)

        """
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "Q, K and V must all be 4-dimensional."
            )

        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise ValueError("Batch sizes must match.")

        if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
            raise ValueError("Number of heads must match.")

        if k.shape[-2] != v.shape[-2]:
            raise ValueError(
                "Key and Value sequence lengths must match."
            )

        if q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "Query and Key dimensions must match."
            )

        d_k = q.shape[-1]
        self.scale = np.sqrt(d_k)

        scores = np.matmul(
            q,
            np.swapaxes(k, -2, -1),
        ) / self.scale

        attention = self.softmax.forward(scores)

        output = np.matmul(attention, v)

        self.q = q
        self.k = k
        self.v = v
        self.attention = attention

        return output

    def backward(
        self,
        grad_output: Array,
    ) -> Tuple[Array, Array, Array]:
        """
        Backpropagate through scaled dot-product attention.

        Returns a tuple : (dQ, dK, dV) of gradients with respect to the Query, Key, and Value inputs.
        """
        if (
            self.q is None
            or self.k is None
            or self.v is None
            or self.attention is None
            or self.scale is None
        ):
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        d_attention = np.matmul(
            grad_output,
            np.swapaxes(self.v, -2, -1),
        )

        d_v = np.matmul(
            np.swapaxes(self.attention, -2, -1),
            grad_output,
        )

        d_scores = self.softmax.backward(d_attention)

        d_q = (
            np.matmul(
                d_scores,
                self.k,
            )
            / self.scale
        )

        d_k = (
            np.matmul(
                np.swapaxes(d_scores, -2, -1),
                self.q,
            )
            / self.scale
        )

        return d_q, d_k, d_v
    


    def zero_grad(self) -> None:
        """
        This module has no trainable parameters.
        """
        self.softmax.zero_grad()
        

    def __call__(
        self,
        q: Array,
        k: Array,
        v: Array,
    ) -> Array:
        return self.forward(q, k, v)