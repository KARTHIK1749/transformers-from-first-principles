"""
Cross Attention : 

This module implements Multi-Head Cross Attention used in the Transformer
decoder.

Unlike Self Attention, Cross Attention receives its Queries (Q) from the
decoder while the Keys (K) and Values (V) originate from the encoder output.
This mechanism allows the decoder to attend over the encoded source sequence
during generation.

Mathematical approach : 

Given

Decoder Hidden States

    X_d ∈ R^(B × L_t × D)

Encoder Output

    X_e ∈ R^(B × L_s × D)

Linear Projections

    Q = X_d W_Q + b_Q

    K = X_e W_K + b_K

    V = X_e W_V + b_V

Scaled Dot Product Attention

                QKᵀ
S = ----------------------------
          √d_k

A = Softmax(S)

Head = AV

Multi-head concatenation

    MultiHead = Concat(head₁,...,head_h)

Output Projection

    Y = MultiHead W_O + b_O



Backward:

Gradients propagate through

Output Projection
        ↓
Attention
        ↓
Q Projection
        ↓
Decoder Input

and simultaneously

K Projection
        ↓
Encoder Output

V Projection
        ↓
Encoder Output

Since both K and V originate from the encoder, their gradients are summed
before being propagated back to the encoder.



------
Decoder Input

    X_d : (B, L_t, D)

Encoder Output

    X_e : (B, L_s, D)

Output

    Y : (B, L_t, D)

where

B   : Batch Size
L_t : Target Sequence Length
L_s : Source Sequence Length
D   : Model Dimension

Applications : 

Used only inside the Transformer Decoder.

Masked Self Attention
          ↓
Cross Attention
          ↓
Feed Forward Network
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from attention.multi_head_attention import MultiHeadAttention

Array = NDArray[np.float64]


class CrossAttention:
    """
    Multi-Head Cross Attention.

    Queries are generated from the decoder input while Keys and Values are
    generated from the encoder output.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
    ) -> None:
        self.attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )



    def forward(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        """
        Compute Cross Attention.

        Parameters
        ----------
        decoder_input : ndarray
            Shape (B, L_t, D)

        encoder_output : ndarray
            Shape (B, L_s, D)

        Returns
        -------
        ndarray
            Shape (B, L_t, D)
        """
        return self.attention.forward(
            q=decoder_input,
            k=encoder_output,
            v=encoder_output,
        )



    def backward(
        self,
        grad_output: Array,
    ) -> Tuple[Array, Array]:
        """
        Backpropagate through Cross Attention.

        Parameters
        ----------
        grad_output : ndarray
            Gradient with respect to the output.

        Returns
        -------
        tuple

        (
            d_decoder_input,
            d_encoder_output
        )
        """
        d_q, d_k, d_v = self.attention.backward(
            grad_output
        )

        d_encoder = d_k + d_v

        return d_q, d_encoder
    


    def zero_grad(self) -> None:
        """
        Reset parameter gradients.
        """
        self.attention.zero_grad()

        

    def __call__(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        return self.forward(
            decoder_input,
            encoder_output,
        )