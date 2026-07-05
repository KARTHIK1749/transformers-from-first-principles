"""
Decoder Block : 

This module implements a single Decoder Block from the original Transformer
architecture.

Each Decoder Block consists of

    Masked Multi-Head Self Attention
                ↓
        Residual Connection
                ↓
        Layer Normalization
                ↓
       Cross Multi-Head Attention
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

Given

X : Decoder Input
E : Encoder Output

Masked Self Attention

    M = MaskedAttention(X)

Residual

    R₁ = X + M

LayerNorm

    N₁ = LayerNorm(R₁)

Cross Attention

    C = CrossAttention(N₁, E)

Residual

    R₂ = N₁ + C

LayerNorm

    N₂ = LayerNorm(R₂)

Feed Forward Network

    F = W₂(ReLU(W₁(N₂)))

Residual

    R₃ = N₂ + F

LayerNorm

    Output = LayerNorm(R₃)

Backward : 

Gradients propagate in the reverse order

LayerNorm₃
      ↓
Residual₃
      ↓
Feed Forward
      ↓
LayerNorm₂
      ↓
Residual₂
      ↓
Cross Attention
      ↓
LayerNorm₁
      ↓
Residual₁
      ↓
Masked Self Attention

Shapes
------

Decoder Input

    X : (B, L_t, D)

Encoder Output

    E : (B, L_s, D)

Output

    Y : (B, L_t, D)

Applications : 

Multiple Decoder Blocks are stacked to build the Transformer Decoder.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from attention.cross_attention import CrossAttention
from attention.masked_attention import MaskedAttention
from layers.LayerNorm import LayerNorm
from layers.linear import Linear
from layers.ReLU import ReLU

Array = NDArray[np.float64]


class DecoderBlock:
    """
    Single Transformer Decoder Block.
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

        self.masked_attention = MaskedAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )

        self.norm1 = LayerNorm(model_dim)

        self.cross_attention = CrossAttention(
            model_dim=model_dim,
            num_heads=num_heads,
        )

        self.norm2 = LayerNorm(model_dim)

        self.ffn1 = Linear(model_dim, ff_dim)
        self.relu = ReLU()
        self.ffn2 = Linear(ff_dim, model_dim)

        self.norm3 = LayerNorm(model_dim)

    def forward(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        """
        Compute the forward pass of the decoder block.
        """
        masked = self.masked_attention(
            decoder_input,
            decoder_input,
            decoder_input,
        )

        x = self.norm1(decoder_input + masked)

        cross = self.cross_attention(
            x,
            encoder_output,
        )

        x = self.norm2(x + cross)

        ffn = self.ffn1(x)
        ffn = self.relu(ffn)
        ffn = self.ffn2(ffn)

        x = self.norm3(x + ffn)

        return x

    def backward(
        self,
        grad_output: Array,
    ) -> tuple[Array, Array]:
        """
        Backpropagate through the decoder block.

        Returns
        -------
        tuple
            (
                d_decoder_input,
                d_encoder_output
            )
        """
        grad = self.norm3.backward(grad_output)

        grad_ffn = grad
        grad_skip = grad

        grad_ffn = self.ffn2.backward(grad_ffn)
        grad_ffn = self.relu.backward(grad_ffn)
        grad_ffn = self.ffn1.backward(grad_ffn)

        grad = grad_skip + grad_ffn

        grad = self.norm2.backward(grad)

        grad_cross = grad
        grad_skip = grad

        grad_decoder_cross, grad_encoder = (
            self.cross_attention.backward(
                grad_cross
            )
        )

        grad = grad_skip + grad_decoder_cross

        grad = self.norm1.backward(grad)

        grad_masked = grad
        grad_skip = grad

        d_q, d_k, d_v = self.masked_attention.backward(
            grad_masked
        )

        grad_decoder = grad_skip + d_q + d_k + d_v

        return grad_decoder, grad_encoder
    

    def zero_grad(self) -> None:
        """
        Reset all parameter gradients.
        """
        self.masked_attention.zero_grad()
        self.cross_attention.zero_grad()
        self.norm1.zero_grad()
        self.norm2.zero_grad()
        self.norm3.zero_grad()
        self.ffn1.zero_grad()
        self.ffn2.zero_grad()


    def __call__(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        return self.forward(
            decoder_input,
            encoder_output,
        )