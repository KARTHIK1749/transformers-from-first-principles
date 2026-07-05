"""
Encoder : 

This module implements the Transformer Encoder by stacking multiple Encoder
Blocks.

The encoder transforms an input sequence into a contextual representation
through repeated applications of self-attention and feed-forward networks.

Architecture:

Input Embeddings
        ↓
Positional Encoding
        ↓
Encoder Block 1
        ↓
Encoder Block 2
        ↓
      ...
        ↓
Encoder Block N
        ↓
Encoder Output

Mathematical approach :

Given the input sequence

    X₀ = Input Embeddings + Positional Encoding

For each encoder block i,

    Xᵢ = EncoderBlockᵢ(Xᵢ₋₁)

The final encoder representation is

    Output = Xₙ

    
Backward:

Gradients propagate through the encoder blocks in reverse order

Encoder Block N
        ↓
Encoder Block N-1
        ↓
      ...
        ↓
Encoder Block 1
        ↓
Encoder Input


------
Input

    X : (B, L, D)

Output

    Y : (B, L, D)

where

B : Batch Size
L : Sequence Length
D : Model Dimension

Applications in Transformer:

The encoder output is consumed by the decoder through Cross Attention.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from encoder.encoder_block import EncoderBlock

Array = NDArray[np.float64]


class Encoder:
    """
    Transformer Encoder.
    """

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        ff_dim: int,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        self.num_layers = num_layers

        self.layers: List[EncoderBlock] = [
            EncoderBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
            )
            for _ in range(num_layers)
        ]

        

    def forward(self, x: Array) -> Array:
        
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad_output: Array) -> Array:
        
        grad = grad_output

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
    


    def zero_grad(self) -> None:
        """
        Reset gradients for every encoder block.
        """
        for layer in self.layers:
            layer.zero_grad()



    def __call__(self, x: Array) -> Array:
        return self.forward(x)