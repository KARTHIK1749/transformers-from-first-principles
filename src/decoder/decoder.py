"""
Decoder : 

This module implements the Transformer Decoder by stacking multiple Decoder
Blocks.

The decoder generates contextual target representations by combining masked
self-attention over the target sequence with cross-attention over the encoder
output.

Architecture : 

Target Embeddings
        ↓
Positional Encoding
        ↓
Decoder Block 1
        ↓
Decoder Block 2
        ↓
      ...
        ↓
Decoder Block N
        ↓
Decoder Output

Each Decoder Block performs

Masked Self Attention
        ↓
Cross Attention
        ↓
Feed Forward Network

Mathematical approach : 

Let

    X₀ = Target Embeddings + Positional Encoding

and

    E = Encoder Output

Then

    Xᵢ = DecoderBlockᵢ(Xᵢ₋₁, E)

The final decoder representation is

    Output = Xₙ

Backward : 

Gradients propagate through the decoder blocks in reverse order.

For each block,

- gradients flow back to the decoder input
- gradients also flow back to the encoder output through Cross Attention

The encoder gradients from every decoder block are accumulated.



------

Decoder Input

    X : (B, L_t, D)

Encoder Output

    E : (B, L_s, D)

Decoder Output

    Y : (B, L_t, D)

Applications : 

The decoder output is typically projected into vocabulary logits using a
final Linear layer followed by Softmax.
"""

from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import NDArray

from decoder.decoder_block import DecoderBlock

Array = NDArray[np.float64]


class Decoder:
    """
    Transformer Decoder.
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

        self.layers: List[DecoderBlock] = [
            DecoderBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
            )
            for _ in range(num_layers)
        ]
        

    def forward(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        """
        Compute the forward pass through all decoder blocks.

        We use two params : decoder_input -> ndarray, encoder_output -> ndarray
        decoder_input : ndarray
            Shape (B, L_t, D)

        encoder_output : ndarray
            Shape (B, L_s, D)

        Returns
        -------
        ndarray
            Shape (B, L_t, D)
        """
        x = decoder_input

        for layer in self.layers:
            x = layer.forward(
                x,
                encoder_output,
            )

        return x
    

    def backward(
        self,
        grad_output: Array,
    ) -> tuple[Array, Array]:
        """
        Backpropagate through the decoder.

        Returns tuple of gradients with respect to the decoder input and encoder output.
        tuple : (d_decoder_input, d_encoder_output)
        """
        grad_decoder = grad_output

        grad_encoder = None

        for layer in reversed(self.layers):
            grad_decoder, encoder_grad = layer.backward(
                grad_decoder
            )

            if grad_encoder is None:
                grad_encoder = encoder_grad
            else:
                grad_encoder += encoder_grad

        return grad_decoder, grad_encoder
    

    def zero_grad(self) -> None:
        """
        Reset gradients for every decoder block.
        """
        for layer in self.layers:
            layer.zero_grad()


    def __call__(
        self,
        decoder_input: Array,
        encoder_output: Array,
    ) -> Array:
        return self.forward(
            decoder_input,
            encoder_output,
        )