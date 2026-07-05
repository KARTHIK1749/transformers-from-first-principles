"""
Transformer : 

This module implements the complete Transformer architecture introduced in

    "Attention Is All You Need"
    Vaswani et al., 2017

The Transformer consists of

    Source Tokens
            │
            ▼
      Token Embedding
            │
            ▼
  Positional Encoding
            │
            ▼
         Encoder
            │
            ▼
    Encoder Representation
            │
            ▼
      Cross Attention
            ▲
            │
 Target Embedding
            │
            ▼
 Positional Encoding
            │
            ▼
         Decoder
            │
            ▼
    Vocabulary Projection
            │
            ▼
         Logits

Mathematical approach :

Encoder

    E = Encoder(
            PE(
                Embedding(src)
            )
        )

Decoder

    D = Decoder(
            PE(
                Embedding(tgt)
            ),
            E
        )

Vocabulary Projection

    Logits = DW_vocab + b_vocab

    
Backward:

Gradients propagate in the reverse order

Vocabulary Projection
        ↓
Decoder
        ↓
Encoder
        ↓
Embeddings

The positional encodings are fixed and therefore do not receive gradients.

------

Source Tokens

    src : (B, Ls)

Target Tokens

    tgt : (B, Lt)

Output

    logits : (B, Lt, vocab_size)

Applications:

This module combines every individual component implemented in the repository
into a complete Transformer model using only NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from decoder.decoder import Decoder
from encoder.encoder import Encoder
from layers.Embedding import Embedding
from layers.linear import Linear
from layers.positional_encoding import PositionalEncoding

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Transformer:
    """
    Complete Transformer architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ff_dim: int,
        max_length: int = 5000,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        self.src_embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=model_dim,
        )

        self.tgt_embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=model_dim,
        )

        self.positional_encoding = PositionalEncoding(
            embedding_dim=model_dim,
            max_length=max_length,
        )

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )

        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
        )

        self.output_projection = Linear(
            model_dim,
            vocab_size,
        )

        

    def forward(
        self,
        src_tokens: IntArray,
        tgt_tokens: IntArray,
    ) -> Array:
        """
        Compute the forward pass of the Transformer.

        we use two params : src_tokens -> ndarray , tgt_tokens -> ndarray
        src_tokens : ndarray
            Source token IDs of shape (B, Ls)

        tgt_tokens : ndarray
            Target token IDs of shape (B, Lt)

        Returns ndarray Vocabulary logits of shape (B, Lt, vocab_size)
        """
        src = self.src_embedding.forward(src_tokens)
        src = self.positional_encoding.forward(src)

        encoder_output = self.encoder.forward(src)

        tgt = self.tgt_embedding.forward(tgt_tokens)
        tgt = self.positional_encoding.forward(tgt)

        decoder_output = self.decoder.forward(
            tgt,
            encoder_output,
        )

        logits = self.output_projection.forward(
            decoder_output
        )

        return logits
    


    def backward(
        self,
        grad_output: Array,
    ) -> tuple[None, None]:
        """
        
        Returns  tuple(None, None)

        Token indices are integers and therefore do not receive gradients.
        """
        grad_decoder = self.output_projection.backward(
            grad_output
        )

        grad_decoder_input, grad_encoder_output = (
            self.decoder.backward(grad_decoder)
        )

        grad_decoder_input = (
            self.positional_encoding.backward(
                grad_decoder_input
            )
        )

        self.tgt_embedding.backward(
            grad_decoder_input
        )

        grad_encoder_input = self.encoder.backward(
            grad_encoder_output
        )

        grad_encoder_input = (
            self.positional_encoding.backward(
                grad_encoder_input
            )
        )

        self.src_embedding.backward(
            grad_encoder_input
        )

        return None, None
    

    def zero_grad(self) -> None:
        """
        Reset gradients of all trainable parameters.
        """
        self.src_embedding.zero_grad()
        self.tgt_embedding.zero_grad()

        self.encoder.zero_grad()
        self.decoder.zero_grad()

        self.output_projection.zero_grad()


    def __call__(
        self,
        src_tokens: IntArray,
        tgt_tokens: IntArray,
    ) -> Array:
        return self.forward(
            src_tokens,
            tgt_tokens,
        )