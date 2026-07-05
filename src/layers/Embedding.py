"""
Embedding Layer:

This module implements the token embedding layer used at the input of the
Transformer architecture.

The embedding layer maps discrete token indices to continuous dense vectors
through a learnable embedding matrix.

Mathematical approach:

Forward:

    Y = E[X]

where

    X -> Token indices
    E -> Embedding matrix
    Y -> Embedded vectors

For each token index i,

    Y_i = E[i]

Backward:

The embedding layer has no gradient with respect to the integer token indices.

Instead, gradients are accumulated into the embedding matrix:

    dE[k] = Σ dY_i

for every occurrence of token k in the input sequence.

If a token appears multiple times, its gradients are summed.


Input:
    X : (...)

Parameters:
    E : (vocab_size, embedding_dim)

Output:
    Y : (..., embedding_dim)

Gradients:
    dE : (vocab_size, embedding_dim)

Applications in Transformer:
The embedding layer converts token IDs into dense vector representations
before positional encodings are added.

Pipeline:

    Token IDs
         ↓
    Embedding
         ↓
Positional Encoding
         ↓
 Transformer Encoder / Decoder

"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Embedding:
    """
    Learnable token embedding layer.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        params we use :
        vocab_size : int
            Number of tokens in the vocabulary.

        embedding_dim : int
            Dimension of each embedding vector.
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        limit = np.sqrt(6.0 / (vocab_size + embedding_dim))

        self.weight: Array = np.random.uniform(
            -limit,
            limit,
            size=(vocab_size, embedding_dim),
        ).astype(np.float64)

        self.dweight: Array = np.zeros_like(self.weight)

        self.input: Optional[IntArray] = None



    def forward(self, x: IntArray) -> Array:
        """
        Look up embeddings for the given token indices.
        """

        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError("Embedding input must contain integer token IDs.")

        if np.any(x < 0) or np.any(x >= self.vocab_size):
            raise ValueError("Token index out of vocabulary range.")

        self.input = x

        return self.weight[x]
    


    def backward(self, grad_output: Array) -> None:
        """
        Accumulate gradients into the embedding matrix.
        """
        if self.input is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        expected_shape = self.input.shape + (self.embedding_dim,)

        if grad_output.shape != expected_shape:
            raise ValueError(
                f"Expected gradient shape {expected_shape}, "
                f"but received {grad_output.shape}."
            )

        self.dweight.fill(0.0)

        np.add.at(
            self.dweight,
            self.input,
            grad_output,
        )

        return None
    


    def zero_grad(self) -> None:
        """
        Reset accumulated embedding gradients.
        """
        self.dweight.fill(0.0)

        

    def __call__(self, x: IntArray) -> Array:
        return self.forward(x)