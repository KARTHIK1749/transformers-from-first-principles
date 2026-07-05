"""
Positional Encoding:

This module implements the fixed sinusoidal positional encoding introduced in
the original Transformer paper:

    "Attention Is All You Need"
    Vaswani et al., 2017

Unlike recurrent architectures, Transformers process all tokens in parallel
and therefore have no inherent notion of sequence order. Positional encodings
inject information about the position of each token into its embedding.

Mathematical approach:
For position pos and embedding dimension i,

                     pos
PE(pos,2i) = sin(-------------)
                10000^(2i/d)

                     pos
PE(pos,2i+1) = cos(-------------)
                 10000^(2i/d)

Forward:
    Y = X + PE

Backward:
    Since positional encodings are constant (non-trainable),

        dX = dY

No gradients are computed for the positional encoding matrix.


Input:
    X : (batch_size, sequence_length, embedding_dim)

Positional Encoding:
    PE : (max_length, embedding_dim)

Output:
    Y : (batch_size, sequence_length, embedding_dim)

Applications in Transformer :
The positional encoding is added immediately after the embedding layer.

Pipeline:

    Token IDs
         ↓
     Embedding
         ↓
Positional Encoding
         ↓
 Multi-Head Attention

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


class PositionalEncoding:
    """
    Fixed sinusoidal positional encoding.
    """

    def __init__(self, embedding_dim: int, max_length: int = 5000) -> None:
        """
        we use two params : embedding_dim -> int , max_length -> int

        embedding_dim : int
            Dimension of the embedding vectors.

        max_length : int, default=5000
            Maximum supported sequence length.
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")

        self.embedding_dim = embedding_dim
        self.max_length = max_length

        positions = np.arange(max_length, dtype=np.float64).reshape(-1, 1)

        div_term = np.exp(
            np.arange(0, embedding_dim, 2, dtype=np.float64)
            * (-np.log(10000.0) / embedding_dim)
        )

        pe = np.zeros((max_length, embedding_dim), dtype=np.float64)

        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)

        self.encoding: Array = pe



    def forward(self, x: Array) -> Array:
        """
        Add positional encodings to the input embeddings.

        """
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(batch_size, sequence_length, embedding_dim)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {embedding_dim}."
            )

        if sequence_length > self.max_length:
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"maximum supported length ({self.max_length})."
            )

        return x + self.encoding[:sequence_length]
    


    def backward(self, grad_output: Array) -> Array:
        """
        Backpropagate through the positional encoding.

        Since positional encodings are constant, the gradient with respect to
        the input is simply the gradient of the output.
        """
        return grad_output


    def zero_grad(self) -> None:
        """
        Positional encoding contains no trainable parameters.
        """
        pass

    

    def __call__(self, x: Array) -> Array:
        return self.forward(x)