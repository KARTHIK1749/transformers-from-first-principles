"""
Layer Normalization:

This module implements Layer Normalization as introduced in:

    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (2016)

Layer Normalization normalizes the features of each token independently across
the embedding dimension. Unlike Batch Normalization, it does not depend on the
batch size and is therefore well suited for sequence models such as
Transformers.

Mathematical approach : 

Forward:

                 x - μ
x̂ = -------------------------------
      sqrt(σ² + ε)

y = γ ⊙ x̂ + β

where

μ = mean(x)
σ² = variance(x)

γ -> learnable scale parameter
β -> learnable shift parameter

Backward:

Given upstream gradient dY,

dβ = Σ dY

dγ = Σ (dY ⊙ x̂)

dx̂ = dY ⊙ γ

Let

N = embedding dimension

Then

dX =
(1 / N) * inv_std *
(
    N * dx̂
    - Σ(dx̂)
    - x̂ * Σ(dx̂ ⊙ x̂)
)

where

inv_std = 1 / sqrt(σ² + ε)

-----
Input:
    X : (..., embedding_dim)

Parameters:
    γ : (embedding_dim,)
    β : (embedding_dim,)

Output:
    Y : (..., embedding_dim)

Gradients:
    dγ : (embedding_dim,)
    dβ : (embedding_dim,)

Applications in Transformer:

Layer Normalization is applied after every residual connection.

Residual
    ↓
LayerNorm
    ↓
Attention / Feed Forward
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class LayerNorm:
    """
    Layer Normalization.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5) -> None:
        """
        we use two params : embedding_dim -> int , eps -> float
        embedding_dim : int
            Size of the last dimension.

        eps : float
            Small constant for numerical stability.
        """
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma: Array = np.ones(
            embedding_dim,
            dtype=np.float64,
        )

        self.beta: Array = np.zeros(
            embedding_dim,
            dtype=np.float64,
        )

        self.dgamma: Array = np.zeros_like(self.gamma)
        self.dbeta: Array = np.zeros_like(self.beta)

        self.input: Optional[Array] = None
        self.normalized: Optional[Array] = None
        self.inv_std: Optional[Array] = None



    def forward(self, x: Array) -> Array:
        """
        Apply Layer Normalization.
        This will return a normalized tensor with the same shape as the input.

        """
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected last dimension {self.embedding_dim}, "
                f"but received {x.shape[-1]}."
            )

        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        inv_std = 1.0 / np.sqrt(variance + self.eps)

        normalized = (x - mean) * inv_std

        self.input = x
        self.normalized = normalized
        self.inv_std = inv_std

        return self.gamma * normalized + self.beta

    def backward(self, grad_output: Array) -> Array:
        """
        Backpropagate through Layer Normalization.

        """
        if (
            self.input is None
            or self.normalized is None
            or self.inv_std is None
        ):
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        if grad_output.shape != self.input.shape:
            raise ValueError(
                "grad_output must have the same shape as the input."
            )

        self.dbeta = np.sum(
            grad_output,
            axis=tuple(range(grad_output.ndim - 1)),
        )

        self.dgamma = np.sum(
            grad_output * self.normalized,
            axis=tuple(range(grad_output.ndim - 1)),
        )

        dx_hat = grad_output * self.gamma

        n = self.embedding_dim

        sum_dx_hat = np.sum(
            dx_hat,
            axis=-1,
            keepdims=True,
        )

        sum_dx_hat_xhat = np.sum(
            dx_hat * self.normalized,
            axis=-1,
            keepdims=True,
        )

        grad_input = (
            self.inv_std
            / n
            * (
                n * dx_hat
                - sum_dx_hat
                - self.normalized * sum_dx_hat_xhat
            )
        )

        return grad_input
    


    def zero_grad(self) -> None:
        """
        Reset parameter gradients.
        """
        self.dgamma.fill(0.0)
        self.dbeta.fill(0.0)

    

    def __call__(self, x: Array) -> Array:
        return self.forward(x)