"""
Linear Layer:

This module implements a fully connected affine transformation used
throughout the Transformer architecture.

Mathematical approach:
Forward:
    Y = XW + b

Backward:
    dX = dY Wᵀ
    dW = Xᵀ dY
    db = Σ dY


Input:
    X : (..., in_features)

Parameters:
    W : (in_features, out_features)
    b : (out_features,)

Output:
    Y : (..., out_features)

Gradients:
    dW : (in_features, out_features)
    db : (out_features,)


Useful in :

- Query projection
- Key projection
- Value projection
- Attention output projection
- Feed Forward Network (FFN)
- Final vocabulary projection
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


class Linear:
    """
    We used the Xavier initialization method to initialize the weights of the linear layer. 
    we use two params : input -> int , output -> int 
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be greater than zero.")

        if out_features <= 0:
            raise ValueError("out_features must be greater than zero.")

        self.in_features = in_features
        self.out_features = out_features

        limit = np.sqrt(6.0 / (in_features + out_features))

        self.W: Array = np.random.uniform(
            -limit,
            limit,
            size=(in_features, out_features),
        ).astype(np.float64)

        self.b: Array = np.zeros(out_features, dtype=np.float64)

        self.dW: Array = np.zeros_like(self.W)
        self.db: Array = np.zeros_like(self.b)

        self.input: Optional[Array] = None
    
    ## Forward pass -> to compute the output of the linear layer given an input
    def forward(self, x: Array) -> Array:
        
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, "
                f"but received {x.shape[-1]}."
            )

        self.input = x

        return x @ self.W + self.b
    
    ## Backward pass -> to compute the gradients of the weights and biases with respect to the loss function
    def backward(self, grad_output: Array) -> Array:
        
        if self.input is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        if grad_output.shape[-1] != self.out_features:
            raise ValueError(
                f"Expected last dimension {self.out_features}, "
                f"but received {grad_output.shape[-1]}."
            )

        grad_input = grad_output @ self.W.T

        self.dW = np.tensordot(
            self.input,
            grad_output,
            axes=(
                tuple(range(self.input.ndim - 1)),
                tuple(range(grad_output.ndim - 1)),
            ),
        )

        self.db = np.sum(
            grad_output,
            axis=tuple(range(grad_output.ndim - 1)),
        )

        return grad_input
    

    ## Zero grad -> to reset the gradients to zero after each step of optimization
    def zero_grad(self) -> None:
        
        self.dW.fill(0.0)
        self.db.fill(0.0)

    def __call__(self, x: Array) -> Array:
        return self.forward(x)



