"""
Softmax Layer:

This module implements the Softmax activation function used throughout the
Transformer architecture.

Mathematical approach:
Forward:
                exp(x_i)
    y_i = ------------------------
           Σ_j exp(x_j)

For numerical stability:

    x_stable = x - max(x)

Backward:
    The Jacobian of the Softmax function is

        J = diag(y) - yyᵀ

    Given the upstream gradient dY, the gradient with respect to the input is

        dX = J · dY

    which can be computed efficiently as

        dX = Y ⊙ (dY - Σ(dY ⊙ Y))

where:
    ⊙ -> Element-wise multiplication


Input:
    X : (..., N)

Output:
    Y : (..., N)

Gradients:
    dX : (..., N)

Applications in Transformer:
Softmax is used to normalize the scaled attention scores into a probability
distribution during the Scaled Dot-Product Attention mechanism.

        QKᵀ
    --------------
       √d_k
          ↓
      Softmax
          ↓
Attention Weights

"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


class Softmax:
    """
    Softmax activation layer.
    """

    def __init__(self, axis: int = -1) -> None:
        """
        We use axis param :
        axis : int, default=-1
            Axis along which Softmax is computed.
        """
        self.axis = axis
        self.output: Optional[Array] = None



    def forward(self, x: Array) -> Array:
        
        x_stable = x - np.max(x, axis=self.axis, keepdims=True)

        exp_x = np.exp(x_stable)

        output = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        self.output = output

        return output
    


    def backward(self, grad_output: Array) -> Array:
        """
        Compute the gradient of the loss with respect to the input.

        """
        if self.output is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        if grad_output.shape != self.output.shape:
            raise ValueError(
                "grad_output must have the same shape as the Softmax output."
            )

        dot = np.sum(
            grad_output * self.output,
            axis=self.axis,
            keepdims=True,
        )

        grad_input = self.output * (grad_output - dot)

        return grad_input


    def zero_grad(self) -> None:
        pass



    def __call__(self, x: Array) -> Array:
        return self.forward(x)