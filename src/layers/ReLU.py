"""
ReLU Activation Layer:

This module implements the Rectified Linear Unit (ReLU) activation function,
one of the most widely used nonlinear activation functions in deep learning.

Mathematical approach:
Forward:
    Y = max(0, X)

Backward:
           { 1, if X > 0
    dY/dX =
           { 0, otherwise

Element-wise Gradient:
    dX = dY ⊙ I(X > 0)

where:
    ⊙ -> Element-wise multiplication
    I -> Indicator function


Input: X 

Output: Y 

Gradients: dX

Applications in Transformer:

ReLU is used as the activation function inside the Position-wise
Feed Forward Network (FFN) of the original Transformer architecture:

    Linear
        ↓
      ReLU
        ↓
    Linear

The activation introduces non-linearity, allowing the FFN to learn
complex feature transformations independently for every token.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


class ReLU:
    """
    Rectified Linear Unit activation layer.
    """

    def __init__(self) -> None:
        self.input: Optional[Array] = None

    ## Forward pass 
    def forward(self, x: Array) -> Array:
        
        self.input = x
        return np.maximum(0.0, x)
    

    ## Backward pass
    def backward(self, grad_output: Array) -> Array:
        
        if self.input is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        if grad_output.shape != self.input.shape:
            raise ValueError(
                "grad_output must have the same shape as the input."
            )

        grad_input = grad_output * (self.input > 0.0)

        return grad_input
    
    ## Zero grad
    def zero_grad(self) -> None:
        pass
    
    ## CAll method to make the layer callable
    def __call__(self, x: Array) -> Array:
        return self.forward(x)