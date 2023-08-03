import deepxde.deepxde as dde
import numpy as np
from typing import Callable, Union, Tuple
from torch import Tensor

def diffusion_reaction(x: Tuple[Tensor, Tensor], y: Tensor, aux: Tensor, D = 0.01, k = 0.01) -> Tensor:
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t - D * dy_xx + k * y**2 - aux
    return out

def advection_diffusion_reation(x: Tuple[Tensor, Tensor], y: Tensor, aux: Tensor) -> Tensor:
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_x = dde.grad.jacobian(y, x[1], j=0)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t + dy_x - aux * dy_xx
    return out

def anti_derivative(x: Tuple[Tensor, Tensor], y: Tensor, aux: Tensor) -> Tensor:
    dy = dde.grad.jacobian(y, x[1], j=0)
    out = dy - aux
    return out

def burgers_equation(x: Tuple[Tensor, Tensor], y: Tensor, aux: Tensor, v = 0.01) -> Tensor:
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_x = dde.grad.jacobian(y, x[1], j=0)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t + y * dy_x - v * dy_xx
    return out

def advection_equation(x: Tuple[Tensor, Tensor], y: Tensor, aux: Tensor) -> Tensor:
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_x = dde.grad.jacobian(y, x[1], j=0)
    out = dy_t - aux * dy_x
    return out