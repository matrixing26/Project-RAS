from numbers import Number
from numpy.typing import NDArray
import numpy as np
import torch
import deepxde.deepxde as dde
from torch import nn, Tensor
from torch.nn import functional as F
from deepxde.deepxde.data.function_spaces import FunctionSpace
import matplotlib.pyplot as plt

def H1norm(array: np.ndarray):
    return np.sqrt(np.sum(array**2, axis = 1) + np.sum((np.diff(array) / 0.01) ** 2, axis = 1))

def L2norm(array: np.ndarray):
    return np.sqrt(np.sum(array**2, axis = 1))

def L2_TO_DATA(test_vxs, train_vxs):
    test_vxs = test_vxs[:, None, :]
    dis = test_vxs - train_vxs
    l22d = np.mean(np.sqrt(np.sum(dis ** 2, axis = 2)), axis = 1)
    return l22d

def dirichlet(inputs: Tensor, outputs: Tensor) -> Tensor:
    """
    This function is to embed the dirichlet boundary.
    The dirichlet boundary is defined as:
        :math:`G(x, t)|_{x=0, x=1} = 0`
        :math:`G(x, t)|_{t=0} = 0`
    
    Args:
        inputs (np.ndarray): vxs and grid
        outputs (np.ndarray): the output of the network

    Returns:
        np.ndarray: this would make the boundary condition of output automatically satisfied.
        
    Shapes:
        inputs: `(B, F)` and `(N, 2)`
        outputs:  `(B, F, N)`
        
        
    """
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, 0], x_trunk[:, 1] # 10201
    # using sine function would have some errors
    scale_factor = (x * (1 - x) * t).unsqueeze(0)
    return scale_factor * (outputs + 1)

def periodic(x_loc: Tensor) -> Tensor:
    x, t = x_loc[:, 0], x_loc[:, 1]
    return torch.stack([t,
                        (2 * torch.pi * x).cos(),
                        (2 * torch.pi * x).sin(),
                        (4 * torch.pi * x).cos(),
                        (4 * torch.pi * x).sin(),
                        ], dim = -1)

def arg_topk(array: np.ndarray, k: int):
    return np.argpartition(array, -k)[-k:]

def plot_data(vx: np.ndarray,           # (N, )
              grid: np.ndarray,         # (N, 2)
              uxt: np.ndarray,          # (N, )
              out: np.ndarray = None,   # (N, )
              ext: np.ndarray = None,   # (N, )
              ):
    # plot-data
    plot_number = 2
    if out is not None:
        plot_number += 2
    if ext is not None:
        plot_number += 1
        
    fig, axs = plt.subplots(1, plot_number, figsize=(plot_number * 5, 5))

    v_space = np.linspace(0, 1, len(vx))

    for i, ax in enumerate(axs):
        ax.set_xlim(0, 1)
        if i != 0:
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
    
    axs[0].scatter(v_space, vx, s=1)
    sc = axs[1].scatter(grid[:, 0], grid[:, 1], c=uxt, cmap='viridis')
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax = axs[1])
    if out is not None:
        sc = axs[2].scatter(grid[:, 0], grid[:, 1], c=out, cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[2])
        sc = axs[3].scatter(grid[:, 0], grid[:, 1], c=np.abs(uxt-out), cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[3])
    if ext is not None:
        sc = axs[4].scatter(grid[:, 0], grid[:, 1], c=ext, cmap='viridis')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=sc.norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax = axs[4])

    plt.tight_layout()
    plt.show()

# Adapted from https://github.com/luo3300612/grid_sample1d, support 1D, 2D, 3D grid
def grid_sample(input: torch.Tensor, 
                grid: torch.Tensor, 
                mode:str = "bilinear", 
                padding_mode:str = "zeros", 
                align_corners: bool | None = None
                ) -> torch.Tensor:
    if grid.shape[-1] == 1:
        assert mode in ["nearest", "linear"], "1D grid only support nearest and linear mode"
        input = input.unsqueeze(-1)
        grid = grid.unsqueeze(1)
        grid = torch.cat([-torch.ones_like(grid), grid], dim=-1)
        out_shape = [grid.shape[0], input.shape[1], grid.shape[2]]
        return F.grid_sample(input, grid, 
                             mode= "bilinear" if mode == "linear" else mode, 
                             padding_mode=padding_mode, 
                             align_corners=align_corners).view(*out_shape)
    else:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def interp_nd(grid: torch.Tensor, 
              points: torch.Tensor, 
              mode: str = "linear", 
              align_corners: bool = True, 
              padding_mode: str = "border"
              ) -> torch.Tensor:
    """
    Using torch to do interpolation.

    Args:
        grid (torch.Tensor): the input function, shape `B, C, D, (H,) (W,)` for batched input, `C, D, (H,) (W,)` for unbatched input, channel dimension is optional.
        points (torch.Tensor): `B, N, dim` for batched input, `N, dim` for unbatched input. the points should be in the range of `[-1, 1]`.
        mode (str, optional): the mode for interpolation. Defaults to "linear".

    Returns:
        torch.Tensor: shape `(B,) (C,) N
    """
    interp_dim = points.shape[-1]
    is_batched = points.dim() == 3
    is_channelled = grid.dim() -1 == is_batched + interp_dim
    
    if not is_channelled:
        grid = grid.unsqueeze(int(is_batched))
        
    if not is_batched:
        grid = grid.unsqueeze(0)
        points = points.unsqueeze(0)
    
    for _ in range(interp_dim - 1):
        points = points.unsqueeze(-2)
    
    grid = grid_sample(grid, points, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    
    for _ in range(interp_dim - 1):
        grid = grid.squeeze(-2)
    
    if not is_batched:
        grid = grid.squeeze(0)
    
    if not is_channelled:
        grid = grid.squeeze(int(is_batched))
    
    return grid

class RFF(FunctionSpace):
    def __init__(self, N = 200, mu: Number = 0, sigma: Number = 1):
        self._N = N
        self._mu = mu
        self._sigma = sigma
    
    def random(self, size: int) -> NDArray[np.float_]:
        """Generate random features for RFF

        Args:
            size (int): batch size

        Returns:
            NDArray[np.float_]: a feature array with shape: (B, 2, N)
        """
        freq = np.random.randn(size, self._N) * self._sigma + self._mu
        phase = np.random.rand(size, self._N) * 2 * np.pi
        return np.stack([freq, phase], axis = 1)
    
    def eval_one(self, feature: NDArray, x: NDArray) -> NDArray[np.float_]:
        """Evaluate the value of a function

        Args:
            feature (_type_): (1, 2, N)
            x (_type_): (1, M) | (M)
            
        Returns:
            NDArray[np.float_]: a feature array with shape: (1, M)
        """
        if len(x.shape) == 1:
            x = x[:, None]
        phase = np.einsum("ij,kr->ijk", feature[:, 0], x) + feature[:, 1][..., None]
        return np.cos(phase).sum(axis = 1) * np.sqrt(2 / self._N)
    
    def eval_batch(self, features: NDArray, xs: NDArray) -> NDArray[np.float_]:
        """Evaluate the value of functions

        Args:
            features (NDArray): (B, 2, N)
            xs (NDArray): (B, M) | (1, M) | (M)

        Returns:
            NDArray[np.float_]: a feature array with shape: (B, M)
        """
        if len(xs.shape) == 1:
            xs = xs[:, None]
        phase = np.einsum("ij,kr->ijk", features[:, 0], xs) + features[:, 1][..., None]
        return np.cos(phase).sum(axis = 1) * np.sqrt(2 / self._N)

class COS(FunctionSpace):
    def __init__(self, N = 20):
        self._N = N
    
    def random(self, size: int) -> NDArray[np.float_]:
        coeff = np.random.randn(size, self._N)
        phase = np.random.rand(size, self._N) * 2 * np.pi
        return np.stack([coeff, phase], axis = 0) # 2, S, N

    def eval_one(self, feature: NDArray, x: NDArray) -> NDArray[np.float_]:
        if len(x.shape) == 1:
            x = x[:, None]
            # G, 1
        coeff, phi = feature[0], feature[1] # S, N
        freq = np.arange(1, self._N + 1)
        phase = np.einsum("i,jk->ij", freq, x) # N, G
        phase = phase[None, ...] + phi[..., None] # S, N, G
        phase = np.cos(phase) # S, N, G
        phase = phase * coeff[..., None] # (S, N, G)
        return phase.sum(axis = 1) * np.sqrt(2 / self._N)
    
    def eval_batch(self, features: NDArray, xs: NDArray) -> NDArray[np.float_]:
        if len(xs.shape) == 1:
            xs = xs[:, None]
            # G, 1
        coeff, phi = features[0], features[1] # S, N
        freq = np.arange(1, self._N + 1)
        phase = np.einsum("i,jk->ij", freq, xs) # N, G
        phase = phase[None, ...] + phi[..., None] # S, N, G
        phase = np.cos(phase) # S, N, G
        phase = phase * coeff[..., None] # (S, N, G)
        return phase.sum(axis = 1) * np.sqrt(2 / self._N)

class RFFCHE(FunctionSpace):
    def __init__(self, N_RFF = 100, N_chebyshev = 10, mu: Number = 0, sigma: Number = 1):
        self._N_RFF = N_RFF
        self._N_chebyshev = N_chebyshev
        self._mu = mu
        self._sigma = sigma
        self.RFF = RFF(N_RFF, mu, sigma)
        self.CHE = dde.data.function_spaces.Chebyshev(N_chebyshev)

    def random(self, size):
        return self.RFF.random(size), self.CHE.random(size)
    
    def eval_one(self, feature, x):
        return (self.RFF.eval_one(feature[0], x) + self.CHE.eval_one(feature[1], x)) / 2
        
    def eval_batch(self, features, xs):
        return (self.RFF.eval_batch(features[0], xs) + self.CHE.eval_batch(features[1], xs)) / 2
    
    