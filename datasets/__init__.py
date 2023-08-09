import deepxde.deepxde as dde
from sklearn import gaussian_process as gp
from scipy import linalg, interpolate

from multiprocessing import Pool
import numpy as np
from typing import Callable, List, Any
from .solver import *

def parallel_solver(func_solver: Callable[[], Any], data: List[Any], num_workers: int = 6) -> List[Any]:
    """
    Solve pde in parallel.
    Actually doing the same thing as `map(func_solver, data)`

    Args:
        func_solver (Callable[[Any], Any]): solver function.
        data (List[Any]): a list of data to be solved.
        num_workers (int, optional): workers number, the more, the faster. Defaults to 6.

    Returns:
        List[Any]: _description_
    """
    if num_workers <= 0:
        result = list(map(func_solver, data))    
    else:
        with Pool(num_workers) as pool:
            result = pool.map(func_solver, data)
    return result

def GRF_get(length_scale: float, Nx: int = 101) -> np.ndarray:
    space = dde.data.GRF(1.0, length_scale = length_scale, N= 1000, interp="cubic")
    vx = space.eval_batch(space.random(1), np.linspace(0, 1, Nx)[:, None])[0]
    return vx

def GRF_pos_get(length_scale: float, Nx: int = 101) -> np.ndarray:
    space = GRF_pos(1.0, length_scale = length_scale, N= 1000, interp="cubic")
    vx = space.eval_batch(space.random(1), np.linspace(0, 1, Nx)[:, None])[0]
    return vx

def makeTesting_dr(size: int = 100, length_scale: float = 0.05) -> str:
    """
    Generate testing dataset.

    Args:
        size (int, optional): the number of input function in testing dataset. Defaults to 100.
        length_scale (float, optional): the length_scale of the testing dataset. Defaults to 0.1.
        
    Returns:
        str: the path of the testing dataset.
    """
    vxs = []
    uxts = []
    for i in range(size):
        print(i, end = " ")
        vxs.append(GRF_get(length_scale))
        xt, uxt = diffusion_reaction_solver(vxs[-1])
        uxts.append(uxt)
    vxs = np.stack(vxs, axis = 0, dtype = np.float32)
    uxts = np.stack(uxts, axis = 0, dtype = np.float32)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = f"datasets/DF_{size}_{length_scale:.2f}_101_101.npz"
    np.savez(path, info = {"size": size, "grid": (101, 101), "grid_sample": "uniform", "length_scale": length_scale}, vxs = vxs, uxts = uxts, xt = xt)
    return path

def makeTesting_adv(size: int = 100, length_scale: float = 0.1, Nx = 101, Nt = 101) -> str:
    space = GRF_pos(1.0, length_scale = length_scale, N= 1000, interp="cubic")
    vxs = space.eval_batch(space.random(size), np.linspace(0, 1, Nx)[:, None])
    xt, uxts = advection_solver(vxs, Nx=Nx, Nt=Nt)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = f"datasets/ADV_{size}_{length_scale:.2f}_101_101.npz"
    np.savez(path, info = {"size": size, "grid": (101, 101), "grid_sample": "uniform", "length_scale": length_scale}, vxs = vxs, uxts = uxts, xt = xt)
    return path

def makeTesting_bur(size: int = 100, length_scale: float = 0.5, Nx = 101, Nt = 101) -> str:
    space = dde.data.GRF(1.0, kernel = "ExpSineSquared", length_scale = length_scale, N= 1000, interp="cubic")
    vxs = space.eval_batch(space.random(size), np.linspace(0, 1, Nx)[:, None])
    xt, uxts = burger_solver(vxs, Nx=Nx, Nt=Nt)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = f"datasets/BUR_{size}_{length_scale:.2f}_101_101.npz"
    np.savez(path, info = {"size": size, "grid": (101, 101), "grid_sample": "uniform", "length_scale": length_scale}, vxs = vxs, uxts = uxts, xt = xt)
    return path
class GRF_pos(dde.data.function_spaces.FunctionSpace):
    """Gaussian random field (Gaussian process) in 1D.

    The random sampling algorithm is based on Cholesky decomposition of the covariance
    matrix.

    Args:
        T (float): `T` > 0. The domain is [0, `T`].
        kernel (str): Name of the kernel function. "RBF" (radial-basis function kernel,
            squared-exponential kernel, Gaussian kernel), "AE"
            (absolute exponential kernel), or "ExpSineSquared" (Exp-Sine-Squared kernel,
            periodic kernel).
        length_scale (float): The length scale of the kernel.
        N (int): The size of the covariance matrix.
        interp (str): The interpolation to interpolate the random function. "linear",
            "quadratic", or "cubic".
    """

    def __init__(self, T=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        elif kernel == "ExpSineSquared":
            K = gp.kernels.ExpSineSquared(length_scale=length_scale, periodicity=T)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, size):
        u = np.random.randn(self.N, size)
        fea = np.dot(self.L, u).T
        print(fea.shape)
        fea = fea - fea.min(axis=1, keepdims=True) + 0.5
        return fea

    def eval_one(self, feature, x):
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), feature)
        f = interpolate.interp1d(
            np.ravel(self.x), feature, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_batch(self, features, xs):
        if self.interp == "linear":
            return np.vstack([np.interp(xs, np.ravel(self.x), y).T for y in features])
        res = map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(xs).T,
            features,
        )
        return np.vstack(list(res)).astype(dde.config.real(np))