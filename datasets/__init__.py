import deepxde.deepxde as dde
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

def makeTesting(size: int = 100, length_scale: float = 0.1) -> str:
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
        vxs.append(GRF_get(0.1))
        xt, uxt = diffusion_reaction_solver(vxs[-1])
        uxts.append(uxt)
    vxs = np.stack(vxs, axis = 0, dtype = np.float32)
    uxts = np.stack(uxts, axis = 0, dtype = np.float32)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = f"datasets/DF_{size}_0.1_101_101.npz"
    np.savez(path, info = {"size": size, "grid": (101, 101), "grid_sample": "uniform", "length_scale": length_scale}, vxs = vxs, uxts = uxts, xt = xt)
    return path
    