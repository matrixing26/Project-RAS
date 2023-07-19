import numpy as np
from typing import Callable, Tuple, Union, List, Any

def solve_ADR(xmin: float, xmax: float, tmin: float, tmax: float, k: Callable[[],Any], v: Callable[[],Any], g: Callable[[],Any], dg: Callable[[],Any], f: Callable[[],Any], u0: Callable[[],Any], Nx: int,  Nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 1D equation: `u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)` with zero boundary condition.

    Args:
        xmin (float): the left boundary of `x`.
        xmax (float): the right boundary of `x`.
        tmin (float): the left boundary of `t`.
        tmax (float): the right boundary of `t`.
        k (Callable[x -> float]): `k(x)` in the equation.
        v (Callable[x -> float]): `v(x)` in the equation.
        g (Callable[u -> float]): `g(u)` in the equation.
        dg (Callable[u -> float]): `g'(u)` in the equation.
        f (Callable[x, t -> float]): `f(x, t)` in the equation.
        u0 (Callable[x -> float]): initial condition `u(x, 0) = u0(x)`.
        Nx (int): grid number of `x`.
        Nt (int): grid number of `t`.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: x, t, u(x, t).
        
    Shapes:
        x: (Nx, )
        t: (Nt, )
        u: (Nx, Nt)
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u

def diffusion_reaction_solver(v: np.ndarray, xmax: float = 1.0, tmax: float = 1.0, D: float = 0.01, k: float = 0.01, Nx: int = 101, Nt: int = 101, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve diffusion reaction equation: `u_t = D * u_xx - v(x) * u_x + k * u` with zero boundary condition.

    Args:
        v (np.ndarray): `v(x)` in the equation.
        xmax (float, optional): the right boundary of `x`. Defaults to 1.0.
        tmax (float, optional): the right boundary of `t`. Defaults to 1.0.
        D (float, optional): `D` in the equation. Defaults to 0.01.
        k (float, optional): `k` in the equation. Defaults to 0.01.
        Nx (int, optional): grid number of `x`. Defaults to 101.
        Nt (int, optional): grid number of `t`. Defaults to 101.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: xt, u(x, t).
 
    Shapes:
        xt: (Nx, Nt, 2)
        u(x, t): (Nx, Nt) 
    """
    x, t, u = solve_ADR(xmin = 0, xmax = xmax, tmin = 0, tmax = tmax, 
                        k = lambda x: D * np.ones_like(x), 
                        v = lambda x: np.zeros_like(x), 
                        g = lambda u: k * u ** 2, 
                        dg = lambda u: 2 * k * u, 
                        f = lambda x, t: np.tile(v[:, None], (1, Nt)), 
                        u0 = lambda x: np.zeros_like(x), 
                        Nx = Nx, Nt = Nt)
    
    xt = np.asarray(np.meshgrid(x, t, indexing = "ij")).transpose([1,2,0]) # shape (2, 101, 101)
    
    return xt, u