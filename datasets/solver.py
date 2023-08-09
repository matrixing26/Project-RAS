import torch
import numpy as np
import math
from typing import Callable, Tuple, Union, List, Any
from torch import nn, Tensor
from torch.nn import functional as F
from numpy.typing import ArrayLike

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

class CVCSolver(nn.Module):
    def __init__(self, xmin: float = 0, xmax: float = 1, tmin: float = 0, tmax: float = 1, Nx: int = 101, Nt: int = 101, upsample: int = 5):
        super().__init__()
        self.xmin = xmin
        self.xmax = xmax
        self.tmin = tmin
        self.tmax = tmax
        self.Nx = Nx
        self.Nt = Nt
        self.upsample = upsample
        self.Mx = (Nx - 1) * upsample + 1
        self.Mt = (Nt - 1) * upsample + 1
        xgrid = np.linspace(xmin, xmax, Nx)
        tgrid = np.linspace(tmin, tmax, Nt)
        self.grid = np.asarray(np.meshgrid(xgrid, tgrid, indexing="ij")).transpose([1,2,0])
        self.register_buffer("Xgrid", torch.linspace(xmin, xmax, self.Mx))
        self.register_buffer("Tgrid", torch.linspace(tmin, tmax, self.Mt))
        self.register_buffer("offdiag", torch.ones(self.Mx - 2).diag(-1))
        self.dx = (xmax - xmin) / (self.Mx - 1)
        self.dt = (xmax - xmin) / (self.Mt - 1)
        self.lam = self.dt / self.dx
        self.register_buffer("tk", torch.ones((self.Mx - 1,self.Mx - 1)).tril().unsqueeze(0))
    
    @torch.no_grad()
    def forward(self, vxs: ArrayLike, g: Callable[[Tensor], Tensor] = lambda t: (t * torch.pi / 2).sin(), f: Callable[[Tensor], Tensor] = lambda x: (x * torch.pi).sin()):
        """
        _summary_

        Args:
            vxs (ArrayLike): should be a batched array of shape B, N
            g (_type_, optional): _description_. Defaults to lambdat:(t * torch.pi).sin().
            f (_type_, optional): _description_. Defaults to lambdax:(x * torch.pi / 2).sin().
        """
        vxs = torch.as_tensor(vxs, dtype = self.Xgrid.dtype, device = self.Xgrid.device)
        vxs = interp_nd(vxs, self.Xgrid[:, None] * 2 - 1, mode = "linear")
        u = torch.empty(vxs.shape[0], self.Mx, self.Mt, dtype = self.Xgrid.dtype, device = self.Xgrid.device)
        u[:, 0] = g(self.Tgrid).unsqueeze(0).expand(u.shape[0], -1)
        u[:, :, 0] = f(self.Xgrid).unsqueeze(0).expand(u.shape[0], -1)
        
        mid = (vxs[:, :-1] + vxs[:, 1:]) / 2
        k = (1 - mid * self.lam) / (1 + mid * self.lam)
        
        t_K = self.tk.repeat(k.shape[0], 1, 1)
        _ = [t_K[:, i:, :i].mul_(-k[:, (i,), None]) for i in range(1, k.shape[1])]
        t_B = u[:, 0, :-1] - u[:, 0, 1:] * k[:, (0,)]
        t_D = torch.diag_embed(k) + self.offdiag.unsqueeze(0)
        # print(t_K.shape, t_D.shape, t_B.shape)
        viu = u[:, 1:]
        for i in range(0, self.Mt - 1):
            buf = torch.einsum("bik,bk->bi", t_D, viu[:, :, i])
            buf[:, 0] += t_B[:, i]
            buf = torch.einsum("bik,bk->bi", t_K, buf)
            viu[:, :, i + 1] = buf
        
        return self.grid, u[:, ::self.upsample, ::self.upsample].cpu().numpy()

class BurgersSolver(nn.Module):
    def __init__(self, xmax: float = 1, tmax: float = 1, nu: float = 0.1, nx: int = 101, nt: int = 101, upsample: tuple[int, int] = (1, 5), umax: float = 5):
        """
        The settings for the Burgers equation solver.

        Args:
            xmax (float, optional): the length of x-axis. Defaults to 1.
            tmax (float, optional): the length of t-axis. Defaults to 1.
            nu (float, optional): the viscosity of the fluid. Defaults to 0.1.
            nx (int, optional): the number of x grid. Defaults to 101.
            nt (int, optional): the number of t grid. Defaults to 101.
            upsample (tuple[int, int], optional): (up_x, up_t), affecting the precision of the solution, must satisfy some conditons. Defaults to None.
            `up_x` should be bigger than `xmax / (nx * nu)` because of the shapeness of the shock wave. `up_t` should fulfill the stability condition (CFL), which is `CFL = nx * umax * tmax / nt / xmax < 1`.
            umax (float, optional): the maximum of the initial condition. Defaults to 5.
        """
        super().__init__()
        self.xmax = xmax
        self.tmax = tmax
        self.nu = nu
        if upsample is None:
            #TODO: automatically choose upsample
            # 0.1 -> (1, 100), 0.01 -> (1, 10), 0.001 -> (1 , 10)
            upx = max(1, math.ceil((xmax / (nx -1)) / nu))
            upt = max(1, math.ceil(10 / nu * upx * (nx - 1) / (nt -1)))
            upsample = (upx, upt)
        self.upsample = upsample
        self.nx = nx
        self.nt = nt
        self.mx = (nx -1) * upsample[0] + 1
        self.mt = (nt - 1) * upsample[1] + 1
        xgrid = np.linspace(0, xmax, nx)
        tgrid = np.linspace(0, tmax, nt)
        self.grid = np.asarray(np.meshgrid(xgrid, tgrid, indexing="ij")).transpose([1,2,0])
        self.register_buffer("X", torch.linspace(0, xmax, self.mx))
        self.dx = xmax / (self.mx - 1)
        self.dt = tmax / (self.mt - 1)
        self.c1 = self.dt / (2 * self.dx)
        self.c2 = self.nu * self.dt / self.dx ** 2
        
    @torch.no_grad()
    def forward(self, u0):
        u0 = torch.as_tensor(u0, device=self.X.device, dtype=self.X.dtype)
        batch = u0.shape[0]
        u0 = interp_nd(u0, self.X[:, None] * 2 - 1)
        u = torch.empty(batch, self.mx - 1, self.mt, device=self.X.device, dtype=self.X.dtype)
        u[...,0] = u0[..., :-1]
        for n in range(0, self.mt - 1):
            u0 = u[..., n]
            u1 = u0.roll(1, dims = 1)
            u2 = u0.roll(-1, dims = 1)
            u[..., n + 1] = u0 - self.c1 * (u2 - u1) * u0 + self.c2 * (u1 + u2 - 2 * u0)
            
        u = torch.cat([u, u[:, (0,), :]], dim = 1)
        u = u[:, ::self.upsample[0], ::self.upsample[1]].cpu().numpy()
        
        return self.grid, u
    
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

def advection_solver(v: np.ndarray, xmax: float = 1.0, tmax: float = 1.0, Nx: int = 101, Nt: int = 101, cuda = True, batchsize: int = 100, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    solver = CVCSolver(xmin = 0, xmax = xmax, tmin = 0, tmax = tmax, Nx = Nx, Nt = Nt, *args, **kwargs)
    if cuda:
        solver = solver.cuda()
    else:
        solver = solver.cpu()
    
    if batchsize is not None:
        split_num = int(np.ceil(v.shape[0] / batchsize))
        vs = np.array_split(v, split_num)
        us  = []
        for v in vs:
            xt, u = solver(v)
            us.append(u)
        u = np.concatenate(us, axis = 0)
    else:
        xt, u = solver(v)
    
    # xt: shape (2, 101, 101) u: shape (b, 101, 101)
    return xt, u

def burger_solver(v: np.ndarray, xmax: float = 1.0, tmax: float = 1.0, nu = 0.1, Nx = 101, Nt = 101, cuda = True, batchsize: int = 100, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    solver = BurgersSolver(xmax = xmax, tmax = tmax, nu= nu, nx = Nx, nt = Nt, upsample = (2, 200),*args, **kwargs)
    if cuda:
        solver = solver.cuda()
    else:
        solver = solver.cpu()
    
    if batchsize is not None:
        split_num = int(np.ceil(v.shape[0] / batchsize))
        vs = np.array_split(v, split_num)
        us  = []
        for v in vs:
            xt, u = solver(v)
            us.append(u)
        u = np.concatenate(us, axis = 0)
    else:
        xt, u = solver(v)
    
    return xt, u

#TODO
def advection_diffusion_equation():
    pass

# -------------------------- nd grid sample --------------------------
# Adapted from https://github.com/luo3300612/grid_sample1d, support 1D, 2D, 3D grid
def grid_sample(input: Tensor, 
                grid: Tensor, 
                mode:str = "bilinear", 
                padding_mode:str = "zeros", 
                align_corners: bool | None = None
                ) -> Tensor:
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

def interp_nd(grid: Tensor, 
              points: Tensor, 
              mode: str = "linear", 
              align_corners: bool = True, 
              padding_mode: str = "border"
              ) -> Tensor:
    """
    Using torch to do interpolation. Support 1D, 2D, 3D grid. And for grid inputs, points can be unbatched or batched, but for unbatched grid inputs, points must be unbatched.

    Args:
        grid (torch.Tensor): the input function, shape `B, (C,) D, (H,) (W,)` for batched input, `(C,) D, (H,) (W,)` for unbatched input, channel dimension is optional.
        points (torch.Tensor): `B, N, dim` for batched input, `N, dim` for unbatched input. the points should be in the range of `[-1, 1]`.
        mode (str, optional): the mode for interpolation. Defaults to "linear".

    Returns:
        torch.Tensor: shape `(B,) (C,) N
    """
    interp_dim = points.shape[-1]
    is_point_batched = points.dim() == 3
    is_grid_batched = grid.dim() > interp_dim
    is_channelled = grid.dim() -1 == is_grid_batched + interp_dim
    
    if not is_channelled:
        grid = grid.unsqueeze(int(is_grid_batched))
        
    if not is_grid_batched:
        grid = grid.unsqueeze(0)
    if not is_point_batched:
        points = points.unsqueeze(0).expand(grid.shape[0], -1, -1)
    
    for _ in range(interp_dim - 1):
        points = points.unsqueeze(-2)
    
    grid = grid_sample(grid, points, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    for _ in range(interp_dim - 1):
        grid = grid.squeeze(-1)
    
    if not is_grid_batched:
        grid = grid.squeeze(0)
    
    if not is_channelled:
        grid = grid.squeeze(int(is_grid_batched))
    
    return grid