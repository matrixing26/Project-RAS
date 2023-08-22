# %%
import torch
from torch import Tensor, nn
import deepxde.deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from utils.PDETriple import PDETripleCartesianProd

# %%
train_path = "datasets/DF/TRAIN_100_0.05_101_101.npz"
test_path = "datasets/DF/TEST_100_0.05_101_101.npz"

train_data = np.load(train_path)
train_vxs = train_data["vxs"].astype(np.float32)
train_grid = train_data["xt"].reshape(-1, 2).astype(np.float32)
train_uxts = train_data["uxts"].reshape(-1, 101 * 101).astype(np.float32)

test_data = np.load(test_path)
test_vxs = test_data["vxs"].astype(np.float32)
test_grid = test_data["xt"].reshape(-1, 2).astype(np.float32)
test_uxts = test_data["uxts"].reshape(-1, 101 * 101).astype(np.float32)

# %%
def dirichlet(inputs: Tensor, outputs: Tensor) -> Tensor:
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, 0], x_trunk[:, 1] # 10201
    # using sine function would have some errors
    scale_factor = (x * (1 - x) * t).unsqueeze(0)
    return scale_factor * (outputs + 1)

def DF(x: Tensor, 
       y: Tensor, 
       aux_vars: Tensor
       ) -> Tensor:
    D = 0.01
    k = 0.01
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    out = dy_t - D * dy_xx + k * y**2 - aux_vars
    return out

def pde_loss(inputs: tuple[Tensor, Tensor], outputs: Tensor):
    vxs, grid = inputs
    dy_t = dde.grad.jacobian(outputs, grid, j=1)
    dy_xx = dde.grad.hessian(outputs, grid, j=0)
    indices = (grid[:, 0] * 101).long()
    aux_vars = vxs[indices]
    print(dy_t.shape, dy_xx.shape, vxs.shape, aux_vars.shape)
    out = dy_t - 0.01 * dy_xx + 0.01 * outputs**2 - aux_vars
    return torch.nn.functional.mse_loss(out, torch.zeros_like(out))

# %%

net = dde.nn.pytorch.DeepONetCartesianProd(
    layer_sizes_branch = [101, 100, 100],
    layer_sizes_trunk = [2, 100, 100, 100],
    activation = "gelu",
    kernel_initializer = "Glorot normal",
)

net.apply_output_transform(dirichlet)

geom = dde.geometry.Interval(0, 1)
time = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, time)

bc = dde.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)

pde = dde.data.TimePDE(geomtime, DF, [bc, ic], num_domain = 200, num_boundary = 40, num_initial = 20, num_test=500)

func_space = dde.data.GRF(length_scale=0.05)
eval_pts = np.linspace(0, 1, 101)[:, None]

data = PDETripleCartesianProd((train_vxs, train_grid), train_uxts, (test_vxs, test_grid), test_uxts)
    
model = dde.Model(data, net)
model.compile("adam", lr = 1E-3, loss = [pde_loss], metrics=["mean l2 relative error"])

losshistory, train_state = model.train(iterations= 50000)
dde.utils.plot_loss_history(losshistory)
# %%
