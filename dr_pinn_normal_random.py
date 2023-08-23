# %%
# %%
import deepxde.deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
from torch import Tensor

from datasets import parallel_solver, diffusion_reaction_solver 
from utils.func import dirichlet
from utils.PDETriple import PDETriple

date = time.strftime("%Y%m%d%H%M%S", time.localtime())
# %%
train_path = "datasets/DF/TRAIN_20_0.10_101_101.npz"
test_path = "datasets/DF/TEST_20_0.10_101_101.npz"

train_data = np.load(train_path)
train_vxs = train_data["vxs"]
train_grid = train_data["xt"].reshape(-1, 2)
train_uxts = train_data["uxts"].reshape(-1, 101 * 101)

test_data = np.load(test_path)
test_vxs = test_data["vxs"][:10]
test_grid = test_data["xt"].reshape(-1, 2)
test_uxts = test_data["uxts"][:10].reshape(-1, 101 * 101)

# %%
def dirichlet(inputs: Tensor, outputs: Tensor) -> Tensor:
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, (0,)], x_trunk[:, (1,)] # 10201
    # using sine function would have some errors
    scale_factor = 10 * (x * (1 - x) * t)
    return scale_factor * (outputs + 1)

def DF(x: tuple[Tensor, Tensor], 
       y: Tensor, 
       aux_vars: Tensor
       ) -> Tensor:
    D = 0.01
    k = 0.01
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t - D * dy_xx + k * y**2 - aux_vars
    # print(out.shape)
    return out.abs()

def pde_loss(inputs: tuple[Tensor, Tensor], outputs: Tensor):
    vxs, grid = inputs
    dy_t = dde.grad.jacobian(outputs, grid, j=1)
    dy_xx = dde.grad.hessian(outputs, grid, j=0)
    indices = (grid[:, (0,)] * 100).long()
    # card
    # aux_vars = vxs[indices]
    aux_vars = torch.gather(vxs, 1, indices)
    out = dy_t - 0.01 * dy_xx + 0.01 * outputs**2 - aux_vars
    return torch.nn.functional.mse_loss(out, torch.zeros_like(out))

# %%

net = dde.nn.pytorch.DeepONet(
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

func_space = dde.data.GRF(length_scale=0.10)
eval_pts = np.linspace(0, 1, 101)[:, None]

data = PDETriple((train_vxs, train_grid), train_uxts, (test_vxs, test_grid), test_uxts, data_format = "CartesianProd")
    
model = dde.Model(data, net)
model.compile("adam", lr = 1E-3, loss = [pde_loss], metrics=["l2 relative error"])

losshistory, train_state = model.train(iterations= 5000, batch_size = 5000)
dde.utils.plot_loss_history(losshistory)
# %%
def plot_train(i):
    train_ivxs = train_vxs[(i,),].repeat(10201, axis = 0)
    train_igrid = train_grid
    train_iuxts = train_uxts[i, :, None]
    v = train_ivxs[0]
    x = np.linspace(0, 1, v.shape[0])
    print(x.shape, v.shape)
    fig, (ax1 ,ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.set_xlim(0, 1)
    ax1.scatter(x, v)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')
    ax2.scatter(train_igrid[:, 0], train_igrid[:, 1], c = train_iuxts.reshape(-1))
    out = model.predict((train_ivxs, train_igrid))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.scatter(train_igrid[:, 0], train_igrid[:, 1], c = out.reshape(-1))
    colorbar = fig.colorbar(ax3.scatter(train_igrid[:, 0], train_igrid[:, 1], c = out.reshape(-1)), ax = ax3)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    delta = np.abs(train_iuxts.reshape(-1) - out.reshape(-1))
    ax4.scatter(train_igrid[:, 0], train_igrid[:, 1], c = delta)
    colorbar = fig.colorbar(ax4.scatter(train_igrid[:, 0], train_igrid[:, 1], c = delta), ax = ax4)

    plt.tight_layout()
    plt.show()
    
def plot_test(i):
    test_ivxs = test_vxs[(i,),].repeat(10201, axis = 0)
    test_igrid = test_grid
    test_iuxts = test_uxts[i, :, None]
    print(test_ivxs.shape, 
          test_igrid.shape, 
          test_iuxts.shape)
    v = test_ivxs[0]
    x = np.linspace(0, 1, v.shape[0])
    print(x.shape, v.shape)
    fig, (ax1 ,ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    ax1.set_xlim(0, 1)
    ax1.scatter(x, v)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')
    ax2.scatter(test_igrid[:, 0], test_igrid[:, 1], c = test_iuxts.reshape(-1))
    print(test_iuxts.shape, test_igrid.shape, test_ivxs.shape)
    out = model.predict((test_ivxs, test_igrid))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.scatter(test_igrid[:, 0], test_igrid[:, 1], c = out.reshape(-1))
    colorbar = fig.colorbar(ax3.scatter(test_igrid[:, 0], test_igrid[:, 1], c = out.reshape(-1)), ax = ax3)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    delta = np.abs(test_iuxts.reshape(-1) - out.reshape(-1))
    ax4.scatter(test_igrid[:, 0], test_igrid[:, 1], c = delta)
    colorbar = fig.colorbar(ax4.scatter(test_igrid[:, 0], test_igrid[:, 1], c = delta), ax = ax4)

    plt.tight_layout()
    plt.show()
    
# %%
print(test_vxs.shape)
plot_test(0)
# %%

# %%
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
func_space = dde.data.GRF(1.0, length_scale = 0.10, N= 1000, interp="linear")

# %%
while len(train_vxs) < 300:
    # generate some vxs to test
    pde_data = dde.data.TimePDE(geomtime, DF, [], num_domain = 20000)
    eval_pts = np.linspace(0, 1, 101)[:, None] # generate 1000 random vxs
    testing_new_data = dde.data.PDEOperatorCartesianProd(pde_data, func_space, eval_pts, 1, [0])
    # testing_model = dde.Model(testing_new_data, net)
    a, _, c = testing_new_data.train_next_batch()
    # print(res, topk_index, res[topk_index])
    topk_vxs = a[0]
    uxts = parallel_solver(diffusion_reaction_solver, topk_vxs, num_workers = 0)
    uxts = np.asarray([u for grid, u in uxts]).reshape(-1, 101 * 101)

    # then add the new data to the training set, and train the model
    train_vxs = np.concatenate([train_vxs, topk_vxs], axis = 0)
    train_uxts = np.concatenate([train_uxts, uxts], axis = 0)
    for i in range(len(train_vxs) - len(topk_vxs), len(train_vxs)):
        pass
        # plot_train(i)
    print(f"Train with: {len(train_vxs)} data")
    data = PDETriple(X_train=(train_vxs, train_grid), y_train=train_uxts, 
                                  X_test=(test_vxs, test_grid), y_test=test_uxts, boundary = [])
    
    model = dde.Model(data, net)
    model.compile("adam", 
                  lr = 1E-3,
                  loss = [pde_loss], 
                  metrics = ["l2 relative error"],
                  decay = None)

    if len(train_vxs) % 20 == 0:
        iters = 10000
    else:
        iters = 1000
    losshistory, train_state = model.train(iterations = iters, batch_size = 5000)
    
    pd_frame = losshistory.to_pandas()
    os.makedirs("results/DF", exist_ok=True)
    if os.path.exists(f"results/DF/loss_history_{date}_rasg_norrand.csv"):
        pd_frame = pd.concat([pd.read_csv(f"results/DF/loss_history_{date}_rasg_norrand.csv"), pd_frame], axis = 0, ignore_index=True)
    pd_frame.to_csv(f"results/DF/loss_history_{date}_rasg_norrand.csv", index=False)
    if len(train_vxs) % 20 == 0:
        dde.utils.plot_loss_history(losshistory)
        plt.show()
        plot_test(0)

losshistory, train_state = model.train(iterations = 10000, batch_size = 5000)
plot_test(0)
# %%
