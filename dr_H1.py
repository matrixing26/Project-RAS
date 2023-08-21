# %%
import deepxde.deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch

from datasets import parallel_solver, diffusion_reaction_solver
from utils.func import dirichlet, H1norm
from utils.PDETriple import PDETripleCartesianProd

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
batchsize = 10
decay = None
iters = 40000
ls = 0.05
lr = 1e-3
size = 100
total_num = 300

train_name = f"datasets/DF/TRAIN_{size}_{ls:.2f}_101_101.npz"
test_name = f"datasets/DF/TEST_{size}_{ls:.2f}_101_101.npz"

# %%
def pde(x, y, aux):
    D = 0.01
    k = 0.01
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t - D * dy_xx + k * y**2 - aux
    return out

def plot_train(i):
    # plot-data
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,5))

    v = train_vxs[i]
    x = np.linspace(0,1,v.shape[0])

    ax1.set_xlim(0,1)
    ax1.scatter(x, v, s=1)

    ut = train_uxts[i]
    xt = train_grid

    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')
    ax2.scatter(xt[...,0], xt[...,1], c=ut)
    colorbar = fig.colorbar(ax2.scatter(xt[...,0], xt[...,1], c=ut), ax=ax2)

    out = model.predict((train_vxs[(i,),...], xt))

    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    ax3.set_aspect('equal')
    ax3.scatter(xt[...,0], xt[...,1], c=out)
    colorbar = fig.colorbar(ax3.scatter(xt[...,0], xt[...,1], c=out), ax=ax3)

    ax4.set_xlim(0,1)
    ax4.set_ylim(0,1)
    ax4.set_aspect('equal')
    ax4.scatter(xt[...,0], xt[...,1], c=ut-out)
    colorbar = fig.colorbar(ax4.scatter(xt[...,0], xt[...,1], c=np.abs(ut-out)), ax=ax4)

    plt.tight_layout()
    plt.show()

def plot_test(i):
    # plot-data
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(20,5))

    v = test_vxs[i]
    x = np.linspace(0,1,v.shape[0])

    ax1.set_xlim(0,1)
    ax1.scatter(x, v, s=1)

    ut = test_uxts[i]
    xt = test_grid

    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_aspect('equal')
    ax2.scatter(xt[...,0], xt[...,1], c=ut)
    colorbar = fig.colorbar(ax2.scatter(xt[...,0], xt[...,1], c=ut), ax=ax2)

    out = model.predict((test_vxs[(i,),...], xt))

    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    ax3.set_aspect('equal')
    ax3.scatter(xt[...,0], xt[...,1], c=out)
    colorbar = fig.colorbar(ax3.scatter(xt[...,0], xt[...,1], c=out), ax=ax3)

    ax4.set_xlim(0,1)
    ax4.set_ylim(0,1)
    ax4.set_aspect('equal')
    ax4.scatter(xt[...,0], xt[...,1], c=ut-out)
    colorbar = fig.colorbar(ax4.scatter(xt[...,0], xt[...,1], c=np.abs(ut-out)), ax=ax4)

    plt.tight_layout()
    plt.show()

# %%
train_data = np.load(train_name)
train_vxs = train_data["vxs"]
train_grid = train_data["xt"].reshape(-1, 2)
train_uxts = train_data["uxts"].reshape(-1, 101 * 101)
del train_data

test_data = np.load(test_name)
test_vxs = test_data["vxs"]
test_grid = test_data["xt"].reshape(-1, 2)
test_uxts = test_data["uxts"].reshape(-1, 101 * 101)
del test_data
print(test_vxs.shape, train_grid.shape, train_uxts.shape)
print(test_vxs.shape, test_grid.shape, test_uxts.shape)

# %%
data = PDETripleCartesianProd(X_train=(train_vxs, train_grid), y_train=train_uxts, X_test=(test_vxs, test_grid), y_test=test_uxts, boundary = [])

# Net
net = dde.nn.pytorch.DeepONetCartesianProd(
    layer_sizes_branch = [101, 100, 100],
    layer_sizes_trunk = [2, 100, 100, 100],
    activation = "gelu",
    kernel_initializer = "Glorot normal",
)

net.apply_output_transform(dirichlet)
net.load_state_dict(torch.load("datasets/DF/PRETRAIN_100_0.05_20230821-10-42-51.pth"))

model = dde.Model(data, net)
model.compile("adam", 
              lr= lr, 
              loss= ["mse"], 
              metrics = ["mean l2 relative error"], 
              decay = None)

# %%
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
func_space = dde.data.GRF(1.0, length_scale = ls, N= 1000, interp="linear")

# %%
while len(train_vxs) < total_num:
    # generate some vxs to test
    pde_data = dde.data.TimePDE(geomtime, pde, [], num_domain = 20000)
    eval_pts = np.linspace(0, 1, 101)[:, None] # generate 1000 random vxs
    testing_new_data = dde.data.PDEOperatorCartesianProd(pde_data, func_space, eval_pts, len(train_vxs), [0])
    # testing_model = dde.Model(testing_new_data, net)
    a, _, c = testing_new_data.train_next_batch()
    h1 = H1norm(a[0])
    print(f"H1 values: {h1.mean():.2e}, Std: {h1.std():.2e}")
    select_num = min(20, total_num - len(train_vxs))
    topk_index = np.argpartition(h1, -select_num)[-select_num:] # select the top 20 vxs
    topk_vxs = a[0][topk_index]
    uxts = parallel_solver(diffusion_reaction_solver, topk_vxs, num_workers = 0)
    uxts = np.asarray([u for grid, u in uxts]).reshape(-1, 101 * 101)

    # then add the new data to the training set, and train the model
    train_vxs = np.concatenate([train_vxs, topk_vxs], axis = 0)
    train_uxts = np.concatenate([train_uxts, uxts], axis = 0)
    
    print(f"Train with: {len(train_vxs)} data")
    data = PDETripleCartesianProd(X_train=(train_vxs, train_grid), y_train=train_uxts, 
                                  X_test=(test_vxs, test_grid), y_test=test_uxts, boundary = [])
    
    model = dde.Model(data, net)
    model.compile("adam", 
                  lr = lr, 
                  metrics = ["mean l2 relative error"],
                  decay = decay,)

    losshistory, train_state = model.train(iterations=iters, batch_size = batchsize)
    
    pd_frame = losshistory.to_pandas()
    os.makedirs("results/DF", exist_ok=True)
    if os.path.exists(f"results/DF/loss_history_{date}_h1.csv"):
        pd_frame = pd.concat([pd.read_csv(f"results/DF/loss_history_{date}_h1.csv"), pd_frame], axis = 0, ignore_index=True)
    pd_frame.to_csv(f"results/DF/loss_history_{date}_h1.csv", index=False)
    dde.utils.plot_loss_history(losshistory)
    plt.show()
    plot_train(0)
    plot_test(0)
    
torch.save(model.state_dict(), f"results/model_{date}_h1.pth")


