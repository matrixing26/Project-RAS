# %%
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from datasets import makeTesting_adv, GRF_pos
from datasets import advection_solver
from utils.PDETriple import PDETripleCartesianProd
from utils.pdes import advection_equation

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
total_training_vx = 300
ls = 0.5
testing_path = f"datasets/ADV_100_{ls:.2f}_101_101.npz"

start_num = 300
check_num = 1000
select_num = 30

lr_start = 1e-3
lr_middle = 1e-3
lr_end = 1e-3

iter_start = 100000
iter_middle = 100000
iter_end = 100000

batch_start = lambda n: n // 20
batch_middle = lambda n: n // 20
batch_end = lambda n: n // 20

decay_start = ("step", 25000, 0.5)
decay_middle = ("step", 25000, 0.5)
decay_end = ("step", 25000, 0.5)

if False:
    makeTesting_adv(length_scale = ls)

# %%
space = GRF_pos(1.0, length_scale = ls, N= 1000, interp="cubic")
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
vxs = space.eval_batch(space.random(start_num), np.linspace(0, 1, 101)[:, None])
# vxs = vxs - vxs.min(axis = 0, keepdims = True) + 0.5
xt, uxts = advection_solver(vxs)
grid = xt.reshape(101 * 101, -1)
uxts = uxts.reshape(-1, 101 * 101)

# %%
train_vxs = vxs
train_grid = grid
train_uxts = uxts
print(train_vxs.shape, train_grid.shape, train_uxts.shape)

test_data = np.load(testing_path)
test_vxs = test_data["vxs"]
test_grid = test_data["xt"].reshape(-1, 2)
test_uxts = test_data["uxts"].reshape(-1, 101 * 101)
del test_data
print(test_vxs.shape, test_grid.shape, test_uxts.shape)

# %%
def dirichlet(inputs, outputs):
    xt = inputs[1]
    x, t = xt[None, :, 0], xt[None, :, 1]
    return 4 * x * t * outputs + (torch.pi * x).sin() + (torch.pi * t / 2).sin()  

# %%
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

data = PDETripleCartesianProd(X_train=(train_vxs, train_grid), y_train=train_uxts, X_test=(test_vxs, test_grid), y_test=test_uxts, boundary = [])

# Net
net = dde.nn.DeepONetCartesianProd([101, 100, 100, 100], [2, 100, 100, 100], "gelu", "Glorot normal")
net.apply_output_transform(dirichlet)

# pre-train
model = dde.Model(data, net)
model.compile("adam", 
              lr= lr_start, 
              loss= ["mse"], 
              metrics = ["mean l2 relative error"], 
              decay = decay_start)

# %%
plot_train(0)
plot_test(0)

# %%
losshistory, train_state = model.train(iterations = iter_start, batch_size = batch_start(len(train_vxs)))
dde.utils.plot_loss_history(losshistory)

losshistory.to_pandas().to_csv(f"results/adv_{date}_random.csv", index=False)

# %%
plot_train(0)
plot_test(0)

torch.save(model.state_dict(), f"results/adv_model_{date}_random.pth")


