# %%
import torch
import numpy as np
import pandas as pdc
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from datasets import advection_solver
from utils.pdes import advection_equation
from utils.func import interp_nd
from datasets import GRF_pos

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
import torch
from torch import nn
from torch.nn import functional as F
from deepxde.deepxde.nn.pytorch import FNN, NN

class Nomad(NN):
    def __init__(self, layer_sizes_branch, 
                 layer_sizes_trunk, 
                 activation = F.gelu, 
                 kernel_initializer = "Glorot normal"):
        super().__init__()
        self.trunk = FNN(layer_sizes_trunk, 
                         activation=activation, 
                         kernel_initializer=kernel_initializer)
        self.branch = FNN(layer_sizes_branch,
                          activation=activation,
                          kernel_initializer=kernel_initializer)
    
    def forward(self, inputs):
        v, x = inputs
        v = self.branch(v)
        v = torch.cat([v, x], dim = -1)
        v = self.trunk(v)
        if self._output_transform is not None:
            v = self._output_transform(inputs, v)
        return v

def adv(inputs, outputs):
    dy_t = dde.grad.jacobian(outputs, inputs[1], j=1)
    dy_x = dde.grad.jacobian(outputs, inputs[1], j=0)
    # 1020100, 101
    ind = [i for _ in range(100) for i in range(101) for _ in range(101)]
    ind = torch.as_tensor(ind, dtype = torch.long)
    ind = ind[..., None]
    vx = torch.gather(inputs[0], 1, ind)
    out = dy_t + vx * dy_x
    return out
    
pde_func = lambda inputs, outputs: adv(inputs, outputs).abs()

def dirichlet(inputs, outputs):
    xt = inputs[1]
    x, t = xt[:, (0,)], xt[:, (1,)]
    return 4 * x * t * outputs + (torch.pi * x).sin() + (torch.pi * t / 2).sin()  

def plot_test():
    i = 0
    test_ivxs = test_vxs[i * 10201: (i + 1) * 10201]
    test_igrid = test_grid[i * 10201: (i + 1) * 10201]
    test_iuxts = test_uxts[i * 10201: (i + 1) * 10201]
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

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

pde = dde.data.TimePDE(geomtime, advection_equation, [], num_domain=250, num_initial=50, num_test=100)

space = GRF_pos(length_scale=0.5)
eval_pts = np.linspace(0, 1, 101)[:, None]
train_vxs = space.eval_batch(space.random(100), eval_pts)
train_grid, train_uxts = advection_solver(train_vxs)

train_grid = train_grid.reshape(101 * 101, -1)
train_grid = np.tile(train_grid, (100, 1))

train_vxs = np.repeat(train_vxs, 10201, axis = 0)
train_uxts = train_uxts.reshape(-1, 1)

testing_path = f"datasets/ADV_100_0.50_101_101.npz"
test_data = np.load(testing_path)
test_vxs = test_data["vxs"]
test_vxs = np.repeat(test_vxs, 10201, axis = 0)
test_grid = test_data["xt"].reshape(-1, 2)
test_grid = np.tile(test_grid, (100, 1))
test_uxts = test_data["uxts"].reshape(-1, 1)

print(train_vxs.shape, train_grid.shape, train_uxts.shape)
print(test_vxs.shape, test_grid.shape, test_uxts.shape)

net = Nomad([101, 100, 100, 100], [102, 100, 100, 1])
# net.apply_output_transform(dirichlet)

data = dde.data.Triple((train_vxs, train_grid), train_uxts, (test_vxs, test_grid), test_uxts)

model = dde.Model(data, net)
model.compile("adam", lr = 1e-3, metrics = ["l2 relative error"], decay=("step", 25000, 0.5))

# %%
# result = model.predict((test_vxs, test_grid), operator = pde_func)
# print(result.mean(), result.std())
plot_test()
# %%
# %%
for _ in range(10):
    model.train(iterations = 10000, batch_size= 5000, test_batch_size = 10000)
    print(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024)
    result = model.predict((test_vxs, test_grid), operator = pde_func)
    print(result.mean(), result.std())
    plot_test()
