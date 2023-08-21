# %%
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from utils.func import arg_topk
from datasets import burger_solver
from deepxde.deepxde.nn.pytorch import FNN, NN

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
total_training_vx = 1000
ls = 1.0
testing_path = f"datasets/BUR_100_{ls:.2f}_101_101.npz"

start_num = 10
check_num = 20
select_num = 1000

lr = 1e-3

iter_start = 100000

batchsize = 5000

decay = ("step", 25000, 0.5)


# %%
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

class DeepONetLNL(NN):
    def __init__(self, layer_sizes_branch, 
                 layer_sizes_trunk, 
                 activation = F.gelu, 
                 kernel_initializer = "Glorot normal"):
        super().__init__()
        self.t1 = FNN(layer_sizes_trunk,
                      activation=activation,
                      kernel_initializer=kernel_initializer)
        b2 = layer_sizes_trunk.copy()
        b2[0] += layer_sizes_branch[-1]
        self.b2 = FNN(b2,
                      activation=activation,
                      kernel_initializer=kernel_initializer)

        self.b1 = FNN(layer_sizes_branch,
                            activation=activation,
                            kernel_initializer=kernel_initializer)
    
    def forward(self, inputs):
        v, x = inputs
        v = self.b1(v)
        vx = torch.cat([v, x], dim = -1)
        vx = self.b2(vx)
        x = self.t1(x)
        output = (vx * x).sum(dim = -1, keepdim = True)
        if self._output_transform is not None:
            output = self._output_transform(inputs, output)
        return output
        
def bur(inputs: tuple[torch.Tensor, torch.Tensor], outputs: torch.Tensor):
    dy_t = dde.grad.jacobian(outputs, inputs[1], j=1)
    dy_x = dde.grad.jacobian(outputs, inputs[1], j=0)
    dy_xx = dde.grad.hessian(outputs, inputs[1], j=0, i=0)
    out = dy_t + outputs * dy_x - 0.1 * dy_xx
    return out
    
pde_func = lambda inputs, outputs: bur(inputs, outputs).abs()

def dirichlet(inputs, outputs):
    return outputs

def plot_test():
    i = 0
    test_ivxs = test_vxs[i * 10201: (i + 1) * 10201]
    test_igrid = test_grid[i * 10201: (i + 1) * 10201]
    test_iuxts = test_uxts[i * 10201: (i + 1) * 10201]
    test_iresult = result[i * 10201: (i + 1) * 10201]
    v = test_ivxs[0]
    x = np.linspace(0, 1, v.shape[0])
    print(x.shape, v.shape)
    fig, (ax1 ,ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))

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
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_aspect('equal')
    delta = np.abs(test_iuxts.reshape(-1) - out.reshape(-1))
    ax5.scatter(test_igrid[:, 0], test_igrid[:, 1], c = delta)
    colorbar = fig.colorbar(ax5.scatter(test_igrid[:, 0], test_igrid[:, 1], c = test_iresult), ax = ax5)

    plt.tight_layout()
    plt.show()
    
def func_bc(x, v):
    return v
# %%


geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.BC(geomtime, func_bc, lambda _, on_boundary: on_boundary)
pde = dde.data.TimePDE(geomtime, pde_func, [], num_domain=250, num_initial=50, num_test=100)

space = dde.data.GRF(length_scale= ls)
eval_pts = np.linspace(0, 1, 101)[:, None]
train_vxs = space.eval_batch(space.random(start_num), eval_pts)
train_grid, train_uxts = burger_solver(train_vxs)

train_grid = train_grid.reshape(101 * 101, -1)
train_grid = np.tile(train_grid, (start_num, 1))

train_vxs = np.repeat(train_vxs, 10201, axis = 0)
train_uxts = train_uxts.reshape(-1, 1)

test_data = np.load(testing_path)
test_vxs = test_data["vxs"]
test_vxs = np.repeat(test_vxs, 10201, axis = 0)
test_grid = test_data["xt"].reshape(-1, 2)
test_grid = np.tile(test_grid, (100, 1))
test_uxts = test_data["uxts"].reshape(-1, 1)

print(train_vxs.shape, train_grid.shape, train_uxts.shape)
print(test_vxs.shape, test_grid.shape, test_uxts.shape)


net = dde.nn.pytorch.DeepONet([101, 100, 100, 100], [2, 100, 100, 100], "gelu", "Glorot normal")
# net = DeepONetLNL([101, 100, 100, 100], [2, 100, 100, 100])
# net = Nomad([101, 100, 100, 100], [102, 100, 100, 1])
# net.apply_output_transform(dirichlet)

data = dde.data.Triple((train_vxs, train_grid), train_uxts, (test_vxs, test_grid), test_uxts)

model = dde.Model(data, net)
model.compile("adam", lr = lr, metrics = ["l2 relative error"], decay = decay)

# %%
# result = model.predict((test_vxs, test_grid), operator = pde_func)
# print(result.mean(), result.std())
# %%
# %%
for _ in range(10):
    model.train(iterations = 10000, batch_size= 5000, test_batch_size = 10000)
    print(f"Memory used: {torch.cuda.max_memory_allocated()/ (1024 ** 3)}Gb")
    result = model.predict((test_vxs, test_grid), operator = pde_func)
    print(result.mean(), result.std())
    plot_test()
    
# %%
model.predict((test_vxs, test_grid), operator = pde_func)
# %%

timepde = dde.data.TimePDE(geomtime, pde_func, [], num_domain=20000)
eval_pts = np.linspace(0, 1, 101)[:, None]
new_data = dde.data.PDEOperator(timepde, space, eval_pts, check_num, [0])
(vxs, xts), _, c = new_data.train_next_batch()
print(vxs.shape, xts.shape, c.shape)
# %%
