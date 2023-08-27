# %%
import numpy as np
import time
import torch
from torch import Tensor
import os

import deepxde.deepxde as dde
from datasets import makeTesting_dr
from utils.PDETriple import PDETriple

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

# %%
batchsize = 5000
decay = None
iter = 20000
ls = 0.1
lr = 1e-3
size = 50

train_name = f"datasets/DF/TRAIN_{size}_{ls:.2f}_101_101.npz"
test_name = f"datasets/DF/TEST_100_{ls:.2f}_101_101.npz"

os.makedirs("datasets/DF", exist_ok = True)

if not os.path.exists(train_name):
    makeTesting_dr(length_scale = ls, size = size, name = train_name)
if not os.path.exists(test_name):
    makeTesting_dr(length_scale = ls, size = 100, name = test_name)
    
# %%
def dirichlet(inputs: Tensor, outputs: Tensor) -> Tensor:
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, (0,)], x_trunk[:, (1,)] # 10201
    # using sine function would have some errors
    scale_factor = 10 * (x * (1 - x) * t)
    return scale_factor * (outputs + 1)

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

net = dde.nn.pytorch.DeepONet(
    layer_sizes_branch = [101, 100, 100],
    layer_sizes_trunk = [2, 100, 100, 100],
    activation = "gelu",
    kernel_initializer = "Glorot normal",
)

net.apply_output_transform(dirichlet)

data = PDETriple(X_train=(train_vxs, train_grid), 
                 y_train=train_uxts,
                 X_test=(test_vxs, test_grid), 
                 y_test=test_uxts, 
                 boundary = []
                 )

model = dde.Model(data, net)
model.compile("adam", 
              lr = lr, 
              loss = "mse", 
              metrics = ["l2 relative error"], 
              decay = decay)

# %%
losshistory, train_state = model.train(iterations = iter, batch_size = batchsize)
dde.utils.plot_loss_history(losshistory)

# %%
torch.save(model.state_dict(), f"datasets/DF/PRETRAIN_{size}_{ls:.2f}_{date}.pth")