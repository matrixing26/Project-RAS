# %%
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os

import dde.deepxde as dde
from dataset import makeTesting_dr
from utils.library import dirichlet
from dataset.PDETriple import PDETripleCartesianProd

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

# %%
batchsize = 10
decay = None
iter = 40000
ls = 0.1
lr = 1e-3
size = 20

train_name = f"datasets/DF/TRAIN_{size}_{ls:.2f}_101_101.npz"
test_name = f"datasets/DF/TEST_{size}_{ls:.2f}_101_101.npz"

os.makedirs("datasets/DF", exist_ok = True)

if not os.path.exists(train_name):
    makeTesting_dr(length_scale = ls, size = size, name = train_name)
if not os.path.exists(test_name):
    makeTesting_dr(length_scale = ls, size = size, name = test_name)
    
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

net = dde.nn.pytorch.DeepONetCartesianProd(
    layer_sizes_branch = [101, 100, 100],
    layer_sizes_trunk = [2, 100, 100, 100],
    activation = "gelu",
    kernel_initializer = "Glorot normal",
)

net.apply_output_transform(dirichlet)

data = PDETripleCartesianProd(X_train=(train_vxs, train_grid), y_train=train_uxts, 
                              X_test=(test_vxs, test_grid), y_test=test_uxts, 
                              boundary = [])

model = dde.Model(data, net)
model.compile("adam", 
              lr = lr, 
              loss = "mse", 
              metrics = ["mean l2 relative error"], 
              decay = decay)

# %%
losshistory, train_state = model.train(iterations = iter, batch_size = batchsize)
dde.utils.plot_loss_history(losshistory)

# %%
torch.save(model.state_dict(), f"datasets/DF/PRETRAIN_{size}_{ls:.2f}_{date}.pth")