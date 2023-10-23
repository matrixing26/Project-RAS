# %%
import numpy as np
import time
import torch
from torch import Tensor
import os

import dde.deepxde as dde
from dataset import makeTesting_bur
from dataset.PDETriple import PDETriple

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

# %%
batchsize = 5000
decay = None
iter = 10000
ls = 1.0
lr = 1e-3
size = 100

train_name = f"datasets/BUR/TRAIN_{size}_{ls:.2f}_101_101.npz"
test_name = f"datasets/BUR/TEST_200_{ls:.2f}_101_101.npz"

os.makedirs("datasets/BUR", exist_ok = True)

if not os.path.exists(train_name):
    makeTesting_bur(length_scale = ls, size = size, name = train_name)
if not os.path.exists(test_name):
    makeTesting_bur(length_scale = ls, size = 200, name = test_name)

# %%
def dirichlet(inputs: Tensor, outputs: Tensor) -> Tensor:
    return outputs

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
print(train_vxs.shape, train_grid.shape, train_uxts.shape)
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
torch.save(model.state_dict(), f"datasets/BUR/PRETRAIN_{size}_{ls:.2f}_{date}.pth")