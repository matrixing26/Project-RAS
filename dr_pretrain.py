# %%
import numpy as np
import time
import torch
from torch import Tensor
import os
from utils.func import COS
import deepxde.deepxde as dde
from datasets.solver import diffusion_reaction_solver
from utils.PDETriple import PDETriple

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())

# %%
batchsize = 5000
decay = None
iter = 20000
ls = 0.1
lr = 1e-3
size = 50

# train_name = f"datasets/DF/TRAIN_{size}_{ls:.2f}_101_101.npz"
# test_name = f"datasets/DF/TEST_1000_{ls:.2f}_101_101.npz"
# save_pth = f"datasets/DF/PRETRAIN_{size}_{ls:.2f}_{date}.pth"
# train_name = f"datasets/DF/TRAIN_{size}_RC1010_101_101.npz"
# test_name = f"datasets/DF/TEST_1000_RC1010_101_101.npz"
# save_pth = f"datasets/DF/PRETRAIN_{size}_RC1010_{date}.pth"
# train_name = f"datasets/DF/TRAIN_{size}_COS.npz" # N = 20
# test_name = f"datasets/DF/TEST_1000_COS.npz"
# save_pth = f"datasets/DF/PRETRAIN_{size}_COS_{date}.pth"
train_name = f"datasets/DF/TRAIN_{size}_POLY.npz" # N = 10
test_name = f"datasets/DF/TEST_1000_POLY.npz"
save_pth = f"datasets/DF/PRETRAIN_{size}_POLY_{date}.pth"

os.makedirs("datasets/DF", exist_ok = True)

# func_space = dde.data.function_spaces.GRF(length_scale = ls)
# func_space = RFFCHE(100, 10, 0, 1/ls)
func_space = dde.data.function_spaces.Chebyshev(N = 10)

if not os.path.exists(train_name):
    vxs = []
    uxts = []
    for i in range(size):
        print(i, end = " ")
        vx = func_space.eval_batch(func_space.random(1), np.linspace(0, 1, 101)[:, None])[0]
        vxs.append(vx)
        xt, uxt = diffusion_reaction_solver(vx)
        uxts.append(uxt)
    vxs = np.stack(vxs, axis = 0, dtype = np.float32)
    uxts = np.stack(uxts, axis = 0, dtype = np.float32)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = train_name or f"datasets/DF_{size}_{ls:.2f}_101_101.npz"
    np.savez(path, info = {"size": size, "grid": (101, 101), "grid_sample": "uniform", "length_scale": ls}, vxs = vxs, uxts = uxts, xt = xt)

if not os.path.exists(test_name):
    vxs = []
    uxts = []
    for i in range(1000):
        print(i, end = " ")
        grf = dde.data.function_spaces.GRF(length_scale = ls)
        vx = func_space.eval_batch(func_space.random(1), np.linspace(0, 1, 101)[:, None])[0]
        vxs.append(vx)
        xt, uxt = diffusion_reaction_solver(vx)
        uxts.append(uxt)
    vxs = np.stack(vxs, axis = 0, dtype = np.float32)
    uxts = np.stack(uxts, axis = 0, dtype = np.float32)
    print("\n",vxs.shape, uxts.shape, xt.shape)
    path = test_name or f"datasets/DF_1000_{ls:.2f}_101_101.npz"
    np.savez(path, info = {"size": 1000, "grid": (101, 101), "grid_sample": "uniform", "length_scale": ls}, vxs = vxs, uxts = uxts, xt = xt)
    
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
torch.save(model.state_dict(), save_pth)