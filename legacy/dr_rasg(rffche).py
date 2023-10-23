# %%
import dde.deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time
import torch
from torch import Tensor

from dataset import parallel_solver, diffusion_reaction_solver
from utils.library import plot_data, RFFCHE
from dataset.PDETriple import PDETriple

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
batchsize = 5000
decay = None
iters = 10000
ls = 0.1
lr = 1e-3
total_num = 200
test_num = 100
test_points = 20000
test_select_num = 1

train_name = "datasets/DF/TRAIN_50_RC1010_101_101.npz"
test_name = "datasets/DF/TEST_1000_RC1010_101_101.npz"
pretrain_path = "datasets/DF/PRETRAIN_50_RC1010_20230917-23-53-50.pth"
modelsave_path = f"results/DF/rasg_{date}.pth"
csv_path = f"results/DF/rasg_{date}.csv"

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

def plotdata(i, data = "train"):
    vx = train_vxs[i] if data == "train" else test_vxs[i]
    grid = train_grid if data == "train" else test_grid
    vx = vx[None, :].repeat(grid.shape[0], axis = 0)
    uxt = train_uxts[i] if data == "train" else test_uxts[i]
    out = model.predict((vx, grid),
                        batch_size= grid.shape[0])
    plot_data(vx[0], grid, out[:, 0], uxt)
    

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

# %%
data = PDETriple(X_train=(train_vxs, train_grid), 
                 y_train=train_uxts, 
                 X_test=(test_vxs, test_grid), 
                 y_test=test_uxts, 
                 boundary = []
                 )

# Net
net = dde.nn.pytorch.DeepONet(
    layer_sizes_branch = [101, 100, 100],
    layer_sizes_trunk = [2, 100, 100, 100],
    activation = "gelu",
    kernel_initializer = "Glorot normal",
)

net.apply_output_transform(dirichlet)
net.load_state_dict(torch.load(pretrain_path))

model = dde.Model(data, net)

model.compile("adam", lr = lr, metrics = ["l2 relative error"], decay = decay)
plotdata(0, "train")
plotdata(0, "test")

# %%
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
func_space = RFFCHE(sigma = 1/ls)

# %%
while len(train_vxs) < total_num:
    # generate some vxs to test
    pde_data = dde.data.TimePDE(geomtime, 
                                DF, 
                                [], 
                                num_domain = test_points)
    
    eval_pts = np.linspace(0, 1, 101)[:, None] # generate 1000 random vxs
    testing_new_data = dde.data.PDEOperatorCartesianProd(pde_data, func_space, eval_pts, 100, [0])
    (vxs, grid), _, auxs = testing_new_data.train_next_batch()
    outs = []
    for vx, aux in zip(vxs, auxs):
        aux = aux[:, None]
        vx = vx[None, :].repeat(test_points, axis = 0)
        out = model.predict((vx, grid), 
                            aux_vars = aux, 
                            operator = DF, 
                            batch_size= test_points)
        outs.append(out[:, 0])
    outs = np.asarray(outs)
    res = np.mean(outs, axis = 1)
    print(f"PDE residuals: {res.mean():.2e}, Std: {res.std():.2e}")
    
    select_num = min(test_select_num, total_num - len(train_vxs))
    topk_index = np.argpartition(res, -select_num)[-select_num:] # 
    topk_vxs = vxs[topk_index]
    uxts = parallel_solver(diffusion_reaction_solver, topk_vxs, num_workers = 0)
    uxts = np.asarray([u for grid, u in uxts]).reshape(-1, 101 * 101)

    # then add the new data to the training set, and train the model
    train_vxs = np.concatenate([train_vxs, topk_vxs], axis = 0)
    train_uxts = np.concatenate([train_uxts, uxts], axis = 0)
    
    print(f"Train with: {len(train_vxs)} data")
    data = PDETriple(X_train=(train_vxs, train_grid), 
                     y_train=train_uxts, 
                     X_test=(test_vxs, test_grid), 
                     y_test=test_uxts, 
                     boundary = [])
    
    model = dde.Model(data, net)
    
    model.compile("adam", 
                  lr = lr, 
                  metrics = ["l2 relative error"],
                  decay = decay,)
    
    losshistory, train_state = model.train(iterations=iters if len(train_vxs) % 10 == 0 else 1000, 
                                           batch_size = batchsize)
    
    pd_frame = losshistory.to_pandas()
    os.makedirs("results/DF", exist_ok=True)
    if os.path.exists(csv_path):
        pd_frame = pd.concat([pd.read_csv(csv_path), pd_frame], axis = 0, ignore_index=True)
    pd_frame.to_csv(csv_path, index=False)
    
    if len(train_vxs) % 10 == 0:
        dde.utils.plot_loss_history(losshistory)
        plotdata(0, "train")
        plotdata(0, "test")
        plt.show()

torch.save(model.state_dict(), modelsave_path)


