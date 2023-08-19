# %%
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from deepxde.deepxde.nn.pytorch import FNN, NN
from datasets import makeTesting_dr
from datasets import parallel_solver, diffusion_reaction_solver
from utils.PDETriple import PDETripleCartesianProd
torch.cuda.set_device(1)
date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%
total_training_vx = 300
ls = 0.05

start_num = 300
check_num = 1000
select_num = 30
solver_worker = 0

lr_start = 1e-3
lr_middle = 1e-3
lr_end = 1e-3

iter_start = 40000
iter_middle = 20000
iter_end = 60000

batch_start = lambda n: n // 5
batch_middle = lambda n: n // 5
batch_end = lambda n: n

decay_start = None
decay_middle = ("inverse time", 5000, 0.4)
decay_end = ("inverse time", 5000, 0.4)

if False:
    makeTesting_dr(length_scale = ls)

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

def pde(x, y, aux):
    D = 0.01
    k = 0.01
    dy_t = dde.grad.jacobian(y, x[1], j=1)
    dy_xx = dde.grad.hessian(y, x[1], j=0)
    out = dy_t - D * dy_xx + k * y**2 - aux
    return out

def dirichlet(inputs, outputs):
    xt = inputs[1]
    x, t = xt[:, (0,)], xt[:, (1,)]
    return 8 * x * (1 - x) * t * (outputs + 1)

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
    print(out.shape)
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
space = dde.data.GRF(1.0, length_scale = ls, N= 1000, interp="cubic")

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
vxs = space.eval_batch(space.random(start_num), np.linspace(0, 1, 101)[:, None])
uxts = parallel_solver(diffusion_reaction_solver, vxs, num_workers = solver_worker)
grid = uxts[0][0].reshape(101 * 101, -1)
uxts = np.asarray([u for grid, u in uxts]).reshape(-1, 101 * 101)


train_vxs = vxs.astype(np.float32)
train_grid = grid.astype(np.float32)
train_uxts = uxts.astype(np.float32)

train_grid = train_grid.reshape(101 * 101, -1)
train_grid = np.tile(train_grid, (300, 1))

train_vxs = np.repeat(train_vxs, 10201, axis = 0)
train_uxts = train_uxts.reshape(-1, 1)

testing_path = f"datasets/DF_100_{ls:.2f}_101_101.npz"
test_data = np.load(testing_path)
test_vxs = test_data["vxs"]
test_vxs = np.repeat(test_vxs, 10201, axis = 0).astype(np.float32)
test_grid = test_data["xt"].reshape(-1, 2)
test_grid = np.tile(test_grid, (100, 1)).astype(np.float32)
test_uxts = test_data["uxts"].reshape(-1, 1).astype(np.float32)

print(train_vxs.shape, train_grid.shape, train_uxts.shape)
print(test_vxs.shape, test_grid.shape, test_uxts.shape)
# %%
data = dde.data.Triple(X_train=(train_vxs, train_grid), y_train=train_uxts, X_test=(test_vxs, test_grid), y_test=test_uxts)

# Net
net = Nomad([101, 100, 100], [102, 100, 100, 100, 1])
#net = dde.nn.pytorch.DeepONet([101, 100, 100], [2, 100, 100, 100], "gelu", "Glorot normal")
net.apply_output_transform(dirichlet)

model = dde.Model(data, net)
model.compile("adam", 
              lr= lr_start, 
              loss= "mse", 
              metrics = ["l2 relative error"], 
              decay = decay_start)


plot_test()
# %%
losshistory, train_state = model.train(iterations = iter_start, batch_size = 5000)
dde.utils.plot_loss_history(losshistory)
# %%
plot_test()
# %%
losshistory.to_pandas().to_csv(f"results/loss_history_{date}_rasg.csv", index=False)

# %%

while len(train_vxs) < total_training_vx:
    # generate some vxs to test
    pde_data = dde.data.TimePDE(geomtime, pde, [], num_domain = 20000)
    eval_pts = np.linspace(0, 1, 101)[:, None] # generate 1000 random vxs
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    func_space = dde.data.GRF(1.0, length_scale = ls, N= 1000, interp="linear")
    testing_new_data = dde.data.PDEOperatorCartesianProd(pde_data, func_space, eval_pts, check_num, [0])
    # testing_model = dde.Model(testing_new_data, net)
    a, _, c = testing_new_data.train_next_batch()
    out = model.predict(a, aux_vars = c, operator = pde)
    
    res = np.mean(np.abs(out), axis = 1)
    print(np.mean(res), np.std(res))
    select_num = min(select_num, total_training_vx - len(train_vxs))
    topk_index = np.argpartition(res, -select_num)[-select_num:] # select the top 20 vxs
    # print(res, topk_index, res[topk_index])
    topk_vxs = a[0][topk_index]
    uxts = parallel_solver(diffusion_reaction_solver, topk_vxs, num_workers = solver_worker)
    uxts = np.asarray([u for grid, u in uxts]).reshape(-1, 101 * 101)

    # then add the new data to the training set, and train the model
    train_vxs = np.concatenate([train_vxs, topk_vxs], axis = 0)
    train_uxts = np.concatenate([train_uxts, uxts], axis = 0)
    
    print(len(train_vxs))
    data = PDETripleCartesianProd(X_train=(train_vxs, train_grid), y_train=train_uxts, X_test=(test_vxs, test_grid), y_test=test_uxts, boundary = [])
    
    model = dde.Model(data, net)
    lr = lr_middle if len(train_vxs) != total_training_vx else lr_end
    decay = decay_middle if len(train_vxs) != total_training_vx else decay_end
    batchsize = batch_middle(len(train_vxs)) if len(train_vxs) != total_training_vx else batch_end(len(train_vxs))
    iterations = iter_middle if len(train_vxs) != total_training_vx else iter_end
    model.compile("adam", 
                  lr = lr, 
                  metrics = ["mean l2 relative error"],
                  decay = decay,)

    losshistory, train_state = model.train(iterations=iterations, batch_size = batchsize)
    
    pd_frame = losshistory.to_pandas()
    pd_frame = pd.concat([pd.read_csv(f"results/loss_history_{date}_rasg.csv"), pd_frame], axis = 0, ignore_index=True)
    pd_frame.to_csv(f"results/loss_history_{date}_rasg.csv", index=False)
    dde.utils.plot_loss_history(losshistory)
    plt.show()
    
torch.save(model.state_dict(), f"results/model_{date}_rasg.pth")


