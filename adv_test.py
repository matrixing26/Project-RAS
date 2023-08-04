# %%
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from datasets import makeTesting_adv
from datasets import parallel_solver, advection_solver
from utils.func import dirichlet, periodic
from utils.PDETriple import PDETripleCartesianProd
from utils.pdes import advection_equation

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

pde = dde.data.TimePDE(geomtime, advection_equation, [], num_domain=250, num_initial=50, num_test=100)

func_space = dde.data.GRF(length_scale=0.1)

eval_pts = np.linspace(0, 1, 100)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=32
)

net = dde.nn.pytorch.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "tanh",
    "Glorot normal",
)

def di(inputs, outputs):
    x_trunk = inputs[1] # x_trunk.shape = (t, 2)
    x, t = x_trunk[:, 0], x_trunk[:, 1] # 10201
    # using sine function would have some errors
    scale_factor = (x * t).unsqueeze(0)
    outputs = scale_factor * (outputs + 1)
    xinit = (torch.pi * x).sin().unsqueeze(0)
    tinit = (0.5 * torch.pi * t).sin().unsqueeze(0)
    outputs = outputs + xinit + tinit
    # B N
    return outputs

net.apply_output_transform(di)