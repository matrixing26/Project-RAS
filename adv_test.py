# %%
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import deepxde.deepxde as dde
from datasets import advection_solver
from utils.PDETriple import PDETripleCartesianProd
from utils.pdes import advection_equation
from datasets import GRF_pos

date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
# dde.config.set_random_seed(2023)

# %%

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

pde = dde.data.TimePDE(geomtime, advection_equation, [], num_domain=250, num_initial=50, num_test=100)

func_space = GRF_pos(length_scale=0.5)

eval_pts = np.linspace(0, 1, 100)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde, func_space, eval_pts, 1000, function_variables=[0], num_test=100, batch_size=32
)

net = dde.nn.pytorch.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [2, 128, 128, 128],
    "gelu",
    "Glorot normal",
)

net.apply_output_transform(torch.nn.functional.sigmoid)