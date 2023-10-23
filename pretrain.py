# %%
import hydra
import numpy as np
import time
import os
import utils

import dde.deepxde as dde
from dataset import PDETriple

# %%

def prepare(cfg, train_path, test_path):
    net = utils.get_model(cfg.pde.model.name, cfg.pde.model.params, cfg.pde.model.input_transform, cfg.pde.model.output_transform)
        
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    
    train_vxs = train_data["funcs"]
    train_grid = train_data["grids"].reshape(-1, 2)
    train_uxts = train_data["out"].reshape(-1, 101 * 101)
    test_vxs = test_data["funcs"]
    test_grid = test_data["grids"].reshape(-1, 2)
    test_uxts = test_data["out"].reshape(-1, 101 * 101)
    
    dts = PDETriple((train_vxs, train_grid), train_uxts, (test_vxs, test_grid), test_uxts)
        
    log = utils.get_logger("Rank 0")
    
    return net, dts, log

def spawn_data(path, size, func_space, solver):
    funcs = []
    out = []
    for i in range(size):
        print(f"Generating {i}/{size}", end = "\r")
        vx = func_space.eval_batch(func_space.random(1), np.linspace(0, 1, 101)[:, None])[0]
        funcs.append(vx)
        uxt, grids = solver(vx)
        out.append(uxt)
    funcs = np.stack(funcs, axis = 0, dtype = np.float32)
    out = np.stack(out, axis = 0, dtype = np.float32)
    np.savez(path, funcs = funcs, out = out, grids = grids)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    solver = utils.get_solver(cfg.pde.solver.name, cfg.pde.solver.params)
    space = utils.get_space(cfg.funcspace.name, cfg.funcspace.params)
    
    os.makedirs(cfg.pde.datasets.workdir, exist_ok=True)
    
    train_path = eval(cfg.pde.datasets.train_path)
    test_path = eval(cfg.pde.datasets.test_path)
    save_path = eval(cfg.pde.datasets.pretrain_path)
    
    if not os.path.exists(train_path):
        print("Making train data")
        spawn_data(train_path, cfg.pde.train.init_train_size, space, solver)
    if not os.path.exists(test_path):
        print("Making test data")
        spawn_data(test_path, cfg.pde.train.test_size, space, solver)
        
    net, data, log = prepare(cfg, train_path, test_path)
    
    model = dde.Model(data, net)
    model.compile("adam", lr = cfg.optimizer.lr, loss = "mse", metrics = ["l2 relative error"], decay = cfg.optimizer.decay)

    # %%
    losshistory, train_state = model.train(iterations = cfg.pde.train.iters, batch_size = cfg.pde.train.batch_size)
    
    utils.model_save(net, save_path)


if __name__ == "__main__":
    main()