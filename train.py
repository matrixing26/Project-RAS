# %%
import hydra
import numpy as np
import pandas as pd
import time
import os
import utils

import dde.deepxde as dde
from dataset import PDETriple

# %%

def prepare(cfg, train_path, test_path, load_path):
    net = utils.get_model(cfg.pde.model.name, cfg.pde.model.params, cfg.pde.model.input_transform, cfg.pde.model.output_transform)
    if cfg.pde.datasets.pretrain_path is not None:
        match_list = utils.model_load(net, load_path)
    else:
        raise ValueError("Pretrain path is None")
        
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

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg):
    date = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
    work_dir = hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']
    pde = utils.get_pde(cfg.pde.pde.name, cfg.pde.pde.params)
    space = utils.get_space(cfg.funcspace.name, cfg.funcspace.params)
    solver = utils.get_solver(cfg.pde.solver.name, cfg.pde.solver.params)
    csv_path = f"{work_dir}/{utils.get_space_name(space)}_{cfg.pde.train.init_train_size}.csv"
    model_path = f"{work_dir}/{utils.get_space_name(space)}_{cfg.pde.train.init_train_size}_{cfg.pde.train.final_train_size}.pth"
    
    train_path = eval(cfg.pde.datasets.train_path)
    test_path = eval(cfg.pde.datasets.test_path)
    load_path = eval(cfg.pde.datasets.pretrain_path)
        
    net, data, log = prepare(cfg, train_path, test_path, load_path)
    # model.compile("adam", lr = cfg.optimizer.lr, loss = "mse", metrics = ["l2 relative error"], decay = cfg.optimizer.decay)
    
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    
    while len(data) < cfg.pde.train.final_train_size:
        pde_data = dde.data.TimePDE(geomtime, pde, [], num_domain = cfg.pde.train.test_point)
        eval_pts = np.linspace(0, 1, 101)[:, None] # generate 1000 random vxs
        testing_space = dde.data.PDEOperator(pde_data, space, eval_pts, 1, [0])
        funcs, labels = utils.active.active_selection(cfg.pde.active.funcs, cfg.pde.active.select, net, testing_space, pde, solver, mode = cfg.pde.active.mode)
        data.add_funcs(funcs, labels)
        
        print(f"Train with: {len(data)} data")
        
        model = dde.Model(data, net)
        model.compile("adam", lr = cfg.optimizer.lr, loss = "mse", metrics = ["l2 relative error"], decay = cfg.optimizer.decay)
        losshistory, _ = model.train(iterations=cfg.pde.train.iters if len(data) % 10 == 0 else cfg.pde.active.iters, batch_size = cfg.pde.train.batch_size)
        pd_frame = losshistory.to_pandas()
        if os.path.exists(csv_path):
            pd_frame = pd.concat([pd.read_csv(csv_path), pd_frame], axis = 0, ignore_index=True)
        pd_frame.to_csv(csv_path, index=False, float_format="%.3e")
        
    utils.model_save(net, model_path)


if __name__ == "__main__":
    main()