import numpy as np
from dde.deepxde import Model

def active_selection(n, m, net, space, operator = None, solver = None, mode = "RASG-L1"):
    """
    _summary_

    Args:
        n (int): the number of total testing functions
        m (int): the number of selected testing functions
        model (deepxde.Model): _description_
        space (dde.data.Funcspace): _description_
        operator (_type_, optional): _description_. Defaults to None.
        solver (_type_, optional): _description_. Defaults to None.
        mode (str, optional): _description_. Defaults to "RAS-G".
    """
    funcs = []
    outs = []
    model = Model(space, net)
    if mode != "random":
        for i in range(n):
            (func, grid), _, aux = space.train_next_batch()
            space.train_x, space.train_y, space.train_aux_vars = None, None, None
            out = model.predict((func, grid), aux_vars = aux, operator = operator)
            if mode == "RASG-L1":
                outs.append(np.linalg.norm(out, ord = 1))
            elif mode == "RASG-L2":
                outs.append(np.linalg.norm(out, ord = 2))
            funcs.append(func[0])
        funcs = np.stack(funcs, axis = 0)
        funcs = funcs[np.argpartition(outs, -m)[-m:]]
    else:
        for i in range(m):
            (func, _), _, _ = space.train_next_batch()
            funcs.append(func[0])
        funcs = np.array(funcs, axis = 0)
        
    if solver is not None:
        out = map(solver, funcs)
        out = np.array([u.reshape(-1) for u, grid in out])
        return funcs, out
    else:
        return funcs


def active_selection_card(n, model, space, operator = None, solver = None, mode = "RAS-G"): ...