import numpy as np
import dde.deepxde as dde
from dde.deepxde import Model
from dde.deepxde.data import Data
from dde.deepxde.data.sampler import BatchSampler
import dde.deepxde.backend as bkd
from torch import Tensor
from typing import Callable, List, Tuple, Union, Optional, Any, Literal
import inspect

class PDETripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, 
                 X_train: tuple[np.ndarray] = ..., 
                 y_train: np.ndarray | None = None, 
                 X_test: tuple[np.ndarray] = ..., 
                 y_test: np.ndarray | None = None, 
                 boundary: List[Tuple[Tuple[int], Callable[[Any], Any]]] = []):
        """
        _summary_

        Args:
            X_train (np.ndarray): _description_
            y_train (np.ndarray): _description_
            X_test (np.ndarray): _description_
            y_test (np.ndarray): _description_
            boundary (List[Tuple[Tuple[int], Callable[[Any], Any]]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        # if len(X_train[0]) * len(X_train[1]) != y_train.size:
        #     raise ValueError("The training dataset does not have the format of Cartesian product.")
        # if len(X_test[0]) * len(X_test[1]) != y_test.size:
        #     raise ValueError("The testing dataset does not have the format of Cartesian product.")
        
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test
        self.boundary = boundary

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True) 
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets: Tensor, outputs: Tensor, loss_fn: List[Callable[[Any], Any]] | Callable[[Any], Any], inputs: Tensor, model: Model, aux = None):
        if not isinstance(loss_fn, list):
            loss_fn = [loss_fn]
            
        fn_losses = []
        for fn in loss_fn:
            losses = []
            if is_mix(fn):
                for i in range(outputs.shape[0]):
                    out = outputs[i, :, None]
                    inp = (inputs[0][i, :, None], inputs[1])
                    tar = targets[i, :, None]
                    if is_y(fn):
                        losses.append(fn(inputs = inp, y_pred = out, y_true = tar))
                    else:
                        losses.append(fn(inputs = inp, targets = tar, outputs = out))
            elif is_pinn(fn):
                for i in range(outputs.shape[0]):
                    out = outputs[i, :, None]
                    inp = (inputs[0][i, :, None], inputs[1])
                    if is_y(fn):
                        losses.append(fn(inputs = inp, y_pred = out))
                    else:
                        losses.append(fn(inputs = inp, outputs = out))
            elif is_data(fn):
                if is_y(fn):
                    losses.append(fn(y_pred = outputs, y_true = targets))
                else:
                    losses.append(fn(targets = targets, outputs = outputs))
            else:
                raise ValueError("The loss function is not valid. The loss function must be a PINN(inputs, outputs), a Data(outputs, targets), or a Mix(inputs, outputs, targets).")
            # Use stack instead of as_tensor to keep the gradients.
            losses = bkd.reduce_mean(bkd.stack(losses, 0))
            fn_losses.append(losses)

        for (indices, fn) in self.boundary:
            losses = []
            if is_mix(fn):
                for i in range(outputs.shape[0]):
                    out = outputs[i, indices, None]
                    inp = (inputs[0][i, :, None], inputs[1])
                    tar = targets[i, indices, None]
                    if is_y(fn):
                        losses.append(fn(inputs = inp, y_pred = out, y_true = tar))
                    else:
                        losses.append(fn(inputs = inp, targets = tar, outputs = out))
            elif is_pinn(fn):
                for i in range(outputs.shape[0]):
                    out = outputs[i, indices, None]
                    inp = (inputs[0][i, :, None], inputs[1])
                    if is_y(fn):
                        losses.append(fn(inputs = inp, y_pred = out))
                    else:
                        losses.append(fn(inputs = inp, outputs = out))
            elif is_data(fn):
                out = outputs[:, indices]
                tar = targets[:, indices]
                if is_y(fn):
                    losses.append(fn(y_pred = out, y_true = tar))
                else:
                    losses.append(fn(targets = tar, outputs = out))
            else:
                raise ValueError("The loss function is not valid. The loss function must be a PINN(inputs, outputs), a Data(outputs, targets), or a Mix(inputs, outputs, targets).")
            # Use stack instead of as_tensor to keep the gradients.
            losses = bkd.reduce_mean(bkd.stack(losses, 0))
            fn_losses.append(losses)
            
        return fn_losses

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            inps, grid = self.train_x[0][indices], self.train_x[1]
            tgts = None if self.train_y is None else self.train_y[indices]
            return (inps, grid), tgts
        else:
            indices_branch = self.branch_sampler.get_next(batch_size[0])
            indices_trunk = self.trunk_sampler.get_next(batch_size[1])
            inps, grid = self.train_x[0][indices_branch], self.train_x[1][indices_trunk]
            tgts = None if self.train_y is None else self.train_y[indices_branch, indices_trunk]
            return (inps, grid), tgts

    def test(self):
        return self.test_x, self.test_y

class PDETriple(Data):
    def __init__(self, 
                 X_train: Tuple[np.ndarray, np.ndarray] = ..., 
                 y_train: Union[np.ndarray, None] = None, 
                 X_test: Tuple[np.ndarray, np.ndarray] = ..., 
                 y_test: Union[np.ndarray, None] = None, 
                 boundary: List[Tuple[Tuple[int], Callable[[Any], Any]]] = [],
                 data_format: Literal["CartesianProd", "Normal"] = "CartesianProd"):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test
        self.boundary = boundary
        self.data_format = data_format
        if data_format == "CartesianProd":
            self.test_x, self.test_y = (self.test_x[0].repeat(len(self.test_x[1]),0), np.tile(self.test_x[1],(len(self.test_x[0]),1))), self.test_y.reshape(-1,1)
            self.train_sampler = BatchSampler(len(self.train_x[0]) * len(self.train_x[1]), shuffle=True)
        else:
            self.train_sampler = BatchSampler(len(self.train_x[0]), shuffle=True)
    
    def __len__(self):
        return len(self.train_x[0])
    
    def add_funcs(self, funcs, outs = None):
        if self.data_format != "CartesianProd":
            self.train_x = (np.concatenate([self.train_x[0], funcs], axis = 0), np.concatenate([self.train_x[1], np.tile(self.train_x[1], (len(funcs), 1))], axis = 0))
            self.train_sampler = BatchSampler(len(self.train_x[0]), shuffle=True)

            if outs is not None:
                self.train_y = np.concatenate([self.train_y, outs], axis = 0)
        else:
            self.train_x = (np.concatenate([self.train_x[0], funcs], axis = 0), self.train_x[1])
            self.train_sampler = BatchSampler(len(self.train_x[0]) * len(self.train_x[1]), shuffle=True)

            if outs is not None:
                self.train_y = np.concatenate([self.train_y, outs], axis = 0)
        
    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if not isinstance(loss_fn, list):
            loss_fn = [loss_fn]
            
        fn_losses = []
        for fn in loss_fn:
            losses = []
            if is_mix(fn):
                if is_y(fn):
                    losses.append(fn(inputs = inputs, y_pred = outputs, y_true = targets))
                else:
                    losses.append(fn(inputs = inputs, targets = targets, outputs = outputs))
            elif is_pinn(fn):
                if is_y(fn):
                    losses.append(fn(inputs = inputs, y_pred = outputs))
                else:
                    losses.append(fn(inputs = inputs, outputs = outputs))
            elif is_data(fn):
                if is_y(fn):
                    losses.append(fn(y_pred = outputs, y_true = targets))
                else:
                    losses.append(fn(targets = targets, outputs = outputs))
            else:
                raise ValueError("The loss function is not valid. The loss function must be a PINN(inputs, outputs), a Data(outputs, targets), or a Mix(inputs, outputs, targets).")
            # Use stack instead of as_tensor to keep the gradients.
            losses = bkd.reduce_mean(bkd.stack(losses, 0))
            fn_losses.append(losses)

        for (indices, fn) in self.boundary:
            losses = []
            inputs = (inputs[0][indices], inputs[1][indices])
            outputs = outputs[indices] if outputs is not None else None
            targets = targets[indices] if targets is not None else None
            if is_mix(fn):
                if is_y(fn):
                    losses.append(fn(inputs = inputs, y_pred = outputs, y_true = targets))
                else:
                    losses.append(fn(inputs = inputs, targets = targets, outputs = outputs))
            elif is_pinn(fn):
                if is_y(fn):
                    losses.append(fn(inputs = inputs, y_pred = outputs))
                else:
                    losses.append(fn(inputs = inputs, outputs = outputs))
            elif is_data(fn):
                if is_y(fn):
                    losses.append(fn(y_pred = outputs, y_true = targets))
                else:
                    losses.append(fn(targets = targets, outputs = outputs))
            else:
                raise ValueError("The loss function is not valid. The loss function must be a PINN(inputs, outputs), a Data(outputs, targets), or a Mix(inputs, outputs, targets).")
            # Use stack instead of as_tensor to keep the gradients.
            losses = bkd.reduce_mean(bkd.stack(losses, 0))
            fn_losses.append(losses)
            
        return fn_losses

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        if self.data_format == "CartesianProd":
            func_ind = indices // self.train_x[1].shape[0]
            grid_ind = indices % self.train_x[1].shape[0]
            funcs, grid = self.train_x[0][func_ind], self.train_x[1][grid_ind]
            out = self.train_y[func_ind, grid_ind][...,None] if self.train_y is not None else None
        else:
            funcs, grid = self.train_x[0][indices], self.train_x[1][indices]
            out = self.train_y[indices] if self.train_y is not None else None
        return (funcs, grid), out

    def test(self):
        return self.test_x, self.test_y
            

def is_pinn(func: Callable[[Any], Any]) -> bool:
    func_sig = inspect.signature(func).parameters
    if ("inputs" in func_sig) and ("outputs" in func_sig or "y_pred" in func_sig):
        return True
    else:
        return False

def is_data(func: Callable[[Any], Any]) -> bool:
    func_sig = inspect.signature(func).parameters
    if ("targets" in func_sig or "y_true" in func_sig) and ("outputs" in func_sig or "y_pred" in func_sig):
        return True
    else:
        return False

def is_mix(func: Callable[[Any], Any]) -> bool:
    if is_pinn(func) and is_data(func):
        return True
    else:
        return False
    
def is_y(func: Callable[[Any], Any]) -> bool:
    func_sig = inspect.signature(func).parameters
    if ("y_true" in func_sig) or ("y_pred" in func_sig):
        return True
    else:
        return False

class boundary():
    def __init__(self, loss_coeff = 1, value = 0):
        self.loss_coeff = loss_coeff
        self.value = value
    
    def __call__(self, targets, outputs):
        return self.loss_coeff * (outputs - self.value).abs().mean()