from typing import overload
from dde.deepxde.data.function_spaces import FunctionSpace, GRF, Chebyshev, PowerSeries, GRF_KL, GRF2D
import numpy as np

class COS(FunctionSpace):
    def __init__(self, N = 20):
        self._N = N
    
    def random(self, size: int) -> np.ndarray:
        coeff = np.random.randn(size, self._N)
        phase = np.random.rand(size, self._N)
        return np.stack([coeff, phase], axis = 0) # 2, S, N

    def eval_one(self, feature: np.ndarray, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            x = x[:, None]
            # G, 1
        coeff, phi = feature[0], feature[1] # S, N
        freq = np.arange(0, self._N + 1)
        phase = np.einsum("i,jk->ij", freq, x) # N, G
        phase = phase[None, ...] + phi[..., None] # S, N, G
        phase = np.cos(phase) # S, N, G
        phase = phase * coeff[..., None] # (S, N, G)
        return phase.sum(axis = 1)
    
    def eval_batch(self, features: np.ndarray, xs: np.ndarray) -> np.ndarray:
        if len(xs.shape) == 1:
            xs = xs[:, None]
            # G, 1
        coeff, phi = features[0], features[1] # S, N
        freq = np.arange(1, self._N + 1)
        phase = np.einsum("i,jk->ij", freq, xs) # N, G
        phase = phase[None, ...] + phi[..., None] # S, N, G
        phase = np.cos(phase) # S, N, G
        phase = phase * coeff[..., None] # (S, N, G)
        return phase.sum(axis = 1) * np.sqrt(2 / self._N)

class UnionSpace(FunctionSpace):
    def __init__(self, *args):
        if not all([isinstance(arg, FunctionSpace) for arg in args]):
            raise TypeError("args must be FunctionSpace")
        self.space = args
        self._len = len(args)
    
    def random(self, size):
        sizes = np.sort(np.random.randint(0, size + 1, len(self.space) - 1))
        sizes = np.concatenate([[0], sizes, [size]])
        out = []
        for i, (low, up) in enumerate(zip(sizes[:-1], sizes[1:])):
            if low == up:
                out.append(None)
            else:
                out.append(self.space[i].random(up-low))
        return out
                
    def eval_one(self, feature, x):
        out = []
        for i, fea in enumerate(feature):
            if fea is not None:
                out.append(self.space[i].eval_one(fea, x))
        out = np.concatenate(out, axis=0)
        return out
    
    def eval_batch(self, features, xs):
        out = []
        for i, fea in enumerate(features):
            if fea is not None:
                out.append(self.space[i].eval_batch(fea, xs))
        out = np.concatenate(out, axis=0)
        return out