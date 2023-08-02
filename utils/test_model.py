import torch
from torch import nn, Tensor
from torch.nn import functional as F
from deepxde.deepxde.nn.pytorch.fnn import FNN
from deepxde.deepxde.nn.pytorch import NN

class normONet(NN):
    def __init__(self):
        super().__init__()
        self.trunk = FNN([2 , 100, 100, 100], F.gelu, "Glorot normal")
        self.branch = FNN([103, 100, 100], F.gelu, "Glorot normal")
        self.norm = FNN([2, 100, 100], F.gelu, "Glorot normal")
        self.b = torch.nn.parameter.Parameter(torch.tensor(0.0))
        self.n = torch.nn.parameter.Parameter(torch.tensor([1.0, 0.0]))
        
    def forward(self, inputs: tuple[Tensor, Tensor]):
        b, t = inputs
        meanb, varb = torch.mean(b, dim = 1, keepdim=True), torch.var(b, dim =1, keepdim=True)
        b = (b - meanb) / torch.sqrt(varb + 1e-5) * self.n[0] + self.n[1]
        n = self.n.unsqueeze(0).repeat(b.shape[0], 1)
        b = torch.cat((b, n), dim = 1)
        t = self.trunk(t)
        b = self.branch(b)
        #n = self.norm(self.n)
        
        # out = torch.einsum("n i, b i, i -> b n", t, b, n) + self.b
        out = torch.einsum("n i, b i -> b n", t, b) + self.b
        return out
    
        
        
        