from torch import nn

class dirichlet_Deeponet(nn.Module):
    def __init__(self, scale_factor=10):
        super().__init__()
        self.scale_factor = scale_factor
    
    def forward(self, inputs, outputs):
        x_trunk = inputs[1] # x_trunk.shape = (t, 2)
        x, t = x_trunk[:, (0,)], x_trunk[:, (1,)] # 10201
        return self.scale_factor * (x * (1 - x) * t) * (outputs + 1)