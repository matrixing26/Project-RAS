from deepxde import nn

def DeepONet(layer_sizes_branch, layer_sizes_trunk, activation="gelu", kernel_initializer="Glorot normal"):
    return nn.pytorch.DeepONet(layer_sizes_branch, layer_sizes_trunk, activation, kernel_initializer)