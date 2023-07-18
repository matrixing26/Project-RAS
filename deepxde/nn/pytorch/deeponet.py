import torch
import numpy as np
from .fnn import FNN
from .nn import NN
from .. import activations


class DeepONetCartesianProd(NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.tensor(0.0, requires_grad=True)

        self.branch_input_size = layer_sizes_branch[0]

        self._X_func_default = None



    @property
    def inputs(self):
        return self._X_func_default

    @inputs.setter
    def inputs(self, value):
        if value[1] is not None:
            raise ValueError("DeepONet does not support setting trunk net input.")
        self._X_func_default = torch.as_tensor(value[0].astype(np.float32)).reshape(-1,self.branch_input_size)
        self._X_func_default.detach()
        #print("already setted")











    def forward(self, inputs):
        is_pinn = False
        x_func = inputs[0]
        x_loc = inputs[1]


        if isinstance(inputs,tuple):#意味着是deeponetcart类型的输入
            #print("x_func",inputs[0].shape,"x_loc",inputs[1].shape)
            x_func = inputs[0]
            x_loc = inputs[1]

        else:#pinn形式
            x_func = self._X_func_default
            x_loc = inputs
            is_pinn = True
            #print("pinn_mode")



                # Branch net to encode the input function
        #print("inputs final",inputs)
        #print("x_func final",x_func.shape)
        x_func = self.branch(x_func)
        #print("branch", x_func.shape)

        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        #print("trunck", x_loc.shape)

        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)

        #print("output",x.shape)

        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)

        if is_pinn == True:
            x = x.transpose(0,1)#如果是pinn，改成pinn的输出格式

        return x


class PODDeepONet(NN):
    """Deep operator network with proper orthogonal decomposition (POD) for dataset in
    the format of Cartesian product.

    Args:
        pod_basis: POD basis used in the trunk net.
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network. If ``None``, then only use POD basis as the trunk net.

    References:
        `L. Lu, X. Meng, S. Cai, Z. Mao, S. Goswami, Z. Zhang, & G. E. Karniadakis. A
        comprehensive and fair comparison of two neural operators (with practical
        extensions) based on FAIR data. arXiv preprint arXiv:2111.05512, 2021
        <https://arxiv.org/abs/2111.05512>`_.
    """

    def __init__(
        self,
        pod_basis,
        layer_sizes_branch,
        activation,
        kernel_initializer,
        layer_sizes_trunk=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)

        self.pod_basis = torch.as_tensor(pod_basis, dtype=torch.float32)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = None
        if layer_sizes_trunk is not None:
            self.trunk = FNN(
                layer_sizes_trunk, self.activation_trunk, kernel_initializer
            )
            self.b = torch.tensor(0.0, requires_grad=True)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]

        x_func = self.branch(x_func)
        if self.trunk is None:
            # POD only
            x = torch.einsum("bi,ni->bn", x_func, self.pod_basis)
        else:
            x_loc = self.activation_trunk(self.trunk(x_loc))
            x = torch.einsum("bi,ni->bn", x_func, torch.cat((self.pod_basis, x_loc), 1))
            x += self.b
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x



class DeepONet(NN):
    """Deep operator network for dataset in the format of

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    """Deep operator network.
        `Lu et al. Learning nonlinear operators via DeepONet based on the universal
        approximation theorem of operators. Nat Mach Intell, 2021.
        <https://doi.org/10.1038/s42256-021-00302-5>`_
        Args:
            layer_sizes_branch: A list of integers as the width of a fully connected
                network, or `(dim, f)` where `dim` is the input dimension and `f` is a
                network function. The width of the last layer in the branch and trunk net
                should be equal.
            layer_sizes_trunk (list): A list of integers as the width of a fully connected
                network.
            activation: If `activation` is a ``string``, then the same activation is used in
                both trunk and branch nets. If `activation` is a ``dict``, then the trunk
                net uses the activation `activation["trunk"]`, and the branch net uses
                `activation["branch"]`.
            trainable_branch: Boolean.
            trainable_trunk: Boolean or a list of booleans.
        """


    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = activations.get(activation)
        if callable(layer_sizes_branch[1]):
            # User-defined network
            self.branch = layer_sizes_branch[1]
        else:
            # Fully connected network
            self.branch = FNN(layer_sizes_branch, activation_branch, kernel_initializer)
        self.trunk = FNN(layer_sizes_trunk, self.activation_trunk, kernel_initializer)
        self.b = torch.tensor(0.0, requires_grad=True)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        if value[1] is not None:
            raise ValueError("DeepONet does not support setting trunk net input.")
        self._X_func_default = value[0]
        self._inputs = self.X_loc





    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Branch net to encode the input function
        x_func = self.branch(x_func)
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc))
        # Dot product
        if x_func.shape[-1] != x_loc.shape[-1]:
            raise AssertionError(
                "Output sizes of branch net and trunk net do not match."
            )
        x = torch.einsum("bi,ni->bn", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x