import torch
import numpy as np
from .nn import NN
from .. import activations
from .. import initializers
from ... import config


class SHALLOW(NN):
    """
    The network where the inner weights are fixed, and outer weights are trainable.
    The inner_weights is a pytorch tensor and admits the dimension size(network) x size(input).
    The layer_sizes should be [net_size, net_size], since the transformation is already done in the network initialization.
    """

    def __init__(
        self, layer_sizes, activation, kernel_initializer, p = 1, inner_weights = None, inner_bias = None, regularization = None,
    ):
        super().__init__()
        if isinstance(layer_sizes, list):
            if not len(layer_sizes) == 3:
                raise ValueError(
                    "This is not a shallow net!"
                )
        else:
            self.activation = activations.get(activation)

        # passing the parameter 
        self.p = p
        if len(regularization) == 3:
            self.alpha = regularization[2]
        else:
            self.alpha = None
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros") 
        
        # construct the hidden layer
        self.hidden = torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=config.real(torch))

        # initilize the innerweights if not given
        if inner_weights is None or inner_bias is None:
            initializer_zero(self.hidden.weight)
            initializer_zero(self.hidden.bias)

            # del self.hidden.weight
            # del self.hidden.bias
            # Register as buffers instead
            # self.hidden.register_buffer('weight')
            # self.hidden.register_buffer('bias')
        else:
            # Delete the existing parameters first
            del self.hidden.weight
            del self.hidden.bias

            if isinstance(inner_weights, np.ndarray):
                inner_weights = torch.tensor(inner_weights, dtype=config.real(torch))
            if isinstance(inner_bias, np.ndarray):
                inner_bias = torch.tensor(inner_bias, dtype=config.real(torch))

            self.hidden.weight = inner_weights.clone().detach()
            self.hidden.bias = inner_bias.clone().detach()
            # Register as buffers instead
            # self.hidden.register_buffer('weight', inner_weights.clone().detach())
            # self.hidden.register_buffer('bias', inner_bias.clone().detach())

        # initializing the output layer
        self.output = torch.nn.Linear(layer_sizes[1], layer_sizes[2], dtype=config.real(torch))
        initializer(self.output.weight)
        initializer_zero(self.output.bias)

        # register the regularization
        self.regularizer = regularization

    def forward(self, x):
        # Manual implementation of linear layer using buffers
        x = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        x = self.activation(x) ** self.p
        x = self.output(x)
        return x


    # Add this method to the SHALLOW class
    def get_hidden_params(self):
        """Return the fixed parameters of the hidden layer."""
        return self.hidden.weight.detach().clone(), self.hidden.bias.detach().clone()