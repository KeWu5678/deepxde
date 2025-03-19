import torch

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
        self, layer_sizes, activation, kernel_initializer, epsilon, regularization=None, inner_weights = None, inner_bias = None
    ):
        super().__init__()
        if isinstance(layer_sizes, list):
            if not len(layer_sizes) == 2:
                raise ValueError(
                    "This is not a shallow net!"
                )
        else:
            self.activation = activations.get(activation)

        # passing the parameter 
        self.epsilon = epsilon
        self.activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        initializer_zero = initializers.get("zeros") 
        
        # construct the hidden layer
        self.hidden = torch.nn.Linear(layer_sizes[0], layer_sizes[1], dtype=config.real(torch))

        # initilize the innerweights if not given
        if inner_weights is None:
            initializer(self.hidden.weight)
            torch.nn.init.normal_(self.hidden.bias, mean=0.0, std=0.1)
            
            # Normalize the weights
            combined_hidden = torch.cat([self.hidden.weight, self.hidden.bias.unsqueeze(1)], dim=1)
            combined_hidden /= torch.norm(combined_hidden, dim=1, keepdim=True)
            
            # Set weights and bias as buffers instead of parameters
            # Delete the existing parameters first
            del self.hidden.weight
            del self.hidden.bias
            
            # Register as buffers instead
            self.hidden.register_buffer('weight', combined_hidden[:, :-1].detach())
            self.hidden.register_buffer('bias', combined_hidden[:, -1].detach())
        else:
            # Delete the existing parameters first
            del self.hidden.weight
            del self.hidden.bias
            
            # Register as buffers instead
            self.hidden.register_buffer('weight', inner_weights.clone().detach())
            self.hidden.register_buffer('bias', inner_bias.clone().detach())

        # initializing the output layer
        self.output = torch.nn.Linear(layer_sizes[1], 1, dtype=config.real(torch))
        initializer(self.output.weight)
        initializer_zero(self.output.bias)

        # register the regularization
        self.regularizer = regularization

    def forward(self, x):
        # Manual implementation of linear layer using buffers
        x = torch.nn.functional.linear(x, self.hidden.weight, self.hidden.bias)
        x = self.activation(x) ** self.epsilon
        x = self.output(x)
        return x


    # Add this method to the SHALLOW class
    def get_hidden_params(self):
        """Return the fixed parameters of the hidden layer."""
        return {
            'weight': self.hidden.weight.clone(),
            'bias': self.hidden.bias.clone()
        }