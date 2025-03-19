"""Package for pytorch NN modules."""

__all__ = [
    "DeepONet",
    "DeepONetCartesianProd",
    "FNN",
    "MIONetCartesianProd",
    "NN",
    "PFNN",
    "PODDeepONet",
    "PODMIONet",
    "SHALLOW",
]

from .deeponet import DeepONet, DeepONetCartesianProd, PODDeepONet
from .mionet import MIONetCartesianProd, PODMIONet
from .fnn import FNN, PFNN
from .shallow import SHALLOW
from .nn import NN
