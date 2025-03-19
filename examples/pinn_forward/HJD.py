"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
# Backend tensorflow.compat.v1 or tensorflow
from deepxde.backend import tf
# Backend pytorch
# import torch
# Backend jax
# import jax.numpy as jnp
# Backend paddle
# import paddle


def gen_traindata():
    data = np.load("../dataset/HJD.npz")
    return data["x"], data["V"], data["dV"]

geom = dde.geometry.Interval(-3, 3)
# Specify the Interval  

def HJB(x, v):
    v1, v2 = v[:, 0:1], v[:, 1:2]
    dv1 = dde.grad.jacobian(v1, x, i=0)
    return dv1 - v2


""" The input transormation by the Steroegraphic Coordinate Transformation """


def input_transform(x):



""" The Greedy Insertion Algorithm """

""" The Training Step"""

geom = dde.geometry.Interval(-3, 3)

def ode(x, v):
    dv = dde.gra

    return dde.grad.jacobian(y, x)
