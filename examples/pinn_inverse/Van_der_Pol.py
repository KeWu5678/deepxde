#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 18:30:52 2024

@author: chaoruiz
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import deepxde as dde
import os
from scipy.spatial import KDTree

"""
Train the network with the dataset (x, V(x), dV/dx)
"""

def gen_traindata():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.dirname(script_dir)
    data_path = os.path.join(examples_dir, "dataset", "VDP_beta_0.5.npy")
    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    return data["x0"], data["dV0"], data["V"]

# Get the raw data
raw_x, raw_dv, raw_v = gen_traindata()

# Create a mask to filter out rows with NaN values in either v or dv
valid_mask = ~(np.isnan(raw_v) | np.any(np.isnan(raw_dv), axis=1))

# Apply the mask to all three arrays
ob_x = raw_x[valid_mask]
ob_dv = raw_dv[valid_mask]
ob_v = raw_v[valid_mask]

# Normalize data
# Save normalization parameters for later use
x_mean = np.mean(ob_x, axis=0)
x_std = np.std(ob_x, axis=0)
v_mean = np.mean(ob_v)
v_std = np.std(ob_v)

# Normalize the coordinates and value function
ob_x_normalized = (ob_x - x_mean) / x_std
ob_v_normalized = (ob_v - v_mean) / v_std

# Print info about filtered data
print(f"Original data points: {len(raw_x)}")
print(f"Valid data points: {len(ob_x)}")
print(f"Removed {len(raw_x) - len(ob_x)} points with NaN values")

def VdV(x, y, ex):
    y = y[:, 0:1]
    v1 = ex[:, 0:1]
    v2 = ex[:, 1:2]
    dy_dx1 = dde.grad.jacobian(y, x, i=0, j = 0)
    dy_dx2 = dde.grad.jacobian(y, x, i=0, j = 1)
    return [
        v1 - dy_dx1,
        v2 - dy_dx2,
    ]

geom = dde.geometry.Rectangle([0, 0], [3, 3])

def aux_function(x):
    """Return the auxiliary variables (dV/dx values) for the given points x."""
    # Print diagnostic information about the input
    print(f"aux_function called with {len(x)} points")
    print(f"First point shape: {x[0].shape}, dtype: {x[0].dtype}")
    print(f"ob_x shape: {ob_x.shape}, dtype: {ob_x.dtype}")
    
    # Check if all points in x are in ob_x
    for i, point in enumerate(x):
        found = False
        for j, ref_point in enumerate(ob_x):
            if np.array_equal(point, ref_point):
                found = True
                break
        if not found:
            #print(f"Point {i} not found in ob_x: {point}")
            #print(f"Closest point in ob_x: {ob_x[np.argmin(np.sum((ob_x - point)**2, axis=1))]}")
            # Print the difference to see how close they are
            closest_idx = np.argmin(np.sum((ob_x - point)**2, axis=1))
            diff = point - ob_x[closest_idx]
            #print(f"Difference: {diff}, magnitude: {np.linalg.norm(diff)}")
    
    # Use a more robust approach with KDTree
    if not hasattr(aux_function, 'kdtree'):
        aux_function.kdtree = KDTree(ob_x)
    
    # Find indices of closest points
    distances, indices = aux_function.kdtree.query(x, k=1)
    
    # Print information about matches
    if np.max(distances) > 0:
        print(f"Max distance: {np.max(distances)}, mean distance: {np.mean(distances)}")
    
    return ob_dv[indices]

def value_function(x):
    """Return the value function V at points x."""
    # Use KDTree for efficient and robust matching
    if not hasattr(value_function, 'kdtree'):
        value_function.kdtree = KDTree(ob_x)
    
    # Find indices of closest points
    _, indices = value_function.kdtree.query(x, k=1)
    
    # Return corresponding V values
    return ob_v[indices].reshape(-1, 1)  # Make sure it's a column vector

data = dde.data.PDE(
    geom,
    VdV,
    [],
    num_domain=0,
    num_boundary=0,
    anchors=ob_x,
    auxiliary_var_function=aux_function,
    solution=value_function
)

# net = dde.nn.SHALLOW([2] + [60], "tanh", "Glorot normal", 1, regularization=('l1', 0.0))
net = dde.nn.FNN([2] + [200] * 4 + [1], "relu", "Glorot uniform", regularization=["l2", 0.01])
model = dde.Model(data, net)
model.compile("adam", lr=0.005, loss="mse", loss_weights=[1.0, 0.01])
losshistory, train_state = model.train(iterations=200000, display_every=1000)


