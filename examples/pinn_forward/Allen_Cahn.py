"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, paddle

Implementation of Allen-Cahn equation example in paper https://arxiv.org/abs/2111.02801.
"""
import deepxde as dde
import numpy as np
from scipy.io import loadmat
import torch
import os
from deepxde.callbacks import Callback

os.environ["DDE_BACKEND"] = "pytorch"

"""
print("Available in dde.nn:", dir(dde.nn))
print("Available in pytorch_nn:", dir(pytorch_nn))
print("Module path:", sys.modules['deepxde.nn.pytorch'].__file__)
print("SHALLOW in pytorch_nn.__dict__:", 'SHALLOW' in pytorch_nn.__dict__)
"""


def gen_testdata():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.dirname(script_dir)
    data_path = os.path.join(examples_dir, "dataset", "Allen_Cahn.mat")
    print(f"Looking for data at: {data_path}")
    data = loadmat(data_path)

    t = data["t"]
    x = data["x"]
    u = data["u"]

    dt = dx = 0.01
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = u.flatten()[:, None]
    return X, y

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
d = 0.001

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - d * dy_xx - 5 * (y - y**3)

data = dde.data.TimePDE(geomtime, pde, [], num_domain=8000, num_boundary=400, num_initial=800)
net = dde.nn.FNN([2] + [60], "tanh", "Glorot normal")

# def output_transform(x, y):
#     return x[:, 0:1]**2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y
# net.apply_output_transform(output_transform)



# [Joe]: The hard constraint ensured. 
# def output_transform(x, y):
#     return x[:, 0:1]**2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1]**2) * y


model = dde.Model(data, net)


class WeightMonitor(Callback):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.initial_weight = net.hidden.weight.detach().clone()
    
    def on_epoch_end(self):
        if self.model.train_state.step % 100 == 0:
            current_weight = self.net.hidden.weight.detach()
            diff = torch.norm(current_weight - self.initial_weight).item()
            print(f"\nStep {self.model.train_state.step} weight diff: {diff}")
            print("requires_grad:", self.net.hidden.weight.requires_grad)

# Create the monitor and add to training
monitor = WeightMonitor(net)

# Add debug output
print("=== DEBUG INFO ===")
print(f"Using nn module from: {dde.nn.__file__}")
print(f"SHALLOW class is available: {'SHALLOW' in dir(dde.nn)}")

# Reduce training iterations and increase verbosity for debugging
model.compile("adam", lr=1e-2)  # Use a smaller learning rate
print("\nStarting training with smaller learning rate...")
losshistory, train_state = model.train(iterations=1000, verbose=1, callbacks=[monitor])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt("test.dat", np.hstack((X, y_true, y_pred)))

# Get the final hidden layer parameters AFTER training
final_hidden_weights = net.hidden.weight.detach().clone()
final_hidden_bias = net.hidden.bias.detach().clone()

# Store these for later access
np.save("hidden_weights.npy", final_hidden_weights.cpu().numpy())
np.save("hidden_bias.npy", final_hidden_bias.cpu().numpy())

# Get final output layer parameters AFTER training
final_output_weights = net.output.weight.detach().clone()
final_output_bias = net.output.bias.detach().clone()

# Count final zeros in output layer
final_zero_weights, _ = count_zeros(final_output_weights, threshold)
final_zero_bias, _ = count_zeros(final_output_bias, threshold)

print("\n=== AFTER TRAINING ===")
print(f"Output weights zeros: {final_zero_weights}/{total_weights} "
      f"({100.0 * final_zero_weights / total_weights:.2f}%)")
print(f"Output bias zeros: {final_zero_bias}/{total_bias} "
      f"({100.0 * final_zero_bias / total_bias:.2f}%)")

print("\n=== COMPARISON ===")
print(f"New zeros in weights: {final_zero_weights - initial_zero_weights}")
print(f"New zeros in bias: {final_zero_bias - initial_zero_bias}")

# Check if hidden weights changed (they shouldn't have, since they're fixed)
weights_changed = not torch.allclose(initial_hidden_weights, final_hidden_weights)
bias_changed = not torch.allclose(initial_hidden_bias, final_hidden_bias)
print("\n=== HIDDEN LAYER CHECK ===")
print(f"Hidden weights changed: {weights_changed}")
print(f"Hidden bias changed: {bias_changed}")
print(f"Hidden weights diff norm: {torch.norm(final_hidden_weights - initial_hidden_weights).item()}")
print(f"Hidden bias diff norm: {torch.norm(final_hidden_bias - initial_hidden_bias).item()}")
print(dde.saveplot.__module__)  # Shows which module contains saveplot






