# Cell
import torch
import numpy as np

import matplotlib

import matplotlib.pyplot as plt


# Cell
def data_func(x):
    return torch.sin(2 * torch.pi * x)
# Cell
def get_data(n):

    x = torch.rand(n).reshape(-1, 1)

    y = data_func(x) + torch.randn((n, 1)) * 0.2

    return x, y


# Cell
# x, y = get_data(10)
# plt.scatter(x.view(-1).numpy(), y.view(-1).numpy())

# plt_x = torch.linspace(0, 1, 100)
# plt_y = data_func(plt_x)
# plt.plot(plt_x, plt_y)
# plt.show()

def get_stacked(x, M):
    x_stacked = torch.stack(
        tuple((x ** i).view(-1) for i in range(M + 1)),
        dim=1
    )

    return x_stacked

# Cell
def model(x_stacked: torch.Tensor, w:torch.Tensor):
    return x_stacked @ w

# Cell
def mean_squared_err(preds, targets):
    return ((preds - targets) ** 2).sum() / 2

# Cell
def solve_grad(x, y, M):
    w = torch.randn((M + 1, 1))
    w.requires_grad = True

    n_epochs = 10000
    lr = 0.01

    x_stacked = get_stacked(x, M)

    def update_weights(w, lr):
        with torch.no_grad():
            w -= lr * w.grad

    def zero_grad(w):
        if w.grad is not None:
            w.grad.zero_()

    for i in range(n_epochs):

        zero_grad(w)

        preds = model(x_stacked, w)

        loss = mean_squared_err(preds, y)
        loss.backward()
        update_weights(w, lr)

        if i % (n_epochs // 10) == 0:
            print(f'Epoch {i}/{n_epochs - 1}')
            print(f'\tLoss: {loss}')

    return w

# Cell
# w_grad = solve_grad(x, y, 10)

# Cell

def closed_form(x: torch.Tensor, y:torch.Tensor, M:int, lam=None):
    # Make T
    x_flat = x.reshape(-1)
    x_stacked = torch.stack(
        tuple(x_flat ** i for i in range(M + 1)),
        dim = 0
    )
    
    T = x_stacked @ y.reshape(-1, 1)

    # Make A
    A_exponents = torch.arange(0, M + 1, 1) + torch.arange(0, M + 1, 1).reshape(-1, 1)
    A_exponents = A_exponents.reshape(M + 1, M + 1, 1)

    A_xs = torch.zeros(M + 1, M + 1, x.size(1)) + x.reshape(-1)
    A = (A_xs ** A_exponents).sum(2)

    # return A, T

    if lam is not None:
        A += torch.eye(M + 1) * lam

    # w = np.linalg.inv(A.numpy()) @ T.numpy()
    # return torch.tensor(w)
    w = torch.linalg.solve(A, T)
    return w, (A, T)


# Cell
# w_exact = closed_form(x, y, 5)
# A.shape, T.shape


# Cell
def plot_results(x: torch.Tensor, y: torch.Tensor, w:torch.Tensor):
    M = w.size(0) - 1
    # x_stacked = get_stacked(x, M)

    plt.scatter(x.numpy(), y.numpy())
    X_plt = torch.linspace(0, 1, 100)
    X_plt_stacked = get_stacked(X_plt, M)
    with torch.no_grad():
        model_out = model(X_plt_stacked, w)

    plt.plot(X_plt, model_out.reshape(-1), label="predicted")
    plt.plot(X_plt, data_func(X_plt), label="data distribution")
    plt.ylim((-1.2, 1.2))
    plt.legend()


    plt.show()

# Cell
# plot_results(x, y, w_grad)

# Cell
# plot_results(x, y, w_exact)

# Cell
x, y = get_data(10)
w, _ = closed_form(x, y, 10, 0.001)
plot_results(x, y, w)

# Cell
