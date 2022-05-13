# Cell
import torch

import matplotlib.pyplot as plt


# Cell
def data_func(x):
    return torch.sin(2 * torch.pi * x)
# Cell
def get_data(n):

    x = torch.linspace(0, 1, n).reshape(-1, 1)

    y = data_func(x) + torch.randn((n, 1)) * 0.2

    return x, y


# Cell
x, y = get_data(10)
plt.scatter(x.view(-1).numpy(), y.view(-1).numpy())

plt_x = torch.linspace(0, 1, 100)
plt_y = data_func(plt_x)
plt.plot(plt_x, plt_y)
plt.show()


# Cell
M = 10

def model(x: torch.Tensor, w:torch.Tensor):
    x_stacked = torch.stack(
        tuple((x ** i).view(-1) for i in range(M + 1)),
        dim=1
    )
    return x_stacked @ w

# Cell
# first try fitting with gradient descent
w = torch.randn((M + 1, 1))
w.requires_grad = True

plt.plot(x, model(x, w).detach())

# Cell
def mean_squared_err(preds, targets):
    return ((preds - targets) ** 2).sum() / 2

# Cell
preds = model(x, w)
mean_squared_err(preds, y)
# Cell
n_epochs = 100000
lr = 0.01

def update_weights(w, lr):
    with torch.no_grad():
        w -= lr * w.grad

def zero_grad(w):
    if w.grad is not None:
        w.grad.zero_()

for i in range(n_epochs):

    zero_grad(w)

    preds = model(x, w)

    loss = mean_squared_err(preds, y)
    loss.backward()
    update_weights(w, lr)

    if i % (n_epochs // 10) == 0:
        print(f'Epoch {i}/{n_epochs - 1}')
        print(f'\tLoss: {loss}')



    


# Cell
def plot_results(x: torch.Tensor, y: torch.Tensor, w:torch.Tensor):
    plt.scatter(x.numpy(), y.numpy())
    X_plt = torch.linspace(0, 1, 100)
    with torch.no_grad():
        model_out = model(X_plt, w)
    plt.plot(X_plt, model_out, label="predicted")
    plt.plot(X_plt, data_func(X_plt), label="data distribution")
    plt.legend()

    plt.show()

# Cell
plot_results(x, y, w)

# Cell
