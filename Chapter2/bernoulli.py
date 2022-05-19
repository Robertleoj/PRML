# Cell
import torch
import matplotlib.pyplot as plt

# Cell
def bernoulli_sample(n, mu):
    return (torch.rand(n) < mu).int()

def bern2_sample(n, mu):
    out = (torch.rand(n) < ((1 + mu) / 2)).int()
    out[out == 0] -= 1
    return out



# Cell
bernoulli_sample(10, 0.6)
# Cell
'''
plot mu against mean
'''


n = 100
m = 100

def plot_stats(fig_title,num_mus, sample_size, sample_func, mean_func, var_func, entropy_func, start, end):

    mus = torch.linspace(start, end, num_mus)
    samples = torch.stack(
        tuple(sample_func(sample_size, mu) for mu in mus),
        dim = 0
    )

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(fig_title)

    # Plot means
    means = samples.float().mean(1)
    real_means = mean_func(mus)
    axs[0].scatter(mus.numpy(), means.numpy(), label='observed', c='b')
    axs[0].plot(mus.numpy(), real_means.numpy(), label='real', c='r', linewidth=4)
    axs[0].legend()
    axs[0].set_title('Mean')

    vars = samples.float().var(1)
    real_vars = var_func(mus)
    axs[1].scatter(mus.numpy(), vars.numpy(), label='observerd', c='b')
    axs[1].plot(mus.numpy(), real_vars.numpy(), label='real', c='r', linewidth=4)
    axs[1].set_title('Variance')
    axs[1].legend()

    p = {}
    for n in start, end:
        p[n] = (samples == n).sum(1) / samples.size(1)

    entropies = - p[start] * torch.log(p[start]) - (p[end]) * torch.log(p[end])
    real_entropy = entropy_func(mus)
    axs[2].scatter(mus.numpy(), entropies.numpy(), label='observerd', c='b')
    axs[2].plot(mus.numpy(), real_entropy.numpy(), label='real', c='r', linewidth=4)
    axs[2].set_title('Entropy')
    axs[2].legend()



    plt.show()

# Cell
plot_stats(
    "Bernoulli",
    n, m, 
    bernoulli_sample, 
    lambda mu: mu, 
    lambda mu: mu * (1 - mu),
    lambda mu: -mu * torch.log(mu) - (1 - mu) * torch.log(1 - mu),
    0, 1
)

# Cell
plot_stats(
    "Other bernoulli",
    n, m,
    bern2_sample,
    lambda mu: mu,
    lambda mu: 1 - mu ** 2,
    lambda mu: -(1 / 2) * ((1 - mu) * torch.log((1 - mu) / 2) + (1 + mu) * torch.log((1 + mu) / 2)),
    -1, 1
)

# Cell
