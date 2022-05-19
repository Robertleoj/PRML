# Cell
from scipy.stats import beta
# from torch.distributions import Beta
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

# Cell
class Agent:
    def __init__(self, num_bandits):
        self.abs = [[1, 1] for _ in range(num_bandits)]
        self.distributions = [beta(1, 1) for _ in range(num_bandits)]
        self.num_bandits = num_bandits

    def pull(self):
        samples = np.array([b.ppf(random.random()) for b in self.distributions])
        return samples.argmax(0)

    def update(self, bandit, result):
        if result:
            self.abs[bandit][0] += 1
        else:
            self.abs[bandit][1] += 1

        a, b = tuple(self.abs[bandit])
        
        self.distributions[bandit] = beta(a, b)

    def plot(self, linspace):
        return [b.pdf(linspace) for b in self.distributions]


def plot(agent, real=None, n= 100, pth=None):
    x = np.linspace(0, 1, n)
    ys = agent.plot(x)

    plt.figure(figsize=(10, 10))
    for i,y in enumerate(ys):
        label = i if real is None else f'{real[i]:.4f}'
        plt.plot(x, y, label=label)
    plt.legend()
    if pth:
        plt.savefig(pth)
    else:
        plt.show()
    plt.close()
    
# Cell
class Bandit:
    def __init__(self, n_bandits):
        self.mus = np.random.rand(n_bandits)

    def pull(self, b):
        return random.random() < self.mus[b]

# Cell
n = 4
b = Bandit(n)
a = Agent(n)
plot(a, b.mus)

for i in range(700):
    pull = a.pull()
    res = b.pull(pull)
    a.update(pull, res)
    plot(a, b.mus, pth=f'figs/fig{i}.jpg')

# Cell
