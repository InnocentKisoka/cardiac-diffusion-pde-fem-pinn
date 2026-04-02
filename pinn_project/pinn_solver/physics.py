import torch

def f(u, config):
    return config.a * (u - config.fr) * (u - config.ft) * (u - config.fd)

def sigma(xy, config):
    x, y = xy[:, 0], xy[:, 1]
    d1 = (x - 0.3)**2 + (y - 0.7)**2 < 0.1**2
    d2 = (x - 0.7)**2 + (y - 0.3)**2 < 0.15**2
    d3 = (x - 0.5)**2 + (y - 0.5)**2 < 0.1**2
    diseased = d1 | d2 | d3
    return torch.where(diseased, config.Sigma_d, config.Sigma_h).unsqueeze(1)
