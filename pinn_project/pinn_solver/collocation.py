import torch
from pinn_solver.config import PINNConfig

def collocate_initial_condition(N_ic: int, config: PINNConfig):
    N_pos = N_ic // 2
    N_neg = N_ic - N_pos

    # Positive: in the (x >= 0.9, y >= 0.9) region
    x_pos = 0.9 + 0.1 * torch.rand(N_pos, 1, device=config.device)
    y_pos = 0.9 + 0.1 * torch.rand(N_pos, 1, device=config.device)
    xy_pos = torch.cat([x_pos, y_pos], dim=1)
    t_pos = torch.zeros(N_pos, 1, device=config.device)
    xyt_pos = torch.cat([xy_pos, t_pos], dim=1)
    target_pos = torch.ones(N_pos, 1, device=config.device)

    # Negative: oversample 3×, then filter
    xy_raw = torch.rand(N_neg * 3, 2, device=config.device)
    mask = ~((xy_raw[:, 0] >= 0.9) & (xy_raw[:, 1] >= 0.9))
    xy_neg = xy_raw[mask][:N_neg]  # take first N_neg valid

    if xy_neg.shape[0] < N_neg:
        raise RuntimeError("Not enough negative samples generated. Try increasing oversampling ratio.")

    t_neg = torch.zeros(N_neg, 1, device=config.device)
    xyt_neg = torch.cat([xy_neg, t_neg], dim=1)
    target_neg = torch.zeros(N_neg, 1, device=config.device)

    # Combine
    xyt = torch.cat([xyt_pos, xyt_neg], dim=0)
    target = torch.cat([target_pos, target_neg], dim=0)

    return xyt, target


def collocate_domain(N_f: int, config: PINNConfig):
    xyt = torch.rand(N_f, 3).to(config.device)
    xyt[:, 2] *= config.T_final
    return xyt

def collocate_diseased_regions(N_f: int, config: PINNConfig):
    num_regions = 3
    base = N_f // num_regions
    remainder = N_f - base * num_regions
    region_counts = [base] * num_regions

    for i in range(remainder):
        region_counts[i % num_regions] += 1

    x_d1 = torch.normal(0.3, 0.15, (region_counts[0], 1)).to(config.device)
    y_d1 = torch.normal(0.7, 0.15, (region_counts[0], 1)).to(config.device)

    x_d2 = torch.normal(0.7, 0.2, (region_counts[1], 1)).to(config.device)
    y_d2 = torch.normal(0.3, 0.2, (region_counts[1], 1)).to(config.device)

    x_d3 = torch.normal(0.5, 0.15, (region_counts[2], 1)).to(config.device)
    y_d3 = torch.normal(0.5, 0.15, (region_counts[2], 1)).to(config.device)

    t_d1 = torch.rand(region_counts[0], 1).to(config.device) * config.T_final
    t_d2 = torch.rand(region_counts[1], 1).to(config.device) * config.T_final
    t_d3 = torch.rand(region_counts[2], 1).to(config.device) * config.T_final

    x_all = torch.cat([x_d1, x_d2, x_d3], dim=0)
    y_all = torch.cat([y_d1, y_d2, y_d3], dim=0)
    t_all = torch.cat([t_d1, t_d2, t_d3], dim=0)

    xyt_focused = torch.cat([x_all, y_all, t_all], dim=1)
    return torch.clamp(xyt_focused, 0, 1)


def collocate_boundary(N_b: int, config: PINNConfig):
    half = N_b // 4
    t = torch.rand(half, 1).to(config.device) * config.T_final

    x0 = torch.zeros(half, 1).to(config.device)
    x1 = torch.ones(half, 1).to(config.device)
    y  = torch.rand(half, 1).to(config.device)

    y0 = torch.zeros(half, 1).to(config.device)
    y1 = torch.ones(half, 1).to(config.device)
    x  = torch.rand(half, 1).to(config.device)

    xb1 = torch.cat([x0, y, t], dim=1)
    xb2 = torch.cat([x1, y, t], dim=1)
    xb3 = torch.cat([x, y0, t], dim=1)
    xb4 = torch.cat([x, y1, t], dim=1)

    xb = torch.cat([xb1, xb2, xb3, xb4], dim=0)
    return xb
