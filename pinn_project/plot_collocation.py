from pinn_solver.config import PINNConfig
import torch 
from pinn_solver.visualize import animate_solution, plot_initial_condition, plot_collocation_points,plot_sigma_collocation
from pinn_solver.collocation import collocate_initial_condition, collocate_boundary, collocate_domain, collocate_diseased_regions

config = PINNConfig()
# Re-generate points (optional, or reuse from training)
xy_ic, _ = collocate_initial_condition(1000, config)
xyt_bc = collocate_boundary(10, config)
xyt_regular = collocate_domain(10, config)
xyt_diseased = collocate_diseased_regions(1000, config)
xyt_f = torch.cat([xyt_regular, xyt_diseased], dim=0)

plot_collocation_points(xy_ic, xyt_bc, xyt_f)
plot_sigma_collocation(config)