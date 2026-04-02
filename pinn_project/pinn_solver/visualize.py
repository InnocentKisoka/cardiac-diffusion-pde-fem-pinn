import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pinn_solver.physics import sigma 

from pinn_solver.config import PINNConfig
from pinn_solver.collocation import (
    collocate_initial_condition,
    collocate_boundary,
    collocate_domain,
    collocate_diseased_regions
)

def plot_initial_condition(model, config, resolution=100, filename="u_init.png"):
    model.eval()
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    T = torch.zeros_like(X)

    pts = torch.stack([X, Y, T], dim=-1).view(-1, 3).to(config.device)
    with torch.no_grad():
        U = model(pts).view(resolution, resolution).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(X, Y, U, levels=100, cmap='inferno')
    fig.colorbar(contour)
    ax.set_title("Initial Condition u(x, y, t=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved initial condition to {filename}")


def animate_solution(model, config, resolution=100,filename="solution.gif", t_max=35):
    model.eval()
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame):
        ax.clear()
        T = torch.full_like(X, float(frame))
        pts = torch.stack([X, Y, T], dim=-1).view(-1, 3).to(config.device)
        with torch.no_grad():
            U = model(pts).view(resolution, resolution).cpu().numpy()
        contour = ax.contourf(X, Y, U, levels=100, cmap='inferno')
        ax.set_title(f"u(x, y, t={frame})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return contour.collections

    ani = animation.FuncAnimation(fig, update, frames=range(0, t_max + 1), interval=200, blit=False)

    from matplotlib.animation import PillowWriter
    from IPython.display import Image, display
    ani.save(filename, writer=PillowWriter(fps=5))
    display(Image(filename=filename))


def plot_collocation_points(xy_ic, xyt_bc, xyt_f, filename="collocation_points.png"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xy_ic[:, 0].cpu(), xy_ic[:, 1].cpu(), s=1, label='Initial Condition', alpha=0.6)
    ax.scatter(xyt_bc[:, 0].cpu(), xyt_bc[:, 1].cpu(), s=1, label='Boundary Condition', alpha=0.6)
    ax.scatter(xyt_f[:, 0].cpu(), xyt_f[:, 1].cpu(), s=1, label='Domain Points', alpha=0.6)
    ax.set_title("Collocation Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect('equal')
    plt.savefig(filename)
    plt.close()
    print(f"✅ Saved collocation points plot to {filename}")

def plot_sigma_collocation(config: PINNConfig, N_ic=1000, N_f=1000, N_bc=1000, save_path="sigma_collocation_plot.png"):
    # Collocate points
    xy_ic, _ = collocate_initial_condition(N_ic, config)       # (N_ic, 3)
    xyt_bc = collocate_boundary(N_bc, config)                  # (N_bc, 3)
    xyt_regular = collocate_domain(N_f // 2, config)           # (N_f/2, 3)
    xyt_diseased = collocate_diseased_regions(N_f // 2, config)  # (N_f/2, 3)
    xyt_f = torch.cat([xyt_regular, xyt_diseased], dim=0)      # (N_f, 3)

    # Combine all (xyt) and take only (x, y)
    all_xy = torch.cat([xy_ic[:, :2], xyt_bc[:, :2], xyt_f[:, :2]], dim=0)  # (x, y)

    # Get sigma values
    sigma_vals = sigma(all_xy, config).squeeze()

    # Plot
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(all_xy[:, 0].cpu(), all_xy[:, 1].cpu(),
                          c=sigma_vals.cpu(), cmap='coolwarm', s=5)
    plt.colorbar(scatter, label='σ (Sigma)')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Collocation Points Colored by Sigma")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_losses(total_loss_list, pde_loss_list, ic_loss_list, bc_loss_list, save_path="loss_plot.png"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_list, label="Total Loss")
    plt.plot(pde_loss_list, label="PDE Loss")
    plt.plot(ic_loss_list, label="IC Loss")
    plt.plot(bc_loss_list, label="BC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Losses Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved training loss plot to {save_path}")
