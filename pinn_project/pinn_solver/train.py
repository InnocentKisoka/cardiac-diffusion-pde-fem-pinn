import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from pinn_solver.losses import (
    pde_loss, initial_condition_loss, boundary_condition_loss
)
from pinn_solver.collocation import (
    collocate_domain, collocate_diseased_regions,
    collocate_initial_condition, collocate_boundary
)
from pinn_solver.config import PINNConfig
from pinn_solver.visualize import plot_losses

def train(model, config: PINNConfig, epochs=10000, N_f=10000, N_ic=10000, N_bc=10000, lr=1e-3,config_name="config"):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tracking losses
    loss_history = {
        "total": [],
        "pde": [],
        "ic": [],
        "bc": []
    }

    for epoch in range(epochs):
        # Collocation points
        xyt_regular = collocate_domain(N_f // 2, config)
        xyt_diseased = collocate_diseased_regions(N_f // 2, config)
        xyt_f = torch.cat([xyt_regular, xyt_diseased], dim=0)

        xyt_ic, target_ic = collocate_initial_condition(N_ic, config)
        xyt_bc = collocate_boundary(N_bc, config)

        # Losses
        loss_f = pde_loss(model, xyt_f, config)
        loss_ic = initial_condition_loss(model, xyt_ic, target_ic)
        loss_bc = boundary_condition_loss(model, xyt_bc, config)
        total_loss = loss_f + loss_ic +  loss_bc

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Save losses
        loss_history["total"].append(total_loss.item())
        loss_history["pde"].append(loss_f.item())
        loss_history["ic"].append(loss_ic.item())
        loss_history["bc"].append(loss_bc.item())

        # Print every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"  [Total Loss: {total_loss.item():.6f}]")
            print(f"  [PDE Loss:   {loss_f.item():.4e}]")
            print(f"  [IC  Loss:   {loss_ic.item():.4e}]")
            print(f"  [BC  Loss:   {loss_bc.item():.4e}]\n")
    output_dir = f"outputs/{config_name}"
    # Save the loss history
    torch.save(loss_history, os.path.join(output_dir,"loss_history.pt"))

    plot_losses(
    loss_history["total"],
    loss_history["pde"],
    loss_history["ic"],
    loss_history["bc"],
    save_path=os.path.join(output_dir,"loss_plot.png")
)
