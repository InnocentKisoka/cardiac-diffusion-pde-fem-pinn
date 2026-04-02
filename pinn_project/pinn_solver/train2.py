import torch
import torch.optim as optim
from pinn_solver.losses import (
    pde_loss, initial_condition_loss, boundary_condition_loss
)
from pinn_solver.collocation import (
    collocate_domain, collocate_diseased_regions,
    collocate_initial_condition, collocate_boundary
)
from pinn_solver.config import PINNConfig


def train(model, config: PINNConfig, epochs=10000, N_f=10000, N_ic=10000, N_bc=10000, lr=1e-3, ic_pretrain_epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n📈 Step 1: Pretraining with IC + Regularization over all collocation points ({ic_pretrain_epochs} epochs)...")
    for epoch in range(ic_pretrain_epochs):
        # Initial condition points
        xyt_ic, target_ic = collocate_initial_condition(N_ic, config)

        # Boundary points
        xyt_bc = collocate_boundary(N_bc, config)

        # Domain points: regular + diseased
        xyt_regular = collocate_domain(N_f // 2, config)
        xyt_diseased = collocate_diseased_regions(N_f // 2, config)
        xyt_domain = torch.cat([xyt_regular, xyt_diseased], dim=0)

        # Model outputs on various regions
        u_ic = model(xyt_ic)
        u_bc = model(xyt_bc)
        u_dom = model(xyt_domain)

        # Losses
        loss_ic = initial_condition_loss(model, xyt_ic, target_ic)
        loss_reg = (u_bc**2).mean() + (u_dom**2).mean()

        loss = loss_ic + 0.1 * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[Pretrain Epoch {epoch}] Total: {loss.item():.4e} | IC: {loss_ic.item():.4e} | Reg: {loss_reg.item():.4e}")

    print("\n🚀 Step 2: Full training with all physics-informed losses...")
    for epoch in range(epochs):
        # Collocation points
        xyt_ic, target_ic = collocate_initial_condition(N_ic, config)
        xyt_bc = collocate_boundary(N_bc, config)
        xyt_regular = collocate_domain(N_f // 2, config)
        xyt_diseased = collocate_diseased_regions(N_f // 2, config)
        xyt_f = torch.cat([xyt_regular, xyt_diseased], dim=0)

        # Compute loss terms
        loss_ic = initial_condition_loss(model, xyt_ic, target_ic)
        loss_bc = boundary_condition_loss(model, xyt_bc, config)
        loss_f = pde_loss(model, xyt_f, config)

        total_loss = loss_f + loss_ic + 0.2 * loss_bc

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}")
            print(f"  [Total Loss: {total_loss.item():.6f}]")
            print(f"  [PDE Loss:   {loss_f.item():.4e}]")
            print(f"  [IC  Loss:   {loss_ic.item():.4e}]")
            print(f"  [BC  Loss:   {loss_bc.item():.4e}]\n")
