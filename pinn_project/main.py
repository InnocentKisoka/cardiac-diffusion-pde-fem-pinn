import torch
import os
from pinn_solver.model import PINN
from pinn_solver.train import train
from pinn_solver.visualize import animate_solution, plot_initial_condition, plot_collocation_points
from pinn_solver.collocation import collocate_initial_condition, collocate_boundary, collocate_domain, collocate_diseased_regions
from pinn_solver.config import PINNConfig1, PINNConfig2, PINNConfig3

# List of tuples with (ConfigClass, config_name)
configs = [
    (PINNConfig1, "config1"),
    (PINNConfig2, "config2"),
    (PINNConfig3, "config3"),
]

for ConfigClass, config_name in configs:
    print(f"\n🔧 Training model with {config_name}...")

    config = ConfigClass()
    model = PINN().to(config.device)

    # Train model
    train(model, config, N_f=50000, N_ic=1000, N_bc=1000, epochs=50000,config_name=config_name)

    # Create output directory
    output_dir = f"outputs/{config_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))

    # Save plots
    print(f"📊 Saving plots for {config_name}...")

    animate_solution(model, config, filename=os.path.join(output_dir, "animation.gif"))
    plot_initial_condition(model, config, filename=os.path.join(output_dir, "initial_condition.png"))
   

    print(f"✅ Done with {config_name}")
