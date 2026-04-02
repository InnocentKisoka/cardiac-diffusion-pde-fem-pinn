import torch
from pinn_solver.physics import sigma, f

def pde_loss(model, xyt, config):
    xyt.requires_grad_(True)
    u = model(xyt)
    grad = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    u_t = grad[:, 2:3]
    grad_u = grad[:, :2]
    sigma_val = sigma(xyt[:, :2], config)
    grad_u_sigma = grad_u * sigma_val
    div_term = torch.autograd.grad(grad_u_sigma.sum(dim=1, keepdim=True), xyt,
                                   grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0][:, :2].sum(dim=1, keepdim=True)
    return ((u_t - div_term + f(u, config))**2).mean()

def initial_condition_loss(model, xyt, target):
    u = model(xyt)
    return ((u - target) ** 2).mean()



def boundary_condition_loss(model, xyt, config):
    xyt.requires_grad_(True)
    u = model(xyt)
    grad = torch.autograd.grad(u, xyt, torch.ones_like(u), create_graph=True)[0]
    grad_u = grad[:, :2]

    # Assuming normal vector = outward unit normal on boundary.
    # We'll fake it here as axis-aligned boundaries for simplicity
    normals = torch.zeros_like(grad_u)
    
    # Simple boundary detection: x=0 or 1 → normal = +/-x̂ ; y=0 or 1 → normal = +/-ŷ
    eps = 1e-5
    on_left   = xyt[:, 0] < eps
    on_right  = xyt[:, 0] > 1 - eps
    on_bottom = xyt[:, 1] < eps
    on_top    = xyt[:, 1] > 1 - eps

    normals[on_left, 0] = -1.0
    normals[on_right, 0] = 1.0
    normals[on_bottom, 1] = -1.0
    normals[on_top, 1] = 1.0

    # Compute n · ∇u
    normal_dot_grad = (normals * grad_u).sum(dim=1, keepdim=True)
    return (normal_dot_grad**2).mean()
