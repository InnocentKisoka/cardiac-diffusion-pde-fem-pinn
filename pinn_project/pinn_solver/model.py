import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_layers=[200,   100, 100,50,50]):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_layers[0]), nn.Tanh()]
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
