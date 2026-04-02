from dataclasses import dataclass
import torch


@dataclass
class PINNConfig:
    Sigma_h: float = 9.5298e-4
    Sigma_d: float =   0.1*9.5298e-4
    a: float = 18.515
    ft: float = 0.2383
    fr: float = 0.0
    fd: float = 1.0
    T_final: float = 15.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



@dataclass
class PINNConfig1:
    Sigma_h: float = 9.5298e-4
    Sigma_d: float =   0.1*9.5298e-4
    a: float = 18.515
    ft: float = 0.2383
    fr: float = 0.0
    fd: float = 1.0
    T_final: float = 35.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PINNConfig2:
    Sigma_h: float = 9.5298e-4
    Sigma_d: float =   9.5298e-4
    a: float = 18.515
    ft: float = 0.2383
    fr: float = 0.0
    fd: float = 1.0
    T_final: float = 35.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PINNConfig3:
    Sigma_h: float = 9.5298e-4
    Sigma_d: float =   10*9.5298e-4
    a: float = 18.515
    ft: float = 0.2383
    fr: float = 0.0
    fd: float = 1.0
    T_final: float = 35.0
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
