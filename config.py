from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    dt: float = 0.05
    horizon: int = 20
    q: tuple = (10., 10., 2.)
    eta: float = 0.1
    terminal: float = 5.
    lower: tuple = (0., -2.)
    upper: tuple = (1., 2.)
    tolerance: float = 1e-5
