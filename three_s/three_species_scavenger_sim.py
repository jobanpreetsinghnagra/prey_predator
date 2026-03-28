#!/usr/bin/env python3
"""
Three-species normalized ODE (prey x, predator y, scavenger z):

  x' = x(1 - b x - y - z)
  y' = y(-c + x)
  z' = z(-e - h z + f x + g y)

Run: python three_species_scavenger_sim.py
Writes three PNGs (prey / predator / scavenger extinction scenarios; see three_s/extinction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ThreeSpeciesParams:
    b: float = 0.1  # prey self-limiting
    c: float = 1.0  # predator death
    e: float = 2.0  # scavenger death
    h: float = 0.1  # scavenger self-limiting
    f: float = 0.005  # prey → scavenger benefit
    g: float = 0.002  # predator → scavenger benefit


def three_species_rhs(t: float, y: np.ndarray, p: ThreeSpeciesParams) -> np.ndarray:
    x, pred, scav = y

    dx = x * (1 - p.b * x - pred - scav)
    dpred = pred * (-p.c + x)
    dscav = scav * (-p.e - p.h * scav + p.f * x + p.g * pred)

    return np.array([dx, dpred, dscav])


def simulate_three_species(
    p: ThreeSpeciesParams,
    x0: float,
    y0: float,
    z0: float,
    t_span: Tuple[float, float] = (0, 100),
    n_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        lambda t, y: three_species_rhs(t, y, p),
        t_span,
        np.array([x0, y0, z0], dtype=float),
        t_eval=t_eval,
        method="RK45",
    )

    return sol.t, sol.y


def plot_three_species(
    t: np.ndarray,
    y: np.ndarray,
    outfile: Optional[str] = None,
    *,
    show: bool = True,
    title: Optional[str] = None,
) -> None:
    x, pred, scav = y

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, x, label="Prey (x)")
    ax.plot(t, pred, label="Predator (y)")
    ax.plot(t, scav, label="Scavenger (z)")

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title(title or "3-Species Lotka–Volterra (with scavenger)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Three-species prey–predator–scavenger ODE.")
    parser.add_argument("--no-show", action="store_true", help="Save figure only, do not open windows.")
    parser.add_argument(
        "-o",
        "--out-prefix",
        default="scavenger_extinction",
        help="Prefix for output PNGs (_prey, _predator, _scavenger appended).",
    )
    args = parser.parse_args()

    show_plots = not args.no_show
    if args.no_show:
        plt.switch_backend("Agg")

    # Three runs from three_s/extinction: prey → 0, predator → 0, scavenger → 0
    scenarios: list[tuple[ThreeSpeciesParams, float, float, float, str, str]] = [
        (
            ThreeSpeciesParams(b=0.5, c=1.0, e=0.5, h=0.05, f=0.1, g=0.05),
            5.0,
            15.0,
            10.0,
            f"{args.out_prefix}_prey.png",
            "Prey extinction (high b, y0, z0; strong predation)",
        ),
        (
            ThreeSpeciesParams(b=0.3, c=3.0, e=0.5, h=0.05, f=0.02, g=0.01),
            2.0,
            5.0,
            2.0,
            f"{args.out_prefix}_predator.png",
            "Predator extinction (high c, low x0 so x < c)",
        ),
        (
            ThreeSpeciesParams(b=0.1, c=1.0, e=2.0, h=0.1, f=0.005, g=0.002),
            10.0,
            5.0,
            2.0,
            f"{args.out_prefix}_scavenger.png",
            "Scavenger extinction (high e, low f, g vs resources)",
        ),
    ]

    for p, x0, y0, z0, outfile, plot_title in scenarios:
        t, y = simulate_three_species(p, x0=x0, y0=y0, z0=z0)
        plot_three_species(t, y, outfile=outfile, show=show_plots, title=plot_title)
        print(f"Wrote {outfile}")


if __name__ == "__main__":
    main()
