#!/usr/bin/env python3
"""
Predator–prey simulations aligned with:
  - Karjanto & Peng (bioRxiv 2025.05.01.651785): classical Lotka–Volterra ODEs.
  - Predator_Prey_Model_Wolves.ipynb (J. S. Butler): plant–deer–wolves Euler model
    with wolf reintroduction in 2020.

Run: python prey_predator_sim.py
     python prey_predator_sim.py --model lotka
     python prey_predator_sim.py --model wolves
     python prey_predator_sim.py --model scavenger
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from three_species_scavenger_sim import (
    ThreeSpeciesParams,
    plot_three_species,
    simulate_three_species,
)


# ---------------------------------------------------------------------------
# Lotka–Volterra (paper): dN/dt = N(α - βP),  dP/dt = P(-γ + δN)
# ---------------------------------------------------------------------------


@dataclass
class LotkaVolterraParams:
    alpha: float = 1.0  # prey intrinsic growth
    beta: float = 0.1  # predation rate on prey
    gamma: float = 1.5  # predator mortality
    delta: float = 0.075  # predator conversion from prey


def lotka_volterra_rhs(t: float, y: np.ndarray, p: LotkaVolterraParams) -> np.ndarray:
    n, pr = y
    dn = n * (p.alpha - p.beta * pr)
    dpr = pr * (-p.gamma + p.delta * n)
    return np.array([dn, dpr])


def simulate_lotka_volterra(
    p: LotkaVolterraParams,
    n0: float,
    p0: float,
    t_span: Tuple[float, float],
    n_points: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, y: lotka_volterra_rhs(t, y, p),
        t_span,
        np.array([n0, p0], dtype=float),
        t_eval=t_eval,
        method="RK45",
        dense_output=True,
    )
    return sol.t, sol.y


def plot_lotka_volterra(
    t: np.ndarray,
    y: np.ndarray,
    outfile: Optional[str],
    *,
    show: bool = True,
) -> None:
    n, pr = y
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, n, label="prey N(t)")
    ax.plot(t, pr, label="predator P(t)")
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.set_title("Lotka–Volterra (Karjanto & Peng, Sec. 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Plant–deer–wolves (notebook): explicit Euler, wolves from 2020
# ---------------------------------------------------------------------------


def simulate_plant_deer_wolves(
    t0: float = 1950.0,
    t1: float = 2100.0,
    h: float = 0.01,
    wolf_intro_year: float = 2020.0,
    # Coefficients from Predator_Prey_Model_Wolves.ipynb
    small_plant: Tuple[float, float, float] = (0.9, -0.02, 0.03),
    deer_const: Tuple[float, float, float] = (0.005, 0.002, -0.001),
    wolves_const: Tuple[float, float, float] = (-0.05, 0.001, 0.0),
    plant0: float = 80.0,
    deer0: float = 45.0,
    wolves_init: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Replicates the notebook loop: before wolf_intro_year, wolves are forced to 0
    for interactions; from the intro year onward the wolf ODE is integrated.
    Deer term uses (plant - 60) as in the wolves notebook cell.
    """
    time = np.arange(t0, t1, h, dtype=float)
    n = len(time)
    plant = np.zeros(n)
    deer = np.zeros(n)
    wolves = np.full(n, wolves_init)
    plant[0] = plant0
    deer[0] = deer0

    a0, a1, a2 = small_plant
    b0, b1, b2 = deer_const
    w0, w1, _ = wolves_const

    for i in range(1, n):
        # Notebook: zero wolves before reintroduction year for the interaction terms.
        if time[i - 1] < wolf_intro_year:
            wolves[i - 1] = 0.0

        if time[i - 1] >= wolf_intro_year:
            wolves[i] = wolves[i - 1] + h * wolves[i - 1] * (w0 + w1 * deer[i - 1])

        plant[i] = plant[i - 1] + h * plant[i - 1] * (
            a0 + a1 * deer[i - 1] + a2 * wolves[i - 1]
        )
        # Same notebook cell uses plant reference 60 throughout (with wolves scenario).
        deer[i] = deer[i - 1] + h * deer[i - 1] * (
            b0 + b1 * (plant[i - 1] - 60.0) + b2 * wolves[i - 1]
        )

    return time, plant, deer, wolves


def plot_plant_deer_wolves(
    time: np.ndarray,
    plant: np.ndarray,
    deer: np.ndarray,
    wolves: np.ndarray,
    outfile: Optional[str],
    *,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time, wolves, "r-", label="wolves", linewidth=2)
    axes[0].set_ylabel("population (arb. units)")
    axes[0].set_title("Wolves (notebook-style reintroduction)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, deer, "b-", label="deer", linewidth=2)
    axes[1].plot(time, plant, "g-", label="plants", linewidth=2, alpha=0.8)
    axes[1].set_xlabel("time (years)")
    axes[1].set_ylabel("population (arb. units)")
    axes[1].set_title("Deer and plants")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Plant–deer–wolves (Predator_Prey_Model_Wolves.ipynb)", y=1.02)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predator–prey simulations (paper + notebook).")
    parser.add_argument(
        "--model",
        choices=("lotka", "wolves", "scavenger", "both"),
        default="both",
        help="Which model to run (default: both = Lotka + plant–deer–wolves).",
    )
    parser.add_argument("--no-show", action="store_true", help="Save figures only, do not open windows.")
    args = parser.parse_args()

    show_plots = not args.no_show
    if args.no_show:
        plt.switch_backend("Agg")

    if args.model in ("lotka", "both"):
        lv_p = LotkaVolterraParams()
        t_lv, y_lv = simulate_lotka_volterra(lv_p, n0=10.0, p0=5.0, t_span=(0.0, 50.0))
        out_lv = "lotka_volterra.png"
        plot_lotka_volterra(t_lv, y_lv, outfile=out_lv, show=show_plots)
        print(f"Wrote {out_lv}")

    if args.model in ("wolves", "both"):
        time, plant, deer, wolves = simulate_plant_deer_wolves()
        out_w = "plant_deer_wolves.png"
        plot_plant_deer_wolves(time, plant, deer, wolves, outfile=out_w, show=show_plots)
        print(f"Wrote {out_w}")

    if args.model == "scavenger":
        p3 = ThreeSpeciesParams()
        t3, y3 = simulate_three_species(p3, x0=10.0, y0=5.0, z0=2.0)
        out_s = "three_species_scavenger.png"
        plot_three_species(t3, y3, outfile=out_s, show=show_plots)
        print(f"Wrote {out_s}")


if __name__ == "__main__":
    main()
