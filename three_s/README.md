# `three_s` — Three-species prey–predator–scavenger simulation

This folder contains a **normalized three-species ODE model** (prey \(x\), predator \(y\), scavenger \(z\)), a Python script that integrates it and writes plots, **extinction scenario notes**, and **example PNG figures** produced by the script.

---

## Folder contents

| Item | Role |
|------|------|
| `three_species_scavenger_sim.py` | Main program: parameters, ODE right-hand side, numerical integration, plotting, CLI. |
| `extinction` | Informal notes (Markdown-style) on how to push each species toward extinction via parameters and initial conditions; mirrors the three built-in scenarios in `main()`. |
| `three_species_scavenger.png` | Example time-series figure in the repo (naming differs from `main()`’s default `--out-prefix`; regenerate or rename as needed). |
| `scavenger_extinction_prey.png` | Saved output for the **prey extinction** scenario (`--out-prefix scavenger_extinction` → `_prey.png`). |
| `scavenger_extinction_predator.png` | Saved output for the **predator extinction** scenario. |
| `scavenger_extinction_scavenger.png` | Saved output for the **scavenger extinction** scenario. |
| `image/` | Additional copies or variants of extinction plots: `prey_extinction.png`, `predator_extinction.png`, `scavenger_extinction.png`. |

**Run the simulator** (from this directory or with a path to the script):

```bash
python three_species_scavenger_sim.py
```

Save figures without opening windows:

```bash
python three_species_scavenger_sim.py --no-show
```

Custom output filename prefix (files become `<prefix>_prey.png`, `_predator.png`, `_scavenger.png`):

```bash
python three_species_scavenger_sim.py --no-show -o my_run
```

---

## Mathematical model

State vector \((x, y, z)\) with **normalized** Lotka–Volterra-style dynamics:

\[
\begin{aligned}
x' &= x\,(1 - b x - y - z) \\
y' &= y\,(-c + x) \\
z' &= z\,(-e - h z + f x + g y)
\end{aligned}
\]

- **Prey** \(x\): logistic-like growth with carrying capacity shaped by \(b\), minus losses to predator and scavenger.
- **Predator** \(y\): grows when prey \(x > c\), decays when \(x < c\).
- **Scavenger** \(z\): benefits from prey and predator (\(f x + g y\)), has baseline loss \(e\) and self-limitation \(h z\).

---

## `ThreeSpeciesParams` (dataclass)

A frozen-style parameter bundle (implemented with `@dataclass`). All fields are **floats** representing dimensionless coefficients in the ODE above.

| Field | Default | Meaning |
|-------|---------|---------|
| `b` | `0.1` | Prey **self-limitation** (crowding / resource competition within prey). Larger \(b\) shrinks effective carrying capacity. |
| `c` | `1.0` | Predator **baseline mortality** (threshold: predator needs \(x > c\) for positive growth). |
| `e` | `2.0` | Scavenger **baseline mortality** / maintenance cost. |
| `h` | `0.1` | Scavenger **self-limitation** (density-dependent scavenger loss). |
| `f` | `0.005` | Conversion from **prey** \(x\) to scavenger growth. |
| `g` | `0.002` | Conversion from **predator** \(y\) to scavenger growth. |

**Inputs:** None (instantiated with defaults or with keyword arguments, e.g. `ThreeSpeciesParams(b=0.5, c=1.0, …)`).

**Outputs:** A single instance holding the six parameters as attributes.

---

## Functions

### `three_species_rhs(t, y, p)`

**Purpose:** Evaluate the ODE derivative \(\mathrm{d}/\mathrm{d}t\,[x, y, z]^\top\) at one time point.

| Argument | Type | Meaning |
|----------|------|---------|
| `t` | `float` | Time (not used in the current equations; kept for compatibility with `solve_ivp`). |
| `y` | `np.ndarray` | State vector **length 3**: `[x, pred, scav]` (prey, predator, scavenger). |
| `p` | `ThreeSpeciesParams` | Model parameters. |

**Returns:** `np.ndarray` of shape `(3,)`: `[dx, dpred, dscav]` — same order as `y`.

---

### `simulate_three_species(p, x0, y0, z0, t_span=..., n_points=...)`

**Purpose:** Numerically integrate the ODE from initial conditions over `t_span` using SciPy’s `solve_ivp` (RK45).

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `p` | `ThreeSpeciesParams` | (required) | Parameters for `three_species_rhs`. |
| `x0` | `float` | (required) | Initial prey population. |
| `y0` | `float` | (required) | Initial predator population. |
| `z0` | `float` | (required) | Initial scavenger population. |
| `t_span` | `Tuple[float, float]` | `(0, 100)` | Integration interval \((t_\text{start}, t_\text{end})\). |
| `n_points` | `int` | `2000` | Number of equally spaced output times (via `t_eval`). |

**Returns:** `Tuple[np.ndarray, np.ndarray]`

1. `t` — 1D array of times (length `n_points`).
2. `y` — 2D array of shape `(3, len(t))`: row 0 = prey \(x(t)\), row 1 = predator \(y(t)\), row 2 = scavenger \(z(t)\).

---

### `plot_three_species(t, y, outfile=None, *, show=True, title=None)`

**Purpose:** Plot the three trajectories vs. time with Matplotlib.

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `t` | `np.ndarray` | (required) | Time samples (same as first return of `simulate_three_species`). |
| `y` | `np.ndarray` | (required) | State history, shape `(3, N)` — same as second return of `simulate_three_species`. |
| `outfile` | `Optional[str]` | `None` | If set, save figure to this path (PNG recommended; DPI 150). |
| `show` | `bool` | `True` | If `True`, display an interactive window; if `False`, close the figure after saving (non-interactive backends). |
| `title` | `Optional[str]` | `None` | Plot title; if `None`, uses a default descriptive string. |

**Returns:** `None` (side effects: figure display and/or file write).

---

### `main()`

**Purpose:** CLI entry point: parses arguments, runs **three** hard-coded scenarios (prey extinction, predator extinction, scavenger extinction), writes PNGs, prints paths.

**Inputs:** None (reads `sys.argv` via `argparse`).

**CLI options:**

| Flag | Effect |
|------|--------|
| `--no-show` | Do not open plot windows; use Agg backend; still saves files if `outfile` is set. |
| `-o` / `--out-prefix` | Prefix for output files (default `scavenger_extinction`). Appends `_prey.png`, `_predator.png`, `_scavenger.png`. |

**Returns:** `None`.

**Important local variables inside `main()`:**

| Variable | Meaning |
|----------|---------|
| `parser` / `args` | Argument parser and parsed namespace (`no_show`, `out_prefix`). |
| `show_plots` | `False` if `--no-show`, else `True`. |
| `scenarios` | List of tuples: `(ThreeSpeciesParams, x0, y0, z0, outfile, plot_title)` for each extinction demo. |
| `p`, `x0`, `y0`, `z0` | Per-scenario parameters and initial populations. |
| `outfile` | Output filename for that scenario’s PNG. |
| `plot_title` | Human-readable title for the plot. |
| `t`, `y` | Time array and state history from `simulate_three_species` for each loop iteration. |

---

## Important variables in `simulate_three_species`

| Variable | Meaning |
|----------|---------|
| `t_eval` | Equally spaced times from `t_span[0]` to `t_span[1]` with `n_points` samples; passed to `solve_ivp(..., t_eval=...)`. |
| `sol` | Return value of `solve_ivp`: solution object; `sol.t` is time, `sol.y` is state history. |

---

## Important variables in `plot_three_species`

| Variable | Meaning |
|----------|---------|
| `x`, `pred`, `scav` | Unpacked rows of `y`: prey, predator, scavenger trajectories. |
| `fig`, `ax` | Matplotlib figure and axes. |

---

## Important variables in `three_species_rhs`

| Variable | Meaning |
|----------|---------|
| `x`, `pred`, `scav` | Unpacked components of state `y`. |
| `dx`, `dpred`, `dscav` | Time derivatives for prey, predator, scavenger. |

---

## Built-in scenarios (in `main()`)

These match the intuition in `extinction`: each run picks parameters and \((x_0, y_0, z_0)\) so one species is driven toward extinction.

1. **Prey extinction** — high `b`, high initial predators/scavengers, stronger `f`, `g`: output `<out_prefix>_prey.png`.
2. **Predator extinction** — high `c`, low `x0` so \(x < c\): output `<out_prefix>_predator.png`.
3. **Scavenger extinction** — high `e`, low `f`, `g`: output `<out_prefix>_scavenger.png`.

---

## Dependencies

- Python 3
- `numpy`
- `scipy` (`scipy.integrate.solve_ivp`)
- `matplotlib`

---

## See also

- Module docstring at the top of `three_species_scavenger_sim.py` for a one-line summary of the ODE and how to run the script.
- `extinction` for narrative tuning tips (log scale, sweeps, reporting).
