"""Step 2d Phase 0b: why does self-recovery fall short, and can CMA-ES do better?

Throwaway diagnostic. It reuses the Phase 0 spike (`step2d_match_spike.py`) for
the core, the free parameters and the loss, and runs at the same budget and seeds
so its numbers compare with the Powell (2/10) and differential-evolution (3/10)
results.

Parts:
  A. Identifiability: for each recovered patch, how far it is from the truth,
     how the loss splits, how sensitive the loss is to each parameter, and
     1-D slices through the truth (local minima).
  B. Optimizers at 10,000 evaluations: CMA-ES (one run), CMA-ES with restarts
     (population doubling), and a staged CMA-ES.
  C. Realistic patches: truth near the schema defaults instead of across the
     whole range, to see whether the random test is the problem.

Pass rule (from the plan, not changed here): some optimizer reaches loss down
more than 95% on 8 of 10 seeds within about 10,000 evaluations.

Run:  .venv/bin/python spikes/step2d_diagnose.py [--part A|B|C]
"""

import argparse
import sys
import time
from pathlib import Path

import cma
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2d_match_spike as spike  # noqa: E402

FREE = spike.FREE
N = len(FREE)
MAX_EVALS = 10_000
SEEDS = range(10)

# Staged CMA-ES: parameter groups, each with its own evaluation budget.
STAGES = [
    (["env_attack", "env_decay", "env_sustain", "env_release", "filter_cutoff"], 2000),
    (["filter_env_amount", "lfo_pitch_depth", "lfo_rate"], 2000),
    (["filter_resonance", "drive"], 1000),
    (FREE, 5000),
]


def loss_terms(render, target):
    """(spectral, envelope, total) with the same weighting as the spike's loss."""
    import torch
    from neural.model import mr_stft_loss

    n = min(len(render), len(target))
    x = render[:n] / (np.sqrt(np.mean(render[:n] ** 2)) + 1e-9)
    y = target[:n] / (np.sqrt(np.mean(target[:n] ** 2)) + 1e-9)
    spectral = float(mr_stft_loss(torch.as_tensor(x, dtype=torch.float32)[None],
                                  torch.as_tensor(y, dtype=torch.float32)[None]))
    env = float(np.mean(np.abs(spike.envelope_db(render[:n]) - spike.envelope_db(target[:n]))) / 20.0)
    return spectral, env, spectral + 0.5 * env


class Counted:
    """The spike's objective, with a global evaluation counter."""

    def __init__(self, target):
        self.target = target
        self.evals = 0
        self.best = np.inf
        self.best_x = None

    def __call__(self, unit):
        unit = np.clip(np.asarray(unit, dtype=float), 0.0, 1.0)
        self.evals += 1
        value = spike.loss(spike.render_unit(unit), self.target)
        if value < self.best:
            self.best, self.best_x = value, unit.copy()
        return value


def make_truths(kind, seeds=SEEDS):
    """Yields (seed, truth unit vector, start unit vector) for one target set."""
    default_unit = np.array([
        (spike.SCHEMA[name]["default"] - spike.SCHEMA[name]["min"])
        / (spike.SCHEMA[name]["max"] - spike.SCHEMA[name]["min"])
        for name in FREE])
    for seed in seeds:
        rng = np.random.default_rng(seed)
        if kind == "random":
            truth = rng.uniform(0.25, 0.75, size=N)
        else:  # realistic: near the schema defaults
            truth = np.clip(default_unit + rng.uniform(-0.15, 0.15, size=N), 0.0, 1.0)
        start = np.clip(truth + rng.uniform(-0.2, 0.2, size=N), 0.0, 1.0)
        yield seed, truth, start


def run_cma(f, x0, sigma, maxfev, seed, popsize=None, idx=None, base=None):
    """One CMA-ES run on the subset `idx` (all parameters when None), at most `maxfev` evaluations."""
    base = np.asarray(x0, dtype=float).copy() if base is None else base.copy()
    if idx is None:
        idx = list(range(N))
    z0 = base[idx]

    def sub(z):
        x = base.copy()
        x[idx] = np.clip(z, 0.0, 1.0)
        return f(x)

    opts = {"bounds": [0.0, 1.0], "maxfevals": maxfev, "seed": seed,
            "verbose": -9, "tolfun": 1e-12, "tolx": 1e-7}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(z0, sigma, opts)
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, [sub(z) for z in candidates])
    return f.best_x


def run_restarts(f, x0, maxfev, seed):
    """CMA-ES with restarts: each restart doubles the population, from a random point after the first."""
    rng = np.random.default_rng(seed)
    default_pop = 4 + int(3 * np.log(N))
    k = 0
    while f.evals < maxfev:
        start = np.asarray(x0, dtype=float) if k == 0 else rng.uniform(0.0, 1.0, size=N)
        remaining = maxfev - f.evals
        if remaining < 2 * default_pop:
            break
        run_cma(f, start, 0.3, remaining, seed * 100 + k, popsize=default_pop * 2 ** k)
        k += 1
    return f.best_x


def run_staged(f, x0, seed):
    """Stage by stage, each group optimized with the others fixed at their best values."""
    x = np.asarray(x0, dtype=float).copy()
    for stage, (names, budget) in enumerate(STAGES):
        idx = [FREE.index(name) for name in names]
        left = MAX_EVALS - f.evals
        if left <= 0:
            break
        run_cma(f, x, 0.3, min(budget, left), seed * 100 + stage, idx=idx, base=f.best_x if f.best_x is not None else x)
        x = f.best_x.copy()
    return f.best_x


def fit_with(optimizer, target, start, seed):
    f = Counted(target)
    f(start)  # the starting point counts as an evaluation
    t0 = time.perf_counter()
    if optimizer == "cma":
        run_cma(f, start, 0.3, MAX_EVALS - 1, seed)
    elif optimizer == "cma_restarts":
        run_restarts(f, start, MAX_EVALS - 1, seed)
    elif optimizer == "cma_staged":
        run_staged(f, start, seed)
    else:
        raise ValueError(optimizer)
    return f, time.perf_counter() - t0


def part_a():
    print("A. Identifiability (single CMA-ES run, random targets)\n")
    print(f"{'seed':>4} {'start':>7} {'best':>8} {'down%':>6} {'spect':>7} {'env':>7}  "
          f"{'max|dx|':>7} {'mean|dx|':>8}  flat params (loss change under +-0.1 is < 1%)")
    slices = {}
    for seed, truth, start in make_truths("random"):
        target = spike.render_unit(truth)
        f, _ = fit_with("cma", target, start, seed)
        initial = Counted(target)(start)
        best = f.best_x
        spectral, env, total = loss_terms(spike.render_unit(best), target)
        dx = np.abs(best - truth)
        base_loss = spike.loss(target, target)  # 0 by construction
        reduction = 1.0 - f.best / initial
        flat = []
        for i, name in enumerate(FREE):
            deltas = []
            for step in (-0.1, 0.1):
                x = truth.copy()
                x[i] = np.clip(x[i] + step, 0.0, 1.0)
                deltas.append(abs(spike.loss(spike.render_unit(x), target) - base_loss))
            if max(deltas) < 0.01:
                flat.append(name)
        print(f"{seed:>4} {initial:7.3f} {f.best:8.4f} {reduction * 100:6.1f} {spectral:7.4f} {env:7.4f}  "
              f"{dx.max():7.3f} {dx.mean():8.3f}  {', '.join(flat) or '-'}")
        if seed == 0:
            slices = one_d_slices(target, truth)
    print("\nOne-dimensional slices through the truth, seed 0 (local minima per parameter):")
    for name, (points, values, minima) in slices.items():
        print(f"  {name:>18}: {minima} local minimum/minima, loss range {values.min():.3f}-{values.max():.3f}")


def one_d_slices(target, truth, points=21):
    """Loss along each parameter with the others at truth. Counts local minima on the grid."""
    grid = np.linspace(0.0, 1.0, points)
    out = {}
    for i, name in enumerate(FREE):
        values = []
        for g in grid:
            x = truth.copy()
            x[i] = g
            values.append(spike.loss(spike.render_unit(x), target))
        values = np.asarray(values)
        inner = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
        out[name] = (grid, values, int(inner.sum()))
    return out


def part_b_or_c(kind):
    label = "random targets (0.25-0.75)" if kind == "random" else "realistic targets (near defaults)"
    print(f"\nTargets: {label}. Budget {MAX_EVALS} evaluations, {len(list(SEEDS))} seeds.")
    for optimizer in ("cma", "cma_restarts", "cma_staged"):
        wins, reductions, seconds = 0, [], []
        for seed, truth, start in make_truths(kind):
            target = spike.render_unit(truth)
            initial = Counted(target)(start)
            f, elapsed = fit_with(optimizer, target, start, seed)
            reduction = 1.0 - f.best / initial
            wins += reduction > 0.95
            reductions.append(reduction)
            seconds.append(elapsed)
            print(f"  {optimizer:>12} seed {seed}: {initial:.3f} -> {f.best:.4f} "
                  f"down {reduction * 100:5.1f}%  evals {f.evals:6d}  {elapsed:5.1f}s", flush=True)
        print(f"  {optimizer:>12}: {wins}/10 above 95%, median reduction "
              f"{np.median(reductions) * 100:.1f}%, median time {np.median(seconds):.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=["A", "B", "C"], default=None,
                        help="A identifiability, B random targets, C realistic targets; default all")
    args = parser.parse_args()
    if args.part in (None, "A"):
        part_a()
    if args.part in (None, "B"):
        print("B. Optimizers on random targets")
        part_b_or_c("random")
    if args.part in (None, "C"):
        print("C. Optimizers on realistic targets")
        part_b_or_c("realistic")


if __name__ == "__main__":
    main()
