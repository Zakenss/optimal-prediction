# Optimal-Prediction DMD for Incomplete Data

Preliminary code that accompanies a review of Katrutsa et al. (2022),
*"Extension of Dynamic Mode Decomposition for Dynamic Systems with Incomplete
Information Based on the t-Model of Optimal Prediction."* The longer-term goal
is to extend that method to Higher-Order DMD (HODMD) for incomplete, noisy data,
with electric power grids as the target application.

## Status

This is an early scaffold, not a finished reproduction. The test system here is a
linear coupled-oscillator stand-in, **not** the nonlinear Hamiltonian system used
in the paper, and the fitted memory term uses a simplified `(A, M)` form rather
than the paper's memory columns. The pipeline is meant for checking the experiment
structure and the optimisation behaviour. Replacing the stand-in with the paper's
nonlinear system and exact memory form is the next step.

## What the code does

The pipeline fits three models on the same data and compares them:

- standard DMD,
- a differentiable t-model fitted with Adam (via JAX autodiff),
- delay-based HODMD.

It then runs four checks:

1. **Reproduction** – compare projection, standard DMD, the analytic t-model, and
   the fitted t-model on the observed variables.
2. **Time-step sensitivity** – sweep the step size and watch where the model breaks down.
3. **Optimisation depth** – vary the number of Adam iterations and inspect convergence.
4. **HODMD baseline** – fit DMD on delay-embedded states and measure the effect of the delay.

It also reports relative prediction error, variance across memory initialisations,
and continuous-time spectra.

## Layout

```
experiments/        experiment drivers (exp1-exp4) and shared setup
experiments/src/    system definition, DMD/HODMD, metrics, t-model
results/            figures and metrics written by run_all.py
run_all.py          runs every experiment and writes results/
requirements.txt    Python dependencies
```

## Running it

```
pip install -r requirements.txt
python run_all.py
```

All output is written to the local `results/` folder. The two figures reported in
the accompanying write-up are `results/visual1_trajectories.png` and
`results/exp3_convergence_landscape.png`.
