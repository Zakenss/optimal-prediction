"""
t-model helpers: analytic rollout and a fitted discrete model.

Analytic form used here (linear stand-in):
  dx/dt = (L_xx + t * K0) x

Fitted discrete form used in the preliminary code:
  x_{k+1} = x_k + dt * A @ x_k + dt * t_k * M @ x_k
  with t_k = k * dt

Note: the paper fits a single operator A with memory columns
g_j(A, n) = j * dt * expm(j * (A - I)) @ n. That form is the next
implementation target. The (A, M) form below is a working stand-in
so the experiment pipeline runs end to end.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)


def analytic_tmodel(L_xx: np.ndarray, K0: np.ndarray, x0, ts: np.ndarray) -> np.ndarray:
    """Integrate dx/dt = (L_xx + t * K0) x with RK4. Returns (len(ts), 2)."""
    L_xx = np.asarray(L_xx)
    K0 = np.asarray(K0)

    def gen(t):
        return L_xx + t * K0

    out = np.empty((len(ts), 2))
    x = np.asarray(x0, dtype=float).copy()
    out[0] = x

    for k in range(1, len(ts)):
        t = ts[k - 1]
        h = ts[k] - ts[k - 1]
        k1 = gen(t) @ x
        k2 = gen(t + 0.5 * h) @ (x + 0.5 * h * k1)
        k3 = gen(t + 0.5 * h) @ (x + 0.5 * h * k2)
        k4 = gen(t + h) @ (x + h * k3)
        x = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[k] = x

    return out


def rollout(params, x0, dt, n_steps):
    """Run the discrete (A, M) model for n_steps. Returns (n_steps, 2)."""
    A, M = params
    x0 = jnp.asarray(x0)

    def step(carry, k):
        x = carry
        t_k = k * dt
        x_next = x + dt * (A @ x) + dt * t_k * (M @ x)
        return x_next, x

    ks = jnp.arange(n_steps)
    _, xs = jax.lax.scan(step, x0, ks)
    return xs


def _loss(params, target, x0, dt):
    pred = rollout(params, x0, dt, target.shape[0])
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=1))


def _tree_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))


def fit_tmodel(
    target: np.ndarray,
    x0,
    dt: float,
    n_iters: int = 1000,
    lr: float = 1e-2,
    seed: int = 0,
    A_init: np.ndarray | None = None,
    memory_scale: float = 0.02,
    checkpoints: tuple | None = None,
):
    """
    Fit matrices A and M with Adam.

    If checkpoints is given, store the prediction and loss at those
    iteration counts during a single run.
    """
    ckpt_set = set(checkpoints or ())
    key = jax.random.PRNGKey(seed)
    kA, kM = jax.random.split(key)

    if A_init is None:
        A = jax.random.normal(kA, (2, 2)) * 0.01
    else:
        A = jnp.asarray(A_init)

    M = jax.random.normal(kM, (2, 2)) * memory_scale
    params = (A, M)

    target_j = jnp.asarray(target)
    x0_j = jnp.asarray(x0)

    opt = optax.adam(lr)
    state = opt.init(params)
    loss_grad = jax.jit(jax.value_and_grad(lambda p: _loss(p, target_j, x0_j, dt)))

    losses = np.empty(n_iters)
    gnorms = np.empty(n_iters)
    ckpt_pred = {}
    ckpt_loss = {}

    for it in range(n_iters):
        loss_val, grads = loss_grad(params)
        losses[it] = float(loss_val)
        gnorms[it] = float(_tree_norm(grads))
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)

        n_done = it + 1
        if n_done in ckpt_set:
            ckpt_pred[n_done] = np.array(rollout(params, x0_j, dt, target.shape[0]))
            ckpt_loss[n_done] = float(_loss(params, target_j, x0_j, dt))

    A_f = np.array(params[0])
    M_f = np.array(params[1])
    pred = np.array(rollout(params, x0_j, dt, target.shape[0]))

    return {
        "A": A_f,
        "M": M_f,
        "pred": pred,
        "losses": losses,
        "grad_norms": gnorms,
        "final_loss": float(losses[-1]),
        "ckpt_pred": ckpt_pred,
        "ckpt_loss": ckpt_loss,
    }


def effective_generator(A: np.ndarray, M: np.ndarray, t: float) -> np.ndarray:
    """Return G(t) = A + t * M."""
    return np.asarray(A) + t * np.asarray(M)


def loss_landscape(params, target, x0, dt, span=1.0, n=41):
    """
    Evaluate the loss on a 2D grid along the two leading Hessian directions
    at the fitted point. Returns (alphas, betas, Z, hessian_eigenvalues).
    """
    A0 = np.asarray(params[0])
    M0 = np.asarray(params[1])
    theta0 = jnp.asarray(np.concatenate([A0.ravel(), M0.ravel()]))
    target_j = jnp.asarray(target)
    x0_j = jnp.asarray(x0)

    def loss_vec(theta):
        A = theta[:4].reshape(2, 2)
        M = theta[4:].reshape(2, 2)
        return _loss((A, M), target_j, x0_j, dt)

    H = np.array(jax.hessian(loss_vec)(theta0))
    H = 0.5 * (H + H.T)
    evals, evecs = np.linalg.eigh(H)
    v1 = evecs[:, -1]
    v2 = evecs[:, -2]

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    Z = np.empty((n, n))
    loss_np = jax.jit(loss_vec)

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            Z[i, j] = float(loss_np(theta0 + a * v1 + b * v2))

    return alphas, betas, Z, evals
