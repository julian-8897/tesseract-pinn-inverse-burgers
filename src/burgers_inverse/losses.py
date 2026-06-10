"""PINN loss components, pointwise residuals, and BRDR adaptive weighting.

Pure, differentiable JAX objectives over ``(viscosity, params_flat)`` evaluated
through a PINN Tesseract. No optimization loop, presentation, or container
lifecycle lives here.
"""

from __future__ import annotations

import jax.numpy as jnp
from tesseract_jax import apply_tesseract

from burgers_inverse.configs import (
    LOSS_WEIGHT_NAMES,
    normalize_loss_weights,
)


def format_loss_weights(loss_weights):
    """Format loss weights for compact CLI output."""
    return ", ".join(f"{name}={loss_weights[name]:g}" for name in LOSS_WEIGHT_NAMES)


def loss_weights_to_array(loss_weights):
    """Return loss weights as a JAX array in canonical component order."""
    weights = normalize_loss_weights(loss_weights)
    return jnp.asarray([weights[name] for name in LOSS_WEIGHT_NAMES])


def validate_brdr_loss_weights(loss_weights):
    """BRDR uses fixed coefficients as positive component scaling factors."""
    weights = loss_weights_to_array(loss_weights)
    if bool(jnp.any(weights <= 0)):
        raise ValueError("BRDR adaptive loss weights require positive fixed weights")


def initialize_brdr_state(pointwise_losses):
    """Initialize pointwise BRDR moments and weights."""
    return {
        "step": 0,
        "moment": {
            name: jnp.zeros_like(losses) for name, losses in pointwise_losses.items()
        },
        "weights": {
            name: jnp.ones_like(losses) for name, losses in pointwise_losses.items()
        },
    }


def update_brdr_state(state, pointwise_losses, beta_c=0.9999, beta_w=0.999, eps=1e-12):
    """Update BRDR state from pointwise squared residual losses.

    BRDR computes inverse residual decay rates as loss / sqrt(EMA(loss^2)),
    normalizes them to unit global mean, and smooths the resulting pointwise
    weights with an EMA.
    """
    step = state["step"] + 1
    corrected_irdr = {}
    new_moment = {}

    bias_correction = 1.0 - beta_c**step
    for name, losses in pointwise_losses.items():
        losses = jnp.asarray(losses)
        moment = beta_c * state["moment"][name] + (1.0 - beta_c) * losses**2
        new_moment[name] = moment
        corrected_irdr[name] = losses / jnp.sqrt(moment / bias_correction + eps)

    all_irdr = jnp.concatenate(
        [jnp.ravel(corrected_irdr[name]) for name in LOSS_WEIGHT_NAMES]
    )
    mean_irdr = jnp.mean(all_irdr)

    new_weights = {}
    for name in LOSS_WEIGHT_NAMES:
        target_weight = corrected_irdr[name] / (mean_irdr + eps)
        weight = beta_w * state["weights"][name] + (1.0 - beta_w) * target_weight
        new_weights[name] = weight

    return {
        "step": step,
        "moment": new_moment,
        "weights": new_weights,
    }


def summarize_brdr_weights(brdr_weights):
    """Return mean BRDR weight by component for logging and plotting."""
    return {name: float(jnp.mean(brdr_weights[name])) for name in LOSS_WEIGHT_NAMES}


def compute_pointwise_losses(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
):
    """Compute pointwise squared residual losses for each PINN constraint."""
    result_obs = apply_tesseract(
        pinn,
        {
            "x": x_obs,
            "t": t_obs,
            "params_flat": params_flat,
        },
    )
    u_pred = result_obs["u_pred"]
    data_losses = (u_pred - u_obs) ** 2

    result_col = apply_tesseract(
        pinn, {"x": x_col, "t": t_col, "params_flat": params_flat}
    )

    u_col = result_col["u_pred"]
    u_x = result_col["u_x"]
    u_t = result_col["u_t"]
    u_xx = result_col["u_xx"]

    residual = u_t + u_col * u_x - viscosity * u_xx
    physics_losses = residual**2

    t_ic = jnp.zeros_like(x_ic)
    result_ic = apply_tesseract(
        pinn,
        {
            "x": x_ic,
            "t": t_ic,
            "params_flat": params_flat,
        },
    )
    u_ic = result_ic["u_pred"]
    u_ic_true = jnp.sin(2 * jnp.pi * x_ic)
    ic_losses = (u_ic - u_ic_true) ** 2

    x_left = jnp.zeros_like(t_bc)
    x_right = jnp.ones_like(t_bc)

    result_left = apply_tesseract(
        pinn,
        {
            "x": x_left,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    result_right = apply_tesseract(
        pinn,
        {
            "x": x_right,
            "t": t_bc,
            "params_flat": params_flat,
        },
    )
    u_left = result_left["u_pred"]
    u_right = result_right["u_pred"]
    bc_losses = (u_left - u_right) ** 2

    return {
        "data": data_losses,
        "physics": physics_losses,
        "ic": ic_losses,
        "bc": bc_losses,
    }


def compute_loss_components(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute total and component PINN losses for the inverse problem.

    Components:
    1. Data loss: fit observations
    2. Physics loss: satisfy PDE residual
    3. Initial condition loss: u(x, 0) = sin(2πx)
    4. Boundary condition loss: periodic BCs u(0,t) = u(1,t)

    All terms are differentiable with respect to viscosity.
    """
    weights = normalize_loss_weights(loss_weights)
    pointwise_losses = compute_pointwise_losses(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
    )

    data_loss = jnp.mean(pointwise_losses["data"])
    physics_loss = jnp.mean(pointwise_losses["physics"])
    ic_loss = jnp.mean(pointwise_losses["ic"])
    bc_loss = jnp.mean(pointwise_losses["bc"])

    raw_losses = jnp.asarray([data_loss, physics_loss, ic_loss, bc_loss])
    if brdr_weights is None:
        weight_array = loss_weights_to_array(weights)
        total_loss = jnp.sum(weight_array * raw_losses)
    else:
        total_loss = sum(
            weights[name] * jnp.mean(brdr_weights[name] * pointwise_losses[name])
            for name in LOSS_WEIGHT_NAMES
        )

    return {
        "total": total_loss,
        "data": data_loss,
        "physics": physics_loss,
        "ic": ic_loss,
        "bc": bc_loss,
    }


def compute_loss(
    viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute scalar total PINN loss for gradient-based optimization."""
    components = compute_loss_components(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )

    return components["total"]


def compute_loss_from_log_viscosity(
    log_viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights=None,
    loss_weights=None,
):
    """Compute loss while optimizing viscosity in log-space."""
    viscosity = jnp.exp(log_viscosity)
    return compute_loss(
        viscosity,
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )


def _loss_from_log_and_params(
    log_viscosity,
    params_flat,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    brdr_weights,
    loss_weights,
):
    """Scalar loss as a function of both optimized variables for value_and_grad."""
    return compute_loss(
        jnp.exp(log_viscosity),
        params_flat,
        x_obs,
        t_obs,
        u_obs,
        x_col,
        t_col,
        x_ic,
        t_bc,
        pinn,
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )
