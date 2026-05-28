"""Tesseract Cross-Framework Autodiff Demo: Inverse Burgers Equation Solver.

Demonstrates Tesseract's pipeline-level automatic differentiation across JAX and PyTorch,
enabling JAX-based optimization of PyTorch PINN models via VJP (Vector-Jacobian Product).
"""

import json
import subprocess
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import seaborn as sns
import streamlit as st
from tesseract_jax import apply_tesseract

from configs import DEFAULT_LOSS_WEIGHTS, LOSS_WEIGHT_NAMES
from inverse_problem import (
    Tesseract,
    compute_loss,
    compute_loss_components,
    compute_loss_from_log_viscosity,
    compute_pointwise_losses,
    generate_observations,
    get_burgers_solver,
    get_initial_params,
    initialize_brdr_state,
    summarize_brdr_weights,
    update_brdr_state,
)

st.set_page_config(
    page_title="Tesseract Cross-Framework Autodiff Demo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)

sns.set_theme(
    context="notebook",
    style="whitegrid",
    palette="deep",
    rc={
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "figure.facecolor": "white",
    },
)

PINN_COLOR = "#1f77b4"
PYTORCH_COLOR = "#ff7f0e"
TRUE_COLOR = "#c44e52"
LOSS_COLOR = "#dd8452"
PARAM_GRAD_COLOR = "#55a868"
LOSS_WEIGHT_COLORS = {
    "data": "#4c72b0",
    "physics": "#dd8452",
    "ic": "#55a868",
    "bc": "#c44e52",
}
FIELD_CMAP = sns.color_palette("vlag", as_cmap=True)
ERROR_CMAP = sns.color_palette("rocket", as_cmap=True)


def finish_axes(ax):
    """Apply consistent plot cosmetics."""
    ax.grid(True, alpha=0.25)
    sns.despine(ax=ax)


@st.cache_data(show_spinner=False)
def cached_observations(
    n_obs,
    true_viscosity,
    noise_std,
    seed,
    domain_x_min=0.0,
    domain_x_max=1.0,
    domain_t_min=0.0,
    domain_t_max=1.0,
):
    """Generate and cache solver-backed observations for repeatable app runs."""
    domain = {
        "x": (domain_x_min, domain_x_max),
        "t": (domain_t_min, domain_t_max),
    }
    x_obs, t_obs, u_obs = generate_observations(
        n_obs,
        true_viscosity,
        domain,
        jax.random.PRNGKey(seed),
        noise_std=noise_std,
    )
    return np.asarray(x_obs), np.asarray(t_obs), np.asarray(u_obs)


def history_frame(history, epoch_key="epoch"):
    """Convert a dict of equal-length histories into a dataframe."""
    return pd.DataFrame(
        {epoch_key: np.arange(len(next(iter(history.values())))), **history}
    )


def docker_image_available(image_name):
    """Return whether a local Tesseract Docker image exists."""
    try:
        result = subprocess.run(
            ["docker", "inspect", image_name, "--type", "image"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0


def pinn_image_name(backend):
    """Map the selected backend to its local Tesseract image."""
    return "pinn_jax" if backend == "jax" else "pinn_pytorch"


def render_tesseract_contract(backend, image_name, trace_enabled):
    """Render the stable host/container contract used by the demo."""
    st.subheader("Tesseract Contract")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Host Optimizer", "JAX / Optax")
    col2.metric("Autodiff Boundary", "Tesseract VJP")
    col3.metric("PINN Container", image_name)
    col4.metric("Trace Collection", "On" if trace_enabled else "Off")

    st.code(
        f"""JAX loss + Optax updates
        |
        v
jax.grad(total_loss)
        |
        v
Tesseract primitive: {image_name}
        |
        v
{backend.upper()} PINN apply/VJP endpoint""",
        language="text",
    )


def render_shared_objective(loss_weights, adaptive_loss_weights):
    """Show the objective that stays fixed while the backend changes."""
    st.subheader("Shared JAX Objective")
    st.latex(r"L = w_d L_{data} + w_f L_{physics} + w_{ic} L_{ic} + w_{bc} L_{bc}")
    weight_df = pd.DataFrame(
        {
            "component": list(loss_weights.keys()),
            "base_weight": [float(value) for value in loss_weights.values()],
        }
    )
    st.dataframe(weight_df, hide_index=True)
    if adaptive_loss_weights:
        st.caption(
            "BRDR is applied inside the same objective by replacing each component's pointwise residual weights."
        )
    else:
        st.caption(
            "These fixed weights are used unchanged for either PINN backend; only the Tesseract image changes."
        )


@dataclass
class GradientFlowMetrics:
    """Track Tesseract gradient flow metrics."""

    epoch: int
    vjp_calls: int
    apply_calls: int
    visc_grad_norm: float
    param_grad_norm: float
    loss_value: float
    shapes: dict[str, tuple]


def initialize_session_state():
    """Initialize all session state variables."""
    if "training" not in st.session_state:
        st.session_state.training = False
    if "trained_viscosity" not in st.session_state:
        st.session_state.trained_viscosity = {}
    if "viscosity_history" not in st.session_state:
        st.session_state.viscosity_history = {}
    if "loss_history" not in st.session_state:
        st.session_state.loss_history = {}
    if "loss_weight_history" not in st.session_state:
        st.session_state.loss_weight_history = {}
    if "params_flat" not in st.session_state:
        st.session_state.params_flat = {}
    if "epoch_times" not in st.session_state:
        st.session_state.epoch_times = {}
    if "gradient_metrics" not in st.session_state:
        st.session_state.gradient_metrics = []
    if "tesseract_trace" not in st.session_state:
        st.session_state.tesseract_trace = {}
    if "show_gradient_inspector" not in st.session_state:
        st.session_state.show_gradient_inspector = False


def generate_solution_grid(true_viscosity, params_flat, pinn, nx=128, nt=64):
    """Generate PINN and solver solutions on a grid for visualization."""
    x = np.linspace(0, 1, nx, endpoint=False, dtype=np.float32)
    t = np.linspace(0, 1, nt, dtype=np.float32)
    X, T = np.meshgrid(x, t)

    # Flatten for tesseract evaluation
    x_flat = jnp.array(X.flatten(), dtype=jnp.float32)
    t_flat = jnp.array(T.flatten(), dtype=jnp.float32)

    result = apply_tesseract(
        pinn, {"x": x_flat, "t": t_flat, "params_flat": params_flat}
    )
    u_pred = np.array(result["u_pred"]).reshape(nt, nx)

    solve_burgers = get_burgers_solver()
    u_solver = solve_burgers(
        jnp.asarray(true_viscosity, dtype=jnp.float32),
        jnp.asarray(x, dtype=jnp.float32),
        jnp.asarray(t, dtype=jnp.float32),
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )

    return X, T, u_pred, np.array(u_solver)


def render_gradient_flow_inspector(backend, gradient_metrics):
    """Render the gradient flow inspector UI."""
    st.markdown(f"""
    ### Cross-Framework Autodiff Pipeline

    This trace shows how **Tesseract lets a JAX gradient flow through the
    {backend.upper()} PINN container** while the outer optimizer remains unchanged.
    """)

    if backend == "pytorch":
        st.markdown("""
        ```
        JAX Optimizer (optax)
                |
        jax.grad(compute_loss)
                |
        Tesseract VJP Endpoint  <cross-framework boundary>
                |
        PyTorch Autograd (torch.autograd.grad)
                |
        PyTorch PINN forward pass
                |
        gradients return through VJP
                |
        JAX receives dL/dlog_nu and dL/dparams_flat
        ```
        """)
    else:
        st.markdown("""
        ```
        JAX Optimizer (optax)
                |
        jax.grad(compute_loss)
                |
        Tesseract Apply/VJP Endpoint
                |
        JAX PINN forward pass
                |
        JAX receives dL/dlog_nu and dL/dparams_flat
        ```
        """)

    if not gradient_metrics:
        st.info(
            "Enable Tesseract trace collection before running to see call counts and gradient norms."
        )
        return

    tab1, tab2, tab3 = st.tabs(["Call Statistics", "Gradient Norms", "Tensor Shapes"])

    with tab1:
        st.subheader("Tesseract API Call Count")

        col1, col2, col3 = st.columns(3)
        latest = gradient_metrics[-1]

        col1.metric(
            "apply() calls per epoch",
            latest.apply_calls,
            help="Forward pass evaluations",
        )
        col2.metric(
            "VJP calls per epoch",
            latest.vjp_calls,
            help="Backward pass gradient evaluations",
        )
        col3.metric("Total AD operations", latest.apply_calls + latest.vjp_calls)

        st.info(f"""
        **PINN Loss Architecture**: Each epoch computes a composite loss with {latest.apply_calls} network evaluations:

        1. **Data loss**: MSE at observation points
        2. **Physics loss**: PDE residual
        3. **Initial condition**: enforce u(x, t=0)
        4. **Boundary left**: periodic boundary value
        5. **Boundary right**: periodic boundary value

        Then **{latest.vjp_calls} VJP calls** compute gradients: dL/dlog_nu and dL/dparams_flat.

        {"VJP calls route through PyTorch autograd" if backend == "pytorch" else "The JAX container uses native JAX autodiff behind the same Tesseract interface."}
        """)

    with tab2:
        st.subheader("Gradient Magnitude Evolution")

        epochs = [m.epoch for m in gradient_metrics]
        visc_grads = [m.visc_grad_norm for m in gradient_metrics]
        param_grads = [m.param_grad_norm for m in gradient_metrics]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        sns.lineplot(
            x=epochs,
            y=visc_grads,
            marker="o",
            color=PINN_COLOR,
            linewidth=2,
            ax=ax1,
        )
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("||∂L/∂log(ν)||")
        ax1.set_title("Log-Viscosity Gradient Norm")
        finish_axes(ax1)

        sns.lineplot(
            x=epochs,
            y=param_grads,
            marker="o",
            color=PARAM_GRAD_COLOR,
            linewidth=2,
            ax=ax2,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("||∂L/∂params||")
        ax2.set_title("Network Parameter Gradient Norm")
        finish_axes(ax2)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)
        col1.metric("Latest ||∂L/∂log(ν)||", f"{visc_grads[-1]:.2e}")
        col2.metric("Latest ||∂L/∂params||", f"{param_grads[-1]:.2e}")

        st.info("""
        **Gradient norms** show the sensitivity of loss to parameters:
        - High norm: steep loss landscape, large updates
        - Decreasing norm: approaching optimum
        - These are computed through Tesseract's VJP endpoint
        """)

    with tab3:
        st.subheader("Tensor Shapes Through Pipeline")

        if latest.shapes:
            st.json(latest.shapes)
        else:
            st.info("No shape information available")

        st.markdown(f"""
        **Data flow through Tesseract**:
        - **Inputs**: x, t, and params_flat
        - **Outputs**: u_pred, u_x, u_t, and u_xx
        - Derivatives are computed by the **{backend.upper()}** backend and exposed through the same Tesseract API
        """)


def train_step(
    backend,
    log_viscosity,
    params_flat,
    brdr_weights,
    visc_opt_state,
    param_opt_state,
    loss_weights,
    x_obs,
    t_obs,
    u_obs,
    x_col,
    t_col,
    x_ic,
    t_bc,
    pinn,
    visc_optimizer,
    param_optimizer,
    update_viscosity=True,
    log_nu_bounds=None,
    epoch=0,
    track_gradients=False,
):
    """Single training step with optional gradient flow tracking."""

    grad_log_visc = jax.grad(compute_loss_from_log_viscosity, argnums=0)
    grad_params = jax.grad(compute_loss, argnums=1)

    v_grad = grad_log_visc(
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
        brdr_weights=brdr_weights,
        loss_weights=loss_weights,
    )
    viscosity = jnp.exp(log_viscosity)
    p_grad = grad_params(
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

    # Compute gradient norms
    visc_grad_norm = float(jnp.linalg.norm(v_grad))
    param_grad_norm = float(jnp.linalg.norm(p_grad))

    # Update log-viscosity so the physical viscosity remains positive.
    if update_viscosity:
        visc_updates, visc_opt_state = visc_optimizer.update(v_grad, visc_opt_state)
        log_viscosity = optax.apply_updates(log_viscosity, visc_updates)
        if log_nu_bounds is not None:
            log_viscosity = jnp.clip(log_viscosity, log_nu_bounds[0], log_nu_bounds[1])
    viscosity = jnp.exp(log_viscosity)

    # Update params
    param_updates, param_opt_state = param_optimizer.update(p_grad, param_opt_state)
    params_flat = optax.apply_updates(params_flat, param_updates)

    # Compute loss
    loss = compute_loss(
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

    metrics = None
    if track_gradients:
        metrics = GradientFlowMetrics(
            epoch=epoch,
            vjp_calls=2,
            apply_calls=5,
            visc_grad_norm=visc_grad_norm,
            param_grad_norm=param_grad_norm,
            loss_value=float(loss),
            shapes={
                "x_obs": tuple(x_obs.shape),
                "t_obs": tuple(t_obs.shape),
                "params_flat": tuple(params_flat.shape),
            },
        )

    return (
        log_viscosity,
        params_flat,
        visc_opt_state,
        param_opt_state,
        float(loss),
        metrics,
    )


def main():
    initialize_session_state()

    st.title("Tesseract Cross-Framework Autodiff Showcase")
    st.subheader(
        "One JAX inverse-problem loop, swappable JAX and PyTorch PINN containers"
    )

    st.markdown("""
    This demo keeps the optimizer, loss function, and inverse-problem orchestration in JAX.
    The only component that changes is the Tesseract PINN container. Tesseract exposes
    backend-specific apply/VJP endpoints as JAX primitives, so `jax.grad` can optimize
    through either backend.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Host Loop", "JAX + Optax")
    col2.metric("Autodiff Boundary", "Tesseract VJP")
    col3.metric("Swappable Containers", "JAX / PyTorch")

    st.code(
        """fixed observations
        |
        v
JAX objective: data + physics + IC + BC losses
        |
        v
jax.grad(total_loss)
        |
        v
Tesseract PINN container: JAX or PyTorch
        |
        v
updates for log_nu and params_flat""",
        language="text",
    )

    with st.expander("Inverse Burgers problem used as the test case"):
        st.markdown(
            "Given noisy observations of the 1D Burgers equation solution, infer the unknown viscosity parameter $\\nu$:"
        )
        st.latex(
            r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}"
        )
        st.caption(
            "The Burgers solver currently generates fixed observations offline; the active Tesseract boundary in this app is the PINN container."
        )

    st.sidebar.header("Configuration")

    backend = st.sidebar.selectbox(
        "Tesseract PINN Container",
        ["jax", "pytorch"],
        help="Select backend implementation. Both expose identical Tesseract endpoints (apply/VJP/JVP), enabling seamless backend switching.",
    )

    seed = st.sidebar.number_input(
        "Seed",
        min_value=0,
        max_value=1_000_000,
        value=123,
        step=1,
        help="Controls observations, collocation points, and model initialization",
    )

    true_viscosity = st.sidebar.slider(
        "True Viscosity $\\nu$ (Ground Truth)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Ground truth viscosity parameter used to generate synthetic observations",
    )

    initial_viscosity = st.sidebar.slider(
        "Initial Viscosity Guess $\\nu_0$",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        help="Initial estimate for gradient-based optimization (typically set below ground truth)",
    )

    n_obs = st.sidebar.slider(
        "Number of Observations",
        min_value=20,
        max_value=200,
        value=100,
        step=20,
        help="Number of spatiotemporal observation points for data loss term",
    )

    noise_level = st.sidebar.slider(
        "Observation Noise (σ)",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01,
        help="Standard deviation of additive Gaussian noise in synthetic observations",
    )

    n_epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        help="Number of training epochs for PINN",
    )

    learning_rate = st.sidebar.slider(
        "Log-Viscosity Learning Rate",
        min_value=0.001,
        max_value=0.2,
        value=0.02,
        step=0.001,
        format="%.3f",
        help="Optimizer learning rate for log(ν)",
    )

    viscosity_warmup_epochs = st.sidebar.slider(
        "Viscosity Warmup Epochs",
        min_value=0,
        max_value=100,
        value=25,
        step=5,
        help="Train the PINN field before updating ν, reducing early overshoot from random derivatives",
    )

    clip_log_viscosity = st.sidebar.checkbox(
        "Constrain Viscosity Range",
        value=True,
        help="Clip ν to a conservative positive range during optimization",
    )
    effective_warmup_epochs = min(viscosity_warmup_epochs, max(0, n_epochs - 1))
    if effective_warmup_epochs != viscosity_warmup_epochs:
        st.sidebar.warning(
            f"Warmup is capped at {effective_warmup_epochs} epochs for this run."
        )

    param_learning_rate = st.sidebar.slider(
        "Network Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        step=0.0001,
        format="%.4f",
        help="Optimizer learning rate for PINN parameters",
    )

    with st.sidebar.expander("Sampling"):
        n_col = st.number_input(
            "Collocation Points",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
        )
        n_ic = st.number_input(
            "Initial-Condition Points",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
        )
        n_bc = st.number_input(
            "Boundary Points",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
        )

    with st.sidebar.expander("Loss Weights"):
        loss_weights = {
            "data": st.number_input(
                "Data",
                min_value=0.0,
                value=float(DEFAULT_LOSS_WEIGHTS["data"]),
                step=0.1,
            ),
            "physics": st.number_input(
                "Physics",
                min_value=0.0,
                value=float(DEFAULT_LOSS_WEIGHTS["physics"]),
                step=0.05,
            ),
            "ic": st.number_input(
                "Initial Condition",
                min_value=0.0,
                value=float(DEFAULT_LOSS_WEIGHTS["ic"]),
                step=0.05,
            ),
            "bc": st.number_input(
                "Boundary",
                min_value=0.0,
                value=float(DEFAULT_LOSS_WEIGHTS["bc"]),
                step=0.05,
            ),
        }

    adaptive_loss_weights = st.sidebar.checkbox(
        "BRDR Adaptive Loss Weights",
        value=False,
        help="Use Balanced Residual Decay Rate pointwise weights for data, physics, initial-condition, and boundary residuals",
    )

    brdr_beta_c = 0.9999
    brdr_beta_w = 0.999
    if adaptive_loss_weights:
        if any(value <= 0 for value in loss_weights.values()):
            st.sidebar.warning("BRDR requires all fixed loss weights to be positive.")
        brdr_beta_c = st.sidebar.slider(
            "BRDR Residual EMA",
            min_value=0.9,
            max_value=0.99999,
            value=0.9999,
            step=0.00001,
            format="%.5f",
            help="EMA factor for BRDR residual-history estimates",
        )
        brdr_beta_w = st.sidebar.slider(
            "BRDR Weight EMA",
            min_value=0.9,
            max_value=0.9999,
            value=0.999,
            step=0.0001,
            format="%.4f",
            help="EMA factor for BRDR pointwise weights",
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tesseract Diagnostics")
    show_gradient_inspector = st.sidebar.checkbox(
        "Collect Tesseract Trace",
        value=True,
        help="Track Tesseract apply/VJP calls, gradient norms, and tensor shapes through the autodiff boundary",
    )

    can_train = not (
        adaptive_loss_weights and any(value <= 0 for value in loss_weights.values())
    )
    if st.sidebar.button(
        "Run Tesseract Inversion", type="primary", disabled=not can_train
    ):
        st.session_state.training = True
        st.session_state.gradient_metrics = []

    if st.session_state.training:
        # Setup
        domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        key = jax.random.PRNGKey(int(seed))

        key_col_x, key_col_t, key_ic, key_bc = jax.random.split(key, 4)
        x_obs_np, t_obs_np, u_obs_np = cached_observations(
            n_obs,
            true_viscosity,
            noise_level,
            int(seed),
        )
        x_obs = jnp.asarray(x_obs_np, dtype=jnp.float32)
        t_obs = jnp.asarray(t_obs_np, dtype=jnp.float32)
        u_obs = jnp.asarray(u_obs_np, dtype=jnp.float32)

        # Make collocation points
        x_col = jax.random.uniform(
            key_col_x, (n_col,), minval=domain["x"][0], maxval=domain["x"][1]
        )
        t_col = jax.random.uniform(
            key_col_t, (n_col,), minval=0.05, maxval=domain["t"][1]
        )

        x_ic = jax.random.uniform(
            key_ic, (n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
        )

        t_bc = jax.random.uniform(key_bc, (n_bc,), minval=0.05, maxval=domain["t"][1])

        # Initialize tesseract
        image_name = pinn_image_name(backend)
        if not docker_image_available(image_name):
            st.error(f"Tesseract image `{image_name}` was not found.")
            st.code("./buildall.sh", language="bash")
            st.session_state.training = False
            return
        pinn = Tesseract.from_image(image_name)
        params_flat = get_initial_params(backend, seed=int(seed))

        # Initialize log-viscosity
        log_viscosity = jnp.log(jnp.asarray(initial_viscosity))
        log_nu_bounds = (
            (
                jnp.log(jnp.asarray(1e-4, dtype=jnp.float32)),
                jnp.log(jnp.asarray(0.5, dtype=jnp.float32)),
            )
            if clip_log_viscosity
            else None
        )
        brdr_state = None
        brdr_weights = None

        visc_optimizer = optax.adam(learning_rate)
        visc_opt_state = visc_optimizer.init(log_viscosity)
        param_optimizer = optax.adam(param_learning_rate)
        param_opt_state = param_optimizer.init(params_flat)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Container", image_name)
        with col2:
            st.metric("Backend", backend.upper())
        with col3:
            st.metric("True Viscosity", f"{true_viscosity:.4f}")
        with col4:
            st.metric("Initial Guess", f"{initial_viscosity:.4f}")

        st.caption(
            "The host JAX objective and Optax optimizers stay fixed; this run changes only the Tesseract PINN container."
        )

        if adaptive_loss_weights:
            st.caption(
                "BRDR updates pointwise residual weights from inverse residual decay rates and tracks component mean weights."
            )
        if effective_warmup_epochs:
            st.caption(
                f"ν is frozen for the first {effective_warmup_epochs} epochs so the PINN field can form before parameter inversion starts."
            )

        st.markdown("---")

        progress_bar = st.progress(0)
        status_text = st.empty()

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_visc = metric_col1.empty()
        metric_error = metric_col2.empty()
        metric_loss = metric_col3.empty()
        metric_time = metric_col4.empty()

        plot_columns = st.columns(3 if adaptive_loss_weights else 2)
        plot_col1 = plot_columns[0]
        plot_col2 = plot_columns[1]
        with plot_col1:
            st.subheader("Viscosity Convergence")
            visc_chart = st.empty()
        with plot_col2:
            st.subheader("Training Loss")
            loss_chart = st.empty()
        if adaptive_loss_weights:
            with plot_columns[2]:
                st.subheader("Loss Weights")
                weight_chart = st.empty()
        else:
            weight_chart = None

        visc_history = [float(initial_viscosity)]
        loss_history = []
        loss_weight_history = {name: [loss_weights[name]] for name in LOSS_WEIGHT_NAMES}
        component_loss_history = {
            name: [] for name in ("total", "data", "physics", "ic", "bc")
        }
        component_loss_epochs = []
        time_history = []

        with pinn:
            for epoch in range(n_epochs):
                start_time = time.time()

                # Track gradients every 5 epochs (or first 10) if inspector enabled
                track_this_epoch = show_gradient_inspector and (
                    epoch % 5 == 0 or epoch < 10
                )

                viscosity = jnp.exp(log_viscosity)
                if adaptive_loss_weights:
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
                    if brdr_state is None:
                        brdr_state = initialize_brdr_state(pointwise_losses)
                    brdr_state = update_brdr_state(
                        brdr_state,
                        pointwise_losses,
                        beta_c=brdr_beta_c,
                        beta_w=brdr_beta_w,
                    )
                    brdr_weights = brdr_state["weights"]
                else:
                    brdr_weights = None

                (
                    log_viscosity,
                    params_flat,
                    visc_opt_state,
                    param_opt_state,
                    loss,
                    metrics,
                ) = train_step(
                    backend,
                    log_viscosity,
                    params_flat,
                    brdr_weights,
                    visc_opt_state,
                    param_opt_state,
                    loss_weights,
                    x_obs,
                    t_obs,
                    u_obs,
                    x_col,
                    t_col,
                    x_ic,
                    t_bc,
                    pinn,
                    visc_optimizer,
                    param_optimizer,
                    update_viscosity=epoch >= effective_warmup_epochs,
                    log_nu_bounds=log_nu_bounds,
                    epoch=epoch,
                    track_gradients=track_this_epoch,
                )

                if metrics and show_gradient_inspector:
                    st.session_state.gradient_metrics.append(metrics)

                epoch_time = time.time() - start_time
                time_history.append(epoch_time)

                visc_val = float(jnp.exp(log_viscosity))
                visc_history.append(visc_val)
                loss_history.append(loss)
                current_weights = (
                    loss_weights
                    if brdr_weights is None
                    else summarize_brdr_weights(brdr_weights)
                )
                for name in LOSS_WEIGHT_NAMES:
                    loss_weight_history[name].append(current_weights[name])

                # Update every 5 epochs
                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    error = abs(visc_val - true_viscosity)
                    rel_error = error / true_viscosity * 100
                    loss_components = compute_loss_components(
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
                    component_loss_epochs.append(epoch + 1)
                    for name in component_loss_history:
                        component_loss_history[name].append(
                            float(loss_components[name])
                        )

                    # Update progress
                    progress = (epoch + 1) / n_epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{n_epochs}")

                    metric_visc.metric(
                        "Current ν",
                        f"{visc_val:.6f}",
                        delta=f"{visc_val - true_viscosity:.6f}",
                    )
                    metric_error.metric("Relative Error", f"{rel_error:.2f}%")
                    metric_loss.metric("Loss", f"{loss_components['total']:.6f}")
                    metric_time.metric("Epoch Time", f"{epoch_time * 1000:.1f}ms")

                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    epochs = np.arange(len(visc_history))
                    sns.lineplot(
                        x=epochs,
                        y=visc_history,
                        label="Inferred ν",
                        color=PINN_COLOR,
                        linewidth=2.3,
                        ax=ax1,
                    )
                    ax1.axhline(
                        true_viscosity,
                        color=TRUE_COLOR,
                        linestyle="--",
                        linewidth=2,
                        label=f"True ν = {true_viscosity}",
                    )
                    if effective_warmup_epochs:
                        ax1.axvline(
                            effective_warmup_epochs,
                            color="#666666",
                            linestyle=":",
                            linewidth=1.5,
                            label="ν warmup end",
                        )
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("Viscosity")
                    ax1.legend(frameon=False)
                    finish_axes(ax1)
                    visc_chart.pyplot(fig1)
                    plt.close(fig1)

                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.lineplot(
                        x=np.arange(len(loss_history)),
                        y=loss_history,
                        color=LOSS_COLOR,
                        linewidth=2.3,
                        ax=ax2,
                    )
                    ax2.set_yscale("log")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Loss (log scale)")
                    finish_axes(ax2)
                    loss_chart.pyplot(fig2)
                    plt.close(fig2)

                    if weight_chart is not None:
                        fig3, ax3 = plt.subplots(figsize=(6, 4))
                        weight_epochs = np.arange(
                            len(next(iter(loss_weight_history.values())))
                        )
                        for name in LOSS_WEIGHT_NAMES:
                            sns.lineplot(
                                x=weight_epochs,
                                y=loss_weight_history[name],
                                label=name,
                                color=LOSS_WEIGHT_COLORS[name],
                                linewidth=2.2,
                                ax=ax3,
                            )
                        ax3.set_yscale("log")
                        ax3.set_xlabel("Epoch")
                        ax3.set_ylabel("Effective Weight")
                        ax3.legend(frameon=False)
                        finish_axes(ax3)
                        weight_chart.pyplot(fig3)
                        plt.close(fig3)

            st.markdown("---")
            st.success("Finished training.")

            final_visc = float(jnp.exp(log_viscosity))
            final_error = abs(final_visc - true_viscosity) / true_viscosity * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Viscosity", f"{final_visc:.6f}")
            col2.metric("True Viscosity", f"{true_viscosity:.6f}")
            col3.metric("Relative Error", f"{final_error:.2f}%")
            col4.metric("Avg Time/Epoch", f"{np.mean(time_history) * 1000:.1f}ms")

            st.session_state.trained_viscosity[backend] = final_visc
            st.session_state.viscosity_history[backend] = visc_history
            st.session_state.loss_history[backend] = loss_history
            st.session_state.loss_weight_history[backend] = loss_weight_history
            st.session_state.params_flat[backend] = params_flat
            st.session_state.epoch_times[backend] = time_history
            latest_trace = (
                st.session_state.gradient_metrics[-1]
                if st.session_state.gradient_metrics
                else None
            )
            st.session_state.tesseract_trace[backend] = {
                "image": image_name,
                "trace_collected": latest_trace is not None,
                "apply_calls": latest_trace.apply_calls if latest_trace else None,
                "vjp_calls": latest_trace.vjp_calls if latest_trace else None,
                "visc_grad_norm": latest_trace.visc_grad_norm if latest_trace else None,
                "param_grad_norm": latest_trace.param_grad_norm
                if latest_trace
                else None,
            }

            run_summary = {
                "backend": backend,
                "tesseract_image": image_name,
                "host_optimizer": "JAX / Optax",
                "autodiff_boundary": "Tesseract VJP",
                "seed": int(seed),
                "true_viscosity": true_viscosity,
                "initial_viscosity": initial_viscosity,
                "final_viscosity": final_visc,
                "relative_error_percent": final_error,
                "epochs": n_epochs,
                "viscosity_warmup_epochs": effective_warmup_epochs,
                "log_nu_learning_rate": learning_rate,
                "param_learning_rate": param_learning_rate,
                "viscosity_clipped": clip_log_viscosity,
                "observations": n_obs,
                "collocation_points": int(n_col),
                "ic_points": int(n_ic),
                "bc_points": int(n_bc),
                "brdr_enabled": adaptive_loss_weights,
                "tesseract_trace_collected": latest_trace is not None,
                "apply_calls_per_traced_epoch": (
                    latest_trace.apply_calls if latest_trace else None
                ),
                "vjp_calls_per_traced_epoch": latest_trace.vjp_calls
                if latest_trace
                else None,
                "avg_epoch_ms": np.mean(time_history) * 1000,
            }
            history_df = pd.DataFrame(
                {
                    "epoch": np.arange(len(visc_history)),
                    "viscosity": visc_history,
                    "loss": [np.nan, *loss_history],
                }
            )
            component_df = pd.DataFrame(
                {
                    "epoch": component_loss_epochs,
                    **component_loss_history,
                }
            )
            weight_df = history_frame(loss_weight_history)

            tabs = st.tabs(
                [
                    "Run Summary",
                    "Inverse Solve",
                    "Solution Field",
                    "BRDR Weights",
                    "Tesseract Trace",
                    "Backend Equivalence",
                ]
            )

            with tabs[0]:
                st.subheader("Run Summary")
                render_tesseract_contract(backend, image_name, show_gradient_inspector)
                render_shared_objective(loss_weights, adaptive_loss_weights)
                st.dataframe(pd.DataFrame([run_summary]), hide_index=True)
                export_cols = st.columns(3)
                export_cols[0].download_button(
                    "Download History CSV",
                    history_df.to_csv(index=False),
                    file_name=f"{backend}_history_seed{int(seed)}.csv",
                    mime="text/csv",
                )
                export_cols[1].download_button(
                    "Download Components CSV",
                    component_df.to_csv(index=False),
                    file_name=f"{backend}_components_seed{int(seed)}.csv",
                    mime="text/csv",
                )
                export_cols[2].download_button(
                    "Download Config JSON",
                    json.dumps(run_summary, indent=2),
                    file_name=f"{backend}_config_seed{int(seed)}.json",
                    mime="application/json",
                )

            with tabs[1]:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                sns.lineplot(
                    data=history_df,
                    x="epoch",
                    y="viscosity",
                    color=PINN_COLOR,
                    linewidth=2.4,
                    ax=ax1,
                )
                ax1.axhline(
                    true_viscosity,
                    color=TRUE_COLOR,
                    linestyle="--",
                    linewidth=2,
                    label=f"True ν = {true_viscosity}",
                )
                if effective_warmup_epochs:
                    ax1.axvline(
                        effective_warmup_epochs,
                        color="#666666",
                        linestyle=":",
                        linewidth=1.5,
                        label="ν warmup end",
                    )
                ax1.set_title("Viscosity Convergence")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("ν")
                ax1.legend(frameon=False)
                finish_axes(ax1)

                component_long = component_df.melt(
                    id_vars="epoch",
                    value_vars=["data", "physics", "ic", "bc"],
                    var_name="component",
                    value_name="loss",
                )
                sns.lineplot(
                    data=component_long,
                    x="epoch",
                    y="loss",
                    hue="component",
                    linewidth=2,
                    ax=ax2,
                )
                ax2.set_yscale("log")
                ax2.set_title("PINN Loss Components")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Raw component loss")
                ax2.legend(frameon=False)
                finish_axes(ax2)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(component_df, hide_index=True)

            with tabs[2]:
                with st.spinner("Generating solution visualization..."):
                    X, T, u_pred, u_ground_truth = generate_solution_grid(
                        true_viscosity, params_flat, pinn
                    )

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                im0 = axes[0].contourf(X, T, u_pred, levels=32, cmap=FIELD_CMAP)
                axes[0].set_xlabel("x")
                axes[0].set_ylabel("t")
                axes[0].set_title(f"PINN Solution (ν={final_visc:.4f})")
                plt.colorbar(im0, ax=axes[0])

                im1 = axes[1].contourf(X, T, u_ground_truth, levels=32, cmap=FIELD_CMAP)
                axes[1].set_xlabel("x")
                axes[1].set_ylabel("t")
                axes[1].set_title(f"Solver Ground Truth (ν={true_viscosity:.4f})")
                plt.colorbar(im1, ax=axes[1])

                error_map = np.abs(u_pred - u_ground_truth)
                im2 = axes[2].contourf(X, T, error_map, levels=32, cmap=ERROR_CMAP)
                axes[2].set_xlabel("x")
                axes[2].set_ylabel("t")
                axes[2].set_title(f"Absolute Error (Max: {error_map.max():.4f})")
                plt.colorbar(im2, ax=axes[2])
                axes[0].scatter(
                    x_obs,
                    t_obs,
                    c="white",
                    edgecolors="#222222",
                    linewidths=0.25,
                    s=14,
                    alpha=0.75,
                    label="Observations",
                )
                axes[0].legend(frameon=False, loc="upper right")
                for ax in axes:
                    finish_axes(ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with tabs[3]:
                if adaptive_loss_weights:
                    st.caption(
                        "BRDR raises weights on residual points whose squared residuals decay slower than the global average."
                    )
                    fig, ax = plt.subplots(figsize=(8, 4))
                    weight_long = weight_df.melt(
                        id_vars="epoch",
                        value_vars=list(LOSS_WEIGHT_NAMES),
                        var_name="component",
                        value_name="mean_weight",
                    )
                    sns.lineplot(
                        data=weight_long,
                        x="epoch",
                        y="mean_weight",
                        hue="component",
                        linewidth=2.2,
                        ax=ax,
                    )
                    ax.set_yscale("log")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Mean BRDR weight")
                    ax.legend(frameon=False)
                    finish_axes(ax)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.dataframe(weight_df.tail(10), hide_index=True)
                else:
                    st.info(
                        "Enable BRDR Adaptive Loss Weights in the sidebar to inspect weight trajectories."
                    )

            with tabs[4]:
                if show_gradient_inspector and st.session_state.gradient_metrics:
                    render_tesseract_contract(
                        backend, image_name, show_gradient_inspector
                    )
                    render_gradient_flow_inspector(
                        backend, st.session_state.gradient_metrics
                    )
                else:
                    st.info(
                        "Enable Tesseract trace collection before running to collect VJP and gradient diagnostics."
                    )

            with tabs[5]:
                other_backend = "pytorch" if backend == "jax" else "jax"
                if other_backend in st.session_state.trained_viscosity:
                    jax_visc = st.session_state.trained_viscosity["jax"]
                    pytorch_visc = st.session_state.trained_viscosity["pytorch"]
                    visc_diff = abs(jax_visc - pytorch_visc)
                    st.subheader("Backend Equivalence Report")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("JAX Result", f"{jax_visc:.6f}")
                    col2.metric("PyTorch Result", f"{pytorch_visc:.6f}")
                    col3.metric("Absolute Difference", f"{visc_diff:.6f}")

                    equivalence_df = pd.DataFrame(
                        [
                            {
                                "check": "same host optimizer",
                                "jax": "JAX / Optax",
                                "pytorch": "JAX / Optax",
                            },
                            {
                                "check": "same loss function",
                                "jax": "compute_loss",
                                "pytorch": "compute_loss",
                            },
                            {
                                "check": "same sampled observations",
                                "jax": f"seed {int(seed)}",
                                "pytorch": f"seed {int(seed)}",
                            },
                            {
                                "check": "different Tesseract image",
                                "jax": pinn_image_name("jax"),
                                "pytorch": pinn_image_name("pytorch"),
                            },
                        ]
                    )
                    st.dataframe(equivalence_df, hide_index=True)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    for backend_name, color in (
                        ("jax", PINN_COLOR),
                        ("pytorch", PYTORCH_COLOR),
                    ):
                        sns.lineplot(
                            x=np.arange(
                                len(st.session_state.viscosity_history[backend_name])
                            ),
                            y=st.session_state.viscosity_history[backend_name],
                            label=backend_name.upper(),
                            linewidth=2.5,
                            color=color,
                            ax=ax1,
                        )
                        sns.lineplot(
                            x=np.arange(
                                len(st.session_state.loss_history[backend_name])
                            ),
                            y=st.session_state.loss_history[backend_name],
                            label=backend_name.upper(),
                            linewidth=2.5,
                            color=color,
                            ax=ax2,
                        )
                    ax1.axhline(
                        true_viscosity,
                        color=TRUE_COLOR,
                        linestyle="--",
                        linewidth=2,
                        label=f"True ν = {true_viscosity}",
                    )
                    ax1.set_title("Viscosity Convergence")
                    ax1.set_xlabel("Epoch")
                    ax1.set_ylabel("ν")
                    ax1.legend(frameon=False)
                    finish_axes(ax1)
                    ax2.set_yscale("log")
                    ax2.set_title("Training Loss")
                    ax2.set_xlabel("Epoch")
                    ax2.set_ylabel("Loss")
                    ax2.legend(frameon=False)
                    finish_axes(ax2)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(
                        f"Train the {other_backend.upper()} Tesseract container next to populate the backend equivalence report."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "current_container": image_name,
                                    "next_container": pinn_image_name(other_backend),
                                    "shared_host_loop": "JAX / Optax",
                                    "shared_objective": "data + physics + IC + BC",
                                }
                            ]
                        ),
                        hide_index=True,
                    )

        st.session_state.training = False

    else:
        # Initial state - show info
        st.info(
            """
        **Configure parameters in the sidebar and click "Run Tesseract Inversion" to begin.**

        **Tesseract showcase workflow:**
        1. Keep the inverse objective and Optax optimizers in JAX
        2. Route PINN apply/VJP calls through the selected Tesseract container
        3. Infer the viscosity parameter and monitor gradient flow across the boundary
        4. Visualize the learned field against the solver-generated ground truth
        5. Switch containers and retrain with the same settings to check backend equivalence
        """
        )

        # Show previous training results if available
        if st.session_state.trained_viscosity:
            st.markdown("---")
            st.subheader("Previous Tesseract Runs")

            trained_backends = list(st.session_state.trained_viscosity.keys())
            cols = st.columns(len(trained_backends))

            for idx, backend_name in enumerate(trained_backends):
                with cols[idx]:
                    st.metric(
                        f"{backend_name.upper()} Backend",
                        f"ν = {st.session_state.trained_viscosity[backend_name]:.6f}",
                    )
                    st.caption(
                        f"{len(st.session_state.viscosity_history[backend_name]) - 1} epochs trained"
                    )

            if len(trained_backends) == 1:
                st.info(
                    "Train the other Tesseract container to populate the backend equivalence report."
                )


if __name__ == "__main__":
    main()
