"""Tesseract Cross-Framework Autodiff Demo: Inverse Burgers Equation Solver.

Demonstrates Tesseract's pipeline-level automatic differentiation across JAX and PyTorch,
enabling JAX-based optimization of PyTorch PINN models via VJP (Vector-Jacobian Product).
"""

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import seaborn as sns
import streamlit as st
from tesseract_jax import apply_tesseract

from configs import DEFAULT_LOSS_WEIGHTS, LOSS_WEIGHT_NAMES
from inverse_problem import (
    Tesseract,
    compute_loss,
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
    st.markdown("---")
    with st.expander("🔍 **Gradient Flow ** (Tesseract Internals)", expanded=True):
        st.markdown(f"""
        ### Cross-Framework Autodiff Pipeline

        This shows how **Tesseract enables JAX gradients to flow through {backend.upper()}**:
        """)

        # Flow diagram
        if backend == "pytorch":
            st.markdown("""
            ```
            JAX Optimizer (optax)
                    ↓
            jax.grad(compute_loss)
                    ↓
            Tesseract VJP Endpoint  ← Cross-framework boundary!
                    ↓
            PyTorch Autograd (torch.autograd.grad)
                    ↓
            PyTorch PINN forward pass
                    ↓
            Gradients flow back through VJP
                    ↓
            JAX receives gradients  ← Back to JAX!
            ```
            """)
        else:
            st.markdown("""
            ```
            JAX Optimizer (optax)
                    ↓
            jax.grad(compute_loss)
                    ↓
            Tesseract Apply Endpoint
                    ↓
            JAX Autograd (jax.grad)
                    ↓
            JAX PINN forward pass
                    ↓
            Gradients computed natively
            ```
            """)

        if not gradient_metrics:
            st.info("Run training to see gradient flow metrics...")
            return

        tab1, tab2, tab3 = st.tabs(
            ["Call Statistics", "Gradient Norms", "Tensor Shapes"]
        )

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
                help="Backward pass (gradient) evaluations",
            )
            col3.metric("Total AD operations", latest.apply_calls + latest.vjp_calls)

            st.info(f"""
            **PINN Loss Architecture**: Each epoch computes a composite loss with {latest.apply_calls} network evaluations:

            1. **Data loss** (1 call) — MSE at observation points
            2. **Physics loss** (1 call) — PDE residual: ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x² = 0
            3. **Initial condition** (1 call) — Enforce u(x, t=0) = sin(2πx)
            4. **Boundary left** (1 call) — Periodic BC: u(0, t)
            5. **Boundary right** (1 call) — Periodic BC: u(1, t) = u(0, t)

            Then **{latest.vjp_calls} VJP calls** compute gradients: ∂L/∂log(ν) and ∂L/∂params

            {"VJP calls route through PyTorch autograd" if backend == "pytorch" else "**Native JAX**: All operations use jax.grad"}
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
            - High norm → steep loss landscape, large updates
            - Decreasing norm → approaching optimum
            - These are computed via Tesseract's VJP (Vector-Jacobian Product)
            """)

        with tab3:
            st.subheader("Tensor Shapes Through Pipeline")

            if latest.shapes:
                st.json(latest.shapes)
            else:
                st.info("No shape information available")

            st.markdown(f"""
            **Data flow through Tesseract**:
            - **Inputs**: x, t (collocation points), params_flat (network weights)
            - **Outputs**: u_pred, u_x, u_t, u_xx (solution + derivatives)
            - All computed via **{backend.upper()} autodiff**, exposed through Tesseract API
            """)


def train_step(
    backend,
    log_viscosity,
    params_flat,
    brdr_weights,
    visc_opt_state,
    param_opt_state,
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
        loss_weights=DEFAULT_LOSS_WEIGHTS,
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
        loss_weights=DEFAULT_LOSS_WEIGHTS,
    )

    # Compute gradient norms
    visc_grad_norm = float(jnp.linalg.norm(v_grad))
    param_grad_norm = float(jnp.linalg.norm(p_grad))

    # Update log-viscosity so the physical viscosity remains positive.
    visc_updates, visc_opt_state = visc_optimizer.update(v_grad, visc_opt_state)
    log_viscosity = optax.apply_updates(log_viscosity, visc_updates)
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
        loss_weights=DEFAULT_LOSS_WEIGHTS,
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

    st.title(
        "Tesseract Inverse 1D Burgers Equation Solver: Cross-Framework Autodiff Demo"
    )

    st.subheader("Backend-agnostic viscosity estimation with PINNs")

    st.markdown("""
    **Goal**: Use Tesseract to run the same inverse 1D Burgers solver with either a JAX or PyTorch PINN backend,
    while keeping one JAX-based optimization pipeline.

    A JAX optimizer computes gradients through the selected backend via Tesseract's VJP interface, showcasing
    pipeline-level automatic differentiation across frameworks.
    """)

    st.markdown(
        "**Problem**: Given noisy observations of the 1D Burgers equation solution, infer the unknown viscosity parameter $\\nu$:"
    )
    st.latex(
        r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}"
    )

    st.sidebar.header("⚙️ Configuration")

    backend = st.sidebar.selectbox(
        "PINN Backend",
        ["jax", "pytorch"],
        help="Select backend implementation. Both expose identical Tesseract endpoints (apply/VJP/JVP), enabling seamless backend switching.",
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
        value=0.1,
        step=0.001,
        format="%.3f",
        help="Optimizer learning rate for log(ν)",
    )

    adaptive_loss_weights = st.sidebar.checkbox(
        "BRDR Adaptive Loss Weights",
        value=False,
        help="Use Balanced Residual Decay Rate pointwise weights for data, physics, initial-condition, and boundary residuals",
    )

    brdr_beta_c = 0.9999
    brdr_beta_w = 0.999
    if adaptive_loss_weights:
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

    # Gradient Flow Inspection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Additional Options")
    show_gradient_inspector = st.sidebar.checkbox(
        "Enable Gradient Flow Inspector",
        value=False,
        help="Track Tesseract API calls (apply/VJP), gradient norms, and tensor shapes through the autodiff pipeline",
    )

    if st.sidebar.button("Train Model", type="primary"):
        st.session_state.training = True
        st.session_state.gradient_metrics = []

    if st.session_state.training:
        # Setup
        domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        key = jax.random.PRNGKey(123)

        key_obs, key_col_x, key_col_t, key_ic, key_bc = jax.random.split(key, 5)
        x_obs, t_obs, u_obs = generate_observations(
            n_obs,
            true_viscosity,
            domain,
            key_obs,
            noise_std=noise_level,
        )

        # Make collocation points
        n_col = 200
        x_col = jax.random.uniform(
            key_col_x, (n_col,), minval=domain["x"][0], maxval=domain["x"][1]
        )
        t_col = jax.random.uniform(
            key_col_t, (n_col,), minval=0.05, maxval=domain["t"][1]
        )

        n_ic = 50
        x_ic = jax.random.uniform(
            key_ic, (n_ic,), minval=domain["x"][0], maxval=domain["x"][1]
        )

        n_bc = 50
        t_bc = jax.random.uniform(key_bc, (n_bc,), minval=0.05, maxval=domain["t"][1])

        # Initialize tesseract
        image_name = "pinn_jax" if backend == "jax" else "pinn_pytorch"
        pinn = Tesseract.from_image(image_name)
        params_flat = get_initial_params(backend, seed=123)

        # Initialize log-viscosity
        log_viscosity = jnp.log(jnp.asarray(initial_viscosity))
        brdr_state = None
        brdr_weights = None

        visc_optimizer = optax.adam(learning_rate)
        visc_opt_state = visc_optimizer.init(log_viscosity)
        param_optimizer = optax.adam(1e-3)
        param_opt_state = param_optimizer.init(params_flat)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Backend", backend.upper())
        with col2:
            st.metric("True Viscosity", f"{true_viscosity:.4f}")
        with col3:
            st.metric("Initial Guess", f"{initial_viscosity:.4f}")

        if adaptive_loss_weights:
            st.caption(
                "BRDR updates pointwise residual weights from inverse residual decay rates and tracks component mean weights."
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
        loss_weight_history = {
            name: [DEFAULT_LOSS_WEIGHTS[name]] for name in LOSS_WEIGHT_NAMES
        }
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
                    DEFAULT_LOSS_WEIGHTS
                    if brdr_weights is None
                    else summarize_brdr_weights(brdr_weights)
                )
                for name in LOSS_WEIGHT_NAMES:
                    loss_weight_history[name].append(current_weights[name])

                # Update every 5 epochs
                if epoch % 5 == 0 or epoch == n_epochs - 1:
                    error = abs(visc_val - true_viscosity)
                    rel_error = error / true_viscosity * 100

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
                    metric_loss.metric("Loss", f"{loss:.6f}")
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
            st.success("Finished Training! ✅")

            final_visc = float(jnp.exp(log_viscosity))
            final_error = abs(final_visc - true_viscosity) / true_viscosity * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Viscosity", f"{final_visc:.6f}")
            col2.metric("True Viscosity", f"{true_viscosity:.6f}")
            col3.metric("Relative Error", f"{final_error:.2f}%")
            col4.metric("Avg Time/Epoch", f"{np.mean(time_history) * 1000:.1f}ms")

            if adaptive_loss_weights:
                st.markdown("**Learned Loss Weights**")
                weight_cols = st.columns(len(LOSS_WEIGHT_NAMES))
                for col, name in zip(weight_cols, LOSS_WEIGHT_NAMES, strict=True):
                    col.metric(name, f"{loss_weight_history[name][-1]:.3g}")

            st.markdown("---")
            st.subheader("Visualizing PINN vs Solver Ground Truth")

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

            st.session_state.trained_viscosity[backend] = final_visc
            st.session_state.viscosity_history[backend] = visc_history
            st.session_state.loss_history[backend] = loss_history
            st.session_state.loss_weight_history[backend] = loss_weight_history
            st.session_state.params_flat[backend] = params_flat
            st.session_state.epoch_times[backend] = time_history

            if show_gradient_inspector and st.session_state.gradient_metrics:
                render_gradient_flow_inspector(
                    backend, st.session_state.gradient_metrics
                )

            st.markdown("---")
            st.success(f"✅ **{backend.upper()} Training Complete!**")

            # Check if we can show cross-backend comparison
            other_backend = "pytorch" if backend == "jax" else "jax"
            if other_backend in st.session_state.trained_viscosity:
                st.markdown("---")
                st.subheader("🔄 Cross-Framework Comparison")
                st.markdown("""
                Both backends have now been trained. Compare how JAX and PyTorch implementations
                converge to the same viscosity estimate, demonstrating Tesseract's backend-agnostic autodiff.
                """)

                # Metrics comparison
                col1, col2, col3 = st.columns(3)

                jax_visc = st.session_state.trained_viscosity["jax"]
                pytorch_visc = st.session_state.trained_viscosity["pytorch"]
                visc_diff = abs(jax_visc - pytorch_visc)

                with col1:
                    st.metric("JAX Result", f"{jax_visc:.6f}")
                    st.caption(
                        f"Avg: {np.mean(st.session_state.epoch_times['jax']) * 1000:.1f}ms/epoch"
                    )

                with col2:
                    st.metric("PyTorch Result", f"{pytorch_visc:.6f}")
                    st.caption(
                        f"Avg: {np.mean(st.session_state.epoch_times['pytorch']) * 1000:.1f}ms/epoch"
                    )

                with col3:
                    st.metric("Absolute Difference", f"{visc_diff:.6f}")
                    st.caption("Convergence agreement")

                # Convergence comparison plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                # Viscosity convergence
                sns.lineplot(
                    x=np.arange(len(st.session_state.viscosity_history["jax"])),
                    y=st.session_state.viscosity_history["jax"],
                    label="JAX Backend",
                    linewidth=2.5,
                    color=PINN_COLOR,
                    alpha=0.8,
                    ax=ax1,
                )
                sns.lineplot(
                    x=np.arange(len(st.session_state.viscosity_history["pytorch"])),
                    y=st.session_state.viscosity_history["pytorch"],
                    label="PyTorch Backend",
                    linewidth=2.5,
                    color=PYTORCH_COLOR,
                    alpha=0.8,
                    ax=ax1,
                )
                ax1.axhline(
                    true_viscosity,
                    color=TRUE_COLOR,
                    linestyle="--",
                    linewidth=2,
                    label=f"Ground Truth: ν = {true_viscosity}",
                    alpha=0.6,
                )
                ax1.set_xlabel("Epoch", fontsize=11)
                ax1.set_ylabel("Inferred Viscosity ν", fontsize=11)
                ax1.set_title(
                    "Viscosity Convergence: Backend Comparison",
                    fontsize=12,
                    fontweight="bold",
                )
                ax1.legend(frameon=False)
                finish_axes(ax1)

                # Loss comparison
                sns.lineplot(
                    x=np.arange(len(st.session_state.loss_history["jax"])),
                    y=st.session_state.loss_history["jax"],
                    label="JAX Backend",
                    linewidth=2.5,
                    color=PINN_COLOR,
                    alpha=0.8,
                    ax=ax2,
                )
                sns.lineplot(
                    x=np.arange(len(st.session_state.loss_history["pytorch"])),
                    y=st.session_state.loss_history["pytorch"],
                    label="PyTorch Backend",
                    linewidth=2.5,
                    color=PYTORCH_COLOR,
                    alpha=0.8,
                    ax=ax2,
                )
                ax2.set_yscale("log")
                ax2.set_xlabel("Epoch", fontsize=11)
                ax2.set_ylabel("Loss (log scale)", fontsize=11)
                ax2.set_title(
                    "Training Loss: Backend Comparison", fontsize=12, fontweight="bold"
                )
                ax2.legend(frameon=False)
                finish_axes(ax2)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.info(f"""
                Both backends produced consistent viscosity estimates (absolute difference: {visc_diff:.6f}).
                This validates that a single JAX-based optimization pipeline can compute gradients
                through both JAX and PyTorch PINN implementations via Tesseract's VJP interface.
                """)
            else:
                st.info(f"""
                **Next Step:** Train the **{other_backend.upper()}** backend to see cross-framework comparison.
                Switch the backend in the sidebar and click "Train Model" again.
                """)

        st.session_state.training = False

    else:
        # Initial state - show info
        st.info(
            """
        **Configure parameters in the sidebar and click "Train Model" to begin.**

        **Demo Workflow:**
        1. Generate synthetic observations from the 1D Burgers equation with known viscosity
        2. Train a Physics-Informed Neural Network (PINN) to infer the unknown viscosity parameter
        3. Monitor real-time convergence of viscosity estimate and loss metrics
        4. Visualize learned solution field and compare with solver ground truth
        5. (Optional) Inspect gradient flow through Tesseract's VJP interface
        6. **Switch backends and retrain with the same hyperparameters to see backend comparison**
        """
        )

        # Show previous training results if available
        if st.session_state.trained_viscosity:
            st.markdown("---")
            st.subheader("📋 Previous Training Results")

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
                    "**Train the other backend** to see cross-framework comparison and validate Tesseract's autodiff portability!"
                )


if __name__ == "__main__":
    main()
