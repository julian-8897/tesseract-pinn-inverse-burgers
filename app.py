"""Streamlit demo: inferring the Burgers viscosity ν from sparse observations.

Solves the inverse problem three ways, each through a separate Tesseract component:

- Solver-adjoint inversion: ``jax.grad`` through the Burgers solver VJP (PDE adjoint).
- PINN inversion: ``jax.grad`` through the PINN VJP, backend ``pinn_jax`` or
  ``pinn_pytorch`` behind one contract.
- FMPE posterior: the apply-only ``fmpe_posterior`` component maps one observation to a
  posterior over (ν, ic_amp, ic_phase).
"""

import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from tesseract_core import Tesseract

from burgers_inverse import (
    TrainingCallback,
    docker_image_available,
    evaluate_pinn_solution_grid,
    get_burgers_solver,
    image_name_for_backend,
    train_inverse,
    train_solver_inverse,
)
from burgers_inverse.configs import (
    DEFAULT_LOSS_WEIGHTS,
    LOSS_WEIGHT_NAMES,
    DataConfig,
    LossWeights,
    ProblemConfig,
    RunConfig,
    TrainingConfig,
)

st.set_page_config(
    page_title="Inverse Burgers with Tesseract",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.stMetric {
    background-color: #f0f2f6;
    padding: 12px 14px;
    border-radius: 8px;
}
div[data-testid="stMetricValue"] { font-size: 1.05rem; }
h1 { letter-spacing: -0.5px; }
section[data-testid="stSidebar"] { min-width: 330px; }
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


def history_frame(history, epoch_key="epoch"):
    """Convert a dict of equal-length histories into a dataframe."""
    return pd.DataFrame(
        {epoch_key: np.arange(len(next(iter(history.values())))), **history}
    )


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
        **Complete epoch telemetry**: this traced epoch made {latest.apply_calls} `apply()` calls across the optimization pass, optional BRDR preparation, and periodic metric evaluation.

        One composite PINN loss evaluates data, physics, initial-condition, and both periodic-boundary point sets. Metric epochs evaluate those components once more; BRDR epochs also evaluate pointwise losses before the gradient pass.

        The epoch made **{latest.vjp_calls} VJP calls** while computing gradients with respect to `log_nu` and `params_flat`.

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


COMPONENT_LOSS_NAMES = ("total", "data", "physics", "ic", "bc")


class StreamlitTrainingCallback(TrainingCallback):
    """Drives the live Streamlit UI and accumulates histories during training.

    Telemetry (apply/VJP call counts, gradient norms, tensor shapes) is read
    directly from the engine's per-epoch records, so the diagnostics reflect
    measured Tesseract behavior rather than hardcoded estimates.
    """

    def __init__(
        self,
        *,
        true_viscosity,
        initial_viscosity,
        adaptive_loss_weights,
        show_gradient_inspector,
        placeholders,
    ):
        self.true_viscosity = true_viscosity
        self.adaptive_loss_weights = adaptive_loss_weights
        self.show_gradient_inspector = show_gradient_inspector
        self.ph = placeholders

        self.warmup_epochs = 0
        self.x_obs = None
        self.t_obs = None
        self.u_obs = None

        self.visc_history = [float(initial_viscosity)]
        self.loss_history = []
        self.time_history = []
        self.loss_weight_history = {name: [] for name in LOSS_WEIGHT_NAMES}
        self.component_loss_history = {name: [] for name in COMPONENT_LOSS_NAMES}
        self.component_loss_epochs = []
        self.gradient_metrics = []

    def on_start(self, context):
        self.warmup_epochs = context["warmup_epochs"]
        self.x_obs = np.asarray(context["x_obs"])
        self.t_obs = np.asarray(context["t_obs"])
        self.u_obs = np.asarray(context["u_obs"])

    def on_epoch(self, record):
        self.visc_history.append(record.viscosity)
        self.loss_history.append(record.loss)
        self.time_history.append(record.epoch_time)
        for name in LOSS_WEIGHT_NAMES:
            self.loss_weight_history[name].append(record.effective_weights[name])

        track_this_epoch = self.show_gradient_inspector and (
            record.epoch % 5 == 0 or record.epoch < 10
        )
        if track_this_epoch:
            self.gradient_metrics.append(
                GradientFlowMetrics(
                    epoch=record.epoch,
                    vjp_calls=record.vjp_calls,
                    apply_calls=record.apply_calls,
                    visc_grad_norm=record.visc_grad_norm,
                    param_grad_norm=record.param_grad_norm,
                    loss_value=record.loss,
                    shapes={
                        "x_obs": tuple(self.x_obs.shape),
                        "t_obs": tuple(self.t_obs.shape),
                        "params_flat": (record.param_count,),
                    },
                )
            )

        if record.loss_components is not None:
            self.component_loss_epochs.append(record.epoch + 1)
            for name in COMPONENT_LOSS_NAMES:
                self.component_loss_history[name].append(record.loss_components[name])

        if record.epoch % 5 == 0 or record.epoch == record.n_epochs - 1:
            self._render_live(record)

    def _render_live(self, record):
        progress = (record.epoch + 1) / record.n_epochs
        self.ph["progress_bar"].progress(progress)
        self.ph["status_text"].text(f"Epoch {record.epoch + 1}/{record.n_epochs}")

        rel_error = abs(record.viscosity - self.true_viscosity) / self.true_viscosity
        displayed_loss = (
            record.loss_components["total"]
            if record.loss_components is not None
            else record.loss
        )
        self.ph["metric_visc"].metric(
            "Current ν",
            f"{record.viscosity:.6f}",
            delta=f"{record.viscosity - self.true_viscosity:.6f}",
        )
        self.ph["metric_error"].metric("Relative Error", f"{rel_error * 100:.2f}%")
        self.ph["metric_loss"].metric("Loss", f"{displayed_loss:.6f}")
        self.ph["metric_time"].metric("Epoch Time", f"{record.epoch_time * 1000:.1f}ms")

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        epochs = np.arange(len(self.visc_history))
        sns.lineplot(
            x=epochs,
            y=self.visc_history,
            label="Inferred ν",
            color=PINN_COLOR,
            linewidth=2.3,
            ax=ax1,
        )
        ax1.axhline(
            self.true_viscosity,
            color=TRUE_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"True ν = {self.true_viscosity}",
        )
        if self.warmup_epochs:
            ax1.axvline(
                self.warmup_epochs,
                color="#666666",
                linestyle=":",
                linewidth=1.5,
                label="ν warmup end",
            )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Viscosity")
        ax1.legend(frameon=False)
        finish_axes(ax1)
        self.ph["visc_chart"].pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            x=np.arange(len(self.loss_history)),
            y=self.loss_history,
            color=LOSS_COLOR,
            linewidth=2.3,
            ax=ax2,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (log scale)")
        finish_axes(ax2)
        self.ph["loss_chart"].pyplot(fig2)
        plt.close(fig2)

        weight_chart = self.ph.get("weight_chart")
        if weight_chart is not None:
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            weight_epochs = np.arange(
                len(next(iter(self.loss_weight_history.values())))
            )
            for name in LOSS_WEIGHT_NAMES:
                sns.lineplot(
                    x=weight_epochs,
                    y=self.loss_weight_history[name],
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


def render_pinn_demo():
    """PINN inversion with swappable JAX / PyTorch Tesseract containers."""
    st.header("PINN inversion (JAX or PyTorch)")
    st.caption(
        "Infer ν through the PINN Tesseract's VJP, with a JAX or PyTorch backend."
    )

    st.markdown(
        """
A PINN fits the velocity field while the outer loop infers ν. Both are optimized in
JAX/Optax; each step takes one `value_and_grad` over (log ν, network params) and routes
it through the PINN Tesseract's VJP. The PINN backend is `pinn_jax` or `pinn_pytorch`,
and the host code is identical either way. Train both and the app reports how far apart
their recovered ν values are.
        """
    )

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
            "The solver generates the noisy observations offline; during training the "
            "PINN container is the component gradients pass through."
        )

    st.sidebar.header("Configuration")

    backend = st.sidebar.selectbox(
        "Tesseract PINN Container",
        ["jax", "pytorch"],
        help="Select backend implementation. Both expose the same apply/VJP inverse-training contract, enabling seamless backend switching.",
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
        # Build a validated run configuration from the sidebar controls. The
        # same RunConfig drives the CLI, so the app and CLI share one engine.
        try:
            config = RunConfig(
                backend=backend,
                problem=ProblemConfig(
                    true_viscosity=float(true_viscosity),
                    initial_viscosity=float(initial_viscosity),
                ),
                data=DataConfig(
                    n_obs=int(n_obs),
                    noise_std=float(noise_level),
                    seed=int(seed),
                ),
                training=TrainingConfig(
                    n_epochs=int(n_epochs),
                    log_nu_learning_rate=float(learning_rate),
                    param_learning_rate=float(param_learning_rate),
                    adaptive_loss_weights=adaptive_loss_weights,
                    brdr_beta_c=float(brdr_beta_c),
                    brdr_beta_w=float(brdr_beta_w),
                    n_col=int(n_col),
                    n_ic=int(n_ic),
                    n_bc=int(n_bc),
                    viscosity_warmup_epochs=int(viscosity_warmup_epochs),
                    clip_log_viscosity=bool(clip_log_viscosity),
                ),
                loss=LossWeights(**loss_weights),
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid configuration: {exc}")
            st.session_state.training = False
            return

        image_name = image_name_for_backend(backend)
        if not docker_image_available(image_name):
            st.error(f"Tesseract image `{image_name}` was not found.")
            st.code("./buildall.sh", language="bash")
            st.session_state.training = False
            return

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
        with plot_columns[0]:
            st.subheader("Viscosity Convergence")
            visc_chart = st.empty()
        with plot_columns[1]:
            st.subheader("Training Loss")
            loss_chart = st.empty()
        if adaptive_loss_weights:
            with plot_columns[2]:
                st.subheader("Loss Weights")
                weight_chart = st.empty()
        else:
            weight_chart = None

        placeholders = {
            "progress_bar": progress_bar,
            "status_text": status_text,
            "metric_visc": metric_visc,
            "metric_error": metric_error,
            "metric_loss": metric_loss,
            "metric_time": metric_time,
            "visc_chart": visc_chart,
            "loss_chart": loss_chart,
            "weight_chart": weight_chart,
        }

        callback = StreamlitTrainingCallback(
            true_viscosity=true_viscosity,
            initial_viscosity=initial_viscosity,
            adaptive_loss_weights=adaptive_loss_weights,
            show_gradient_inspector=show_gradient_inspector,
            placeholders=placeholders,
        )

        pinn = Tesseract.from_image(image_name)
        with pinn:
            result = train_inverse(
                config, pinn=pinn, callback=callback, metrics_every=5
            )

            st.markdown("---")
            st.success("Finished training.")

            final_visc = result["final_viscosity"]
            final_error = result["relative_error"]
            params_flat = result["params_flat"]
            x_obs, t_obs, u_obs = result["observations"]
            effective_warmup_epochs = result["warmup_epochs"]

            visc_history = callback.visc_history
            loss_history = callback.loss_history
            time_history = callback.time_history
            loss_weight_history = callback.loss_weight_history
            component_loss_history = callback.component_loss_history
            component_loss_epochs = callback.component_loss_epochs
            st.session_state.gradient_metrics = callback.gradient_metrics

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
                    "Backend Consistency",
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
                    X, T, u_pred, u_ground_truth = evaluate_pinn_solution_grid(
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
                    rel_spread = (
                        visc_diff / true_viscosity * 100 if true_viscosity else 0.0
                    )
                    st.subheader("Backend Consistency Report")
                    st.caption(
                        "The two backends start from independent random initializations "
                        "(JAX PRNG vs. torch.manual_seed), so they are not bit-for-bit "
                        "equivalent. This is a consistency check: under the same objective, "
                        "data, and seed, both Tesseract backends recover ν to within a small "
                        "spread — evidence the autodiff contract is backend-agnostic."
                    )
                    col1, col2, col3 = st.columns(3)
                    col1.metric("JAX Result", f"{jax_visc:.6f}")
                    col2.metric("PyTorch Result", f"{pytorch_visc:.6f}")
                    col3.metric(
                        "|Δν| (% of true ν)",
                        f"{visc_diff:.6f}",
                        delta=f"{rel_spread:.2f}%",
                        delta_color="off",
                    )

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
                                "jax": image_name_for_backend("jax"),
                                "pytorch": image_name_for_backend("pytorch"),
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
                        f"Train the {other_backend.upper()} Tesseract container next to populate the backend consistency report."
                    )
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "current_container": image_name,
                                    "next_container": image_name_for_backend(
                                        other_backend
                                    ),
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
        **Set the parameters in the sidebar and press "Run Tesseract Inversion" to start.**

        The outer loop stays in JAX/Optax and routes the PINN's apply/VJP calls through
        the container you picked. It infers ν, plots the learned field against the solver
        ground truth, and reports the measured apply/VJP counts. Run the JAX and PyTorch
        builds with the same settings to compare their recovered ν.
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
                    "Train the other Tesseract container to populate the backend consistency report."
                )


class SolverStreamlitCallback(TrainingCallback):
    """Live Streamlit progress for solver-adjoint inversion (no neural net)."""

    def __init__(self, *, true_viscosity, initial_viscosity, placeholders):
        self.true_viscosity = true_viscosity
        self.ph = placeholders
        self.visc_history = [float(initial_viscosity)]
        self.loss_history = []
        self.time_history = []
        self.observations = None

    def on_start(self, context):
        self.observations = context["observations"]

    def on_epoch(self, record):
        self.visc_history.append(record.viscosity)
        self.loss_history.append(record.loss)
        self.time_history.append(record.epoch_time)
        if record.epoch % 5 == 0 or record.epoch == record.n_epochs - 1:
            self._render_live(record)

    def _render_live(self, record):
        progress = (record.epoch + 1) / record.n_epochs
        self.ph["progress_bar"].progress(progress)
        self.ph["status_text"].text(f"Epoch {record.epoch + 1}/{record.n_epochs}")
        rel_error = abs(record.viscosity - self.true_viscosity) / self.true_viscosity
        self.ph["metric_visc"].metric(
            "Current ν",
            f"{record.viscosity:.6f}",
            delta=f"{record.viscosity - self.true_viscosity:.6f}",
        )
        self.ph["metric_error"].metric("Relative Error", f"{rel_error * 100:.2f}%")
        self.ph["metric_loss"].metric("Data Loss", f"{record.loss:.3e}")
        self.ph["metric_time"].metric("Epoch Time", f"{record.epoch_time * 1000:.1f}ms")

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            x=np.arange(len(self.visc_history)),
            y=self.visc_history,
            label="Inferred ν",
            color=PINN_COLOR,
            linewidth=2.3,
            ax=ax1,
        )
        ax1.axhline(
            self.true_viscosity,
            color=TRUE_COLOR,
            linestyle="--",
            linewidth=2,
            label=f"True ν = {self.true_viscosity}",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Viscosity")
        ax1.legend(frameon=False)
        finish_axes(ax1)
        self.ph["visc_chart"].pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.lineplot(
            x=np.arange(len(self.loss_history)),
            y=self.loss_history,
            color=LOSS_COLOR,
            linewidth=2.3,
            ax=ax2,
        )
        ax2.set_yscale("log")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Data loss (log scale)")
        finish_axes(ax2)
        self.ph["loss_chart"].pyplot(fig2)
        plt.close(fig2)


FMPE_BUNDLE_PATH = "tesseracts/fmpe_posterior/posterior.pkl"


@st.cache_resource(show_spinner=False)
def load_fmpe_bundle(path):
    """Load and validate the trained FMPE posterior bundle (cached)."""
    from pathlib import Path

    if not Path(path).is_file():
        return None
    from burgers_inverse.fmpe_posterior import load_model

    return load_model(path)


@st.cache_resource(show_spinner=False)
def get_fmpe_component():
    """Load the local fmpe_posterior Tesseract API (apply-only component)."""
    from burgers_inverse.component_loader import load_tesseract_api

    return load_tesseract_api("fmpe_posterior")


def render_app_header():
    """Shared framing and component overview shown above every method."""
    st.title("Inverse Burgers with Tesseract")
    st.subheader("Infer the Burgers viscosity from sparse observations, three ways")
    st.markdown(
        """
This app infers the viscosity ν of the 1D viscous Burgers equation from sparse, noisy
observations of the velocity field, and solves that inverse problem three ways.

Each method is a separate Tesseract component: a differentiable Burgers solver, a
physics-informed neural network with JAX and PyTorch builds, and a trained
flow-matching posterior. They share one typed `apply`/`vector_jacobian_product`
interface, so the calling code is the same whether the component is JAX or PyTorch and
whether it returns a point estimate or a posterior.
        """
    )

    with st.expander("The PDE"):
        st.latex(
            r"\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x}"
            r" = \nu\,\frac{\partial^2 u}{\partial x^2}"
        )
        st.caption(
            "u(x, t) is the velocity field on [0, 1], periodic, with initial condition "
            "u(x, 0) = sin(2πx). ν is the viscosity being inferred."
        )

    st.markdown("##### Components")
    card1, card2, card3 = st.columns(3)
    with card1.container(border=True):
        st.markdown("**🌊 Solver** &nbsp; `burgers_solver`", unsafe_allow_html=True)
        st.write(
            "Pseudospectral Burgers solver in JAX (FFT derivatives, Diffrax time "
            "stepping). Differentiable, so the inverse loop backprops through it."
        )
        st.caption("Differentiable · JAX")
    with card2.container(border=True):
        st.markdown(
            "**🧠 PINN** &nbsp; `pinn_jax` · `pinn_pytorch`", unsafe_allow_html=True
        )
        st.write(
            "MLP with Fourier features, trained on the PDE residual. Same model and "
            "the same VJP contract in a JAX build and a PyTorch build."
        )
        st.caption("Differentiable · JAX or PyTorch")
    with card3.container(border=True):
        st.markdown("**📊 Posterior** &nbsp; `fmpe_posterior`", unsafe_allow_html=True)
        st.write(
            "Flow-matching posterior trained offline on simulated observations. "
            "Apply-only: maps one observation to samples of (ν, ic_amp, ic_phase)."
        )
        st.caption("Apply-only · PyTorch")

    st.caption(
        "Pick a method in the sidebar. Each routes the same inverse problem through a "
        "different component."
    )


def render_solver_demo():
    """Solver-adjoint inversion: differentiate through the solver Tesseract VJP."""
    st.header("Solver-adjoint inversion")
    st.caption("Backprop the data-fit loss through the solver's VJP to recover ν.")
    st.markdown(
        """
Optimize log ν to minimize ‖solver(ν) − u_obs‖² at the observation points. `jax.grad`
backpropagates this loss through the solver Tesseract's VJP, which is the PDE adjoint.
No neural network is involved. The observations come from the same solver, so the
inverse problem is well-posed and ν is recovered to within the observation noise. This
is the PDE-constrained baseline for the PINN method.
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Host Loop", "JAX + Optax")
    col2.metric("Autodiff Boundary", "Solver Tesseract VJP")
    col3.metric("Tesseract", "burgers_solver")
    st.code(
        """noisy sparse observations u_obs
        |
        v
JAX objective: ||solver(nu)[sensors] - u_obs||^2
        |
        v
jax.grad(loss) w.r.t. log_nu
        |
        v
Tesseract solver container: burgers_solver VJP (PDE adjoint)
        |
        v
update for log_nu""",
        language="text",
    )

    st.sidebar.header("Configuration")
    seed = st.sidebar.number_input(
        "Seed", min_value=0, max_value=1_000_000, value=123, step=1, key="solver_seed"
    )
    true_viscosity = st.sidebar.slider(
        "True Viscosity $\\nu$ (Ground Truth)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        key="solver_true_nu",
    )
    initial_viscosity = st.sidebar.slider(
        "Initial Viscosity Guess $\\nu_0$",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        key="solver_init_nu",
    )
    n_obs = st.sidebar.slider(
        "Number of Observations",
        min_value=20,
        max_value=400,
        value=200,
        step=20,
        key="solver_n_obs",
    )
    noise_level = st.sidebar.slider(
        "Observation Noise (σ)",
        min_value=0.0,
        max_value=0.1,
        value=0.02,
        step=0.01,
        key="solver_noise",
    )
    n_epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=300,
        value=80,
        step=10,
        key="solver_epochs",
    )
    learning_rate = st.sidebar.slider(
        "Log-Viscosity Learning Rate",
        min_value=0.001,
        max_value=0.3,
        value=0.05,
        step=0.001,
        format="%.3f",
        key="solver_lr",
    )
    clip_log_viscosity = st.sidebar.checkbox(
        "Constrain Viscosity Range", value=True, key="solver_clip"
    )

    if not st.sidebar.button(
        "Run Solver-Adjoint Inversion", type="primary", key="solver_run"
    ):
        st.info(
            "Set the parameters in the sidebar and press **Run Solver-Adjoint "
            "Inversion**. Requires the `burgers_solver` image."
        )
        return

    try:
        config = RunConfig(
            backend="jax",
            problem=ProblemConfig(
                true_viscosity=float(true_viscosity),
                initial_viscosity=float(initial_viscosity),
            ),
            data=DataConfig(
                n_obs=int(n_obs), noise_std=float(noise_level), seed=int(seed)
            ),
            training=TrainingConfig(
                n_epochs=int(n_epochs),
                log_nu_learning_rate=float(learning_rate),
                viscosity_warmup_epochs=0,
                clip_log_viscosity=bool(clip_log_viscosity),
            ),
        )
    except (TypeError, ValueError) as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    if not docker_image_available("burgers_solver"):
        st.error("Tesseract image `burgers_solver` was not found.")
        st.code("./buildall.sh", language="bash")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Container", "burgers_solver")
    col2.metric("Method", "Solver adjoint")
    col3.metric("True Viscosity", f"{true_viscosity:.4f}")
    col4.metric("Initial Guess", f"{initial_viscosity:.4f}")
    st.markdown("---")

    progress_bar = st.progress(0)
    status_text = st.empty()
    m1, m2, m3, m4 = st.columns(4)
    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.subheader("Viscosity Convergence")
        visc_chart = st.empty()
    with plot_cols[1]:
        st.subheader("Data Loss")
        loss_chart = st.empty()

    callback = SolverStreamlitCallback(
        true_viscosity=true_viscosity,
        initial_viscosity=initial_viscosity,
        placeholders={
            "progress_bar": progress_bar,
            "status_text": status_text,
            "metric_visc": m1.empty(),
            "metric_error": m2.empty(),
            "metric_loss": m3.empty(),
            "metric_time": m4.empty(),
            "visc_chart": visc_chart,
            "loss_chart": loss_chart,
        },
    )

    solver = Tesseract.from_image("burgers_solver")
    with solver:
        result = train_solver_inverse(
            config, solver=solver, callback=callback, metrics_every=5
        )

    st.markdown("---")
    st.success("Finished solver-adjoint inversion.")
    final_visc = result["final_viscosity"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Viscosity", f"{final_visc:.6f}")
    c2.metric("True Viscosity", f"{true_viscosity:.6f}")
    c3.metric("Relative Error", f"{result['relative_error']:.2f}%")
    c4.metric("Avg Time/Epoch", f"{np.mean(callback.time_history) * 1000:.1f}ms")

    st.subheader("Solver Field at Recovered ν vs. Ground Truth")
    with st.spinner("Rendering solver fields..."):
        obs = result["observations"]
        x_grid = np.asarray(obs.x_grid)
        t_grid = np.asarray(obs.t_grid)
        solve_burgers = get_burgers_solver()
        one = np.float32(1.0)
        zero = np.float32(0.0)
        u_pred = np.asarray(
            solve_burgers(np.float32(final_visc), x_grid, t_grid, one, zero)
        )
        u_true = np.asarray(
            solve_burgers(np.float32(true_viscosity), x_grid, t_grid, one, zero)
        )
    X, T = np.meshgrid(x_grid, t_grid)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].contourf(X, T, u_pred, levels=32, cmap=FIELD_CMAP)
    axes[0].set_title(f"Solver Field (recovered ν={final_visc:.4f})")
    plt.colorbar(im0, ax=axes[0])
    axes[0].scatter(
        np.asarray(obs.x_obs),
        np.asarray(obs.t_obs),
        c="white",
        edgecolors="#222222",
        linewidths=0.25,
        s=14,
        alpha=0.75,
        label="Observations",
    )
    axes[0].legend(frameon=False, loc="upper right")
    im1 = axes[1].contourf(X, T, u_true, levels=32, cmap=FIELD_CMAP)
    axes[1].set_title(f"Ground Truth (ν={true_viscosity:.4f})")
    plt.colorbar(im1, ax=axes[1])
    err = np.abs(u_pred - u_true)
    im2 = axes[2].contourf(X, T, err, levels=32, cmap=ERROR_CMAP)
    axes[2].set_title(f"Absolute Error (Max: {err.max():.4f})")
    plt.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        finish_axes(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_fmpe_demo():
    """Amortized FMPE posterior: the apply-only posterior Tesseract (UQ)."""
    st.header("FMPE posterior")
    st.caption("Map one observation to a posterior over the Burgers parameters.")
    st.markdown(
        """
The solver-adjoint and PINN methods return a point estimate of ν. This method returns a
posterior over (ν, ic_amp, ic_phase) from a single observation. The flow-matching
network was trained offline (`sbi` + `zuko`) on simulated (parameters → observation)
pairs and runs forward only, so the `fmpe_posterior` Tesseract exposes `apply` with no
VJP. Set the ground-truth parameters on the left; the solver builds the matching
observation and the network infers the parameters back.
        """
    )

    bundle = load_fmpe_bundle(FMPE_BUNDLE_PATH)
    if bundle is None:
        st.warning(
            f"No trained posterior at `{FMPE_BUNDLE_PATH}`. It is gitignored (large, "
            "reproducible). Train it:"
        )
        st.code("make train-posterior", language="bash")
        return

    contract = bundle["metadata"]["contract"]
    sensors = bundle["sensors"]
    param_names = list(contract["param_names"])
    prior_low = list(contract["prior_low"])
    prior_high = list(contract["prior_high"])
    noise_std = float(contract["noise_std"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Tesseract", "fmpe_posterior")
    col2.metric("Endpoint", "apply only")
    col3.metric("Sensors (obs dim)", str(int(contract["observation_dim"])))
    st.caption(
        f"Posterior contract `model_id`: `{bundle['metadata']['model_id'][:16]}…`"
    )

    st.sidebar.header("Ground-truth parameters")
    st.sidebar.caption(
        "Set the true (ν, ic_amp, ic_phase) within the prior. The solver builds a noisy "
        "observation at the fixed sensor layout; the posterior infers the parameters back."
    )
    nice_labels = {
        "nu": "Viscosity ν",
        "ic_amp": "IC amplitude",
        "ic_phase": "IC phase",
    }
    theta_true = []
    for i, name in enumerate(param_names):
        lo, hi = float(prior_low[i]), float(prior_high[i])
        theta_true.append(
            st.sidebar.slider(
                nice_labels.get(name, name),
                min_value=lo,
                max_value=hi,
                value=float((lo + hi) / 2.0),
                step=(hi - lo) / 100.0,
                key=f"fmpe_{name}",
            )
        )
    obs_seed = st.sidebar.number_input(
        "Observation noise seed",
        min_value=0,
        max_value=1_000_000,
        value=1234,
        step=1,
        key="fmpe_obs_seed",
    )
    sample_seed = st.sidebar.number_input(
        "Posterior sampling seed",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
        key="fmpe_sample_seed",
    )
    inference_target = st.sidebar.selectbox(
        "Posterior deployment",
        ("In-process", "Tesseract image", "Remote Tesseract URL"),
        key="fmpe_inference_target",
        help=(
            "Use the local Python API for development, start the packaged image, "
            "or query an already-served Tesseract endpoint."
        ),
    )
    remote_url = None
    fmpe_image = "fmpe_posterior"
    if inference_target == "Tesseract image":
        fmpe_image = st.sidebar.text_input(
            "Posterior image",
            value="fmpe_posterior",
            key="fmpe_image",
            help="Local, registry-tagged, or digest-pinned Tesseract image reference.",
        )
    if inference_target == "Remote Tesseract URL":
        remote_url = st.sidebar.text_input(
            "Tesseract URL",
            value="http://127.0.0.1:8000",
            key="fmpe_remote_url",
        )

    if not st.sidebar.button("Sample Posterior", type="primary", key="fmpe_run"):
        st.info(
            "Set the ground-truth parameters in the sidebar and press **Sample "
            "Posterior**."
        )
        return

    if inference_target == "Tesseract image" and not docker_image_available(fmpe_image):
        st.error(f"Tesseract image `{fmpe_image}` was not found.")
        st.code(
            "make train-posterior\nuv run tesseract build tesseracts/fmpe_posterior",
            language="bash",
        )
        return
    if inference_target == "Remote Tesseract URL" and not remote_url.strip():
        st.error("Remote Tesseract URL must not be empty.")
        return

    with st.spinner("Simulating the measurement and sampling the posterior..."):
        from burgers_inverse.fmpe_posterior import (
            observation_from_theta,
            query_posterior_tesseract,
        )

        observation = (
            np.asarray(
                observation_from_theta(
                    tuple(theta_true), sensors, noise_std=noise_std, seed=int(obs_seed)
                )
            )
            .ravel()
            .astype(np.float32)
        )

        if inference_target == "In-process":
            component = get_fmpe_component()
            out = component.apply(
                component.InputSchema(observation=observation, seed=int(sample_seed))
            )
        else:
            out = query_posterior_tesseract(
                observation,
                seed=int(sample_seed),
                image=fmpe_image,
                url=remote_url if inference_target == "Remote Tesseract URL" else None,
            )

    def output_value(name):
        return out[name] if isinstance(out, dict) else getattr(out, name)

    samples = np.asarray(output_value("samples"), dtype=np.float64)
    mean = np.asarray(output_value("mean"), dtype=np.float64)
    q05 = np.asarray(output_value("q05"), dtype=np.float64)
    q95 = np.asarray(output_value("q95"), dtype=np.float64)

    st.success(f"Drew {samples.shape[0]} posterior samples in one apply() call.")
    st.caption(f"Deployment path: {inference_target}")

    st.subheader("Posterior marginals")
    fig, axes = plt.subplots(1, len(param_names), figsize=(6 * len(param_names), 4))
    if len(param_names) == 1:
        axes = [axes]
    for i, name in enumerate(param_names):
        ax = axes[i]
        sns.histplot(samples[:, i], bins=40, color=PINN_COLOR, stat="density", ax=ax)
        ax.axvspan(q05[i], q95[i], color=PARAM_GRAD_COLOR, alpha=0.15, label="90% CI")
        ax.axvline(
            theta_true[i], color=TRUE_COLOR, linestyle="--", linewidth=2, label="Truth"
        )
        ax.axvline(
            mean[i], color="#333333", linestyle=":", linewidth=1.5, label="Post. mean"
        )
        ax.set_title(nice_labels.get(name, name))
        ax.set_xlabel(name)
        ax.legend(frameon=False)
        finish_axes(ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Joint posterior (corner)")
    sample_df = pd.DataFrame(samples, columns=param_names)
    grid = sns.pairplot(
        sample_df, corner=True, diag_kind="hist", plot_kws=dict(s=6, alpha=0.15)
    )
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax = grid.axes[i][j]
            if ax is None:
                continue
            if i == j:
                ax.axvline(
                    theta_true[i], color=TRUE_COLOR, linestyle="--", linewidth=1.5
                )
            elif j < i:
                ax.axvline(
                    theta_true[j], color=TRUE_COLOR, linestyle="--", linewidth=1.0
                )
                ax.axhline(
                    theta_true[i], color=TRUE_COLOR, linestyle="--", linewidth=1.0
                )
    st.pyplot(grid.figure)
    plt.close(grid.figure)

    st.subheader("Per-parameter summary")
    summary = pd.DataFrame(
        {
            "parameter": param_names,
            "truth": [float(v) for v in theta_true],
            "posterior_mean": mean,
            "q05": q05,
            "q95": q95,
            "covered_90%": [
                bool(q05[i] <= theta_true[i] <= q95[i]) for i in range(len(param_names))
            ],
        }
    )
    st.dataframe(summary, hide_index=True)

    st.info(
        "**The ν marginal is overconfident.** Joint coverage (TARP) and the two IC "
        "marginals pass SBC, but the ν marginal fails (c2st ≈ 0.63), so its credible "
        "interval is narrower than the true uncertainty. Diagnostics: README UQ section "
        "and `scripts/fmpe_diagnostics.py`."
    )


def main():
    initialize_session_state()
    render_app_header()
    st.markdown("---")

    method = st.sidebar.radio(
        "Method",
        ("Solver-adjoint", "PINN (JAX or PyTorch)", "Posterior (FMPE)"),
        index=1,
        help="Each choice puts a different Tesseract component at the center of the "
        "same inverse problem.",
    )
    st.sidebar.markdown("---")

    if method == "Solver-adjoint":
        render_solver_demo()
    elif method.startswith("Posterior"):
        render_fmpe_demo()
    else:
        render_pinn_demo()


if __name__ == "__main__":
    main()
