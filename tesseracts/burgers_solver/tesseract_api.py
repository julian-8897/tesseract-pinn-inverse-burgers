# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable pseudospectral solver for periodic 1D viscous Burgers.

Solves:
    u_t + u u_x = nu u_xx

The implementation uses Fourier derivatives in space and Diffrax for time
integration. It is intentionally small and JAX-native so Tesseract VJP/JVP
endpoints can differentiate through the solver with respect to viscosity and
initial condition parameters.
"""

from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

DEFAULT_DT0 = 1e-3
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-5
DEFAULT_MAX_STEPS = 100_000


class InputSchema(BaseModel):
    """Input schema for the Burgers solver."""

    nu: Differentiable[Array[(), Float32]] = Field(
        description="Positive viscosity coefficient"
    )
    x_grid: Array[(None,), Float32] = Field(
        description="Uniform periodic spatial grid on [0, 1)"
    )
    t_grid: Array[(None,), Float32] = Field(description="Monotone saved output times")
    ic_amp: Differentiable[Array[(), Float32]] = Field(
        default=1.0,
        description="Initial condition amplitude",
    )
    ic_phase: Differentiable[Array[(), Float32]] = Field(
        default=0.0,
        description="Initial condition phase shift in radians",
    )


class OutputSchema(BaseModel):
    """Output schema for the Burgers solver."""

    u_field: Differentiable[Array[(None, None), Float32]] = Field(
        description="Solution field with shape (len(t_grid), len(x_grid))"
    )
    u_final: Differentiable[Array[(None,), Float32]] = Field(
        description="Final solution snapshot"
    )
    energy: Differentiable[Array[(None,), Float32]] = Field(
        description="Mean kinetic energy 0.5 * mean(u^2) at each saved time"
    )


def initial_condition(x_grid, ic_amp, ic_phase):
    """Periodic sinusoidal initial condition."""
    return ic_amp * jnp.sin(2.0 * jnp.pi * x_grid + ic_phase)


def spectral_derivatives(u, x_grid):
    """Return u_x and u_xx using FFT derivatives on a periodic grid."""
    nx = u.shape[0]
    dx = x_grid[1] - x_grid[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx) / dx
    u_hat = jnp.fft.fft(u)
    u_x = jnp.fft.ifft(1j * k * u_hat).real
    u_xx = jnp.fft.ifft(-(k**2) * u_hat).real
    return u_x, u_xx


def dealias_field(field):
    """Apply the 2/3 spectral filter to a field represented on a uniform grid."""
    nx = field.shape[0]
    mode_numbers = jnp.fft.fftfreq(nx) * nx
    keep = jnp.abs(mode_numbers) <= (nx // 3)
    return jnp.fft.ifft(jnp.fft.fft(field) * keep).real


def burgers_rhs(u, nu, x_grid):
    """Compute du/dt for viscous Burgers."""
    u_x, u_xx = spectral_derivatives(u, x_grid)
    nonlinear = dealias_field(u * u_x)
    return -nonlinear + nu * u_xx


def burgers_vector_field(_, u, args):
    """Diffrax-compatible vector field wrapper."""
    nu, x_grid = args
    return burgers_rhs(u, nu, x_grid)


def solve_burgers(nu, x_grid, t_grid, ic_amp, ic_phase):
    """Solve Burgers and return saved snapshots on t_grid."""
    u0 = initial_condition(x_grid, ic_amp, ic_phase)
    term = diffrax.ODETerm(burgers_vector_field)
    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=t_grid)
    stepsize_controller = diffrax.PIDController(
        rtol=DEFAULT_RTOL,
        atol=DEFAULT_ATOL,
    )

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t_grid[0],
        t1=t_grid[-1],
        dt0=DEFAULT_DT0,
        y0=u0,
        args=(nu, x_grid),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=DEFAULT_MAX_STEPS,
    )
    return sol.ys


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """Run the Burgers solver."""
    nu = jnp.asarray(inputs["nu"], dtype=jnp.float32)
    x_grid = jnp.asarray(inputs["x_grid"], dtype=jnp.float32)
    t_grid = jnp.asarray(inputs["t_grid"], dtype=jnp.float32)
    ic_amp = jnp.asarray(inputs["ic_amp"], dtype=jnp.float32)
    ic_phase = jnp.asarray(inputs["ic_phase"], dtype=jnp.float32)

    u_field = solve_burgers(nu, x_grid, t_grid, ic_amp, ic_phase)
    energy = 0.5 * jnp.mean(u_field**2, axis=1)
    return {
        "u_field": u_field,
        "u_final": u_field[-1],
        "energy": energy,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply the solver."""
    return apply_jit(inputs.model_dump())


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Jacobian-vector product computation."""
    return jvp_jit(
        inputs.model_dump(), tuple(jvp_inputs), tuple(jvp_outputs), tangent_vector
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Vector-Jacobian product computation."""
    return vjp_jit(
        inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector
    )


def abstract_eval(abstract_inputs):
    """Calculate output shapes."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: {"shape": x.shape, "dtype": str(x.dtype)}
        if is_shapedtype_struct(x)
        else x,
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


@eqx.filter_jit
def jac_jit(inputs: dict, jac_inputs: tuple[str], jac_outputs: tuple[str]):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


if __name__ == "__main__":
    n_points = 128
    nu = 0.05

    x_grid = jax.numpy.linspace(0, 1, n_points, endpoint=False)
    t_grid = jax.numpy.linspace(0, 0.2, 32)
    ic_amp = 1.0
    ic_phase = 0.0

    u_sol = solve_burgers(nu, x_grid, t_grid, ic_amp, ic_phase)

    print(u_sol)
