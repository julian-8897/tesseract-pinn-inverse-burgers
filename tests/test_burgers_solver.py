"""Smoke tests for the differentiable 1D Burgers pseudospectral solver."""

import pathlib
import sys

import jax
import jax.numpy as jnp


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tesseracts" / "burgers_solver"))

from tesseract_api import (  # noqa: E402
    InputSchema,
    apply_jit,
    burgers_rhs,
    solve_burgers,
    vector_jacobian_product,
)


def make_grid(nx=64, nt=16):
    x_grid = jnp.linspace(0.0, 1.0, nx, endpoint=False, dtype=jnp.float32)
    t_grid = jnp.linspace(0.0, 0.2, nt, dtype=jnp.float32)
    return x_grid, t_grid


def test_solver_shapes_and_finiteness():
    x_grid, t_grid = make_grid()
    out = apply_jit(
        {
            "nu": jnp.array(0.05, dtype=jnp.float32),
            "x_grid": x_grid,
            "t_grid": t_grid,
            "ic_amp": jnp.array(1.0, dtype=jnp.float32),
            "ic_phase": jnp.array(0.0, dtype=jnp.float32),
        }
    )

    assert out["u_field"].shape == (16, 64)
    assert out["u_final"].shape == (64,)
    assert out["energy"].shape == (16,)
    assert jnp.all(jnp.isfinite(out["u_field"]))
    assert jnp.all(jnp.isfinite(out["energy"]))


def test_viscous_energy_dissipates():
    x_grid, t_grid = make_grid(nt=32)
    u_field = solve_burgers(
        jnp.array(0.05, dtype=jnp.float32),
        x_grid,
        t_grid,
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    energy = 0.5 * jnp.mean(u_field**2, axis=1)

    assert float(energy[-1]) < float(energy[0])


def test_rhs_matches_known_derivative_at_t0():
    x_grid, _ = make_grid()
    nu = jnp.array(0.05, dtype=jnp.float32)
    u = jnp.sin(2.0 * jnp.pi * x_grid)
    rhs = burgers_rhs(u, nu, x_grid)

    expected = (
        -u * (2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x_grid))
        - nu * (2.0 * jnp.pi) ** 2 * u
    )

    assert float(jnp.max(jnp.abs(rhs - expected))) < 1e-3


def test_gradient_through_solver_wrt_nu():
    x_grid, t_grid = make_grid(nt=24)

    def final_energy(nu):
        u_field = solve_burgers(
            nu,
            x_grid,
            t_grid,
            jnp.array(1.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        )
        return 0.5 * jnp.mean(u_field[-1] ** 2)

    grad_nu = jax.grad(final_energy)(jnp.array(0.05, dtype=jnp.float32))

    assert jnp.isfinite(grad_nu)
    assert float(grad_nu) < 0.0


def test_tesseract_vjp_endpoint_wrt_nu():
    x_grid, t_grid = make_grid(nt=12)
    inputs = InputSchema(
        nu=jnp.array(0.05, dtype=jnp.float32),
        x_grid=x_grid,
        t_grid=t_grid,
        ic_amp=jnp.array(1.0, dtype=jnp.float32),
        ic_phase=jnp.array(0.0, dtype=jnp.float32),
    )
    out = apply_jit(inputs.model_dump())
    cotangent = {"u_final": jnp.ones_like(out["u_final"]) / out["u_final"].size}

    vjp = vector_jacobian_product(inputs, {"nu"}, {"u_final"}, cotangent)

    assert "nu" in vjp
    assert jnp.isfinite(vjp["nu"])


if __name__ == "__main__":
    test_solver_shapes_and_finiteness()
    test_viscous_energy_dissipates()
    test_rhs_matches_known_derivative_at_t0()
    test_gradient_through_solver_wrt_nu()
    test_tesseract_vjp_endpoint_wrt_nu()
    print("burgers solver smoke tests passed")
