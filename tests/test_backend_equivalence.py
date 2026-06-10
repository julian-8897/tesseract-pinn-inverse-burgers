"""Backend contract tests for the JAX and PyTorch PINN implementations."""

import jax
import numpy as np

from burgers_inverse.component_loader import load_tesseract_api

jax_api = load_tesseract_api(
    "pinn_jax",
    module_name="pinn_jax_api_for_equivalence",
)
torch_api = load_tesseract_api(
    "pinn_pytorch",
    module_name="pinn_torch_api_for_equivalence",
)


def test_fixed_fourier_features_match_across_backends():
    jax_b_x, jax_b_t = jax_api.fixed_fourier_features()
    torch_b_x, torch_b_t = torch_api.fixed_fourier_features()

    np.testing.assert_allclose(np.asarray(jax_b_x), torch_b_x.numpy())
    np.testing.assert_allclose(np.asarray(jax_b_t), torch_b_t.numpy())


def test_backends_have_same_trainable_parameter_contract():
    jax_model = jax_api.PINNNet(jax.random.PRNGKey(123))
    torch_model = torch_api.PINNNet(seed=123)

    jax_params = np.asarray(jax_api.flatten_params(jax_model))
    torch_params = torch_api.flatten_params(torch_model)
    torch_trainable_count = sum(p.numel() for p in torch_model.parameters())

    assert jax_params.shape == torch_params.shape
    assert jax_params.size == torch_trainable_count
    assert jax_params.size == 16769


def test_pytorch_vjp_returns_trainable_mlp_gradient_only():
    model = torch_api.PINNNet(seed=123)
    params_flat = torch_api.flatten_params(model)
    inputs = torch_api.InputSchema(
        x=np.linspace(0.1, 0.9, 5, dtype=np.float32),
        t=np.linspace(0.05, 0.5, 5, dtype=np.float32),
        params_flat=params_flat,
    )
    out = torch_api.apply(inputs)
    cotangent = {"u_pred": np.ones_like(out["u_pred"], dtype=np.float32)}

    vjp = torch_api.vector_jacobian_product(
        inputs,
        {"params_flat"},
        {"u_pred"},
        cotangent,
    )

    assert set(vjp) == {"params_flat"}
    assert vjp["params_flat"].shape == params_flat.shape
    assert np.linalg.norm(vjp["params_flat"]) > 0.0
