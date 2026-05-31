import pathlib
import sys

import jax
import jax.numpy as jnp
import pytest
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import inverse_problem as ip  # noqa: E402

BACKENDS = ("jax", "pytorch")


@pytest.mark.parametrize("backend", BACKENDS)
def test_container_apply_and_vjp(backend):
    image = ip.image_name_for_backend(backend)
    if not ip.docker_image_available(image):
        pytest.skip(f"Tesseract image '{image}' not built; run ./buildall.sh")

    params = ip.get_initial_params(backend, seed=0)
    x = jnp.linspace(0.1, 0.9, 5, dtype=jnp.float32)
    t = jnp.linspace(0.05, 0.5, 5, dtype=jnp.float32)

    pinn = Tesseract.from_image(image)
    with pinn:
        out = apply_tesseract(pinn, {"x": x, "t": t, "params_flat": params})
        for key in ("u_pred", "u_x", "u_t", "u_xx"):
            assert out[key].shape == x.shape

        def loss(p):
            result = apply_tesseract(pinn, {"x": x, "t": t, "params_flat": p})
            return jnp.sum(result["u_pred"] ** 2)

        grad = jax.grad(loss)(params)
        assert grad.shape == params.shape
        assert jnp.all(jnp.isfinite(grad))
        assert float(jnp.linalg.norm(grad)) > 0.0
