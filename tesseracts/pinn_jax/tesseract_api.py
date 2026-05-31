# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PINN (Physics-Informed Neural Network) Tesseract API.
Simple MLP implementation of PINN using JAX and Equinox with Fourier feature encoding.
Maps (x, t) → u(x, t) for solving PDEs.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

FOURIER_FEATURE_SEED = 0


def fixed_fourier_features(n_fourier_features=32):
    """Return backend-independent fixed Fourier frequencies."""
    rng = np.random.default_rng(FOURIER_FEATURE_SEED)
    b_x = rng.normal(size=n_fourier_features).astype(np.float32) * 2.0
    b_t = rng.normal(size=n_fourier_features).astype(np.float32) * 2.0
    return jnp.asarray(b_x), jnp.asarray(b_t)


class InputSchema(BaseModel):
    """Input schema for PINN."""

    x: Differentiable[Array[(None,), Float32]] = Field(
        description="Spatial coordinates (collocation points)"
    )
    t: Differentiable[Array[(None,), Float32]] = Field(
        description="Time coordinates (same shape as x)"
    )
    params_flat: Differentiable[Array[(None,), Float32]] = Field(
        description="Flattened neural network parameters"
    )


class OutputSchema(BaseModel):
    """Output schema for PINN."""

    u_pred: Differentiable[Array[(None,), Float32]] = Field(
        description="Predicted solution u(x, t)"
    )
    u_x: Differentiable[Array[(None,), Float32]] = Field(
        description="Spatial derivative ∂u/∂x computed via autodiff"
    )
    u_t: Differentiable[Array[(None,), Float32]] = Field(
        description="Time derivative ∂u/∂t computed via autodiff"
    )
    u_xx: Differentiable[Array[(None,), Float32]] = Field(
        description="Second spatial derivative ∂²u/∂x² computed via autodiff"
    )


class PINNNet(eqx.Module):
    """Simple MLP for PINN with Fourier feature encoding."""

    layers: list
    # Fourier feature frequencies for positional encoding
    B_x: jax.Array
    B_t: jax.Array

    def __init__(self, key, hidden_sizes=None, n_fourier_features=32):
        """Initialize trainable MLP with fixed Fourier features."""
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        # Fixed Fourier feature frequencies keep the trainable parameter contract
        # identical across the JAX and PyTorch backends.
        self.B_x, self.B_t = fixed_fourier_features(n_fourier_features)

        # Input: 2*n_fourier_features (sin + cos for x and t) + 2 (raw x, t)
        input_dim = 4 * n_fourier_features + 2
        layer_sizes = [input_dim] + hidden_sizes + [1]
        layer_keys = jax.random.split(key, len(layer_sizes) - 1)

        self.layers = []
        for i, (in_size, out_size) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:], strict=True)
        ):
            self.layers.append(eqx.nn.Linear(in_size, out_size, key=layer_keys[i]))

    def __call__(self, x, t):
        """Forward pass for single point."""
        # Fourier feature encoding
        x_proj = x * self.B_x
        t_proj = t * self.B_t

        # Concatenate: [x, t, sin(x*B), cos(x*B), sin(t*B), cos(t*B)]
        features = jnp.concatenate(
            [
                jnp.array([x, t]),
                jnp.sin(x_proj),
                jnp.cos(x_proj),
                jnp.sin(t_proj),
                jnp.cos(t_proj),
            ]
        )

        h = features
        for layer in self.layers[:-1]:
            h = layer(h)
            h = jax.nn.tanh(h)

        return self.layers[-1](h).squeeze()


def flatten_params(model):
    """Flatten trainable MLP parameters to a single array.

    Fourier feature frequencies are fixed backend-independent constants, not
    trainable parameters.
    """
    leaves, _ = jax.tree_util.tree_flatten(eqx.filter(model.layers, eqx.is_array))
    return jnp.concatenate([leaf.flatten() for leaf in leaves])


def unflatten_params(params_flat, reference_key=None):
    """Unflatten parameters back to model structure."""
    if reference_key is None:
        reference_key = jax.random.PRNGKey(0)
    reference_model = PINNNet(reference_key)

    trainable_layers = eqx.filter(reference_model.layers, eqx.is_array)
    static_layers = eqx.filter(
        reference_model.layers,
        eqx.is_inexact_array,
        inverse=True,
    )
    leaves, treedef = jax.tree_util.tree_flatten(trainable_layers)
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]

    unflattened_leaves = []
    start = 0
    for shape, size in zip(shapes, sizes, strict=True):
        unflattened_leaves.append(params_flat[start : start + size].reshape(shape))
        start += size

    params_tree = jax.tree_util.tree_unflatten(treedef, unflattened_leaves)
    layers = eqx.combine(params_tree, static_layers)

    return eqx.tree_at(lambda model: model.layers, reference_model, layers)


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    """PINN forward pass with derivatives computed via autodiff."""
    x = jnp.array(inputs["x"])
    t = jnp.array(inputs["t"])
    params_flat = jnp.array(inputs["params_flat"])

    model = unflatten_params(params_flat)

    def u_fn(x_val, t_val):
        return model(x_val, t_val)

    u_pred = jax.vmap(u_fn)(x, t)

    # spatial/temporal partial derivatives via autodiff
    u_x = jax.vmap(jax.grad(u_fn, argnums=0))(x, t)

    u_t = jax.vmap(jax.grad(u_fn, argnums=1))(x, t)

    def u_x_fn(x_val, t_val):
        return jax.grad(u_fn, argnums=0)(x_val, t_val)

    u_xx = jax.vmap(jax.grad(u_x_fn, argnums=0))(x, t)

    return {"u_pred": u_pred, "u_x": u_x, "u_t": u_t, "u_xx": u_xx}


def apply(inputs: InputSchema) -> OutputSchema:
    """Apply PINN."""
    out = apply_jit(inputs.model_dump())
    return out


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    """Jacobian-vector product (JVP) computation."""
    return jvp_jit(
        inputs.model_dump(), tuple(jvp_inputs), tuple(jvp_outputs), tangent_vector
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    """Vector-Jacobian product (VJP) computation."""
    return vjp_jit(
        inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape."""

    def is_shapedtype_dict(x):
        return type(x) is dict and (x.keys() == {"shape", "dtype"})

    def is_shapedtype_struct(x):
        return isinstance(x, jax.ShapeDtypeStruct)

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
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Helper functions
#


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
