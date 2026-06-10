"""Tesseract cross-framework PINN inverse problem for Burgers viscosity.

Public API surface. The implementation is split into cohesive submodules:

- :mod:`burgers_inverse.constants`     -- shared discretization grid
- :mod:`burgers_inverse.checkpointing` -- resumable deterministic training state
- :mod:`burgers_inverse.components`    -- Tesseract access + image guards
- :mod:`burgers_inverse.observations`  -- solver-backed observation samplers
- :mod:`burgers_inverse.losses`        -- PINN losses + BRDR weighting
- :mod:`burgers_inverse.engine`        -- shared inverse-training engine
- :mod:`burgers_inverse.experimental`  -- KdV/discrepancy sidebar (negative result)
- :mod:`burgers_inverse.reporting`     -- console tables, callbacks, artifacts
- :mod:`burgers_inverse.cli`           -- command-line orchestration

The amortized FMPE posterior lives in :mod:`burgers_inverse.fmpe_posterior` and is
imported explicitly (it pulls in ``sbi``/``torch``), not re-exported here.
"""

from __future__ import annotations

from burgers_inverse.checkpointing import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    config_fingerprint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from burgers_inverse.cli import (
    build_run_config,
    compare_backends,
    compare_methods,
    run_inverse_problem,
    run_seed_sweep,
    run_single_backend,
    run_solver_inverse,
    write_cli_artifacts,
)
from burgers_inverse.components import (
    TesseractImageNotFoundError,
    docker_image_available,
    ensure_image_available,
    evaluate_pinn_solution_grid,
    get_burgers_solver,
    get_initial_params,
    image_name_for_backend,
)
from burgers_inverse.configs import (
    DEFAULT_LOSS_WEIGHTS,
    DEFAULT_NOISE_STD,
    LOSS_WEIGHT_NAMES,
    ComponentConfig,
    DataConfig,
    FMPEConfig,
    LossWeights,
    ProblemConfig,
    RunConfig,
    TrainingConfig,
    loss_weights_from_mapping,
    normalize_loss_weights,
)
from burgers_inverse.constants import MIN_OBS_TIME, SOLVER_NT, SOLVER_NX
from burgers_inverse.engine import (
    EpochRecord,
    InverseStrategy,
    PINNStrategy,
    SolverAdjointStrategy,
    StepResult,
    TesseractCallCounter,
    TrainingCallback,
    _run_inverse_training,
    build_training_inputs,
    count_tesseract_calls,
    make_inverse_strategy,
    solver_inverse_loss,
    train_inverse,
    train_solver_inverse,
)
from burgers_inverse.experimental import (
    HybridDiscrepancyStrategy,
    generate_kdv_observations,
    hybrid_discrepancy_loss,
    solve_kdv_burgers,
    train_hybrid_inverse,
)
from burgers_inverse.losses import (
    compute_loss,
    compute_loss_components,
    compute_loss_from_log_viscosity,
    compute_pointwise_losses,
    format_loss_weights,
    initialize_brdr_state,
    loss_weights_to_array,
    summarize_brdr_weights,
    update_brdr_state,
    validate_brdr_loss_weights,
)
from burgers_inverse.observations import (
    GridObservations,
    generate_grid_observations,
    generate_observations,
)
from burgers_inverse.reporting import (
    MetricsRecorderCallback,
    RichProgressCallback,
    SolverInverseCallback,
    summarize_seed_results,
)

__all__ = [
    # constants
    "SOLVER_NX",
    "SOLVER_NT",
    "MIN_OBS_TIME",
    # configs
    "RunConfig",
    "ComponentConfig",
    "ProblemConfig",
    "DataConfig",
    "TrainingConfig",
    "LossWeights",
    "FMPEConfig",
    "DEFAULT_LOSS_WEIGHTS",
    "DEFAULT_NOISE_STD",
    "LOSS_WEIGHT_NAMES",
    "normalize_loss_weights",
    "loss_weights_from_mapping",
    # checkpointing
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_VERSION",
    "config_fingerprint",
    "save_training_checkpoint",
    "load_training_checkpoint",
    # components
    "get_burgers_solver",
    "get_initial_params",
    "image_name_for_backend",
    "docker_image_available",
    "ensure_image_available",
    "evaluate_pinn_solution_grid",
    "TesseractImageNotFoundError",
    # observations
    "generate_observations",
    "generate_grid_observations",
    "GridObservations",
    # losses
    "compute_loss",
    "compute_loss_components",
    "compute_loss_from_log_viscosity",
    "compute_pointwise_losses",
    "format_loss_weights",
    "loss_weights_to_array",
    "validate_brdr_loss_weights",
    "initialize_brdr_state",
    "update_brdr_state",
    "summarize_brdr_weights",
    # engine
    "train_inverse",
    "train_solver_inverse",
    "make_inverse_strategy",
    "InverseStrategy",
    "PINNStrategy",
    "SolverAdjointStrategy",
    "StepResult",
    "EpochRecord",
    "TrainingCallback",
    "TesseractCallCounter",
    "count_tesseract_calls",
    "build_training_inputs",
    "solver_inverse_loss",
    "_run_inverse_training",
    # experimental
    "solve_kdv_burgers",
    "generate_kdv_observations",
    "hybrid_discrepancy_loss",
    "HybridDiscrepancyStrategy",
    "train_hybrid_inverse",
    # reporting
    "summarize_seed_results",
    "RichProgressCallback",
    "MetricsRecorderCallback",
    "SolverInverseCallback",
    # cli
    "build_run_config",
    "run_inverse_problem",
    "run_solver_inverse",
    "run_single_backend",
    "compare_backends",
    "compare_methods",
    "run_seed_sweep",
    "write_cli_artifacts",
]
