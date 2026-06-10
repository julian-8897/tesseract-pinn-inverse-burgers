"""Shared discretization constants for the inverse Burgers pipeline.

Named generically rather than after any one experiment so cross-module imports
read honestly: the same grid backs the forward solver, the observation samplers,
and the FMPE sensor layout.
"""

from __future__ import annotations

# Solver discretization grid.
SOLVER_NX = 128
SOLVER_NT = 64

# Smallest observation/collocation time. Sensors and collocation points avoid
# t≈0, where the initial condition dominates and the inverse signal is weak.
MIN_OBS_TIME = 0.05
