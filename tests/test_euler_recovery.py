"""Unit tests for Euler recovery campaign modes (ptc_ser, full_backward_euler, persistent_backward_euler)."""
import pytest
import numpy as np
from kflame.flame.solver import SolveOptions, solve_free_flame
from kflame.flame.problem import FreeFlameProblem
from kflame.chemistry.backend import NativeSpeciesBackend


from kflame.flame.config import FlameCase


def test_solve_options_transient_mode_defaults():
    opts = SolveOptions()
    assert opts.transient_solver_mode == "ptc_ser"
    assert opts.persistent_be_steps == 5
    assert opts.persistent_be_stall_threshold == 0.8


def test_solve_options_custom_mode():
    opts = SolveOptions(
        transient_solver_mode="persistent_backward_euler",
        persistent_be_steps=3,
        persistent_be_stall_threshold=0.75,
    )
    assert opts.transient_solver_mode == "persistent_backward_euler"
    assert opts.persistent_be_steps == 3
    assert opts.persistent_be_stall_threshold == 0.75


def test_modes_run_without_error():
    case = FlameCase(
        fuel="CH4", P=101325.0,
        transport_model="multicomponent", soret_enabled=False,
    )

    p = FreeFlameProblem(case, n_points=8)
    p.assume_finite_y = True
    p.backend = NativeSpeciesBackend(p)

    for mode in ("ptc_ser", "full_backward_euler", "persistent_backward_euler"):
        opts = SolveOptions(
            verbose=False,
            max_refine_passes=0,
            require_grid_convergence=False,
            transient_solver_mode=mode,
            persistent_be_steps=2,
        )
        x, ok, report = solve_free_flame(p, options=opts)
        assert isinstance(report, dict)
        assert "stage_A" in report
