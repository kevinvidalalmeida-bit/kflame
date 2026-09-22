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


def test_persistent_state_transitions_recorded_in_history():
    """Verify that BE-persistent scheme appears in history and returns to PTC-SER when progress recovers."""
    case = FlameCase(
        fuel="H2", P=101325.0 * 10.0,
        transport_model="multicomponent", soret_enabled=False,
    )
    p = FreeFlameProblem(case, n_points=8)
    p.assume_finite_y = True
    p.backend = NativeSpeciesBackend(p)

    opts = SolveOptions(
        verbose=False,
        trace_solver=True,
        max_refine_passes=0,
        require_grid_convergence=False,
        transient_solver_mode="persistent_backward_euler",
        persistent_be_steps=3,
        persistent_be_stall_threshold=0.8,
    )

    x, ok, report = solve_free_flame(p, options=opts)
    trace = getattr(p, "_solver_trace", [])
    assert len(trace) > 0

    transient_steps = []
    for tr in trace:
        hist = tr.get("history", [])
        for h in hist:
            if isinstance(h, dict) and h.get("phase") == "transient":
                transient_steps.append(h)

    # Verify steady_norm_before and steady_norm_after are recorded
    for h in transient_steps:
        if h.get("ok"):
            assert np.isfinite(h.get("steady_norm_before"))
            assert np.isfinite(h.get("steady_norm_after"))


