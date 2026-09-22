"""Global nonlinear experiments: no fuel, mechanism, pressure or grid dispatch.

Research basis and limitations are recorded in the validation report. These
are hypotheses to measure, not convergence guarantees borrowed from papers.
"""
import inspect
import kflame.flame.solver as solver


def build_candidate(name):
    newton = inspect.getsource(solver.newton_solve)
    hybrid = inspect.getsource(solver._hybrid_newton)
    def replace(source, old, new):
        if source.count(old) != 1:
            raise RuntimeError('Solver source changed; review nonlinear experiment')
        return source.replace(old, new)
    if name == 'defect_corrections':
        # Spend extra corrections only while the transient nonlinear residual
        # remains above 10% of its initial value. Cap work at four corrections.
        # The final steady-state Newton convergence test is never changed.
        newton = replace(newton, '    status = -1\n',
                         '    status = -1\n    initial_ptc_norm = None\n')
        newton = replace(newton, '        normf = float(np.linalg.norm(f, ord=np.inf))\n',
            '        normf = float(np.linalg.norm(f, ord=np.inf))\n'
            '        if initial_ptc_norm is None:\n            initial_ptc_norm = normf\n')
        newton = replace(newton,
            '                converged = bool(alpha >= 1.0 - 1.0e-14 and s1 < tol)',
            '                converged = bool((alpha >= 1.0 - 1.0e-14 and s1 < tol)\n'
            '                                 or normf_try <= 0.1 * initial_ptc_norm)')
        hybrid = replace(hybrid, 'max_iter=1 if use_ptc else opts.transient_max_iter,',
                         'max_iter=4 if use_ptc else opts.transient_max_iter,')
        # After several corrections x != x_old. The last transient norm is NOT
        # the initial steady norm; using it in SER would be mathematically wrong.
        hybrid = replace(hybrid, 'float(last_record.get("normF", float("nan")))',
                         '(float(hist_ts[0].get("normF", float("nan"))) if hist_ts else float("nan"))')
    elif name == 'defect_step':
        # A nonlinear model-defect controller, NOT a temporal error estimator.
        # The already computed transient residual measures the failure of the
        # linear prediction. No extra residual, LU, or Jacobian age change.
        newton = replace(newton, '            if local_refresh_info:\n',
            '            if residual_damping:\n'
            '                record["transient_defect_ratio"] = normf_try / max(normf, 1e-300)\n'
            '            if local_refresh_info:\n')
        hybrid = replace(hybrid, '                    dt = dt_try * ser_factor',
            '                    defect = float(last_record.get("transient_defect_ratio", 1.0))\n'
            '                    dt = dt_try * float(np.clip(0.9 * np.sqrt(0.1 / max(defect, 1e-12)), 0.2, 2.0))')
    elif name == 'scaled_ser':
        # Residual units differ by component. Use one fixed scaling (x_old) for
        # both sides of the SER ratio, restricted to differential equations.
        newton = replace(newton, '            if local_refresh_info:\n',
            '            if residual_damping:\n'
            '                record["ser_norm"] = weighted_norm(f * mask, x_old, problem, rdt=rdt)\n'
            '            if local_refresh_info:\n')
        hybrid = replace(hybrid, 'float(last_record.get("normF", float("nan")))',
                         'float(last_record.get("ser_norm", float("nan")))')
        hybrid = replace(hybrid,
            '            steady_norm_after = _residual_inf(problem, x_ts) if ptc_step_ok else float("nan")',
            '            steady_norm_after = (weighted_norm(\n'
            '                residual(x_ts, problem) * build_transient_mask(problem.n_points, problem.n_species,\n'
            '                    solve_energy=bool(problem.solve_energy)), x_old, problem, rdt=rdt)\n'
            '                if ptc_step_ok else float("nan"))')
    else:
        raise ValueError(name)
    namespace = dict(vars(solver))
    exec(compile(newton, '<global nonlinear experiment>', 'exec'), namespace)
    exec(compile(hybrid, '<global nonlinear experiment>', 'exec'), namespace)
    return namespace['newton_solve'], namespace['_hybrid_newton']
