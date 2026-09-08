"""Audit completeness, strict final metrics, profiles and paired cold timings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import cantera as ct

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'V2'))
from analyze_soret_validation import crossing


def profile_error(folder, label, name, repetition, case):
    a = dict(np.load(folder/f'{label}_{name}_{repetition}.npz'))
    b = dict(np.load(folder/f'{label}_baseline_{repetition}.npz'))
    if all(np.array_equal(a[k], b[k]) for k in ('z', 'T', 'Y', 'u')):
        return dict(bitwise_equal=True, Su_relative=0., E2_T=0., E2_qdot=0., E2_Y_max=0., passed=True)
    gas = ct.Solution(case['mech'])
    for f in (a, b):
        heat = []
        for t, y in zip(f['T'], f['Y'].T):
            gas.TPY = t, case['P'], y
            heat.append(gas.heat_release_rate)
        f['qdot'] = np.array(heat)
    target = case['T_in']+.25*(min(a['T'].max(), b['T'].max())-case['T_in'])
    for f in (a, b):
        f['aligned'] = f['z']-crossing(f['z'], f['T'], target)
    lo, hi = max(a['aligned'][0], b['aligned'][0]), min(a['aligned'][-1], b['aligned'][-1])
    z = np.unique(np.r_[a['aligned'], b['aligned'], lo, hi])
    z = z[(z >= lo) & (z <= hi)]
    def e2(qa, qb):
        av = np.interp(z, a['aligned'], qa)
        bv = np.interp(z, b['aligned'], qb)
        return float(np.sqrt(np.trapezoid((av-bv)**2, z)/max(np.trapezoid(bv**2, z), 1e-300)))
    active = np.flatnonzero(b['Y'].max(axis=1) >= 1e-5)
    errors = dict(bitwise_equal=False, Su_relative=float(abs(a['u'][0]/b['u'][0]-1)),
                  E2_T=e2(a['T'], b['T']), E2_qdot=e2(a['qdot'], b['qdot']),
                  E2_Y={gas.species_names[k]: e2(a['Y'][k], b['Y'][k]) for k in active})
    errors['E2_Y_max'] = max(errors['E2_Y'].values(), default=0.)
    errors['passed'] = bool(errors['Su_relative'] <= .001 and errors['E2_T'] <= .005
                            and errors['E2_qdot'] <= .05 and errors['E2_Y_max'] <= .02)
    return errors


def audit(folder):
    path = folder/'summary.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    command = data['command']
    count = int(command[command.index('--repeat')+1]) if '--repeat' in command else 1
    def declared(flag, default):
        if flag not in command:
            return default
        values = []
        for value in command[command.index(flag)+1:]:
            if value.startswith('--'):
                break
            values.append(value)
        return values
    labels = declared('--cases', ['ch4_1', 'h2_1'])
    variants = declared('--variants', ['baseline', 'thermal', 'blocked', 'powers', 'serial'])
    result = dict(directory=str(folder), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=data['source_sha256'], kernel_sha256=data['kernel_sha256'],
                  expected_repetitions=count, cases={})
    for label in labels:
        baseline = {r['repetition']: r for r in data['runs'] if r['case'] == label and r['variant'] == 'baseline'}
        modes = {}
        for name in variants:
            runs = {r['repetition']: r for r in data['runs'] if r['case'] == label and r['variant'] == name}
            warm = [r for r in data['warmup_runs'] if r['case'] == label and r['variant'] == name]
            complete = set(runs) == set(range(count)) and len(warm) == 1
            valid = complete and warm[0]['accepted'] and all(r['accepted'] for r in runs.values())
            def within(value, limit):
                return value is not None and np.isfinite(value) and value <= limit
            strict = all(within(r['stages'][-1].get('weighted_step_norm_final'), 1) and within(r.get('Finf'), 1e4)
                         and r['stages'][-1].get('grid_converged', False) for r in runs.values()) if runs else False
            errors = {str(i): profile_error(folder, label, name, i, r['configuration'])
                      for i, r in runs.items() if i in baseline and r['accepted'] and baseline[i]['accepted']}
            times = np.array([runs[i]['time_s'] for i in sorted(runs)])
            exceptions = [e for r in runs.values() for e in r.get('strategy_trace', []) if 'exception' in e]
            execution_verified = bool(runs) and not exceptions
            if name == 'mkl-blocks':
                execution_verified = execution_verified and all(
                    len(r.get('strategy_trace', [])) == 1 and
                    r['strategy_trace'][0].get('kind') == 'mkl_blocks' and
                    r['strategy_trace'][0].get('getrf',0) > 0 and
                    r['strategy_trace'][0].get('getrs',0) > 0 for r in runs.values())
            if name == 'simd-nasa':
                execution_verified = execution_verified and data.get('simd_evidence',{}).get('packed_fp64_operations',0) > 0
            if name in ('limited-spatial', 'limited-spatial-adapt'):
                execution_verified = execution_verified and all(
                    len(r.get('strategy_trace',[])) == 1 and
                    not r['strategy_trace'][0].get('assembly_errors') and
                    r['strategy_trace'][0].get('assembly_calls',0) > 0 for r in runs.values())
            if name in ('nonlinear-blocks', 'nonlinear-blocks-global'):
                execution_verified = execution_verified and all(
                    len(r.get('strategy_trace', [])) == 1 and
                    r['strategy_trace'][0].get('kind') == 'nonlinear_blocks' and
                    r['strategy_trace'][0].get('local_trials', 0) > 0
                    for r in runs.values())
                if name.endswith('-global'):
                    execution_verified = execution_verified and all(
                        r['strategy_trace'][0].get('global_descent') and
                        np.all(np.diff([r['strategy_trace'][0]['merit_initial']]+
                                       r['strategy_trace'][0]['accepted_merits']) < 0)
                        for r in runs.values())
            if name in ('exact-residual', 'generated-chemistry'):
                if name == 'exact-residual' and label.startswith('ch4'):
                    # Deliberate no-op control: exact transport is not lagged.
                    execution_verified = execution_verified and all(not r['strategy_trace'] for r in runs.values())
                else:
                    kind = 'exact_residual_shared' if name == 'exact-residual' else 'generated_chemistry'
                    execution_verified = execution_verified and all(
                        any(e.get('kind') == kind for e in r.get('strategy_trace', []))
                        for r in runs.values())
                if name == 'exact-residual':
                    def residual_calls(r):
                        return sum(s.get('profile', {}).get('residual_full', {}).get('count', 0)
                                   for s in r['stages'])
                    execution_verified = execution_verified and all(
                        i in baseline and residual_calls(baseline[i])-residual_calls(r) ==
                        sum(e.get('kind') == 'exact_residual_shared' for e in r.get('strategy_trace', []))
                        for i, r in runs.items())
            if name == 'thermal':
                profiles = [s['profile'] for r in runs.values() for s in r['stages'] if 'profile' in s]
                execution_verified = execution_verified and bool(profiles) and all(
                    p.get('jacobian_build', {}).get('count', 0) > 0 and
                    p['jacobian_build']['count'] == p.get('jacobian_precompute_thermochem', {}).get('count')
                    for p in profiles)
            elif name in ('two-grid', 'mesh-budget', 'dependencies', 'compiled-lu', 'transport-action', 'transport-action-shared', 'transport-homotopy'):
                execution_verified = execution_verified and all(r.get('strategy_trace') for r in runs.values())
                if name == 'compiled-lu':
                    execution_verified = execution_verified and all(
                        sum(e.get('kind') == 'compiled_lu' for e in r['strategy_trace']) > 0 and
                        sum(e.get('kind') == 'compiled_lu' for e in r['strategy_trace']) ==
                        sum(s.get('profile', {}).get('linear_factorize', {}).get('count', 0) for s in r['stages'])
                        for r in runs.values())
                if name == 'dependencies':
                    execution_verified = execution_verified and all(
                        sum(e.get('kind') == 'dependencies' for e in r.get('strategy_trace', [])) ==
                        sum(s.get('profile', {}).get('jacobian_build', {}).get('count', 0) for s in r['stages'])
                        for r in runs.values())
                if name == 'transport-homotopy':
                    execution_verified = execution_verified and all(
                        any(e.get('kind') == 'homotopy_stage' and e.get('accepted') for e in r['strategy_trace'])
                        for r in runs.values())
            mode = dict(complete=complete, all_accepted=bool(valid), strict_final_metrics=bool(strict),
                        warmup=warm, n=len(runs), profiles=errors,
                        accuracy_passed=bool(valid and len(errors) == count and all(e['passed'] for e in errors.values())),
                        execution_verified=bool(execution_verified), candidate_errors=exceptions)
            if name.startswith('limited-spatial'):
                mode['interpretation'] = ('Different spatial operator. Errors against baseline are diagnostic only; '
                                          'equal-accuracy speedup requires an independently refined reference. '
                                          'Embedded reconstruction monitor is not a certified error estimator.')
            if times.size:
                mode.update(raw_s=times.tolist(), median_s=float(np.median(times)),
                            quartiles_s=np.quantile(times, [.25, .75]).tolist())
            if valid and set(baseline) == set(runs) and all(r['accepted'] for r in baseline.values()):
                b = np.array([baseline[i]['time_s'] for i in sorted(runs)])
                mode['reduction_percent'] = float(100*(1-np.median(times)/np.median(b)))
                if count >= 2:
                    samples = np.random.default_rng(20260906).integers(0, count, size=(20000, count))
                    ratios = np.median(times[samples], axis=1)/np.median(b[samples], axis=1)
                    mode['paired_ratio_ci95'] = np.quantile(ratios, [.025, .975]).tolist()
            modes[name] = mode
        result['cases'][label] = modes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = dict(campaigns=[audit(p) for p in args.directories],
                  limitations=['Cold flame, warmed compiler; no seed reuse.',
                               'These are within-V2 ablations, not new Cantera timings.',
                               'One repetition gives screening evidence, not a confidence interval.',
                               'The two-grid variant is a prototype, not a complete recursive FAS solver.',
                               'Final residual and weighted norm do not prove mesh/domain independence.'])
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    for campaign in result['campaigns']:
        for label, modes in campaign['cases'].items():
            for name, mode in modes.items():
                print(label, name, {k: mode.get(k) for k in ['n', 'all_accepted', 'accuracy_passed', 'median_s', 'paired_ratio_ci95']})


if __name__ == '__main__':
    main()
