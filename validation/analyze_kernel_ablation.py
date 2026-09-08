"""Summarize every paired ablation run, including warm-up/rejection policy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for directory in args.directories:
        path = directory/'summary.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        command = data['command']
        expected = int(command[command.index('--repeat')+1]) if '--repeat' in command else 1
        cases = []
        for case in sorted({r['case'] for r in data['runs']}):
            rows = [r for r in data['runs'] if r['case'] == case]
            grouped = {name: {r['repetition']: r for r in rows if r['variant'] == name}
                       for name in {r['variant'] for r in rows}}
            complete = all(set(records) == set(range(expected)) for records in grouped.values())
            accepted = complete and all(r['accepted'] for r in rows)
            baseline = grouped['baseline']
            statistics = {}
            for name, records in sorted(grouped.items()):
                times = np.array([records[i]['time_s'] for i in sorted(records)])
                stat = dict(n=len(times), raw_s=times.tolist(), median_s=float(np.median(times)),
                            quartiles_s=np.quantile(times, [.25, .75]).tolist(),
                            all_same_mesh_as_baseline=all(r['same_grid_as_baseline'] for r in records.values()))
                if stat['all_same_mesh_as_baseline']:
                    stat['max_absolute_profile_differences'] = {
                        k: max(r['differences_on_identical_grid'][k] for r in records.values())
                        for k in ('z', 'T', 'Y', 'u')}
                if accepted and set(records) == set(baseline):
                    b = np.array([baseline[i]['time_s'] for i in sorted(records)])
                    rng = np.random.default_rng(20260905)
                    indices = rng.integers(0, len(times), size=(20000, len(times)))
                    ratios = np.median(times[indices], axis=1) / np.median(b[indices], axis=1)
                    stat.update(reduction_percent=float(100*(1-np.median(times)/np.median(b))),
                                paired_ratio_ci95=np.quantile(ratios, [.025, .975]).tolist())
                statistics[name] = stat
            effective_options = dict(rows[0]['options'])
            # Earlier raw records stored the pre-call options; native_solve
            # applies this explicitly recorded bootstrap override in all runs.
            effective_options['multicomponent_bootstrap'] = bool(rows[0]['bootstrap'])
            effective_options['bootstrap_mesh_factor'] = 2.0
            cases.append(dict(case=case, configuration=rows[0]['configuration'],
                              effective_options=effective_options,
                              complete=complete, expected_repetitions=expected,
                              all_accepted=accepted, statistics=statistics))
        reports.append(dict(directory=str(directory), summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            full_stack_warmup=data['full_stack_warmup'], warmup_runs=data['warmup_runs'],
                            reference_kinetics=data.get('reference_kinetics', 'helper-grouped screening'),
                            reference_kernel_sha256=data.get('reference_kernel_sha256'),
                            source_sha256=data['source_sha256'], cases=cases))
    result = dict(campaigns=reports, limitations=[
        'Cold flame initialization, not cold compilation: predeclared warm-up is separate and never a seed.',
        'Bootstrap intervals with three pairs are exploratory, not a substitute for a larger campaign.',
        'Kernel/same-mesh equivalence is not a mesh independence or experimental validation study.',
        'These are within-V2 ablations; they do not measure Cantera performance.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
