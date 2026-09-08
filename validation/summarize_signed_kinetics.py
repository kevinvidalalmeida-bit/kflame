"""Freeze evidence for the signed-kinetics correction; never discard failed pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(folder):
    summary_path = folder / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    rows = []
    reference = {}
    for run in summary['runs']:
        name, repetition = run['solver'], run['repetition']
        path = folder / f'{name}_{repetition}.npz'
        record = {k: v for k, v in run.items() if k not in ('stages',)}
        if path.exists():
            record['profile_sha256'] = digest(path)
            with np.load(path) as data:
                fields = {k: data[k].copy() for k in data.files}
            if name not in reference:
                reference[name] = fields
            ref = reference[name]
            record['same_mesh_as_first_repetition'] = np.array_equal(fields['z'], ref['z'])
            record['repeat_max_absolute_difference'] = {
                k: float(np.max(np.abs(fields[k] - ref[k])))
                if fields[k].shape == ref[k].shape else None for k in fields}
        rows.append(record)
    validation_path = folder / 'validation.json'
    return dict(directory=str(folder), summary_sha256=digest(summary_path),
                case=summary['case'], options=summary['options'], command=summary['command'],
                source_sha256=summary.get('source_sha256'),
                mechanism_sha256=summary.get('mechanism_sha256'),
                versions=summary.get('versions'), threads=summary.get('threads'),
                jit_policy=summary.get('jit_policy'), profile_cache=summary.get('profile_cache'),
                rows=rows, validation_sha256=digest(validation_path) if validation_path.exists() else None,
                validation=json.loads(validation_path.read_text(encoding='utf-8'))
                if validation_path.exists() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--subject', default='Native Soret: signed trial-state kinetics correction')
    args = parser.parse_args()
    result = dict(
        subject=args.subject,
        limitations=['Cold means no saved flame profiles; disk JIT cache is retained.',
                     'Each solver adapts its own mesh. Equal thresholds do not imply equal error.',
                     'Agreement between solvers is not experimental validation or mesh independence.',
                     'Timing comparisons across revisions without interleaving are diagnostic only.'],
        campaigns=[collect(folder) for folder in args.directories])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
