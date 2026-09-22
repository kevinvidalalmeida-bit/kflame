"""Paired cold-profile flames. One candidate at a time against analytic default.

Warmups excluded; four Numba threads, one BLAS thread; all acceptance outcomes
and exact profile comparisons saved. No concurrent heavy jobs during timings.
"""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
from pathlib import Path
import json
import hashlib
import platform
import sys
from importlib.metadata import version
from dataclasses import asdict
import time
import numpy as np
from residual_lu_candidates import variant, workspace_factorize
from kflame.benchmarks.soret import benchmark_options, native_solve, compact_result, json_safe
from kflame.flame.config import FlameCase
from kflame.flame.equations import BlockTridiagJacobian, factorize, solve_linear


def verify_lu():
    rng = np.random.default_rng(985)
    n,k = 11,12
    diag = rng.normal(size=(n,k,k)) + 12*np.eye(k)[None]
    lower,upper = rng.normal(size=(n-1,k,k)),rng.normal(size=(n-1,k,k))
    diag[2,0,0] = 0.001
    j = BlockTridiagJacobian(lower,diag,upper)
    snapshot = [a.copy() for a in (lower,diag,upper)]
    rhs = rng.normal(size=n*k)
    reference = factorize(j)
    for order in ('C','F'):
        candidate = workspace_factorize(j,order)
        for key in ('lu_blocks_array','pivots_array','cprime'):
            np.testing.assert_array_equal(candidate[key],reference[key])
        np.testing.assert_array_equal(solve_linear(candidate,rhs),solve_linear(reference,rhs))
        # Lifetime and fallback: a later factorization must not overwrite this one.
        workspace_factorize(BlockTridiagJacobian(lower,diag+np.eye(k)[None],upper),order)
        candidate['compiled_substitution'] = False
        np.testing.assert_allclose(solve_linear(candidate,rhs),solve_linear(reference,rhs),atol=1e-13,rtol=1e-13)
    for actual,expected in zip((j.lower,j.diag,j.upper),snapshot):
        np.testing.assert_array_equal(actual,expected)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--candidate',choices=['carry','retry','workspace_c','workspace_f','small_serial','combined','backward_euler','defect_corrections','scaled_ser','defect_step'],required=True)
    ap.add_argument('--cases',nargs='+',choices=['CH4_1','CH4_10','H2_1','H2_10'],default=['CH4_1','CH4_10','H2_1','H2_10'])
    ap.add_argument('--pairs',type=int,default=3)
    args=ap.parse_args()
    if args.pairs < 1:
        ap.error('--pairs must be positive')
    verify_lu()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out=dict(protocol=__doc__,candidate=args.candidate,options=asdict(benchmark_options(True)),cases={})
    root=Path(__file__).resolve().parents[1]
    sources=sorted((root/'src/kflame').rglob('*.py')) + [Path(__file__),Path(__file__).with_name('residual_lu_candidates.py'),Path(__file__).with_name('global_nonlinear_candidates.py')]
    out['environment']=dict(python=sys.version,platform=platform.platform(),command=sys.argv,
        versions={name:version(name) for name in ('numpy','scipy','numba','llvmlite')},
        threads={key:os.environ.get(key) for key in ('NUMBA_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    def save(): args.output.write_text(json.dumps(json_safe(out),indent=2),encoding='utf-8')
    for name in args.cases:
        fuel,pressure=name.split('_')
        multi=fuel=='H2'
        case=FlameCase(fuel=fuel,P=101325*float(pressure),transport_model='multicomponent' if multi else 'mixture-averaged',
                       soret_enabled=multi,ratio=2.5,slope=.04,curve=.08,prune=.003)
        record=out['cases'][name]=dict(case=asdict(case),warmups={},pairs=[])
        def run(mode):
            with variant(mode):
                start=time.perf_counter()
                result=native_solve(case,benchmark_options(True),bootstrap=multi,bootstrap_mesh_factor=2.)
                wall=time.perf_counter()-start
            summary=compact_result(result)
            summary['wall_s']=wall
            print(name,mode,round(wall,4),'accepted',result['accepted'],flush=True)
            return summary,{k:result[k] for k in ('z','T','u','Y')}
        modes=('baseline',args.candidate)
        for mode in modes:
            record['warmups'][mode],_=run(mode)
            save()
        for i in range(args.pairs):
            pair,fields={},{}
            for mode in (modes if i%2==0 else modes[::-1]):pair[mode],fields[mode]=run(mode)
            pair['identical_fields']=all(np.array_equal(fields[modes[0]][k],fields[modes[1]][k]) for k in fields[modes[0]])
            if not pair['identical_fields']:
                pair['max_abs_diff']={k:float(np.max(abs(fields[modes[0]][k]-fields[modes[1]][k])))
                    if fields[modes[0]][k].shape==fields[modes[1]][k].shape else None for k in fields[modes[0]]}
            record['pairs'].append(pair)
            save()
        record['median_wall_s']={m:float(np.median([p[m]['wall_s'] for p in record['pairs']])) for m in modes}
        record['reduction_percent']=100*(1-record['median_wall_s'][modes[1]]/record['median_wall_s'][modes[0]])
        record['all_identical']=all(p['identical_fields'] for p in record['pairs'])
        record['all_accepted']=all(p[m]['accepted'] for p in record['pairs'] for m in modes)
        save()
        print(name,'SUMMARY',record['median_wall_s'],record['reduction_percent'],record['all_identical'],flush=True)


if __name__=='__main__':main()
