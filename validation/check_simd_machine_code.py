"""Local exactness and machine-code audit, not flame timing evidence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','4')
from pathlib import Path
import json
from benchmark_cold_strategies import selected_kernels,kernel_checks
from cold_simd import vector_evidence


def main():
    kernels=selected_kernels(['baseline','simd-nasa'])
    result=dict(checks=kernel_checks(kernels),evidence=vector_evidence(kernels['simd-nasa']))
    output=Path(__file__).with_name('simd_machine_code.json')
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['evidence'],indent=2))


if __name__=='__main__':
    main()
