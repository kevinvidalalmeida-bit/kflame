# TFM - Flujo Principal Cantera vs V2

## Comando principal

Desde la raiz del proyecto:

```powershell
python .\V2\run_saved_comparison.py --loglevel 0
```

Para ver el avance del solver V2:

```powershell
python .\V2\run_saved_comparison.py --loglevel 0 --verbose-ours
```

Para ver tambien la salida de Cantera:

```powershell
python .\V2\run_saved_comparison.py --loglevel 1 --verbose-ours
```

## Salidas

Cada corrida se guarda en:

```text
V2/comparison_runs/run_YYYYMMDD_HHMMSS/
```

Ese directorio contiene:

- `summary.json`
- `comparison_data.npz`
- `profiles_cantera.csv`
- `profiles_ours.csv`
- `plots/`

## Estructura limpia

- `V2/`: solver 1-D y comparación de referencia.
- `FGM/scripts/generate_fgm_tables_native.py`: generador FGM con el backend
  nativo CPU.
- `FGM/resultados/`: resultados finales conservados.
- `TESIS/`: material de la tesis conservado.

Los perfiles, cachés, respaldos, binarios compilados y scripts de ensayo se
eliminaron del árbol de trabajo. La ruta de producción usa Newton amortiguado
tipo Cantera, Euler implícito (BE), Jacobiano block-tridiagonal y
SciPy/LAPACK. En el generador FGM, los cambios de paso BE reutilizan la ultima
LU exacta como precondicionador GMRES; ante fallo se vuelve automaticamente a
una factorizacion exacta.
El generador FGM usa por defecto `block_tridiag + recycled_gmres`; GMRES prueba
cuatro iteraciones con la LU anterior y se vuelve a LU exacta cuando no basta.
Para evitar sobre-hilos en los bloques densos pequenos, V2 fija
`OPENBLAS_NUM_THREADS=1` si el usuario no lo define. En corridas FGM
secuenciales y en V2, el kernel de quimica usa 4 hilos de Numba por defecto;
puede modificarse con `NUMBA_NUM_THREADS` o, en FGM, con
`--numba-kinetics-threads`.

Las variantes medidas y descartadas están registradas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md). No deben volver a
añadirse sin una validación end-to-end de FGM reproducible.
