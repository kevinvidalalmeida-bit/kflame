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
tipo Cantera, continuación pseudo-transitoria PTC-SER con rescate BE,
Jacobiano block-tridiagonal y SciPy/LAPACK. En cada paso PTC se hace una sola
corrección lineal y el paso temporal se adapta con la reducción del residual;
si esa corrección falla, el modo `auto` vuelve al Euler implícito totalmente
convergido. En el generador FGM, la ruta fría usa por defecto
`block_tridiag + direct` y desactiva los grids fijos intermedios; esto evita
resolver y refinar varias veces la misma llama. `recycled_gmres` sigue
disponible explícitamente para continuaciones donde resulte beneficioso.
La opción `--auto-bootstrap-grids` conserva el bootstrap 12/24/48 para
diagnóstico y reproducibilidad histórica.
Para evitar sobre-hilos en los bloques densos pequenos, V2 fija
`OPENBLAS_NUM_THREADS=1` si el usuario no lo define. En corridas FGM
secuenciales y en V2, el kernel de quimica usa 4 hilos de Numba por defecto;
puede modificarse con `NUMBA_NUM_THREADS` o, en FGM, con
`--numba-kinetics-threads`.

El modo robusto y rápido es `--pseudo-transient-mode auto`. Para reproducir
la ruta anterior se puede usar `--pseudo-transient-mode fully_implicit`; el
modo `linear_ser` desactiva el rescate BE y se conserva para diagnóstico.
Estas opciones están disponibles tanto en el generador FGM como en
`V2/run_saved_comparison.py`.

Las variantes medidas y descartadas están registradas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md). No deben volver a
añadirse sin una validación end-to-end de FGM reproducible.

Para barridos FGM repetidos, el generador conserva las semillas V2 aceptadas
en `output-root/_v2_seed_cache` y activa automaticamente procesos paralelos
cuando todas las semillas del barrido ya existen. Es la ruta recomendada para
produccion: mantiene la misma malla y criterio de convergencia, y evita pagar
el bootstrap frio en cada regeneracion. La comparacion estricta fria y sus
limitaciones estan documentadas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md).
