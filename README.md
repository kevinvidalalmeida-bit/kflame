# TFM - Flujo Principal Cantera vs V2

## Comando principal

Desde la raíz del proyecto:

```powershell
python .\V2\run_saved_comparison.py --loglevel 0
```

Para ver el avance del solver V2:

```powershell
python .\V2\run_saved_comparison.py --loglevel 0 --verbose-ours
```

Para ver también la salida de Cantera:

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
- `FGM/scripts/generate_fgm_tables_cantera.py`: generador FGM de referencia.
- `FGM/scripts/fgm_common.py`: matemática y utilidades compartidas por ambos
  generadores, sin código duplicado.
- `FGM/resultados/`: salidas generadas localmente; Git las ignora y no se
  versionan.
- `TESIS/`: material de la tesis conservado.

Los perfiles, cachés, respaldos, binarios compilados y scripts de ensayo se
eliminaron del árbol de trabajo. La ruta de producción usa Newton amortiguado
tipo Cantera, continuación pseudo-transitoria PTC-SER con rescate BE,
Jacobiano block-tridiagonal y SciPy/LAPACK. En cada paso PTC se hace una sola
corrección lineal y el paso temporal se adapta con la reducción del residual;
si esa corrección falla, el solver vuelve al Euler implícito totalmente
convergido. En el generador FGM, la ruta fría usa por defecto
`block_tridiag + direct` y desactiva los grids fijos intermedios; esto evita
resolver y refinar varias veces la misma llama. `recycled_gmres` sigue
disponible explícitamente para continuaciones donde resulte beneficioso.
La opción `--auto-bootstrap-grids` conserva el bootstrap 12/24/48 para
diagnóstico y reproducibilidad histórica.
Para evitar sobre-hilos en los bloques densos pequeños, V2 fija
`OPENBLAS_NUM_THREADS=1` si el usuario no lo define. En corridas FGM
secuenciales y en V2, el kernel de química usa 4 hilos de Numba por defecto;
puede modificarse con `NUMBA_NUM_THREADS` o, en FGM, con
`--numba-kinetics-threads`.

PTC-SER con rescate BE es la única ruta pseudo-transitoria expuesta. Los
selectores de métodos usados durante la evaluación se retiraron después de
validar la configuración final.

Las variantes medidas y descartadas están registradas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md). No deben volver a
añadirse sin una validación end-to-end de FGM reproducible.

Para barridos FGM repetidos, el generador conserva las semillas V2 aceptadas
en `output-root/_v2_seed_cache` y activa automáticamente procesos paralelos
cuando todas las semillas del barrido ya existen. Es la ruta recomendada para
producción: mantiene la misma malla y criterio de convergencia, y evita pagar
el bootstrap frío en cada regeneración. La comparación estricta fría y sus
limitaciones están documentadas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md).
