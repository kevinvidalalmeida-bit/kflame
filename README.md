# TFM - Solver V2 y generación FGM nativos

## Comando principal

Desde la raíz del proyecto, instalar las dependencias nativas y generar una tabla:

```powershell
python -m pip install -r requirements-native.txt
python .\FGM\scripts\generate_fgm_tables_native.py --phi-values 0.7,0.9,1,1.1,1.4 --save-raw-profiles
```

Esta es la ruta por defecto sin Cantera: mezcla, equilibrio HP, solución,
posprocesado, Bilger y tablas usan código nativo. Se incluyen `gri30.yaml` y
`h2o2.yaml`; otros mecanismos compatibles se proporcionan mediante `--mech ruta.yaml`.
Las tablas se guardan en `fgm_runs/run_..._fgm_native/`. Para un barrido frío sin
semillas guardadas, añadir `--disable-seed-cache --parallel-workers 1`.

Para Soret nativo:

```powershell
python .\V2\benchmark_soret_native.py --native-only --bootstrap --bootstrap-mesh-factor 2
```

Los comparadores y diagnósticos de referencia requieren instalar opcionalmente
`requirements-test.txt`. Para comparar ambas implementaciones:

```powershell
python .\V2\run_saved_comparison.py --loglevel 1 --verbose-ours
```

## Salidas

Cada corrida del comparador se guarda en:

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
- `FGM/resultados/`: salidas voluminosas generadas localmente; Git las ignora.
- `evidence/tfm_20260903/`: resúmenes compactos y versionados de la evidencia
  utilizada en la tesis.
- `TESIS/TFM_FGM_FINAL.tex`: manuscrito maestro de la tesis.
- `output/pdf/TFM_FGM_FINAL.pdf`: PDF final verificado.

Los perfiles, cachés, respaldos, binarios compilados y scripts de ensayo se
eliminaron del árbol de trabajo. La ruta de producción usa Newton amortiguado
tipo Cantera, continuación pseudo-transitoria PTC-SER con rescate BE,
Jacobiano block-tridiagonal y una ruta lineal híbrida. SciPy/LAPACK conserva la
factorización pivotada de cada bloque; un kernel Numba fusiona las sustituciones
hacia delante y atrás a lo largo de la malla. En cada paso PTC se hace una sola
corrección lineal y el paso temporal se adapta con la reducción del residual;
si esa corrección falla, el solver vuelve al Euler implícito totalmente
convergido. En el generador FGM, la ruta fría usa por defecto
`block_tridiag + direct` y desactiva los grids fijos intermedios; esto evita
resolver y refinar varias veces la misma llama. La ruta `recycled_gmres` fue
retirada porque el perfil mostró retrocesos frecuentes a LU directa y mayor
tiempo total sin mejorar la solución.
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

La auditoría actual, con un entorno sin Cantera y sin acceso a polinomios
preexportados, está en
[validation/NATIVE_ALL_STAGES_20260920.md](validation/NATIVE_ALL_STAGES_20260920.md).
La auditoría anterior se conserva en
[validation/NATIVE_INDEPENDENCE_20260919.md](validation/NATIVE_INDEPENDENCE_20260919.md).
Los nombres históricos `acceptance_criterion="cantera"`, `cantera_seed_grid` y
`cantera_local` describen algoritmos implementados en V2; no cargan la librería.
La opción `--transport-backend cantera-reference` sí la carga explícitamente.
Viscosidad, conductividad y difusión se ajustan nativamente desde los datos del
mecanismo y las tablas moleculares de colisión; ya no se lee el archivo histórico
`cantera_transport_poly_coeffs.json`. Los ajustes y parámetros moleculares
inmutables se reutilizan automáticamente entre mallas con claves basadas en sus
entradas numéricas. Las tablas de colisión, los mecanismos y las formulaciones
conservan su procedencia y atribución. Independencia de ejecución y generación
de ajustes no significa que los datos científicos sean de autoría exclusiva.
Las figuras requieren Matplotlib adicional, pero no Cantera; las figuras
comparativas sí necesitan perfiles de referencia previamente guardados.

La interpretación científica de los resultados, las pruebas todavía necesarias
para un artículo y las extensiones matemáticas candidatas se mantienen en
[`PAPER_ROADMAP.md`](PAPER_ROADMAP.md).

La continuación FGM trata cada perfil previo como una aproximación local: lo
reutiliza solo cuando la razón entre valores consecutivos de `phi` está entre
`1/1.15` y `1.15`. Los saltos mayores reinician desde el arranque robusto y
anulan también la secante anterior. Esto evita que una semilla lejana fuerce
una trayectoria no lineal con una malla adaptativa innecesariamente densa.

Para barridos FGM repetidos, el generador conserva las semillas V2 aceptadas
en `output-root/_v2_seed_cache` y activa automáticamente procesos paralelos
cuando todas las semillas del barrido ya existen. Es la ruta recomendada para
producción: mantiene la misma malla y criterio de convergencia, y evita pagar
el bootstrap frío en cada regeneración. La clave de las semillas incluye el
contenido SHA-256 del mecanismo, no solo su ruta. Las semillas antiguas se
conservan, pero esta actualización genera una clave nueva y exige un primer
barrido sin ellas. La comparación estricta fría y sus
limitaciones están documentadas en
[`DECISIONES_DESCARTADAS.md`](DECISIONES_DESCARTADAS.md).
