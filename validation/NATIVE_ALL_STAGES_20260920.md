# Independencia nativa por etapas — 20/09/2026

La ruta productiva V2/FGM es nativa por defecto y se ejecutó en un entorno virtual
nuevo **sin Cantera instalado**, bloqueando también la lectura del archivo
histórico `cantera_transport_poly_coeffs.json`. No hay recuperación mediante
Cantera cuando un cálculo nativo falla. Los comparadores explícitos mantienen
su dependencia opcional de referencia.

## Dependencia adicional eliminada

La auditoría del día 19 demostraba independencia de importación, pero el
transporte promediado todavía consumía ajustes exportados de GRI30. Indexarlos
por nombre de especie podía ocultar cambios posteriores de parámetros del
mecanismo. `NativeTransport` ahora construye viscosidad, difusión binaria y
conductividad desde los datos solicitados, sin coeficientes preexportados.

La conductividad utiliza capacidad calorífica NASA, colisiones moleculares,
relajación rotacional e intercambio de energía interna: polinomios de grado
cuatro en log(T), mínimos cuadrados relativos y 50 muestras del intervalo
térmico común. Se corrigió también el truncamiento de las constantes Debye y
permitividad para especies polares; no se relajaron tolerancias de pruebas.
Las ecuaciones se contrastaron con la fuente primaria
[GasTransport.cpp, Cantera 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/GasTransport.cpp).

Los mecanismos, las tablas Monchick--Mason y las formulaciones conservan sus
licencias y atribución. Independencia computacional no significa autoría
exclusiva de esos datos ni ausencia de datos científicos tabulados. El JSON
histórico se conserva como antecedente, pero ya no participa en producción.
El guard de auditoría rechaza abrirlo con rutas de texto o bytes.

## Etapas comprobadas

| Etapa | Implementación y comprobación |
|---|---|
| Lectura e inicialización | YAML local, mezcla fresca y equilibrio HP NASA-7 nativos; backend nativo por defecto |
| Termoquímica y cinética | Kernels nativos; pruebas de estados perturbados y comparación independiente opcional |
| Transporte promediado | Ajustes generados desde parámetros del mecanismo, sin el archivo histórico |
| Multicomponente y Soret | Colisiones y resolución nativas; H2/aire a 10 atm aceptada sin Cantera |
| Newton, PTC, Jacobiano, LU y malla | V2/Numba/SciPy; mismos criterios y tolerancias |
| Continuación y procesos hijos | Cinco CH4/aire aceptadas sin semillas persistentes; cinco regeneradas con dos procesos |
| Bilger e inversión Z→phi | Dos objetivos Z resueltos y aceptados con cálculo nativo |
| Tabulación y exportación | Campos nativos, 53 especies; FlameMaster completo y cinco archivos individuales |
| Diagnósticos y figuras | Calor liberado y flujo elemental nativos; figuras FGM generadas con Cantera bloqueado |

`plot_thesis_results.py` ya no importa Cantera para reconstruir qdot o nombres
de especies. Las figuras comparativas necesitan perfiles de referencia
guardados, y Matplotlib es una dependencia opcional de dibujo.
`analyze_soret_validation.elemental_flux_diagnostics(..., 'native')` tampoco
importa Cantera; su programa principal sigue siendo un comparador explícito.
Las figuras se verificaron con guard en el entorno habitual; el solver,
la tabulación, Soret, la inversión Z y la exportación, en el entorno limpio.

Siguen siendo referencias opcionales `--transport-backend cantera-reference`,
`benchmark_soret_native.py --no-native-only`, `run_saved_comparison.py`, el
generador Cantera y sus diagnósticos dedicados. Los nombres históricos de
criterios o mallas que contienen «cantera» describen código local, no llamadas
a esa biblioteca.

## Optimización añadida

Los parámetros de pares moleculares y los ajustes de conductividad se guardan
en cachés inmutables, limitadas a ocho entradas. Sus claves incluyen todas las
entradas numéricas usadas, incluyendo NASA, geometría y relajación para la
conductividad; no solo nombres de especies. No se almacena aquí un estado de
llama ni se congela una propiedad de estado entre iteraciones.

Las semillas FGM persistentes incluyen ahora SHA-256 del contenido YAML en su
clave. Cambiar el mecanismo en la misma ruta evita reutilizarlo como un acierto
de caché exacto. Las semillas antiguas permanecen, sin borrar; la nueva clave
requiere un primer barrido que regenere las necesarias.

`benchmark_transport_cache.py` compara caché activada frente a recalcular los
mismos datos nativos. Un calentamiento excluido por modo, tres parejas
alternadas por combustible, ambas cachés nuevas vacías al comenzar cada llama.
Los ajustes existentes de viscosidad/difusión y la compilación están calientes.
Incluye construcción y solución, no importaciones ni dibujo; cuatro hilos
Numba y un hilo BLAS configurados antes de importar NumPy.

| Caso a 1 atm | Recalculando | Con caché | Reducción de medianas |
|---|---:|---:|---:|
| CH4, promediado | 3.1828 s | 3.0641 s | 3.7% |
| H2, Soret y bootstrap | 3.3343 s | 3.1239 s | 6.3% |

Las 12 corridas medidas fueron aceptadas; todas las parejas conservaron z, u,
T e Y **bit a bit**. Una pareja H2 fue más lenta con caché: hay variabilidad y
tres parejas no establecen una garantía estadística. Evidencia:
`native_transport_cache_20260920.json`. Los porcentajes **no** comparan con
los polinomios exportados antiguos ni deben sumarse a la mejora previa del
Jacobiano descrita en `OPTIMIZATION_STAGES_20260920.md`.

## Evidencia de independencia y exactitud

Entorno `tmp/native_clean_20260920`: Python 3.11.0, NumPy 2.4.6, SciPy 1.17.1,
Numba 0.65.1, PyYAML 6.0.3; distribución Cantera ausente. El guard se hereda
mediante PYTHONPATH, también en los procesos paralelos. Resumen, diferencias y
hashes de código: `native_all_stages_20260920.json`.

- Suite completa con referencia instalada: 60 tests y 127 subtests correctos;
  cuatro avisos preexistentes sobre `np.row_stack`.
- Entorno limpio con guard: 60 tests descubiertos, 45 ejecutados correctamente
  y 15 comparaciones de referencia omitidas explícitamente. Las pruebas nativas
  de aceptación, PTC, malla, Soret y productos cinéticos no se omiten.
- GRI30 y H2/O2 frente a referencia a 300, 700, 1000, 1800 y 2800 K:
  viscosidad pura/mezcla, conductividad y difusión dentro de rtol=2e-7.
- Barrido sin semillas: phi=[0.7,0.9,1,1.1,1.4], cinco aceptadas, mallas
  [271,254,264,274,259]. Se mantiene continuación interna entre mezclas cercanas.
- Frente al barrido anterior con polinomios exportados: mismas mallas físicas;
  máximo cambio tabulado T=6.43e-8 K, Y=1.50e-11 y Su=2.82e-11 m/s.
  No son resultados bit a bit: se cambiaron los ajustes y constantes.
- Regeneración paralela: cinco aceptadas y dos procesos, mismas cantidades de
  nodos. Corrige otra vez las semillas: no debe afirmarse identidad con el
  barrido frío. Sus tiempos únicos, 14.77 s y 3.12 s, son pruebas funcionales,
  no una medición aislada de aceleración por paralelismo.
- H2/Soret a 10 atm: aceptada, Su=1.3484847717 m/s, 241 nodos,
  error máximo sum(Y)-1=2.88e-12 y desviación relativa de caudal=9.44e-11.
- qdot nativo sobre ese perfil frente a la referencia: error relativo de norma
  L2=4.51e-15, comprobación puntual excluida del tiempo de solve.

El diagnóstico Soret de flujo elemental reconstruido en puntos medios no es
el residual discreto upwind ni una certificación de independencia de malla.
En este perfil, la desviación relativa máxima de H es aproximadamente 1.99%;
queda registrada. Requiere un estudio de refinamiento para una conclusión
física más fuerte. Independencia del software no certifica automáticamente
exactitud de discretización ni mecanismos arbitrarios.

## Reproducir

```powershell
python -m venv tmp/native_check
.\tmp\native_check\Scripts\python.exe -m pip install -r requirements-native.txt
$env:PYTHONPATH=(Join-Path (Get-Location) 'validation/no_cantera')
.\tmp\native_check\Scripts\python.exe -m unittest discover -s tests -v
.\tmp\native_check\Scripts\python.exe FGM/scripts/generate_fgm_tables_native.py --phi-values '0.7,0.9,1,1.1,1.4' --parallel-workers 1 --output-root tmp/native_check_runs --run-name cold --save-raw-profiles
.\tmp\native_check\Scripts\python.exe FGM/scripts/generate_fgm_tables_native.py --phi-values '0.7,0.9,1,1.1,1.4' --parallel-workers 2 --output-root tmp/native_check_runs --run-name parallel --save-raw-profiles
.\tmp\native_check\Scripts\python.exe V2/benchmark_soret_native.py --bootstrap --bootstrap-mesh-factor 2 --pressure-atm 10 --output tmp/native_check_runs/soret10
```

Usar un output-root nuevo para probar ausencia de semillas persistentes.
Para forzar otro barrido sin ellas, añadir `--disable-seed-cache`.
No quitar el guard durante la auditoría. Las comparaciones explícitas se
ejecutan en otra consola sin ese PYTHONPATH y con `requirements-test.txt`.
El job CI nativo instala solo las dependencias científicas y ejecuta la suite
y una llama con ambos bloqueos.
