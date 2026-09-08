# Soret: trazabilidad del arranque y química en estados de Newton

Fecha: 2026-09-05. Diagnóstico y corrección; no afirmación de superioridad universal.

## Qué hace Cantera 3.2

Fuentes primarias inspeccionadas en la etiqueta `v3.2.0`:

- `interfaces/cython/cantera/_onedim.pyx`: la ruta automática resuelve primero
  con transporte promediado por mezcla, sin Soret; restaura después el modelo
  multicomponente, Soret y las tolerancias solicitadas y vuelve a resolver/refinar.
- `src/transport/MultiTransport.cpp`: conductividad y difusión térmica comparten
  la solución del sistema de transporte; se reutiliza mientras no cambien T/X.
- `src/oneD/MultiNewton.cpp`, `src/numerics/SteadyStateSystem.cpp`: Newton
  amortiguado, Jacobiano reutilizado y pasos Euler implícito de respaldo.
- `include/cantera/kinetics/StoichManager.h`: para reacciones elementales con
  molecularidad de 1 a 3 conserva un factor de concentración negativo; con dos
  o más factores negativos anula el producto. En órdenes generales se anula
  una tasa con un participante no positivo. No se puede describir como una
  extensión polinómica sin restricciones.

URL base de las fuentes: https://github.com/Cantera/cantera/tree/v3.2.0

La observación pública `cantera_trace_20260905_004735.json` midió 34,1450 s de
pared a 10 atm. El último estado estacionario promediado por mezcla se observó
a 28,9736 s; el final multicomponente/Soret, a 34,1430 s. La diferencia incluye
la transición y el control de malla: no es un cronómetro exclusivo del kernel
Soret. Las estadísticas registraron 240 Jacobianos. Sus tiempos internos de
CPU no son una partición del tiempo de pared.

## Fallo localizado

Las trazas de V2 sitúan el estancamiento inicial antes de activar Soret. La
evaluación química recortaba `C_k < 0` a cero, aunque Newton admite fracciones
másicas hasta -1e-7. Para una reacción elemental con un solo participante
negativo, ese recorte elimina la dependencia de la tasa respecto de esa
concentración en ese lado de cero. El Jacobiano pierde el término que debería
participar en la corrección de la especie traza.

Contraejemplo reproducido: H2/aire fresco, 1200 K, 10 atm, h2o2.yaml,
Y_H=-1e-8 sin renormalizar. La diferencia máxima de producción másica era
6,47725 kg/(m3 s). La producción de H era aproximadamente 0,00313 en V2
frente a 0,20717 kg/(m3 s) en Cantera. Los estados físicos positivos no revelaban
este problema.

## Corrección nativa

Se conserva el producto logarítmico de concentraciones positivas. En estados
de prueba negativos se evalúa la magnitud con el valor absoluto y se aplica
el factor -1, 0 o 1 que define la continuación anterior. La concentración
efectiva de tercer cuerpo conserva las concentraciones con signo. No se
consulta Cantera dentro de la química, ni se alteran tolerancias, malla o
predicado de aceptación. Esta continuación no autoriza composiciones finales
no físicas y no modifica la química del interior del dominio positivo.

La corrección cubre las rutas NumPy, Numba densa, Numba dispersa y termoquímica
fusionada. `tests/test_native_kinetics_signed.py` fallaba en las 16 combinaciones
de ruta/mecanismo/presión antes de corregir. Ahora compara fuentes, conservación
de masa y derivada a ambos lados de cero, para h2o2.yaml/GRI30 a 1/10 atm,
incluyendo varias especies negativas y protección ante órdenes fraccionarios.
Las 49 pruebas del repositorio pasan, incluida la coherencia del residual
completo y local congelado en un estado negativo de prueba. No sustituyen
validación de una llama.

## Evidencia de trayectoria (no estadística pareada)

El diagnóstico previo `benchmark_20260905_004310` necesitó 70,3916 s usando
rescate BE tras agotar el presupuesto PTC. La química corregida en
`benchmark_20260905_091426` converge con PTC-SER normal en 27,7425 s;
Cantera en esa pareja tarda 32,1764 s. El arranque intermedio ocupa 22,3339 s
y el corrector final 5,2348 s. Los Jacobianos del arranque se reducen de 1729
a 529 y los del corrector de 32 a 17. Son trayectorias con distinto rescate,
no una ablación aislada que asigne toda la reducción porcentual a un kernel.

La malla base da Su=1,34848446 m/s frente a 1,33287505 m/s (1,1711% de
diferencia). Por tanto, ganar esta pareja de tiempos no basta para una
comparación a error controlado. El refinamiento previo con perfiles propios
redujo la diferencia a 0,0767%; esa prueba era sembrada, no un arranque frío.
La campaña de malla fina en frío se evaluó separadamente como sigue.

## Campañas posteriores a la corrección

GRI30, mezcla estequiométrica, 300 K; sin semillas de perfiles. Se conserva
caché JIT en disco, y la primera ejecución se incluye. Tiempos de resolución,
no de instalación/importación. BLAS=1, Numba=4; cada solver adapta su malla.

| Caso | Pares | Mediana V2 [s] | Mediana Cantera [s] | Reducción |
| --- | ---: | ---: | ---: | ---: |
| H2, Soret, 1 atm; slope=.04/curve=.08 | 3 | 5.19348 | 9.59952 | 45.90% |
| CH4, mezcla promediada sin Soret, 1 atm | 3 | 3.79508 | 18.92336 | 79.94% |
| H2, Soret, 10 atm; slope=.01/curve=.02 | 7 | 45.87342 | 78.66561 | 41.69% |

Directorios respectivos: `benchmark_20260905_091634`,
`benchmark_20260905_091756`, `benchmark_20260905_092520`.
El último tiene 854/868 nodos y dominio final 0.06 m en ambos solvers.
Las 14 soluciones fueron aceptadas y cada solver reprodujo exactamente su
malla y perfiles en todas las repeticiones. Cuartiles V2 [44.89091,49.86478] s;
Cantera [77.73092,79.15299] s. Bootstrap pareado de la razón de medianas,
20000 remuestreos, semilla 20260904: IC95 [0.56495,0.66745]. No se excluyeron
las repeticiones V2 de 52.51/58.10 s ni la de Cantera de 90.77 s.

En la malla fina de 10 atm: Su V2=1.3732138304 frente a 1.3718961788 m/s
(diferencia 0.09605%). E2 de perfiles alineados e integrados espacialmente:
T=0.004055%, calor=0.14283%, máximo de especies activas=0.43261% (NO).
Residual V2 completo=0.35918, sin relajar la guarda de 1e4.
El balance elemental reconstruido de H difiere 0.7145% en V2 y 0.7501% en
Cantera; es un diagnóstico continuo sensible a malla, no una cancelación exacta.

La malla intermedia fría (`benchmark_20260905_091946`, slope=.02/curve=.04)
dio 36.3182/51.3332 s y Su=1.3572549984/1.3604072405 m/s, con 440/455 nodos.
Al refinar de nuevo, las velocidades todavía cambian aproximadamente 1.18%
(V2) y 0.84% (Cantera). Aunque los dos solvers concuerden al 0.096%, esto NO
demuestra independencia de malla al 0.1%. Es necesario un nivel más fino o
extrapolación espacial y ampliar dominio/rango físico antes de afirmar exactitud
general. La significación temporal demostrada se limita a esta configuración.

Evidencia compacta con huellas de código, mecanismo, perfiles y comandos:
`validation/soret_signed_kinetics_20260905.json` (generada por
`validation/summarize_signed_kinetics.py`). Los tres pares de control a 1 atm
son preliminares y no sustituyen siete pares de esa nueva revisión.

## Experimentos que no se promueven

- BE completo: 108,9757 s (`benchmark_20260905_003417`).
- Cambio temprano PTC/BE por estancamiento: 83,3789 s a 10 atm, pero regresión
  CH4/1 atm hasta aproximadamente 14 s frente a aproximadamente 9 s anteriores.
- BE con bloques de 10 pasos: agotó 120 s (`benchmark_20260905_004827`).
- Pesos fijos en el amortiguamiento: 78,3502 s con rescate conservador
  (`benchmark_20260905_005037`), peor que 70,3916 s.

Estas opciones se retiraron del solver y del benchmark durante la limpieza
posterior del 2026-09-05; sus comandos históricos ya no son interfaces vigentes.
Permanece PTC-SER con BE interno tras rechazo. La corrección química es general,
no una regla específica para 10 atm. Véase `V2/PIPELINE.md` para el alcance exacto
de la limpieza y las hipótesis de optimización todavía no implementadas.
