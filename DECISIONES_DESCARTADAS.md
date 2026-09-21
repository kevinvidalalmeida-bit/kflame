# Decisiones descartadas de optimización

Nota de organización (20/09/2026): este historial se conserva. La ruta actual
es el paquete `src/kflame/`; los informes están en `docs/validation/` y las
instrucciones actuales en `docs/migration.md`. Las rutas de campañas y código
citadas dentro de las entradas históricas corresponden a sus instantáneas.

Este documento conserva los resultados negativos para no reintroducir rutas
que ya se midieron. Las referencias de tiempo corresponden a GRI-Mech 3.0,
CH4/aire, 300 K, 1 atm y transporte mixture-averaged.

**Selección actual tras la limpieza solicitada:** los candidatos de las dos
últimas rondas de arranque frío quedan archivados, también los inconcluyentes
(`dependencies` y `compiled-lu`). Las menciones históricas a futuras
confirmaciones no los mantienen activos. El benchmark usa solo `baseline`
por defecto; otras variantes requieren `--reproduce-archived`. Se conservan
fuentes y mediciones como evidencia. Véase `validation/NEXT_IMPROVEMENTS.md`.
Las condiciones distintas del caso histórico anterior (H2, Soret, 10 atm)
se indican expresamente en cada campaña.

## No reincorporar

| Variante | Evidencia | Decisión |
| --- | --- | --- |
| DLL C++ block-Thomas | El backsolve aislado fue más rápido, pero el FGM completo solo redujo un caso sembrado de 20.05 s a 18.51 s. La LU C++ completa fue 22 % más lenta que LAPACK y dio pasos lineales inestables para el Jacobiano rígido. | Eliminada. Usar SciPy/LAPACK. |
| Block-Thomas Numba | En continuación certificada tardó 26.55 s frente a 25.07 s con SciPy/LAPACK y terminó en otra malla. | Eliminado. |
| Jacobiano químico analítico experimental | El microbenchmark redujo el ensamblado, pero una continuación real requirió 39.35 s frente a 20.77 s de diferencias finitas. | Eliminado. |
| BDF2 / BE-BDF2 pseudo-transitorio | La llama fría completa tomó 64.68 s frente a 57.79 s con Euler implícito (BE). | Eliminado; no confundir con PTC-SER, que mantiene el desplazamiento BE pero evita converger cada subproblema transitorio. |
| Damping Armijo | Mejoró un bootstrap aislado, pero la llama completa subió a 60.30 s frente a 57.79 s. | Eliminado; se mantiene el damping estilo Cantera. |
| Vida fija de Jacobiano = 80 | Phi=1.1 desde semilla fue rápido, pero phi=1->1.3 y la cadena hasta phi=1.3 no convergieron en 60 s. | No usar globalmente; FGM usa edad 20 y la ruta histórica conserva 40. |
| GPU/CuPy | Las transferencias CPU-GPU superaron el ahorro: Jacobiano CPU 0.153 s frente a GPU 0.219 s. | Eliminado; KFLAME usa solo CPU/Numba. No mantener una dependencia ni un backend GPU experimental. |
| Reaplicar isoterma persistente de anclaje | El anclaje actual ya conserva T=781.381 K. Forzarlo de nuevo dio la misma malla, 136 pasos y el mismo residual, con 37.18 s frente a 36.43 s. | Sin cambio; no acelera. |
| Detector por residual precondicionado GMRES | Tras cuatro iteraciones, el residual precondicionado era bajo incluso en llamadas que fallaban la convergencia real; no activo ningun corte util. | Eliminado; usar una sonda fija de 4 iteraciones. |
| Tangente con geometria termica secante | En pasos de 2 % en \(\log\phi\), la segunda transicion empeoro de 0.94 s a 1.18 s a 1 atm. A 10 atm bajo de 2.65 s a 2.57 s en una sola corrida, pero termino con \(\|F\|_\infty=9.00\times10^3\), frente a \(1.38\times10^3\) para el tangente crudo. No hay mejora general ni margen suficiente frente a la guarda de \(10^4\). | Retirado; conservar el tangente crudo y registrar el resultado negativo. |
| Transferencia de semilla por longitud de arco ponderada | Con 64 nodos, concentrar la semilla por la longitud de arco de \((u,T,Y_k)\) produjo dos rechazos del corrector a 1 atm, expansiones de dominio y reinicios fríos. El barrido completo \(\phi=1,1.02,1.04\) subió a 31.16 s de solve, frente a 12.80 s con el tangente y transferencia por índice. | Eliminada; la geometría adaptativa heredada por índice conserva mejor la cobertura asintótica. |
| Predictor tangente con curvatura cuadrática | Usa el tangente de Jacobiano en \(\log\phi\) y el flamelet anterior para estimar \(x''\). Fue neutro a 1 atm (0.82 s frente a 0.94 s en la segunda transición, dentro de variación), pero a 10 atm aumentó la segunda transición de 2.65 s a 4.90 s: 25 Jacobianos, 122 LU y 144 pruebas de damping, frente a 11, 35 y 48 con el tangente de primer orden. | Eliminado; el perfil tiene curvatura numérica/espacial suficiente para que la extrapolación de segundo orden empeore la trayectoria. |
| Inicializador 0D con reactor piloto | Se mantuvieron ecuaciones, malla, certificado tipo Cantera y guarda \(\|F\|_\infty\le10^4\). La trayectoria homogénea se inició en el interior de la capa térmica y se usó solo como semilla. A 1 atm el solve frío subió de 8.00 s a 8.22 s; a 10 atm, de 20.70 s a 23.79 s. Las soluciones finales fueron iguales dentro de redondeo, pero a 10 atm el predictor indujo 445 Jacobianos y 2351 LU, frente a 357 y 1638 en el baseline. | Eliminado. Una trayectoria 0D no reproduce el equilibrio difusión--química del frente y empeora la cuenca de Newton aun cuando sea físicamente admisible como semilla. |
| Recorrido FGM bidireccional desde \(\phi\simeq1\) | En el barrido estricto \(\phi=[0.7,0.9,1.0,1.1,1.4]\), el ancla fría \(\phi=1\) y dos ramas con tangente del Jacobiano certificaron las nueve filas finales (cinco solicitadas y cuatro puentes). Sin embargo, la rama rica sufrió tres rechazos antes de un reinicio frío: 12 intentos y 56.55 s de solve intentado (37.31 s solo en filas aceptadas), frente a 21.96 s para la continuación fija con el mismo predictor y certificado. | Eliminado. Un ancla central no compensa los puentes y reintentos cuando la región rica presenta mayor curvatura; conservar el orden solicitado con región de confianza fija. |
| Condensación exacta de \(u_j\) en un flujo másico global | En el Jacobiano físico de una llama certificada de 261 nodos, la transformación \([u,T,Y]\mapsto[\dot m,T,Y_{1:K-1}]\) reprodujo el producto lineal con error relativo \(3.24\times10^{-16}\) y redujo el bloque de 55 a 53 variables. Sin embargo, la LU por bloques del sistema bordeado dejó residual lineal \(2.95\times10^{-1}\) para RHS de prueba (\(1.24\) tras escalado); SuperLU pivotado lo redujo a \(6.0\times10^{-3}\), pero tardó 0.069 s frente a 0.041 s de la ruta block-LU completa. | No integrar. La reducción es algebraicamente válida, pero exige una estrategia lineal más robusta y no gana tiempo en el tamaño actual. |
| Schur exacto de segmento para refresco local | Se reutilizó la LU exacta del prefijo, se eliminó el sufijo desde la salida y se refactorizó solo el intervalo de bloques refrescados. La prueba algebraica coincidió con una matriz densa. En la transición certificada a 10 atm \(\phi=1\to1.02\to1.04\), conservó exactamente perfiles, malla, residual y \(\Su\), pero subió los tiempos a 1.073 y 1.306 s, frente a 1.004 y 1.204 s con la LU completa. El factor central costó 0.0139--0.0145 s y la preparación/solves de interfaz anularon el ahorro. | Eliminado. El refresco local actual se activa una vez por transición; no hay suficientes actualizaciones sobre la misma matriz para amortizar el Schur de dos lados. |
| Pseudo-arclength bordeado directo | En CH4/aire, \(\phi:1.00\to1.02\), el corrector aumentado resolvió una semilla puente y luego se certificaron el puente y el objetivo con KFLAME normal. A 1 atm el trayecto fue 10.46 s frente a 0.961 s del tangente directo; a 10 atm, 13.30 s frente a 1.110 s. Las dos rutas certificaron todas las llamas. Las diferencias finales de \(\Su\) fueron \(3.87\times10^{-5}\) (1 atm) y \(1.99\times10^{-5}\) (10 atm) relativas; proceden de trayectorias certificadas distintas bajo la misma guarda operacional, no de una relajación de criterios. | Implementación retirada para ramas suaves de FGM. La evidencia se conserva: el sistema aumentado disperso es el tratamiento matemáticamente correcto si \(J\) se vuelve singular cerca de un pliegue; el atajo Schur \(J^{-1}\) no lo es. Reabrir únicamente con una familia que presente una rama o pliegue físico real. |

| Escalado físico bilateral del sistema lineal | La ecuación de Newton se equilibró exactamente como \(RJD_x\), con las escalas de la norma ponderada. Conservó \(\Su\), malla y certificado, pero la llama fría pasó de 9.55 a 10.32 s a 1 atm y de 27.60 a 27.78 s a 10 atm. | Eliminado. La condición algebraica no era el cuello dominante; el coste de escalar y la trayectoria no lineal superan cualquier beneficio numérico. |
| Inicializador de frente difusión--reacción | Se construyó una semilla \(\tanh\) con \(\delta=\sqrt{\alpha_T\tau_q}\) y \(S_0=\sqrt{\alpha_T/\tau_q}\), donde \(\tau_q\) procede de la liberación de calor local, sin integrar un reactor 0D. A 10 atm estimó \(S_0=0.00206\) m/s y \(\delta=1.09\) mm; el corrector terminó en 17.86 s, pero no certificó la llama, frente a 27.60 s y certificación del arranque actual. | Eliminado. Una tasa local de mezcla fresca a temperatura elevada no representa el tiempo efectivo de propagación; no debe usarse como inicializador general. |
| Continuación de presión en \(\log p\) | Los puentes 1, 1.5, 2.25, 3.375, 5 y 7.5 atm se certificaron con copia/secante y resolvieron 5 atm en 8.11 s. Sin embargo, el predictor 7.5\(\to\)10 atm fue rechazado; el arranque frío de respaldo produjo 10 atm en 45.82 s y el intento completo costó 55.32 s. | No promocionar como ruta general. Es útil solo en el intervalo donde el corrector acepta; no mejora de forma robusta el extremo de alta presión. |
| Relajación de malla basada en error físico | Frente a la malla estricta de 288 nodos a 10 atm, \((\mathrm{slope},\mathrm{curve},\mathrm{prune})=(0.05,0.10,0.005)\) bajó a 227 nodos pero dio \(\Delta\Su=0.321\,\%\) y \(E_2(Y)=2.22\,\%\). La versión moderada \((0.045,0.09,0.004)\) produjo 263 nodos y errores \(E_2(T)=0.060\,\%\), \(E_2(Y)=0.771\,\%\), pero \(\Delta\Su=0.105\,\%\) y 32.31 s frente a 28.89 s de la estricta. | Eliminada. Menos nodos no compensan la trayectoria más larga y la versión amplia incumple la puerta de fidelidad. |
| AD directa con JAX sobre el backend nativo | JAX está disponible, pero el kernel actual Numba/NumPy exige escalares concretos: `jax.jacfwd` sobre `eval_node_thermo_kinetics_into` termina con `ConcretizationTypeError`. Una ruta AD requeriría reimplementar termoquímica, cinética, transporte y residual en JAX; no sería una derivada del solver productivo. El Jacobiano químico analítico parcial ya fue más lento en una continuación completa. | No integrar una segunda implementación AD. Reabrir solo si se adopta un generador de kernels que produzca las mismas ecuaciones y pase pruebas direccionales antes de un benchmark end-to-end. |

## Ruta de producción retenida

- Residual y Jacobiano local por diferencias finitas con backend nativo CPU y
  Numba para la termoquímica/cinética repetitiva.
- Precomputación vectorizada de termoquímica para las perturbaciones del
  Jacobiano: en pares alternados a 10 atm, mantenerla dio una mediana de
  28.61 s frente a 29.12 s sin ella, con la misma llama certificada. Esta
  agrupación ya explota la independencia espacial; añadir un coloreado externo
  no reduce más las evaluaciones químicas del kernel actual.
- Jacobiano block-tridiagonal y factorización/solves SciPy-LAPACK.
- Factorización LU exacta después de cada actualización pseudo-transitoria,
  igual que la política de referencia de Cantera; la reutilización aproximada
  de LU se retiró al no aportar una mejora reproducible.
- Newton amortiguado compatible con Cantera y PTC linealmente implícito con
  SER; regreso automático al pseudo-transitorio BE totalmente convergido si
  una corrección PTC es rechazada.
- Continuación por perfiles y predictor secante; `max_jac_age=20` para FGM
  frío y 40 solo en la ruta histórica de semillas convergidas.
- Refresco local del Jacobiano en el corrector FGM: solo después de que el
  damping normal no contraiga, con selección por defecto de linealización,
  vecinos espaciales, nueva LU exacta y el certificado sin relajar. La ruta
  fría y la etapa de malla estricta permanecen sin este rescate.
- `OPENBLAS_NUM_THREADS=1` para los bloques densos pequeños.
- Arranque de 12 nodos como opción validada para la FGM fría; conservar 8
  cuando se requiera reproducibilidad histórica.

Una nueva variante solo debe añadirse de nuevo con una comparación end-to-end
en un barrido FGM, misma malla/criterio de aceptación y mejora reproducible.

## Ablación certificada y promovida

| Variante | Evidencia inicial | Estado y siguiente prueba |
| --- | --- | --- |
| Refresco local certificado por defecto de linealización | Ante una prueba de damping no contractiva, mide \(F(x+\alpha s)-F(x)-\alpha Js\), actualiza solo los bloques y vecinos cuyo defecto relativo es alto, refactoriza una LU exacta y exige el mismo test de contracción. En CH4/aire, \(\phi=[1,1.02,1.04]\), 10 atm, siete pares alternados dieron una mediana de \(2.023\times\) en las transiciones, con IC bootstrap pareado 95\,\% \([1.943,2.056]\). Cada pareja conservó certificación; se actualizaron 79 bloques en dos eventos por barrido. Los errores máximos frente al baseline fueron \(\Delta \Su=1.36\times10^{-8}\), \(E_2(T)=2.10\times10^{-9}\), \(E_2(Y)=3.94\times10^{-8}\) y \(E_2(\dot q)=6.23\times10^{-8}\). A 1 atm no hubo activación y los perfiles fueron idénticos. | Promovido como opción por defecto del generador FGM, limitada al corrector de continuación y sin exportar una LU quasi-Newton como tangente exacta. `--no-local-jacobian-refresh` conserva el baseline para futuras ablaciones. |

## Seleccion vigente para FGM

Barrido con GRI-Mech 3.0, CH4/aire, 300 K, 1 atm, phi = 0.9, 1.0 y 1.1;
los dos ultimos flamelets usaron el perfil y la malla del anterior.

| Ruta | Tiempo total de los 3 flamelets |
| --- | ---: |
| numba_local + LU directa | 27.95 s |
| numba_local + LU reciclada/GMRES | 28.30 s |
| block_tridiag + LU reciclada/GMRES | 19.65 s |
| Cantera | 17.24 s |

Ese cuadro conserva la comparacion historica entre backends KFLAME. La sonda GMRES
de cuatro iteraciones redujo aquella corrida KFLAME de 20.40 s a 17.83 s frente a
un limite de doce iteraciones, sin cambio material de la solucion. La nueva
ruta fria validada mas abajo usa LU directa: en el primer flamelet las sondas
GMRES fallaban y despues se reconstruia la LU exacta, por lo que el coste de la
sonda no se amortizaba.

Esta seleccion es de rendimiento entre backends KFLAME. La velocidad de llama KFLAME
difirio de Cantera entre 3.7 % y 4.7 % en este mallado, por lo que no constituye
todavia una certificacion fisica de la tabla FGM.

## Auditoria KFLAME y optimizaciones validadas (2026-08-10)

Primero se reviso el contenido historico de `RESPALDOS/KFLAME.rar`. El respaldo
contiene el solver Newton/transitorio, residual, Jacobiano, mallado, backend de
especies, comparacion con Cantera y una referencia `freeflame_ref.npz`; no
aparecio un benchmark adicional que cambiara las decisiones anteriores.

Las mejoras seguras incorporadas son limitar OpenBLAS a un hilo y limitar
Numba a cuatro hilos por defecto para el kernel sobre mallas pequenas. Tambien
se eliminaron copias temporales de rho/cp/lambda y propiedades en cada
residual, sin cambiar las ecuaciones. En la comparacion oficial final, con el
mismo caso CH4/aire phi=1, se obtuvo:

| Ruta | Tiempo | Su | Nodos |
| --- | ---: | ---: | ---: |
| Cantera | 32.35 s | 0.418984 | 34 |
| KFLAME | 19.45 s | 0.425526 | 36 |

Esto equivale a 1.66x para KFLAME, con una diferencia relativa de Su de 1.56 %,
el mismo residual final y la misma malla que la referencia. El reporte
completo queda en `KFLAME/comparison_runs/run_20260809_235927/summary.json`.

Para el generador FGM, un benchmark corto de tres phi mostro que la carga
secuencial pequena es mas rapida con cuatro hilos de Numba que con el reparto
automatico actual: 24 hilos = 20.1 s, 1 = 17.7 s, 4 = 13.2 s y 8 = 14.3 s.
Por eso el valor automatico secuencial pasa a 4; el argumento
`--numba-kinetics-threads` conserva el control manual y el modo paralelo sigue
repartiendo hilos entre procesos.

También se probaron dos rutas experimentales nuevas y quedaron descartadas:

- damping basado en reduccion del residual: mas lento en la llama completa y
  en el mini-barrido FGM;
- salto adaptativo de GMRES hacia LU exacta: cambio la trayectoria de Newton,
  aumento las reconstrucciones y fue mas lento.

Sus selectores públicos se retiraron del código de producción. El chequeo de
reducción del residual se conserva únicamente dentro de la corrección
PTC-SER, donde evita una segunda resolución lineal y forma parte del método
validado.

## Comparacion estricta FGM: misma malla y convergencia (2026-08-10)

Se repitio un barrido GRI30, CH4/aire, phi = 0.9, 1.0 y 1.1 con width =
0.03 m, transporte mixture-averaged, los mismos criterios base
(ratio=3, slope=0.08, curve=0.12, prune=0.01) y el mismo refinamiento tight
(ratio=2.5, slope=0.04, curve=0.08, prune=0.003). En ambos casos se exigio
convergencia de malla y se uso el criterio de aceptacion cantera.

| Ruta | Total | Tiempos por phi | Nodos finales |
| --- | ---: | --- | --- |
| Cantera | 18.72 s | 10.24 / 2.95 / 4.63 s | 261 / 267 / 280 |
| KFLAME frio, sin cache | 50.82 s | 42.65 / 2.22 / 4.83 s | 267 / 276 / 291 |

Las velocidades de llama de KFLAME fueron 0.338924, 0.378937 y 0.382446 m/s,
frente a 0.338613, 0.378519 y 0.381789 m/s de Cantera; la diferencia maxima
relativa fue aproximadamente 0.17 %. Los tres flamelets KFLAME fueron aceptados,
con residual bruto inferior a 1e4 y norma ponderada del paso inferior a 1.

La causa de la diferencia temporal esta localizada en el primer flamelet
frio: una vez disponible una semilla KFLAME convergida, resolver en la malla final
de Cantera tomo 1.53 s en total para los tres casos. Esto es una medicion del
solver sobre una semilla ya certificada, no una comparacion end-to-end.

La ruta de produccion para barridos repetidos es la cache persistente
`output-root/_kflame_seed_cache` y el paralelismo automatico. Con las mismas
opciones estrictas, despues de generar las tres semillas KFLAME, el barrido
paralelo de tres procesos tomo 6.1 s, con los mismos nodos y aceptacion. Ese
tiempo no debe presentarse como una comparacion fria: representa el escenario
real de regenerar o explorar una tabla FGM despues del primer barrido.

Tambien se probaron 12 y 24 nodos iniciales. En una llama aislada 12 nodos
parecio reducir el tiempo, pero en el barrido completo subio a 58.4 s; 24
nodos tambien empeoro el arranque. Se conserva 8 como default para no cambiar
la trayectoria validada. No se agrega ninguna semilla de Cantera al cache KFLAME,
porque eso ocultaria el coste real del arranque y no seria una comparacion
honesta de los solvers.

## Diagnósticos de arranque Soret (2026-09-04)

Se probó sustituir la semilla tanh por una rampa con mesetas exactamente
constantes, usando H2/aire, h2o2.yaml, 10 atm, multicomponente/Soret y arranque
intermedio nativo. No resolvió el fallo: 34.53 s, salida rechazada y dominio
15.36 m; el diagnóstico tanh previo también fue rechazado (33.81 s, 30.72 m).
El código de la rampa se retiró. Datos: `benchmark_20260904_194655` y
`benchmark_20260904_182840`, bajo `resultados/soret_native_validation`.

Se comprobó además que el valor efectivo de la velocidad inicial ya era
1 m/s, aunque la función auxiliar declara 0.3 m/s por defecto. El ensayo
`benchmark_20260904_182626` repitió esa configuración; no constituye una
estrategia distinta ni una mejora. La velocidad se determina finalmente como
autovalor, no se prescribe en la llama con energía resuelta.

La revisión posterior corrigió dos problemas de control: comprobar el dominio
con temperatura prescrita y tratar la restauración de una malla anterior como
convergencia de refinamiento. Estas correcciones no rebajan el certificado ni
se justifican como aceleraciones sin comparación adicional.

## Optimizacion del arranque frio KFLAME (2026-08-10)

El perfil del arranque mostro que el bootstrap anterior refinaba cada malla
intermedia y repetia el ciclo solve/refine. Se cambio la ruta de produccion a
refinamiento adaptativo directo desde la malla inicial de 8 nodos y LU directa
para BE. Los grids 12/24/48 permanecen como opciones de diagnostico; la ruta
`recycled_gmres` se retiro despues del perfil final descrito abajo.

Con el mismo caso y criterios estrictos de la tabla anterior, el comando real
del generador obtuvo:

| Ruta | Tiempo de flamelets | Nodos | Su [m/s] |
| --- | ---: | --- | --- |
| Cantera | 18.72 s | 261 / 267 / 280 | 0.338613 / 0.378519 / 0.381789 |
| KFLAME optimizado frio | 18.52 s | 247 / 256 / 268 | 0.338981 / 0.378772 / 0.381947 |

Los tres flamelets KFLAME fueron aceptados con convergencia de malla, residual
guardado y norma ponderada dentro de los limites. El runtime end-to-end de
construccion de la tabla fue 19.7 s, incluyendo la interpolacion FGM. La
diferencia de Su maxima fue aproximadamente 0.11 %.

La misma configuracion con las tres semillas persistentes y paralelismo
automatico termino en 5.8 s. Por tanto, KFLAME queda practicamente equiparado a
Cantera en la primera generacion y claramente por delante en regeneraciones
repetidas, sin usar perfiles de Cantera como semillas.

## Experimentos HPC adicionales descartados (2026-08-10)

Se repitio el barrido estricto de tres flamelets con la configuracion actual
(GRI30, CH4/aire, 300 K, 1 atm, width=0.03 m, malla inicial de 8 puntos,
convergencia de malla obligatoria y criterio `cantera`). El baseline de esta
sesion fue 20.62 s de flamelets y 21.9 s end-to-end; la variacion respecto a
la medicion anterior se debe al arranque/JIT y a la carga de la maquina.

| Experimento | Tiempo end-to-end | Resultado | Decision |
| --- | ---: | --- | --- |
| `parallel-cold`, 3 procesos, 4 hilos/proceso | 42.1 s | 3/3 aceptados | Eliminar |
| `parallel-cold`, 3 procesos, 1 hilo/proceso | 50.1 s | 3/3 aceptados | Eliminar |
| Sin precomputacion termoquimica del Jacobiano | 25.3 s | 3/3 aceptados | Mantener precomputacion |
| `recycled_gmres` en BE | 26.2 s | 3/3 aceptados | Mantener LU directa |
| `banded_lapack` | 48.9 s | 3/3 aceptados | Mantener `block_tridiag` |
| BDF2 adaptativo experimental | 28.2 s | 3/3 aceptados | Eliminar |
| Reutilizacion de buffers del residual | 27.0 s | 3/3 aceptados | Eliminar |

Todos los experimentos descartados fueron retirados del codigo despues de la
prueba. Ninguno se activa por defecto ni debe reintroducirse sin una nueva
comparacion end-to-end.

La afinacion del camino cacheado se comparo con una instantanea identica de
las tres semillas KFLAME: 2 workers terminaron en 5.8 s, 3 workers con 4 hilos
Numba por proceso en 4.1 s y 3 workers con 1 hilo en 4.4 s. Se conserva la
configuracion automatica actual: para tres flamelets resuelve con 3 workers y
reparte 4 hilos de Numba por proceso en esta maquina.

## PTC linealmente implícito con SER (2026-08-26)

El cuello matemático del fallback anterior era resolver por Newton hasta
convergencia cada subproblema de Euler implícito. La ruta nueva aplica una
sola corrección pseudo-transitoria

```text
(J_F(x_n) - M/dt_n) s_n = -F(x_n),    x_(n+1) = x_n + s_n
```

y actualiza el paso con switched evolution relaxation (SER):

```text
dt_(n+1) = clip(1.1 dt_n ||F(x_n)|| / ||F(x_(n+1))||).
```

La ruta de producción intenta primero PTC-SER y usa el BE totalmente
implícito anterior después de una corrección rechazada. Es la única ruta
expuesta; los selectores experimentales se retiraron después de validar el
resultado final.

Con GRI30, CH4/aire, 300 K, 1 atm, malla inicial de 8 puntos, refinamiento
estricto, convergencia de malla obligatoria, criterio `cantera` y caché
desactivada, se obtuvo:

| Barrido | Modo anterior | PTC-SER final | Reducción end-to-end | Resultado |
| --- | ---: | ---: | ---: | --- |
| phi = 0.9, 1.0, 1.1 | 22.54 s | 12.89 s | 42.8 % | 3/3 aceptados; 247 / 256 / 268 nodos |
| phi = 0.7, 0.875, 1.05, 1.225, 1.4 | 212.7 s | 139.3 s | 34.5 % | 5/5 aceptados; mallas idénticas |

En el barrido de tres llamas, el tiempo exclusivo de solver bajó de 21.41 a
11.75 s (45.1 %). Frente al registro estricto de Cantera de 18.72 s, KFLAME tarda
37.2 % menos. La diferencia máxima PTC-SER frente al modo anterior fue
6.32e-10 m/s en `Su`, 0.564 K en temperatura y 0.117 % del pico global de
liberación de calor. En el barrido amplio fue 1.52e-10 m/s, 0.533 K y 0.098 %,
respectivamente; coinciden aceptación, forma de las tablas y número de nodos.

El comparador principal KFLAME también conservó 36 nodos, `Su=0.425526 m/s` y
residual final 4.23: bajó de 17.37 s con la ruta anterior a 12.64 s con
PTC-SER (27.2 %). En esa corrida Cantera tomó 34.62 s, por lo que KFLAME fue
2.74x más rápido.

La formulación sigue la continuación pseudo-transitoria linealmente implícita
y el control SER descritos por Kelley y Keyes, y coincide con la estructura
de un paso usada por `TSPSEUDO` de PETSc:

- https://repository.lib.ncsu.edu/items/222848f9-65e0-4e5e-9a72-ee1d96d75857
- https://petsc.org/release/src/ts/impls/pseudo/posindep.c.html

## Limpieza y consolidación final (2026-08-26)

Se retiraron del repositorio 140 archivos de corridas FGM antiguas (60.14
MiB). `FGM/resultados/`, `KFLAME/comparison_runs/` y los cachés de Python/Numba
quedan ignorados y se regeneran localmente; la tesis y los mecanismos se
conservan.

Los dos generadores FGM compartían diez funciones idénticas de parseo,
fracción de mezcla, variable de progreso e interpolación adaptativa. Ahora
viven una sola vez en `FGM/scripts/fgm_common.py`. También se retiraron el
plumbing BDF2 ya descartado, helpers sin llamadas, imports muertos, marcas de
archivos fusionados y el selector público del damping experimental.

La inversión Bilger `Z -> phi` ahora reutiliza una sola fase de Cantera en vez
de recargar el mecanismo en cada iteración. Para tres objetivos bajó de 5.740
s a 0.0145 s (396x), con diferencia máxima de `phi` igual a cero. Esta mejora
afecta al preprocesamiento de tablas `Z-grid`, no al tiempo físico de resolver
cada llama.

La validación estricta final de KFLAME para phi = 0.9, 1.0 y 1.1 aceptó los tres
flamelets con 247 / 256 / 268 nodos y `Su` = 0.338981 / 0.378772 / 0.381947
m/s. Todos los arreglos de la tabla fueron idénticos bit a bit a la referencia
anterior, salvo `solve_time`. La corrida caliente tomó 13.4 s end-to-end; la
primera corrida después de borrar el caché tomó 26.1 s por la recompilación
única de Numba.

La comparación principal final obtuvo 34.84 s para Cantera y 12.41 s para KFLAME
(2.81x), con KFLAME convergido en 36 nodos, `Su=0.425526 m/s` y residual final
4.23. El generador Cantera también pasó una prueba completa en modo `Z-grid`.

## Perfil matematico y continuacion local (2026-08-26)

El perfil del comparador principal identifico un camino lineal no rentable:
`recycled_gmres` ejecuto 873 sondas y 801 terminaron reconstruyendo la LU
exacta (91.8 % de fallback). En corridas alternadas sobre el mismo problema,
LU directa promedio 7.86 s frente a 10.75 s de GMRES reciclado, una reduccion
de 26.9 %, con la misma malla y diferencias maximas de 1.22e-11 K en `T` y
2.92e-14 m/s en `Su`. Por eso se retiro por completo el solver, sus opciones,
estadisticas e imports; la ruta de produccion es solo block-tridiagonal directa.

Tambien se corrigio el paso PTC linealmente implicito. Cuando una correccion
no reducia el residual, el control podia caer por error en la rama Newton y
resolver un segundo sistema con la misma factorizacion. Ahora reduce `alpha`
y prueba el mismo paso; asi cada iteracion PTC hace realmente una sola
correccion lineal, como exige la formulacion documentada.

El perfil instrumentado final del caso principal fue:

| Componente | Tiempo | Llamadas |
| --- | ---: | ---: |
| Construccion de Jacobiano | 2.408 s | 224 |
| Residual completo | 1.815 s | 6486 |
| Damping de Newton/PTC | 1.742 s | 1558 |
| Factorizacion lineal | 1.506 s | 1094 |
| Solve lineal | 0.959 s | 4459 |

El total KFLAME fue 7.33 s, con 36 nodos, `Su=0.4255264047 m/s` y residual final
4.23; Cantera tomo 32.99 s en la misma corrida (4.50x). La suma de los
componentes no debe interpretarse como particion exclusiva porque varias
regiones perfiladas estan anidadas.

Se midieron, pero no se conservaron, las siguientes variantes matematicas:

| Variante | Resultado | Decision |
| --- | --- | --- |
| Newton-Krylov inexacto con forcing 0.1--0.5 | Menos LU, pero 6.90--8.87 s y sin mejora repetible frente a directa | Retirar |
| Mantener `dt` SER 2--4 pasos para reutilizar LU | 11.10 / 15.48 s; intervalo 4 no convergio | Retirar |
| Edad maxima de Jacobiano 30--60 | 7.71--8.68 s frente a 6.78 s con 20 | Conservar 20 |
| Incremento SER 1.3--1.5 | Rapido en una llama, pero 16.22--17.94 s en FGM frente a 11.84 s con 1.1 | Conservar 1.1 |

La mejora adicional vino de explotar que la continuacion en `phi` es local.
En el barrido estricto `phi = 0.7, 0.9, 1.0, 1.1, 1.4`, continuar sin limite
hizo que los saltos 0.7->0.9 y 1.1->1.4 tardaran 84.53 y 61.38 s y crecieran
las mallas a 320 y 381 nodos. Se incorporo una region de confianza
multiplicativa del 15 % tanto para reutilizar el perfil como para mantener el
historial secante. Fuera de ella se usa el arranque frio robusto.

Con esa regla, el barrido KFLAME bajo de 163.7 a 36.4 s end-to-end (77.8 %, 4.50x),
acepto 5/5 flamelets y uso 266 / 247 / 256 / 268 / 247 nodos. Cantera, con los
mismos valores de `phi` y criterios de refinamiento, tomo 93.9 s end-to-end;
la suma exclusiva de solves fue 35.38 s para KFLAME y 93.23 s para Cantera (KFLAME
2.64x mas rapido). Cada solver conservo su propia malla adaptativa; la
diferencia maxima de `Su` fue 0.75 % y ocurrio en el extremo `phi=1.4`.

## 2026-09-05: diagnóstico Soret y química en estados de Newton

No se promueven las copias aisladas de política de Cantera: BE completo
(108,98 s), BE con bloques de 10 pasos (timeout 120 s), pesos fijos del
amortiguamiento (78,35 s frente a 70,39 s con rescate conservador) y cambio
temprano PTC/BE (regresión en CH4 a 1 atm). Los modos diagnósticos permanecen
desactivados por defecto para reproducir el experimento; no son optimizaciones
productivas ni se borran sus resultados negativos.

Se identificó una corrección distinta de esos ensayos: dejar de recortar a
cero toda concentración negativa admisible durante Newton. El tratamiento
nativo conserva el término restaurador elemental de una sola concentración
negativa, evita productos espurios de varias negativas y conserva la suma
con signo para el tercer cuerpo. La química de estados positivos no cambia.
Las cuatro rutas de cinética pasan las pruebas locales de fuentes y derivadas.
PTC-SER normal vuelve a resolver H2/GRI30 a 10 atm; no se necesita el rescate
conservador para ese ensayo corregido. La precisión espacial y los tiempos
pareados se estudian por separado. Evidencia y fuentes:
`validation/SORET_SIGNED_KINETICS_DIAGNOSIS.md`.

## Limpieza posterior del flujo principal (2026-09-05)

Por solicitud del autor se eliminaron del código del solver PTC-auto,
PTC-rescue, BE exclusivo, pesos congelados de Newton y la mezcla centrada/upwind.
También se retiraron sus opciones de las interfaces de comparación/FGM y se
actualizó el diagnóstico de refinamiento para usar la ruta principal.
El BE interno tras rechazo PTC, la LU directa, el refresco local certificado
de FGM y Soret nativo permanecen. Las configuraciones antiguas de variantes
eliminadas ya no se ejecutan con esta revisión; las trazas y métricas históricas
se conservan. Esto sustituye la indicación anterior de mantener esos modos
diagnósticos desactivados dentro del solver.

Se suprimieron asignaciones repetidas de buffers auxiliares de las
perturbaciones, sin alterar ecuaciones ni tolerancias. La revisión detallada
y las siguientes hipótesis de rendimiento se mantienen en `KFLAME/PIPELINE.md`.

## Bloques LU directamente en orden Fortran (2026-09-05)

Se probó crear el bloque de Schur en memoria Fortran antes de la actualización
por bloques. Los factores fueron idénticos, pero el microbenchmark alternado
de nueve repeticiones dio 0.02603/0.02833 s para 261 bloques y
0.08743/0.09527 s para 854 bloques (orden C anterior/Fortran candidato).
Se retiró ese cambio: evitar una copia de LAPACK no compensa necesariamente
el coste de actualizar la matriz con otra disposición de memoria.

Una prueba diferente conserva el orden C y llama a GETRF/GETRS directamente,
sin envoltorios repetidos de alto nivel. No es reutilización inexacta de LU ni
un cambio del pivotado; sus resultados se registran por separado.

## Cribado de estrategias de arranque frío (2026-09-06)

Pruebas aisladas de la ruta productiva; no se añaden interruptores
experimentales a KFLAME. Evidencia: `validation/cold_strategy_audit_20260906.json`.

| Variante | Evidencia y alcance | Decisión actual |
|---|---|---|
| Perturbaciones por lotes de unos 2 MiB | Siete parejas a 1 atm: intervalos de razón temporal incluyen uno en CH4 y H2/Soret. Cribado a 10 atm: empeora CH4, mejora H2 en una sola pareja. | No adoptar como mejora general; conservar auditoría de memoria/tiempo. |
| Potencias compartidas de Kc | Exactitud bit a bit en los casos medidos; sin ganancia concluyente en siete parejas a 1 atm y sin ganancia en la pareja medida de cada caso a 10 atm. | No incorporar la versión ensayada. |
| Química totalmente secuencial | Más lenta en las dos llamas a 1 atm. Calibración de GRI30, de 1 a 256 estados, selecciona umbral serial cero. | Mantener el paralelismo actual; no atribuir ruido temporal al despachador idéntico. |
| Presupuesto no lineal por discrepancia de interpolación | Conserva la certificación final y los errores de perfiles del cribado a 1/10 atm. No hay ahorro consistente; no es un estimador demostrado del error de discretización. | No incorporar sin evidencia adicional. |
| Corrección FAS de dos mallas | Correcciones gruesas efectivamente ejecutadas, sin excepciones; precisión aceptable, pero mayor tiempo medido en los cuatro casos. H2/Soret a 10 atm: 48.64 frente a 66.81 s en una pareja. | Retirar de la candidatura inmediata a producción. No generalizar el resultado a todo multigrid no lineal. |

La reutilización térmica del Jacobiano NO se incluye en estos descartes:
tiene intervalos favorables en siete parejas a 1 atm, pero la confirmación
a 10 atm fue inconcluyente en ambos casos; no se activa globalmente.
El ensamblado directo `streamed` tampoco demostró ahorro general en el cribado
de los cuatro casos, conservando perfiles idénticos.
Todos estos tiempos son comparaciones internas de KFLAME, no nuevas
comparaciones con Cantera. Los prototipos y datos quedan como material de
auditoría, fuera del código productivo.

## Segunda ronda de arranque frío (2026-09-06)

Evidencia detallada en `validation/COLD_FOLLOWUP_20260906.md`. No se cambian
guardas, tolerancias ni producción para obtener estos resultados.

- **Homotopía de transporte 0 -> 0.5 -> 1:** el prototipo conserva el
  corrector final del modelo objetivo, pero añade coste y no supera el
  filtro comparativo de velocidad (0.3003% y 0.2278% en H2/Soret a 1/10 atm,
  frente al límite 0.1%). Se descarta esta implementación, no toda homotopía.
- **Acción directa de difusión y resolución compartida:** flujo local
  verificado sin inversa explícita. Tres parejas con la resolución compartida
  no demuestran mejora general: H2 1 atm empeora; el intervalo a 10 atm
  incluye ausencia de mejora. No se incorpora a producción.
- **Dependencias exactas del Jacobiano:** no se descarta matemáticamente.
  Tres parejas muestran ahorro en CH4/1 atm y H2/Soret/10 atm, pero no lo
  confirman en los otros dos casos. Sigue fuera de producción; no se crean
  reglas por presión para seleccionar únicamente resultados favorables.

La compilación del recorrido completo de LU se prueba aparte: no es la
disposición Fortran descartada ni reutilización de factores envejecidos.
Se contabilizará también su compilación por proceso antes de adoptarla.

## Reutilización térmica por bloque revisada (2026-09-20)

La evaluación del 06/09 sobre reutilización térmica no se extrapola al kernel
actual. La nueva implementación agrupa las perturbaciones [u,T,Y...] por nodo,
evalúa dos conjuntos de factores térmicos y conserva toda la dependencia
composicional de cinética/falloff. Tres parejas alternadas en cada uno de
CH4/1 atm, CH4/10 atm, H2-Soret/1 atm y H2-Soret/10 atm redujeron las medianas
totales un 11.6%, 8.7%, 6.6% y 7.9%, respectivamente. Las 24 corridas medidas
fueron aceptadas y las 12 parejas conservaron perfiles bit a bit. Se activa
en la precomputación nativa del Jacobiano, sin cambiar tolerancias ni física.
La compilación inicial no está incluida en esas ganancias. Protocolo, alcance
y evidencia: `validation/OPTIMIZATION_STAGES_20260920.md`.
