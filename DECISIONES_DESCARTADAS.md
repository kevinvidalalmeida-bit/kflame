# Decisiones descartadas de optimización

Este documento conserva los resultados negativos para no reintroducir rutas
que ya se midieron. Las referencias de tiempo corresponden a GRI-Mech 3.0,
CH4/aire, 300 K, 1 atm y transporte mixture-averaged.

## No reincorporar

| Variante | Evidencia | Decisión |
| --- | --- | --- |
| DLL C++ block-Thomas | El backsolve aislado fue más rápido, pero el FGM completo solo redujo un caso sembrado de 20.05 s a 18.51 s. La LU C++ completa fue 22 % más lenta que LAPACK y dio pasos lineales inestables para el Jacobiano rígido. | Eliminada. Usar SciPy/LAPACK. |
| Block-Thomas Numba | En continuación certificada tardó 26.55 s frente a 25.07 s con SciPy/LAPACK y terminó en otra malla. | Eliminado. |
| Jacobiano químico analítico experimental | El microbenchmark redujo el ensamblado, pero una continuación real requirió 39.35 s frente a 20.77 s de diferencias finitas. | Eliminado. |
| BDF2 / BE-BDF2 pseudo-transitorio | La llama fría completa tomó 64.68 s frente a 57.79 s con Euler implícito (BE). | Eliminado; no confundir con PTC-SER, que mantiene el desplazamiento BE pero evita converger cada subproblema transitorio. |
| Damping Armijo | Mejoró un bootstrap aislado, pero la llama completa subió a 60.30 s frente a 57.79 s. | Eliminado; se mantiene el damping estilo Cantera. |
| Vida fija de Jacobiano = 80 | Phi=1.1 desde semilla fue rápido, pero phi=1->1.3 y la cadena hasta phi=1.3 no convergieron en 60 s. | No usar globalmente; FGM usa edad 20 y la ruta histórica conserva 40. |
| GPU/CuPy | Las transferencias CPU-GPU superaron el ahorro: Jacobiano CPU 0.153 s frente a GPU 0.219 s. | Eliminado; la ruta de produccion es solo CPU. |
| Reaplicar isoterma persistente de anclaje | El anclaje actual ya conserva T=781.381 K. Forzarlo de nuevo dio la misma malla, 136 pasos y el mismo residual, con 37.18 s frente a 36.43 s. | Sin cambio; no acelera. |
| Detector por residual precondicionado GMRES | Tras cuatro iteraciones, el residual precondicionado era bajo incluso en llamadas que fallaban la convergencia real; no activo ningun corte util. | Eliminado; usar una sonda fija de 4 iteraciones. |
| Tangente con geometria termica secante | En pasos de 2 % en \(\log\phi\), la segunda transicion empeoro de 0.94 s a 1.18 s a 1 atm. A 10 atm bajo de 2.65 s a 2.57 s en una sola corrida, pero termino con \(\|F\|_\infty=9.00\times10^3\), frente a \(1.38\times10^3\) para el tangente crudo. No hay mejora general ni margen suficiente frente a la guarda de \(10^4\). | Retirado; conservar el tangente crudo y registrar el resultado negativo. |
| Transferencia de semilla por longitud de arco ponderada | Con 64 nodos, concentrar la semilla por la longitud de arco de \((u,T,Y_k)\) produjo dos rechazos del corrector a 1 atm, expansiones de dominio y reinicios fríos. El barrido completo \(\phi=1,1.02,1.04\) subió a 31.16 s de solve, frente a 12.80 s con el tangente y transferencia por índice. | Eliminada; la geometría adaptativa heredada por índice conserva mejor la cobertura asintótica. |
| Predictor tangente con curvatura cuadrática | Usa el tangente de Jacobiano en \(\log\phi\) y el flamelet anterior para estimar \(x''\). Fue neutro a 1 atm (0.82 s frente a 0.94 s en la segunda transición, dentro de variación), pero a 10 atm aumentó la segunda transición de 2.65 s a 4.90 s: 25 Jacobianos, 122 LU y 144 pruebas de damping, frente a 11, 35 y 48 con el tangente de primer orden. | Eliminado; el perfil tiene curvatura numérica/espacial suficiente para que la extrapolación de segundo orden empeore la trayectoria. |
| Pseudo-arclength bordeado directo | En CH4/aire, \(\phi:1.00\to1.02\), el corrector aumentado resolvió una semilla puente y luego se certificaron el puente y el objetivo con V2 normal. A 1 atm el trayecto fue 10.46 s frente a 0.961 s del tangente directo; a 10 atm, 13.30 s frente a 1.110 s. Las dos rutas certificaron todas las llamas. Las diferencias finales de \(\Su\) fueron \(3.87\times10^{-5}\) (1 atm) y \(1.99\times10^{-5}\) (10 atm) relativas; proceden de trayectorias certificadas distintas bajo la misma guarda operacional, no de una relajación de criterios. | No promover para ramas suaves de FGM. Se conserva solo como experimento de robustez: el sistema aumentado disperso es el tratamiento matemáticamente correcto si \(J\) se vuelve singular cerca de un pliegue; el atajo Schur \(J^{-1}\) no lo es. Reabrir únicamente con una familia que presente una rama o pliegue físico real. |

## Ruta de producción retenida

- Residual y Jacobiano local por diferencias finitas con backend nativo CPU y
  Numba para la termoquímica/cinética repetitiva.
- Jacobiano block-tridiagonal y factorización/solves SciPy-LAPACK.
- En cambios consecutivos de paso BE, LU anterior como precondicionador GMRES
  con una sonda de 4 iteraciones y regreso automatico a LU exacta.
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

Ese cuadro conserva la comparacion historica entre backends V2. La sonda GMRES
de cuatro iteraciones redujo aquella corrida V2 de 20.40 s a 17.83 s frente a
un limite de doce iteraciones, sin cambio material de la solucion. La nueva
ruta fria validada mas abajo usa LU directa: en el primer flamelet las sondas
GMRES fallaban y despues se reconstruia la LU exacta, por lo que el coste de la
sonda no se amortizaba.

Esta seleccion es de rendimiento entre backends V2. La velocidad de llama V2
difirio de Cantera entre 3.7 % y 4.7 % en este mallado, por lo que no constituye
todavia una certificacion fisica de la tabla FGM.

## Auditoria V2 y optimizaciones validadas (2026-08-10)

Primero se reviso el contenido historico de `RESPALDOS/V2.rar`. El respaldo
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
| V2 | 19.45 s | 0.425526 | 36 |

Esto equivale a 1.66x para V2, con una diferencia relativa de Su de 1.56 %,
el mismo residual final y la misma malla que la referencia. El reporte
completo queda en `V2/comparison_runs/run_20260809_235927/summary.json`.

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
| V2 frio, sin cache | 50.82 s | 42.65 / 2.22 / 4.83 s | 267 / 276 / 291 |

Las velocidades de llama de V2 fueron 0.338924, 0.378937 y 0.382446 m/s,
frente a 0.338613, 0.378519 y 0.381789 m/s de Cantera; la diferencia maxima
relativa fue aproximadamente 0.17 %. Los tres flamelets V2 fueron aceptados,
con residual bruto inferior a 1e4 y norma ponderada del paso inferior a 1.

La causa de la diferencia temporal esta localizada en el primer flamelet
frio: una vez disponible una semilla V2 convergida, resolver en la malla final
de Cantera tomo 1.53 s en total para los tres casos. Esto es una medicion del
solver sobre una semilla ya certificada, no una comparacion end-to-end.

La ruta de produccion para barridos repetidos es la cache persistente
`output-root/_v2_seed_cache` y el paralelismo automatico. Con las mismas
opciones estrictas, despues de generar las tres semillas V2, el barrido
paralelo de tres procesos tomo 6.1 s, con los mismos nodos y aceptacion. Ese
tiempo no debe presentarse como una comparacion fria: representa el escenario
real de regenerar o explorar una tabla FGM despues del primer barrido.

Tambien se probaron 12 y 24 nodos iniciales. En una llama aislada 12 nodos
parecio reducir el tiempo, pero en el barrido completo subio a 58.4 s; 24
nodos tambien empeoro el arranque. Se conserva 8 como default para no cambiar
la trayectoria validada. No se agrega ninguna semilla de Cantera al cache V2,
porque eso ocultaria el coste real del arranque y no seria una comparacion
honesta de los solvers.

## Optimizacion del arranque frio V2 (2026-08-10)

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
| V2 optimizado frio | 18.52 s | 247 / 256 / 268 | 0.338981 / 0.378772 / 0.381947 |

Los tres flamelets V2 fueron aceptados con convergencia de malla, residual
guardado y norma ponderada dentro de los limites. El runtime end-to-end de
construccion de la tabla fue 19.7 s, incluyendo la interpolacion FGM. La
diferencia de Su maxima fue aproximadamente 0.11 %.

La misma configuracion con las tres semillas persistentes y paralelismo
automatico termino en 5.8 s. Por tanto, V2 queda practicamente equiparado a
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
las tres semillas V2: 2 workers terminaron en 5.8 s, 3 workers con 4 hilos
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
11.75 s (45.1 %). Frente al registro estricto de Cantera de 18.72 s, V2 tarda
37.2 % menos. La diferencia máxima PTC-SER frente al modo anterior fue
6.32e-10 m/s en `Su`, 0.564 K en temperatura y 0.117 % del pico global de
liberación de calor. En el barrido amplio fue 1.52e-10 m/s, 0.533 K y 0.098 %,
respectivamente; coinciden aceptación, forma de las tablas y número de nodos.

El comparador principal V2 también conservó 36 nodos, `Su=0.425526 m/s` y
residual final 4.23: bajó de 17.37 s con la ruta anterior a 12.64 s con
PTC-SER (27.2 %). En esa corrida Cantera tomó 34.62 s, por lo que V2 fue
2.74x más rápido.

La formulación sigue la continuación pseudo-transitoria linealmente implícita
y el control SER descritos por Kelley y Keyes, y coincide con la estructura
de un paso usada por `TSPSEUDO` de PETSc:

- https://repository.lib.ncsu.edu/items/222848f9-65e0-4e5e-9a72-ee1d96d75857
- https://petsc.org/release/src/ts/impls/pseudo/posindep.c.html

## Limpieza y consolidación final (2026-08-26)

Se retiraron del repositorio 140 archivos de corridas FGM antiguas (60.14
MiB). `FGM/resultados/`, `V2/comparison_runs/` y los cachés de Python/Numba
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

La validación estricta final de V2 para phi = 0.9, 1.0 y 1.1 aceptó los tres
flamelets con 247 / 256 / 268 nodos y `Su` = 0.338981 / 0.378772 / 0.381947
m/s. Todos los arreglos de la tabla fueron idénticos bit a bit a la referencia
anterior, salvo `solve_time`. La corrida caliente tomó 13.4 s end-to-end; la
primera corrida después de borrar el caché tomó 26.1 s por la recompilación
única de Numba.

La comparación principal final obtuvo 34.84 s para Cantera y 12.41 s para V2
(2.81x), con V2 convergido en 36 nodos, `Su=0.425526 m/s` y residual final
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

El total V2 fue 7.33 s, con 36 nodos, `Su=0.4255264047 m/s` y residual final
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

Con esa regla, el barrido V2 bajo de 163.7 a 36.4 s end-to-end (77.8 %, 4.50x),
acepto 5/5 flamelets y uso 266 / 247 / 256 / 268 / 247 nodos. Cantera, con los
mismos valores de `phi` y criterios de refinamiento, tomo 93.9 s end-to-end;
la suma exclusiva de solves fue 35.38 s para V2 y 93.23 s para Cantera (V2
2.64x mas rapido). Cada solver conservo su propia malla adaptativa; la
diferencia maxima de `Su` fue 0.75 % y ocurrio en el extremo `phi=1.4`.
