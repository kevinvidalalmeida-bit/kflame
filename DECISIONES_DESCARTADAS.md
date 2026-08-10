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
| BDF2 / BE-BDF2 pseudo-transitorio | La llama fría completa tomó 64.68 s frente a 57.79 s con Euler implícito (BE). | Eliminado; BE es el único esquema expuesto. |
| Damping Armijo | Mejoró un bootstrap aislado, pero la llama completa subió a 60.30 s frente a 57.79 s. | Eliminado; se mantiene el damping estilo Cantera. |
| Vida fija de Jacobiano = 80 | Phi=1.1 desde semilla fue rápido, pero phi=1->1.3 y la cadena hasta phi=1.3 no convergieron en 60 s. | No usar globalmente; continuación conservadora con edad 40. |
| GPU/CuPy | Las transferencias CPU-GPU superaron el ahorro: Jacobiano CPU 0.153 s frente a GPU 0.219 s. | Eliminado; la ruta de produccion es solo CPU. |
| Reaplicar isoterma persistente de anclaje | El anclaje actual ya conserva T=781.381 K. Forzarlo de nuevo dio la misma malla, 136 pasos y el mismo residual, con 37.18 s frente a 36.43 s. | Sin cambio; no acelera. |
| Detector por residual precondicionado GMRES | Tras cuatro iteraciones, el residual precondicionado era bajo incluso en llamadas que fallaban la convergencia real; no activo ningun corte util. | Eliminado; usar una sonda fija de 4 iteraciones. |

## Ruta de producción retenida

- Residual y Jacobiano local por diferencias finitas con backend nativo CPU y
  Numba para la termoquímica/cinética repetitiva.
- Jacobiano block-tridiagonal y factorización/solves SciPy-LAPACK.
- En cambios consecutivos de paso BE, LU anterior como precondicionador GMRES
  con una sonda de 4 iteraciones y regreso automatico a LU exacta.
- Newton amortiguado compatible con Cantera y pseudo-transitorio Euler
  implícito (BE).
- Continuación por perfiles y predictor secante; `max_jac_age=40` solo para
  semillas convergidas.
- `OPENBLAS_NUM_THREADS=1` para los bloques densos pequeños.
- Arranque de 12 nodos como opción validada para la FGM fría; conservar 8
  cuando se requiera reproducibilidad histórica.

Una nueva variante solo debe añadirse de nuevo con una comparación end-to-end
en un barrido FGM, misma malla/criterio de aceptación y mejora reproducible.

## Seleccion vigente para FGM

Barrido con GRI-Mech 3.0, CH4/aire, 300 K, 1 atm, phi = 0.9, 1.0 y 1.1;
los dos ultimos flamelets usaron el perfil y la malla del anterior.

| Ruta | Tiempo total de los 3 flamelets |
| --- | ---: |
| numba_local + LU directa | 27.95 s |
| numba_local + LU reciclada/GMRES | 28.30 s |
| block_tridiag + LU reciclada/GMRES | 19.65 s |
| Cantera | 17.24 s |

Se mantiene `block_tridiag + recycled_gmres` como predeterminado del
generador: es 29.7 % mas rapido que la mejor alternativa V2 medida, aunque
todavia 14.0 % mas lento que Cantera en este barrido. La sonda GMRES de cuatro
iteraciones redujo la corrida V2 de 20.40 s a 17.83 s frente a un limite de
doce iteraciones, sin cambio material de la solucion. En continuidad, phi=1.0
y phi=1.1 usaron una sola malla, sin expansion del dominio y sin cambios de
refinamiento; no se repitio bootstrap ni remallado.

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

Tambien se probaron dos rutas experimentales nuevas y quedaron desactivadas:

- damping basado en reduccion del residual: mas lento en la llama completa y
  en el mini-barrido FGM;
- salto adaptativo de GMRES hacia LU exacta: cambio la trayectoria de Newton,
  aumento las reconstrucciones y fue mas lento.

Ambas quedan disponibles solo como opciones experimentales para futuras
pruebas, no como camino de produccion.

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
