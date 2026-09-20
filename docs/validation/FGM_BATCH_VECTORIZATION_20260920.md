# Preparación vectorizada del Jacobiano para FGM

Se eliminan los bucles Python nodo×especie en la construcción de los estados
perturbados y dos copias completas redundantes. Las perturbaciones conservan
el mismo signo, magnitud, disposición [u,T,Y...] y tratamiento de ceros. No
cambian el método numérico, las tolerancias ni los criterios de aceptación.

## Barrido completo, no microbenchmark

CH4/aire, GRI30, 1 atm, phi=[0.7,0.9,1,1.1,1.4], transporte promediado,
cuatro hilos Numba y uno BLAS, sin semillas persistentes y con Cantera bloqueado.
Cada corrida inicia otro proceso y construye/exporta la tabla completa. Se
excluye un calentamiento por variante; tres parejas alternan su orden.

| Pareja | Bucle anterior | Preparación vectorizada |
|---|---:|---:|
| 1 | 18.4365 s | 15.4286 s |
| 2 | 15.5930 s | 15.8860 s |
| 3 | 20.3678 s | 18.3501 s |
| Mediana | 18.4365 s | 15.8860 s |

Reducción de medianas del runtime FGM: **13.8%**. La segunda pareja no mejora;
hay variación considerable y tres parejas no garantizan rendimiento universal.
Las 30 llamas medidas fueron aceptadas. Todas las parejas produjeron tablas
bit a bit idénticas en coordenadas, velocidades, nodos y campos físicos.
El tiempo de proceso completo también se registra, separado del runtime FGM.
Evidencia: `fgm_batch_vectorization_20260920.json`.

El oráculo del bucle anterior queda únicamente en un test y el benchmark lo
inyecta para la ablación; no se añade un selector experimental a producción.
Las pruebas incluyen estados con signo, ceros, buffers no contiguos y varios
tamaños de mecanismo. La primera compilación tras reorganizar los módulos es
un coste adicional de instalación/arranque, no se presenta como aceleración.
