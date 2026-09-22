# Auditoria del arranque H2/GRI30 a 10 atm

## Hallazgo

El exceso de llamadas de KFlame se concentra en intentos pseudo-transitorios
en la malla inicial. La diferencia principal comprobada es el esquema de
pseudo-tiempo: produccion usa una correccion PTC-SER, con BE convergido como
rescate tras rechazo. Cantera converge el subproblema de Euler implicito.
Esto no implica que BE sea globalmente mas rapido: los experimentos completos
demuestran la compensacion entre coste por paso y numero de pasos.

## Comparacion diagnostica

Cantera instalado: **3.2.0**. No se instalo ni se midio 4.0; la documentacion
consultada de desarrollo indicaba 4.0.0a2. GRI30, H2/aire, phi=1, 300 K,
10 atm, ancho inicial 0.03 m, multicomponente/Soret. Ambos usan rtol=1e-4,
atol estacionario=1e-9 y transitorio=1e-11, limite de 500 pasos y criterios
finales de malla ratio=2.5, slope=.04, curve=.08, prune=.003.

Cantera `auto=True` resuelve primero con mezcla y sin Soret. Usa una rampa
inicial lineal entre las posiciones relativas .3 y .5, equilibrio HP y
velocidad inicial de 1 m/s. KFlame tambien calcula equilibrio HP nativo y
usa 1 m/s, pero con perfil tanh. En esta comparacion KFlame afloja SOLO la
malla del bootstrap por factor 2, restaurando el refinamiento final;
Cantera auto refina su fase de mezcla con los criterios finales. No son
trayectorias ni modelos de transporte numericamente identicos.

La traza KFlame registra tres fases fallidas con 500 pasos aceptados cada
una en 9 nodos, mas rechazos. Sus residuales son 2946, 5113 y 4028 llamadas;
las fases de temperatura prescrita y otras transiciones completan 12641
llamadas antes de resolver en 17 nodos. La primera sintesis del informe
solo mostraba dos fallos porque una interrupcion por dominio demasiado
estrecho omitia parte del historial; la instrumentacion por fase recupera
el tercer intento.

Cantera termina su primera malla de 9 nodos tras 60 pasos aceptados y
registra 1507 evaluaciones fuera del Jacobiano en esa fila de estadisticas.
En todo el solve registra 5067 evaluaciones fuera del Jacobiano y 240
Jacobianos. Sus callbacks cuentan 180 pasos transitorios aceptados;
`time_step_stats` suma solo 130 en este caso con expansion de dominio.
Por ello no se identifica sin mas ese contador con todos los pasos efectivos.

Las llamadas KFlame incluyen comprobaciones adicionales del residual y las
estadisticas Cantera excluyen evaluaciones para diferencias finitas del
Jacobiano. Los recuentos no representan el mismo trabajo por llamada.
En las corridas diagnosticas instrumentadas Cantera tardo 39.10 s y
KFlame 12.90 s: **no hay evidencia aqui de que Cantera reduzca el tiempo
total**, aunque si evita la trayectoria larga del arranque.

Cantera obtuvo 243 nodos y Su=1.332875049 m/s; KFlame, 241 nodos y
Su=1.348484783 m/s, diferencia aproximada 1.17%. Esta auditoria de algoritmo
no sustituye una comparacion de transporte/discretizacion ni demuestra
igualdad de los resultados de ambos paquetes.

## Ablacion controlada dentro de KFlame

Se cambio solo el subproblema pseudo-transitorio a BE convergido, manteniendo
SER/PTC original como baseline, edades del Jacobiano, ecuaciones, semilla,
tolerancias y aceptacion final. Primero se diagnostico, despues se midieron
tres parejas alternadas por caso con calentamientos excluidos.

| Caso | PTC-SER, mediana s | BE, mediana s | Reduccion |
| --- | ---: | ---: | ---: |
| CH4/1 atm | 2.9561 | 4.3127 | -45.89% |
| CH4/10 atm | 5.9693 | 7.2423 | -21.32% |
| H2/Soret/1 atm | 2.2335 | 2.1387 | 4.24% |
| H2/Soret/10 atm | 11.5011 | 6.5428 | 43.11% |

Las 24 llamas fueron aceptadas. Todas las parejas conservaron exactamente
la malla. En H2/10 atm el bootstrap baja de 15123 a 3575 residuales y de
506 a 229 Jacobianos; el corrector final conserva 41 residuales y 9
Jacobianos. La maxima diferencia de temperatura en las cuatro parejas
representativas es 2.81e-10 K; de Su relativa, 1.03e-12; de Y, 1.27e-12.
Se trata de acuerdo a redondeo, no igualdad bit a bit.

Cambiar solo tanh por rampa lineal no elimina los tres intentos largos:
el bootstrap aun usa 14841 residuales. Se descarta esa explicacion como
causa principal. Tampoco se alteraron las tandas de 20 pasos de KFlame;
Cantera usa tandas de 10, pero la mejora BE se obtuvo sin copiar ese valor.

**Decision:** no sustituir PTC-SER globalmente ni seleccionar BE por
combustible/presion. La evidencia motiva estudiar control global del esfuerzo
no lineal basado en progreso/defecto. El usuario solicito expresamente una
estrategia general beneficiosa para ambos combustibles.

## Estado de la campa?a

La comparaci?n y sus cifras se conservan como diagn?stico hist?rico. Las
variantes de Euler impl?cito y el benchmark de sustituci?n se retiraron porque
no eran una pol?tica global para CH4 y H2. `benchmarks/audit_cantera_startup.py`
contin?a siendo la herramienta de auditor?a nativa/Cantera sin rutas de
arranque alternativas.
