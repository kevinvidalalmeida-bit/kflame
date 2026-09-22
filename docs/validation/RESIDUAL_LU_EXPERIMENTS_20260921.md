# Residual, LU y paralelismo despues del Jacobiano analitico

## Protocolo

Cribado local del 21/09/2026 en Windows, Python 3.11, NumPy 2.4.6,
SciPy 1.17.1 y Numba 0.65.1. Cuatro hilos Numba y uno BLAS/MKL.
Por candidato/caso: un calentamiento excluido de cada modo y tres parejas
alternando el orden. Cada llama arranca sin un perfil previamente resuelto;
los kernels y datos del mecanismo ya estan calientes. No se ejecutan trabajos
pesados en paralelo. Ambos modos usan el Jacobiano espacial analitico actual.

GRI30, phi=1, 300 K; CH4/1 atm con transporte de mezcla y H2/10 atm con
multicomponente/Soret, incluida su llama intermedia de mezcla. Misma fisica,
tolerancias, edades del Jacobiano y criterios de aceptacion. Las mediciones
incluyen los mismos contadores ligeros de perfil en ambos modos; no cProfile.

## Resultados del primer cribado

Tiempos medianos end-to-end en segundos. Reduccion positiva significa ahorro.
Las diferencias pequenas no constituyen evidencia de mejora estable.

| Experimento | CH4: base / candidato | Reduccion | H2: base / candidato | Reduccion |
| --- | ---: | ---: | ---: | ---: |
| Conservar residual en reconstruccion y reintento | 1.8827 / 1.8814 | 0.07% | 11.7615 / 13.1201 | -11.55% |
| Espacios de trabajo LU en orden C | 1.9366 / 2.0212 | -4.37% | 12.8867 / 12.3682 | 4.02% |
| Espacios de trabajo LU en orden Fortran | 2.0191 / 2.1039 | -4.20% | 13.5673 / 14.3322 | -5.64% |
| Termoquimica serial para hasta 16 nodos | 2.7118 / 2.8647 | -5.64% | 12.4111 / 13.4726 | -8.55% |

Las 48 llamas medidas fueron aceptadas; las 24 parejas conservaron z, T, u
e Y bit a bit. El residual conservado reduce llamadas CH4 de 1401 a 1338,
y H2 de 15123 a 14729 en el bootstrap y 41 a 40 en el corrector.
Esa reduccion de llamadas no acredita una reduccion del tiempo total.
El buffer C beneficia H2 en este cribado pero perjudica CH4; no se promueve
una seleccion por combustible basada en dos puntos de prueba.

Ninguno de estos cuatro candidatos se incorpora al solver. Sus funciones
viven exclusivamente en `benchmarks/residual_lu_candidates.py`. El ensayo
de residual construye una copia instrumentada de Newton y comprueba que los
puntos de sustitucion siguen existiendo; no quedan banderas experimentales
ni ramas nuevas en el Newton de produccion. Se conserva la reutilizacion
original de residual de pasos aceptados, ya validada anteriormente.

La prueba LU comprueba pivoteo, solucion compilada y alternativa, propiedad
de los factores tras otra factorizacion y preservacion de la matriz original.
Se conservan dos subtimers opcionales en el residual para separar propiedades
termoquimicas de transporte y flujo de caras.

## Cuello de botella observado

El perfil con cProfile, que no es una medida de velocidad sin instrumentacion,
registra en el bootstrap H2/10 atm: 15123 residuales (4.297 s inclusivos),
506 Jacobianos (1.083 s) y 2776 factorizaciones (3.141 s). Las propiedades
nodales ocupan 2.016 s y las caras 1.587 s, incluidos en el residual.
El corrector multicomponente final tiene solo 41 residuales y 9 Jacobianos.
Por ello se abre una auditoria del arranque y pseudo-tiempo frente a Cantera,
documentada en `CANTERA_STARTUP_20260921.md`.

## Reproducibilidad

- `python benchmarks/benchmark_residual_lu.py --candidate retry --cases CH4_1 H2_10 --pairs 3 --output runs/new/retry.json`
- Sustituir candidato por `workspace_c`, `workspace_f` o `small_serial`.
- Resumen versionable: `residual_lu_screening_20260921.json`, con cada pareja,
  medianas, aceptacion, igualdad de campos y contadores.
- Datos locales completos: `runs/residual_lu_experiments/`; incluye
  `profile_baseline.json`, `environment.json` y copia de las fuentes
  `source_before_isolation/` usadas antes de retirar las ramas experimentales
  del solver. El benchmark actualizado registra versiones y hashes por corrida.

Hay variacion apreciable entre ejecuciones, especialmente en H2. Estas tres
parejas sirven como cribado local y no justifican promesas universales de tiempo.
