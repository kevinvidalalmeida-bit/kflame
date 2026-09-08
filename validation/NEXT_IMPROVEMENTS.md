# Selección activa y nuevas hipótesis de rendimiento

## Limpieza solicitada

La ruta activa es la descrita en `V2/PIPELINE.md`: química nativa con productos
estequiométricos especializados, Jacobiano local precomputado, LU pivotada
SciPy/LAPACK, Newton/PTC-SER/BE, adaptación y certificación completas. Se
mantienen el transporte/Soret nativo y la continuación FGM ya validados.

Todos los candidatos de las dos últimas rondas quedan ARCHIVADOS, incluidos
los de resultado inconcluyente. No tener evidencia de ahorro general basta
para excluirlos de producción; no demuestra que toda su familia matemática
sea inútil. No se crean selectores por presión para rescatar cifras favorables.

El benchmark de esa investigación ahora selecciona únicamente `baseline`
por defecto y no construye ni compila kernels experimentales. Solicitar
una variante histórica archivada requiere `--reproduce-archived`, solo para reproducción
histórica. Los candidatos nuevos autorizados se seleccionan explícitamente;
ninguno se activa por defecto. Los archivos fuente, tests, perfiles y resultados se conservan:
no se borran resultados negativos ni se rompen sus hashes. No hay imports
de estos prototipos desde el solver V2 ni desde los generadores FGM.

Quedan fuera de la lista activa: reutilización térmica/dependencias exactas,
LU de recorrido compilado, acción directa/compartida de transporte, homotopía
0/0.5/1, FAS, presupuestos de malla, potencias compartidas, lotes acotados,
ensamblado streamed y granularidad serial. También se mantienen los descartes
anteriores de POD, GMRES, BDF2, LU aproximada, condensación bordeada, etc.

Validación de esta limpieza: 67 tests aprobados. Dos contratos nuevos
comprueban que el modo normal no construye candidatos archivados y que la
reproducción requiere autorización explícita en la línea de comandos.

## Propuestas nuevas y seguimiento

El usuario autorizó inicialmente las dos primeras. Ambas se implementaron como prototipos
aislados; la generación de grandes funciones estequiométricas no terminó
GRI30 dentro del presupuesto local de unos diez minutos y no se adopta.
El residual compartido terminó el cribado sin ahorro general demostrado.
Estado y evidencia en `RESIDUAL_CODEGEN_REPORT.md`. Posteriormente autorizó
la tercera: barrido local por bloques en la primera malla y variante con
descenso global, documentados en `NONLINEAR_BLOCKS_REPORT.md`. Son
prototipos aislados, no nuevas rutas productivas.

### 1. Compartir el residual exacto ya calculado en la aceptación

Evidencia local: en `V2/solver.py`, la rama `ok_ss` evalúa el residual con
transporte exacto para su norma y vuelve a evaluarlo al preparar la
linealización de continuación. El estado no cambia entre ambas llamadas.
Además, el argumento `source_residual` se calcula incluso si el receptor
retorna sin almacenar una linealización válida.

Propuesta: producir el vector exacto una vez, usar su norma para certificar y
transferir ese mismo vector cuando proceda. Primero comprobar las mutaciones
de caché/transporte y que ningún consumidor modifica el arreglo. No usar una
caché aproximada: invalidar al cambiar estado, malla, física, pseudo-tiempo
o backend. Mantener el cálculo necesario para la guarda aunque no se vaya
a reutilizar la linealización. Esto no es el experimento descartado de
reutilizar buffers: elimina evaluaciones, no solo asignaciones.

Medir número de llamadas exactas evitadas y tiempo de las etapas, perfiles
idénticos y regresión con/sin Soret. Ahorro probablemente acotado; no asignar
un porcentaje sin ejecutarlo. Es la primera prioridad por alcance pequeño.

### 2. Generación estática de química específica del mecanismo

Generar índices y operaciones de cada reacción como código, en vez de
interpretar repetidamente tablas de participantes y tipos. Conservar todas
las especies, reacciones, falloff, unidades y semántica de trazas negativas
usada por Newton. No proponer otro Jacobiano analítico ni reducir el mecanismo.
La generación es general a partir del mecanismo, no una regla especial de
1/10 atm. Separar generación/compilación del coste recurrente.

La [herramienta KinetiX](https://github.com/bogdandanciu/KinetiX) es evidencia
de esta familia de generación para química, termodinámica y transporte,
no de aceleración garantizada de V2. Su infraestructura de benchmark declara
Windows no soportado: no es un reemplazo directo del entorno actual. Una
prueba empezaría solo por fuentes químicas, comparando contra el kernel
productivo, antes de plantear una integración de transporte o Soret.

Riesgos: crecimiento de código, tiempo de compilación, presión de caché y
diferencias por reordenamiento. Requiere igualdad de propiedades, pruebas de
estados admisibles de Newton y finalmente llamas completas.

### 3. Preacondicionamiento no lineal local en la misma malla

Estudiar unos pocos barridos no lineales de bloques espaciales con los
vecinos fijos, antes de una corrección global, para reducir los desequilibrios
más locales. La idea pertenece a Gauss-Seidel/Schwarz no lineal; está
documentada en [PETSc SNES](https://petsc.org/release/overview/nonlinear_solve_table/).
No exige instalar PETSc para un primer prototipo y no supone que esa
biblioteca será más rápida que el solver actual.

Es distinto de GMRES lineal, de FAS en dos mallas y de añadir otra etapa de
homotopía de transporte. Hay que comprobar primero que los subproblemas
locales sean resolubles respetando continuidad, anclaje y suma de especies.
El corrector y el certificado finales siguen usando las ecuaciones completas
acopladas, incluido Soret. El trabajo local adicional debe reducir el coste
total, no solo el número de Jacobianos. Riesgo alto por acoplamiento entre nodos.

## Puerta común

Cada propuesta se ensaya aislada, sin incorporar un nuevo selector a V2.
No hay porcentaje prometido ni afirmación de novedad matemática. Exigir
certificación y errores del protocolo existente, siete parejas alternadas
en 1/10 atm con/sin Soret antes de promoción, y una comparación actual con
Cantera bajo las mismas condiciones. Separar arranque sin perfiles de
primer proceso con compilación fría. No confundir ambas medidas.

## Ronda autorizada BLAS/SIMD/espacial (2026-09-07)

Seguimiento en `BLAS_SIMD_SPATIAL_REPORT.md`. Se implementaron adaptadores
aislados MKL GETRF/GETRS, SIMD NASA entre estados y reconstrucción limitada
de mayor orden con monitor lineal/cuadrático opcional. El cribado BLAS/SIMD
terminó sin ventaja general demostrada: ambos empeoraron H2/Soret a 10 atm
en la pareja disponible. No se promueven ni se añaden reglas por presión.

La primera integración espacial tuvo un respaldo silencioso al residual
original y su campaña fue invalidada. La versión corregida tiene pruebas
de ejecución real, fallo explícito ante respaldo y coherencia del Jacobiano
quasi-Newton. La campaña corregida y sus límites quedan identificados en el
informe. El estimador adjunto de error y la comparación a igual error físico
siguen pendientes; un monitor de reconstrucción no los sustituye.
