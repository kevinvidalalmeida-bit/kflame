# Barrido no lineal por bloques espaciales: prototipo y evidencia

## Definición precisa

Se prueba un único barrido multiplicativo de bloques solapados de dos nodos
en la primera malla de cada llamada de solución fría. Las correcciones de un
bloque ven los cambios aceptados de los anteriores. Con Soret y bootstrap,
ese primer barrido pertenece al arranque nativo de mezcla promediada; el
corrector multicomponente/Soret posterior sigue intacto. No se afirma haber
probado barridos en todas las mallas ni directamente sobre todos los estados
multicomponente finos.

Para el bloque espacial I se mantiene fijo el complemento y se aproxima
F_I(x)=0 con hasta dos correcciones Newton modificadas:

    B_I s_I = -F_I(x),    B_I = J_II(x_inicial).

B_I incluye los dos bloques diagonales y ambos acoplamientos vecinos. Es un
subproblema de 2(K+2) incógnitas, no K+2 ecuaciones desacopladas. Se conservan
continuidad, anclaje, condiciones de frontera y todas las especies incluidas
en esas filas. El Jacobiano local usa la aproximación productiva de transporte,
pero cada prueba de estado evalúa el residual no lineal completo y exacto de
transporte. No se ha añadido regularización artificial para forzar una LU
local singular: ese bloque se omite y se registra.

Se limita el paso con las mismas cotas de estado de V2 y se prueban hasta
cuatro amortiguamientos por corrección. La norma auxiliar de aceptación usa
escalas de fila fijas max(|F_i(x_inicial)|,1). Solo se usan en este prototipo,
no sustituyen las normas ni tolerancias del certificado. Cada actualización
reduce el residual de su bloque. Al terminar, si el residual completo en
esas escalas no mejora, se restaura toda la semilla original. También se
restaura ante excepción o límite temporal. Su coste siempre cuenta.

Esto es un prefiltro no lineal local del arranque. NO es todavía una
formulación de Newton sobre un operador no lineal preacondicionado, ni un
algoritmo ASPIN completo. Es una prueba mínima de si un barrido tipo
Gauss-Seidel/Schwarz no lineal aproxima una semilla más barata para V2.
No emplea GMRES, FAS, una segunda malla ni un modelo químico reducido.

## Pruebas y protocolo

76 pruebas automáticas aprobadas antes de medir, incluidas cuatro nuevas:
submatriz con ambos acoplamientos, reducción de residual y preservación de
entrada, invariancia de una raíz, singularidades/límite temporal con rollback.

Campaña `20260907_070948`: baseline y candidato, una pareja por caso CH4/H2 a
1/10 atm. CH4 sin Soret y H2 con transporte multicomponente/Soret. Calentamiento
separado, sin perfiles guardados, sin benchmarks simultáneos, mismas opciones
y fuentes guardadas. La compilación no se presenta como arranque totalmente
frío del proceso. Se exige el certificado completo y los mismos filtros de
errores de perfiles. Una pareja sirve únicamente para cribado.

## Estado

Ambos cribados completos terminaron; ninguna promoción productiva. Cada traza registra
actualizaciones locales, intentos, factorizaciones singulares, méritos inicial
y final, aceptación global y tiempo añadido. Una corrida que solo vuelve a
la semilla original no demuestra utilidad del preacondicionamiento, aunque
termine aceptada por el solver posterior.

Observación del calentamiento H2/Soret a 10 atm: seis actualizaciones locales
hicieron crecer el mérito global de 8.9283 a 23.4598. El barrido se rechazó
y restauró la semilla. No basta con reducir F_I para todos los bloques
visitados: los acoplamientos fuera del bloque importan. Los tiempos de una
ruta restaurada no deben presentarse como ahorro de iteraciones gracias
a una mejor inicialización.

## Resultado del primer cribado

`20260907_070948` terminó. Ocho perfiles auditados por hashes y métricas
finales (`nonlinear_blocks_integrity.json`); todos pasan los filtros de
precisión (`nonlinear_blocks_screening.json`). Una pareja por caso:

| Caso | V2 actual [s] | Barrido local [s] | Jacobianos actual/candidato | LU globales actual/candidato |
|---|---:|---:|---:|---:|
| CH4, 1 atm | 4.0749 | 3.7008 | 84/81 | 342/337 |
| H2/Soret, 1 atm | 4.4155 | 4.5519 | 67/65 | 208/206 |
| CH4, 10 atm | 27.2604 | 25.1288 | 196/189 | 928/807 |
| H2/Soret, 10 atm | 39.0903 | 39.4734 | Sin ventaja de semilla: rollback | Sin ventaja de semilla: rollback |

Los conteos de Jacobiano incluyen el construido para el barrido. Las LU
globales no incluyen las ocho factorizaciones densas locales adicionales,
que constan en la traza y sí están incluidas en el tiempo total. El barrido
costó 9.25/9.33/13.31 ms en los tres casos aceptados. La primera malla tuvo
nueve nodos por inserción del anclaje: no se confunde con los ocho nodos
solicitados antes del anclaje.

En H2/10 atm, la corrida medida también restauró toda la semilla. Por tanto
el resultado no apoya una aceleración general de esta versión. No se promueve.

## Variante con descenso global por actualización

El defecto observado es específico del criterio local: una mejora de F_I
puede empeorar F fuera del bloque. Se ensaya por separado la misma barrida,
pero cada prueba debe reducir tanto su mérito local como el mérito global
en las escalas iniciales. Se mantiene también la guarda al final del barrido.
Esta regla no depende de combustible ni presión. No modifica el damping
del solver Newton productivo: solo el prefiltro experimental de la semilla.

La traza guarda los méritos de todas las actualizaciones aceptadas; se
verifica su descenso estricto. 77 pruebas unitarias aprobaron antes de esta
segunda campaña, incluida la monotonía de la variante global.
`20260907_071615` contiene el cribado independiente de los cuatro casos.
No se mezclan tiempos absolutos de ambas campañas ni se suman sus ganancias.

### Cierre del segundo cribado

Ocho perfiles y 34 fuentes auditados; todos los casos pasan aceptación y
precisión comparativa. Informes: `nonlinear_blocks_global_screening.json` y
`nonlinear_blocks_global_integrity.json`. Una pareja por caso, no un estudio
estadístico concluyente:

| Caso | V2 actual [s] | Barrido con descenso global [s] |
|---|---:|---:|
| CH4, 1 atm | 3.7719 | 3.7700 |
| H2/Soret, 1 atm | 4.9219 | 4.9968 |
| CH4, 10 atm | 21.7949 | 23.2060 |
| H2/Soret, 10 atm | 41.6576 | 43.5742 |

En H2/10 atm ahora sí se acepta el barrido: cinco actualizaciones reducen
el mérito de 8.9283 a 8.5819. Sin embargo, el perfil final coincide con el
baseline y el tiempo total no mejora. Reducir ese mérito de inicialización
no garantiza ahorrar trabajo hasta la llama certificada.

Decisión: ninguna de las dos variantes justifica promoción productiva ni se
incluye como propuesta todavía no ensayada. Se conservan trazas y prototipos
para reproducibilidad. Estos resultados no descartan toda la familia de
preacondicionamiento no lineal; solo cierran estas dos realizaciones.
