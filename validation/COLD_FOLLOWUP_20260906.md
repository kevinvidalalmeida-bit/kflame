# Segunda ronda: tres hipótesis concretas de arranque frío

Esta campaña continúa la auditoría anterior sin activar candidatos en V2.
No constituye una comparación temporal nueva con Cantera.

Decisión posterior del usuario: archivar todos los candidatos de esta ronda,
incluidos los inconcluyentes. Las propuestas de confirmación al final del
informe conservan su contexto histórico, pero ya no son una cola activa.
Solo `baseline` corre por defecto; la reproducción histórica requiere
`--reproduce-archived`. Ver `NEXT_IMPROVEMENTS.md`.

## Hipótesis y variantes

1. `dependencies`: reutilización térmica por índices conocidos de la perturbación
   y reutilización de las propiedades base en la columna de velocidad. Se
   conserva la columna completa del residual, incluidos convección y continuidad.
   No cambia la química ni se congela la dependencia de composición.
2. `transport-action`: el residual multicomponente no desactualizado calcula
   el flujo por acción de un sistema lineal; la eliminación térmica obtiene
   A^-1 B resolviendo A X=B. El Jacobiano conserva su matriz completa de
   coeficientes y las actualizaciones racionales certificadas anteriores.
   La eliminación térmica sigue teniendo múltiples RHS: no desaparece todo
   el coste cúbico. El gradiente molar se hace de suma cero al nivel de redondeo;
   no se promete igualdad bit a bit con la multiplicación matricial anterior.
   La primera versión resuelve A para el flujo y para B por separado;
   `transport-action-shared` resuelve A [v,X]=[g,B] en una sola factorización.
   Ambas versiones se conservan como ablaciones separadas, no se sobrescriben
   los tiempos de la primera con los de la segunda.
3. `transport-homotopy`: prototipo de continuación del residual
   F_alpha=(1-alpha)F_mezcla+alpha F_multi/Soret, con alpha=0, 0.5, 1.
   Cada Jacobiano intermedio combina las dos linealizaciones locales con los
   mismos pesos. No es todavía un controlador adaptativo del parámetro.
   El corrector final usa únicamente alpha=1 y las guardas productivas.
   El tiempo de las etapas intermedias, incluyendo una eventual falla, cuenta.

Las estrategias 2 y 3 son relevantes para el cambio a transporte multicomponente;
no se etiquetará como aceleración una ejecución de CH4 promediada por mezcla
en la que esos cambios no se activen.

## Protocolo

- Fuentes y configuraciones exactas junto a cada ejecución en
  `resultados/cold_strategies/`.
- Sin semillas persistentes; JIT y calentamiento completo separados. Los tiempos
  de inicialización y extremo a extremo se guardan además del solve.
- Baseline/candidato con el mismo refinamiento y dominio inicial por pareja.
  Guardas sin cambios: norma ponderada final <=1, residual <=1e4, malla aceptada.
- Pruebas locales de dependencias, flujo, conductividad y extremos de homotopía.
- Cribado de llamas completas a 1 y 10 atm. Repeticiones alternadas de candidatos
  prometedores; no promover por una sola pareja favorable.
- Errores alineados: Su <=0.1%, T <=0.5%, especies activas <=2%, calor <=5%.
- Registrar fallos de warmup y ejecución del candidato. Una caída a la ruta base
  no se puede presentar como prueba exitosa del algoritmo experimental.
- Sin benchmarks CPU simultáneos. No modificar políticas del sistema ni borrar
  corridas desfavorables. El cribado y la confirmación se analizan por separado.

## Estado

Implementados y medidos los tres prototipos aislados. Una cuarta prueba de
recorrido LU compilado se evalúa por separado, sin activación productiva.

- `cold_followup_local_checks.json`: pruebas locales completadas en GRI30 y
  h2o2 a 1/10 atm, incluyendo una traza negativa admisible. Las propiedades
  perturbadas son idénticas; diferencias máximas de flujo inferiores a 2e-17
  en esos estados. La conductividad coincide dentro de 3e-17.
- El primer intento del test local de homotopía usó por error el Jacobiano
  disperso predeterminado. Se corrigió el test para usar `block_tridiag`, igual
  que la campaña. No fue un fallo de convergencia de llama ni se contó como tal.
- `20260906_114911`: cribado baseline/dependencies terminado. Ocho llamas
  medidas aceptadas, perfiles idénticos bit a bit. CH4 1 atm: 3.9640/3.7158 s;
  H2/Soret 1 atm: 5.4810/5.0818 s; CH4 10 atm: 29.2011/26.5698 s;
  H2/Soret 10 atm: 53.4450/49.5550 s. Son parejas únicas.
- `20260906_115711`: cribado de acción de transporte y homotopía, H2/Soret
  a 1/10 atm terminado. No se usa un caso sin activación como evidencia favorable.
  A 1 atm: baseline 6.4433 s, acción 6.4568 s, homotopía 8.8326 s.
  A 10 atm: baseline 55.1980 s, acción 52.3688 s, homotopía 68.1603 s.
  La homotopía converge pero NO supera el filtro comparativo de precisión
  (velocidad de llama), además de ser más lenta: no se adopta este prototipo.
  La acción conserva perfiles; su pareja favorable a 10 atm no demuestra
  todavía ahorro estadístico. La variante con factorización compartida se
  ensaya por separado.
- 62 pruebas automáticas aprobadas después de añadir dos contratos nuevos:
  pesos iguales en residual/Jacobiano de la homotopía y layout correcto de
  las perturbaciones conservando la columna de velocidad.
- La verificación posterior del flujo con factorización compartida también
  pasó en GRI30/h2o2, 1/10 atm. Se comprobó además el flujo ordinario con
  Soret desactivado. Las diferencias máximas de flujo siguieron por debajo
  de 2e-17 en los estados locales ensayados.
- `20260906_120654`: confirmación exploratoria H2/Soret, tres parejas por
  candidato (`dependencies`, `transport-action-shared`) contra el mismo
  baseline, terminada. El código productivo conserva los 22 hashes de inicio de campaña.
  Dependencies: 1 atm 6.4948/6.5564 s, IC95 de razón [0.9727,1.0153];
  10 atm 62.3211/56.6771 s, IC95 [0.8836,0.9302]. Perfiles idénticos.
  Acción compartida: 1 atm 6.4948/6.7998 s, IC95 [1.0074,1.0788];
  10 atm 62.3211/61.0914 s, IC95 [0.9489,1.0261]. No hay mejora general.
  Evidencia: `cold_followup_confirmation_h2.json`. Tres parejas son exploratorias.
- `20260906_164509`: confirmación CH4 1/10 atm terminada, tres parejas:
  1 atm 4.3887/4.2299 s, IC95 de razón [0.7903,0.9638];
  10 atm 24.6053/23.2240 s, IC95 [0.9304,1.0280]. Perfiles idénticos.
  No mezclar los tiempos absolutos con los de H2 ni con el registro anterior
  de Cantera. Evidencia: `cold_followup_confirmation_ch4.json`.

La confirmación exploratoria del candidato `dependencies` se fija en tres
parejas alternadas por caso, sin mezclar el cribado. No sustituye la campaña
de siete parejas requerida para una decisión de producción/artículo.

## Interpretación matemática y límites

Para el Jacobiano, el número de estados térmicos se reduce a dos por nodo,
independientemente del número de especies. El número de estados químicos sigue
siendo proporcional al número de perturbaciones de composición: no se reduce
artificialmente la rigidez, no se elimina ninguna especie y no se usa una
semilla de otra llama. El ahorro global pertenece a la combinación; esta ronda
no atribuye un porcentaje causal separado al índice directo y a la columna u.

La acción de difusión usa la identidad de gradiente molar de suma cero. El
Jacobiano necesita todavía la respuesta a muchas direcciones, por lo que se
mantiene su representación matricial congelada y su actualización racional
exacta. Optimizar únicamente las evaluaciones de flujo del residual no elimina
la construcción de esas matrices: el ahorro posible en la llama completa está
limitado por la fracción de tiempo realmente sustituida.

La homotopía probada evalúa ambos residuales y ambas linealizaciones en la etapa
intermedia. El fallo del filtro de velocidad (0.3003% a 1 atm, 0.2278% a 10 atm)
no equivale a afirmar que el modelo físico intermedio sea inválido; indica que
la trayectoria y malla finales no conservan la precisión comparativa exigida.
No se adoptará este prototipo ni se relajará el filtro para presentarlo como éxito.

## Cuarta prueba: recorrido completo de LU compilado

Se conserva la eliminación por bloques y el pivotado de GETRF/GETRS. La nueva
prueba mueve el bucle de bloques a Numba y mantiene la actualización de Schur
en orden C; no repite la prueba descartada de construir todos los bloques en
orden Fortran. No reutiliza factores obsoletos ni aproxima el sistema lineal.

La interfaz pública Cython de SciPy se llama mediante direcciones resueltas
en cada proceso. Se comprueba la firma LP64 antes de llamar; otras firmas se
rechazan. Los pivotes se convierten explícitamente de base uno a base cero.
Las direcciones no se serializan en una caché JIT persistente.

`compiled_lu_local_checks.json`: factores y pivotes idénticos, residual lineal
relativo máximo 6.37e-16, singularidad rechazada. Nueve alternancias de una
factorización aislada, bloques de tamaño 55:

| Bloques | Baseline | Recorrido compilado | Reducción local |
|---|---:|---:|---:|
| 261 | 0.02427 s | 0.02050 s | 15.5% |
| 854 | 0.08035 s | 0.06740 s | 16.1% |

El cribado completo `20260906_165412` comparó cuatro llamas sin semillas,
con calentamiento separado. IMPORTANTE: este prototipo usa `cache=False`
por sus direcciones dinámicas. Su compilación adicional se paga una vez
por proceso y aparece en el calentamiento. Una ventaja en tiempo de solve
caliente NO demuestra ventaja desde el primer proceso sin calentamiento.
No se promoverá sin considerar ambos costes y sin repeticiones alternadas.

Referencias técnicas de la interfaz (no justifican una novedad matemática):
- [SciPy Cython LAPACK](https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html).
- [Numba, funciones Cython externas](https://numba.pydata.org/numba-doc/dev/extending/high-level.html).

El cribado CH4/10 atm dio 22.8907/27.6416 s. Se conservan 196 construcciones
de Jacobiano, 928 factorizaciones y 3652 resoluciones lineales. Los tiempos
del Jacobiano (11.6512/14.0373 s) y residual (4.7237/6.1509 s), que esta
variante no modifica, también aumentan. Esto muestra variación de duración
con igual trabajo contado, pero no identifica su causa ni permite corregir
estadísticamente el resultado a posteriori. No se descarta la corrida lenta.

En ese baseline, factorizar ocupa 4.0508 de 22.8907 s (17.7%). A modo de
estimación, incluso trasladar íntegramente un ahorro local del 16% a todas
esas factorizaciones reduciría el total aproximadamente 2.8%, si el resto
fuese constante. Es una estimación condicional de coste, no una predicción
validada ni una aceleración medida; las regiones instrumentadas están
anidadas y no deben sumarse como una partición exclusiva.

Resultados finales del cribado LU, una pareja por caso, tiempos de solve
sin semillas pero con JIT/entorno caliente:

| Caso | Baseline [s] | LU compilada [s] | Reducción total |
|---|---:|---:|---:|
| CH4, 1 atm, sin Soret | 4.8142 | 4.5316 | 5.9% |
| H2, 1 atm, multicomponente/Soret | 5.1157 | 5.0603 | 1.1% |
| CH4, 10 atm, sin Soret | 22.8907 | 27.6416 | -20.8% |
| H2, 10 atm, multicomponente/Soret | 50.0019 | 48.2074 | 3.6% |

Los cuatro perfiles son idénticos bit a bit, todas las llamadas de
factorización quedan cubiertas por la traza y no hubo excepciones ni
fallback silencioso. Evidencia: `cold_compiled_lu_screening.json`.
No se demuestra ahorro general ni estadístico: no se activa en producción.
Su interés queda limitado a una confirmación futura con orden alternado y
coste de compilación explícito. No se combina con otro candidato para
ocultar esta ablación desfavorable.

## Cierre y decisión de esta ronda

- Cinco campañas terminadas; 52 perfiles medidos con hashes, fuentes,
  dimensiones, finitud y métricas finales comprobados mediante
  `cold_followup_integrity_20260906.json`. Esa integridad no sustituye el
  filtro de precisión comparativa: la homotopía falló dicho filtro.
- 65 pruebas automáticas aprobadas (`python -m unittest discover -s tests -q`,
  13.467 s), incluidas conservación de entradas, pivotado, singularidad,
  restauración del contexto experimental y paso de matrices no bloque a
  su ruta original. La advertencia `row_stack` preexistente no es un fallo.
- Los 21 archivos Python V2 incluidos en las fuentes de inicio conservan
  sus hashes. No se incorporó ninguno de estos prototipos al pipeline
  productivo. Se preservan los cambios anteriores del usuario.
- No hay una nueva comparación temporal con Cantera en esta ronda.
- Los candidatos con señal favorable son dependencias exactas y recorrido
  LU compilado, pero sus resultados no autorizan adopción general. Las
  decisiones siguen siendo por evidencia conjunta, no por reglas de presión.
- Antes de promover: siete parejas alternadas en los cuatro casos,
  intervalos favorables sin empeorar certificación, y medición separada del
  primer proceso/JIT y del solve sin semillas. Una mejora local de LU no
  basta para afirmar una reducción reproducible del tiempo total.
