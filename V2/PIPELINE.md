# Flujo principal de V2 y próximos cuellos de botella

Estado tras la limpieza solicitada: ninguna variante de las dos últimas
rondas se incorpora a producción. Todos esos candidatos quedan archivados;
el benchmark normal solo ejecuta `baseline`. Las tres nuevas hipótesis,
todavía sin implementar, están en `validation/NEXT_IMPROVEMENTS.md`.

## Ruta retenida (limpieza del 2026-09-05)

1. Construir el problema de llama libre y el anclaje.
2. Evaluar termoquímica/cinética nativas, con continuación consistente de las
   concentraciones negativas admisibles durante Newton.
3. Convección upwind, difusión conservativa y cierre de transporte solicitado.
4. Newton amortiguado; si falla, una corrección PTC-SER. Si esa corrección se
   rechaza, BE implícito de respaldo. No hay controladores BE alternativos.
5. Jacobiano local por bloques, LU directa SciPy/LAPACK y sustitución compilada.
6. Refinar y comprobar dominio. Aceptar únicamente con residual completo,
   norma ponderada y criterios espaciales; Soret se reevalúa sin transporte
   desactualizado antes de certificar.
7. En FGM, reutilización local de perfiles y refresco local certificado del
   Jacobiano del corrector. Se conserva porque tiene evidencia favorable.

Para Soret frío se conserva el arranque nativo promediado por mezcla seguido
del corrector multicomponente/Soret, con refinamiento final estricto.
La inicialización mezcla/equilibrio todavía usa Cantera; los kernels Soret,
termoquímicos y cinéticos no. No se afirma independencia de toda la aplicación.

## Eliminado del solver y sus interfaces

- Modos PTC-auto, PTC-rescue y BE exclusivo; permanece el BE interno de respaldo.
- Pesos de Newton congelados y controles CLI de esos experimentos.
- Mezcla de convección centrada/upwind: diez ramas duplicadas en residual y
  Jacobiano. Los metadatos antiguos upwind=1 siguen siendo legibles; solicitar
  otra mezcla produce un error explícito, no una reinterpretación silenciosa.
- La selección experimental de secuencias BE del benchmark Soret.

Las configuraciones antiguas de esas variantes ya no se ejecutan con esta
revisión. Sus métricas, trazas, huellas y decisiones se conservan como evidencia
histórica; no se borran los resultados científicos. El código eliminado no se
mantiene escondido bajo un interruptor. No se tocaron manuscritos de Downloads
ni archivos ajenos a V2/FGM.

Se conservaron herramientas de comparación, auditoría, pruebas y campañas:
instrumentar un método no equivale a introducir otra ruta de solución.
Los estudios paramétricos de investigación fuera del solver no se presentan
como optimizaciones productivas. El cierre denso de transporte y los backends
de referencia usados por pruebas se conservan como oráculos de validación.

## Mejora de implementación incluida

Los buffers auxiliares que se pasaban al kernel local se construían con
`dict.get(clave, np.zeros(...))`: Python evalúa ese valor por defecto incluso
si la clave existe. Ahora se crean una vez por caché de Jacobiano. No cambia
ninguna operación de las ecuaciones. Esto reduce asignaciones de memoria,
pero no se atribuye una aceleración porcentual sin ablación pareada.

## Perfil que motivó las nuevas mejoras

La traza posterior a la limpieza, H2/Soret a 10 atm con slope=.01 y curve=.02
(`benchmark_20260905_095619`), divide 45.454 s de V2 en 26.549 s de arranque
promediado por mezcla y 18.721 s de corrector multicomponente, más orquestación.
El Jacobiano consume 12.945 s/532 construcciones en el arranque y
11.741 s/22 construcciones en el corrector. Dentro de esos tiempos, la
precomputación termoquímica ocupa 6.812 y 5.849 s. La LU ocupa 4.915 y 1.562 s.
Son regiones anidadas: no sumar química al Jacobiano. Esto prioriza el trabajo
por construcción del Jacobiano, no solo reducir sus llamadas ni sustituir LU.
La traza es diagnóstica y añade instrumentación; no es una ablación causal.

## Cambios implementados tras el perfil

1. **Productos estequiométricos especializados en el kernel fusionado.** Se
   precalcula la lista de participantes para órdenes enteros totales hasta tres.
   Sus concentraciones se multiplican directamente; los órdenes generales y
   resultados extremos que perderían representabilidad conservan la vía
   logarítmica. Se mantiene el suelo numérico anterior, la extensión con signo
   de los estados de Newton y el tercer cuerpo. No cambia el modelo químico ni
   se añade un Jacobiano analítico. Las rutas NumPy/densa/dispersa siguen siendo
   oráculos de comparación independientes del nuevo kernel fusionado.
2. **GETRF/GETRS sin envoltorios repetidos.** Se mantiene el bloque de trabajo
   en orden C y el pivotado LAPACK. Se evita la selección repetida de rutinas
   y los envoltorios de alto nivel por bloque, comprobando explícitamente
   los códigos de error. No se reutilizan factorizaciones inexactas. Crear
   directamente todos los bloques en orden Fortran fue más lento y se retiró.

También se retiró el método termoquímico sin cinética y su kernel duplicado,
sin consumidores en el repositorio, que habían quedado de la ruta de
Jacobiano analítico descartada, así como los multiplicadores sin consumidores
del antiguo escalado lineal. No se eliminó ningún resultado histórico.

La confirmación usó siete pares alternados a 1 atm, con calentamiento completo
separado y sin reutilizar perfiles. La referencia restituye también el orden
anterior de evaluación química y los envoltorios LU anteriores:

| Caso | Referencia anterior | Combinación | Reducción combinada |
| --- | ---: | ---: | ---: |
| CH4, mezcla promediada, sin Soret | 4.9522 s | 4.5983 s | 7.15% |
| H2, multicomponente y Soret | 6.9245 s | 6.5162 s | 5.90% |
| H2, multicomponente y Soret, 10 atm (3 pares) | 53.9952 s | 48.0511 s | 11.01% |

IC95 bootstrap pareado de la razón combinación/baseline: [0.91133,0.95255]
para CH4 y [0.92865,0.96370] para H2. Los 28 solves medidos fueron aceptados
y preservaron sus mallas, con diferencias máximas de T inferiores a 1e-8 K.
La confirmación adicional a 10 atm aceptó los seis solves medidos, con 854
nodos en ambas variantes y diferencia máxima de T de 2.74e-10 K. Su IC95
bootstrap de la razón fue [0.88274,0.90347], pero tres pares siguen siendo
evidencia exploratoria, no la campaña definitiva de un artículo.
Evidencia conjunta: `validation/kernel_ablation_confirmed_20260905.json`.
Son ablaciones internas de V2; no comparaciones nuevas con Cantera. La carga
y frecuencia de la máquina variaron entre campañas, por lo que no se comparan
tiempos absolutos de revisiones medidas en momentos distintos.

Verificación final: 55 pruebas aprobadas, incluidos pivotado, singularidad,
conservación, fuentes químicas con signos admisibles, órdenes generales y
productos extremos. Las fuentes se contrastaron con Cantera para GRI30 y
H2/O2 entre 300 y 2400 K, a 1, 10 y 100 atm; esto no equivale a haber validado
llamas completas a 100 atm. Las llamas de rendimiento usan entrada a 300 K.
Entorno: SciPy 1.17.1, Numba 0.65.1, NumPy 2.4.6 y Cantera 3.2.0.

Una pareja directa posterior con Cantera, H2/Soret a 10 atm y los mismos
criterios finos, dio V2 46.8486 s y Cantera 81.7868 s, ambos aceptados por
el protocolo. Nodos: 854/868; diferencia de velocidad: 0.09605%. Es un
control individual, no una mediana ni un intervalo estadístico nuevo. Véase
`validation/kernel_optimized_cantera_20260905.json`. El JIT conserva caché;
no se presenta como compilación completamente fría. Sigue pendiente cerrar
la independencia de malla/dominio de la familia, separada de esta regresión.

El cribado inicial de tres variantes queda separado en
`validation/kernel_ablation_screening_20260905.json`: reconstruía los
productos logarítmicos con otra agrupación de operaciones. No se mezclan
sus tiempos con la confirmación ni se presentan como la revisión anterior
exacta. LAPACK solo no mostró evidencia concluyente de ahorro total en cada
caso; no atribuirle el porcentaje de la combinación.

## Auditoría de nuevas estrategias en frío (2026-09-06)

Las variantes de esta auditoría permanecen en `validation/`; no modifican
el flujo productivo descrito arriba. El estado y las evidencias completas
se encuentran en `validation/COLD_STRATEGIES_STATUS.md` y
`validation/cold_strategy_audit_20260906.json`.

Se probaron reutilización térmica exacta del Jacobiano, lotes acotados de
perturbaciones, potencias compartidas de equilibrio, granularidad de hilos,
presupuesto no lineal ligado a interpolación y una corrección FAS de dos mallas.
Todas tienen cribados de llamas completas a 1 atm; las primeras tres y las
dos variantes numéricas también a 10 atm, con CH4 sin Soret y H2 con Soret.
La granularidad calibrada para GRI30 seleccionó conservar paralelismo en
todos los tamaños medidos: no es un nuevo algoritmo más rápido.

La reutilización térmica conserva las diferencias finitas: agrupa las
temperaturas exactamente iguales dentro de un ensamblado, calcula NASA,
Arrhenius, Kc y la parte térmica de Troe una vez y vuelve a evaluar todas
las dependencias de composición. No es el Jacobiano analítico descartado
ni una caché persistente de perfiles. En siete parejas a 1 atm redujo las
medianas CH4 de 4.1501 a 3.8242 s y H2/Soret de 6.0027 a 5.4141 s, con
perfiles guardados bit a bit idénticos. La confirmación a 10 atm terminó:
CH4 23.6311/24.6731 s y H2/Soret 51.4042/47.4205 s (baseline/candidato).
Ambos intervalos pareados incluyen uno; no se activa como mejora general.
El ensamblado directo por lotes también terminó sin ahorro general.
La siguiente ronda está en `validation/COLD_FOLLOWUP_20260906.md`.

Las potencias compartidas y los lotes menores no demostraron una ventaja
general. El presupuesto no lineal y FAS conservaron la precisión del cribado,
pero no mejoraron consistentemente el tiempo. Esto descarta su adopción
inmediata, no la familia matemática completa. Hay 60 pruebas unitarias
aprobadas, incluidas cinco nuevas pruebas de guardas del laboratorio.

## Otras hipótesis que todavía no se activan

1. **Biblioteca BLAS, solo como ablación de entorno.** SciPy ya usa LAPACK
   optimizado. Comparar otra distribución BLAS en un entorno aislado puede
   cambiar rendimiento, pero no justifica sustituir globalmente dependencias.
   Registrar proveedor, hilos, versiones y repetir los mismos problemas.

Fuentes técnicas:
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lapack.get_lapack_funcs.html
- https://numba.readthedocs.io/en/stable/user/parallel.html
- https://www.cantera.org/3.2/cxx/d4/dc8/group__Stoichiometry.html

No reabrir POD, GMRES reciclado, BDF2, low-rank de desplazamientos diagonales,
escalado físico, GPU de lotes pequeños ni JAX directo: véase el registro de
descartes. No activar fastmath indiscriminadamente en el solver certificado.

Una nueva mejora entra solo después de pruebas locales, perfiles/malla y
certificación, y tiempos alternados a 1 y 10 atm. Mantener separado el coste
de JIT del arranque sin perfiles; nunca ocultar una regresión en una media global.

## Cierre de la segunda ronda (2026-09-06)

`validation/COLD_FOLLOWUP_20260906.md` contiene 52 perfiles auditados y 65
pruebas automáticas aprobadas. La homotopía intermedia se rechaza por tiempo
y precisión comparativa; la acción directa de difusión no demuestra ahorro
general. Las dependencias exactas del Jacobiano muestran señales favorables
en dos casos de cuatro, sin habilitar reglas por presión.

La prueba adicional de recorrido LU compilado conserva factores/pivotes y
los perfiles finales exactamente. Reduce el microbenchmark ~16%, pero
el cribado de llamas muestra tres parejas favorables y una desfavorable.
Su coste JIT por proceso también debe contabilizarse. No se promueve.
Los prototipos permanecen fuera de V2 y el código productivo queda intacto
respecto del comienzo de esta ronda.
