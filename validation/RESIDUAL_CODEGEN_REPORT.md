# Dos propuestas autorizadas: residual exacto y química generada

## Alcance

Prototipos aislados en `validation/`, sin cambios al solver productivo ni a
las tolerancias. No se reabren las variantes archivadas. El benchmark admite
los dos nuevos nombres de forma explícita; el valor por defecto sigue siendo
`baseline`. No se realiza todavía una nueva comparación con Cantera.

### Residual exacto compartido

En la aceptación estacionaria con transporte desactualizado permitido durante
Newton, se calcula `F_exact(x)` una sola vez. Su norma controla el mismo
rechazo exacto anterior; si se acepta, ese vector se entrega a la preparación
de la continuación. El consumidor copia el arreglo. No se reutiliza ningún
valor después de un callback, cambio de malla, estado o parámetro. Un rechazo
descarta el vector y obliga a evaluar el nuevo estado.

La prueba NO introduce una caché global ni suprime el certificado final.
Tampoco elimina todavía llamadas cuyo consumidor decide no guardar una
linealización: ese cambio es separable y no debe mezclarse con esta ablación.
En CH4 sin transporte lagged, conserva exactamente la única evaluación
anterior: esas corridas son controles de no regresión, no evidencia de ahorro
algorítmico en una rama que no se activa.

### Primera etapa de generación específica del mecanismo

Se genera código de índices y coeficientes constantes para las sumas de
Gibbs de reacción y la acumulación de producción de especies. Se mantiene
el orden original de cada suma y de las contribuciones de reacciones a una
especie. La termoquímica, Arrhenius, tercer cuerpo, falloff/Troe, reversibilidad
y productos másicos con extensión de signo siguen siendo las expresiones
productivas. No se eliminan especies ni reacciones.

Esto es una primera especialización de la topología estequiométrica, NO una
implementación completa de KinetiX ni la generación de todas las leyes de
reacción. Se evita prometer los beneficios de una generación total basándose
en esta parte. Las diferencias finitas del Jacobiano no se sustituyen por
derivadas analíticas.

La caché de compilación en memoria usa contenido de índices, coeficientes y
conteos, número de especies y hash del código original; no solo el tamaño
del mecanismo. El cálculo de esa clave cuenta en cada llamada cronometrada.
Presión, temperatura y demás datos físicos siguen siendo argumentos actuales.
Los kernels generados usan `cache=False`: la compilación por proceso se
registra separada y puede impedir el ahorro en un proceso completamente frío.
Los fuentes generados se guardan en el reporte de comprobación local.

## Protocolo

- Comprobación local previa en h2o2 y GRI30 a 1/10 atm: 36 estados por presión,
  desde 300 a 2400 K, ambos lados del cambio NASA y trazas H=-1e-8.
- Guardas de pruebas: reutilización del mismo vector, rechazo sin reutilizar
  un residual previo, control sin lag, orden de sumas generadas, aislamiento
  respecto de los candidatos archivados.
- Primer cribado: una pareja por variante contra V2 actual en CH4/H2,
  1/10 atm; H2 usa multicomponente/Soret, CH4 mezcla promediada sin Soret.
- Sin semillas guardadas; calentamiento completo separado; sin benchmarks
  simultáneos. Compilación inicial de los kernels generados registrada antes
  del calentamiento de llamas, no confundida con tiempo de solve.
- Malla y tolerancias iguales dentro de cada caso. Residual final <=1e4,
  norma ponderada <=1, malla aceptada y filtros de errores de perfil intactos.
- Una pareja solo criba. Ninguna promoción por un tiempo aislado favorable.

## Estado de ejecución

La sesión anterior no dejó un reporte local final ni un nuevo directorio de
campaña y ya no estaba en ejecución al reanudar. No se contabiliza como ensayo
completado ni se inventa una causa de terminación. Se reinició la misma prueba.
H2/h2o2 pasó la igualdad local exacta. La segunda ejecución fue detenida
explícitamente por el agente tras observar al menos 611 s de comprobación
local total: GRI30 seguía sin completar. Su baseline local sí había terminado;
la etapa generada no devolvió resultado. No es un fallo del solver de llamas
ni una medición de velocidad del kernel caliente. Evidencia transcrita de
salida observada y hash del generador:
`generated_chemistry_interrupted_check.json`.

La compilación/comprobación inicial de h2o2 fue 9.866 s; a 10 atm reutilizó
el kernel en memoria. Ambos ensayos locales dieron diferencias máximas cero
en rho, cp, omega y entalpías. No se interpreta el tiempo de una sola llamada
caliente como un microbenchmark estadístico.

Decisión para esta versión de generación: no adoptar. El despliegue de
sumas como funciones grandes tiene un coste de compilación inadecuado para
la prioridad de primer proceso frío. No se concluye que todo code generation
sea inútil ni que un generador precompilado externo tenga el mismo coste.
No hay comparación GRI30 completa ni aceleración demostrada de esta variante.

El residual se evalúa independientemente en `20260906_173355`, una pareja
en cada uno de cuatro casos. 72 tests aprobaron antes de medir, incluidas
las cinco guardas nuevas; no se ejecutaron tests en paralelo al benchmark.

## Resultado del residual compartido

La campaña terminó: ocho perfiles verificados en `exact_residual_integrity.json`
y análisis en `exact_residual_screening.json`. Todos los perfiles guardados son
idénticos y las guardas finales se conservan. Tiempos de solve de una pareja:

| Caso | V2 actual [s] | Compartido [s] | Evaluaciones evitadas |
|---|---:|---:|---:|
| CH4, 1 atm | 5.7657 | 5.7294 | 0 |
| H2/Soret, 1 atm | 7.4845 | 7.4188 | 7 |
| CH4, 10 atm | 31.0744 | 31.0496 | 0 |
| H2/Soret, 10 atm | 48.8424 | 56.4270 | 10 |

La traza coincide exactamente con la disminución de llamadas al residual:
1203 -> 1196 en H2/1 atm y 17141 -> 17131 en H2/10 atm. Los 67/536
Jacobianos y 208/2773 LU de esos casos no cambian. CH4 es un control no-op.
La menor cantidad de trabajo no demuestra aquí menor tiempo total: la pareja
de 10 atm fue más lenta. No se descarta esa medida ni se atribuye una causa
externa no comprobada. No se incorpora el prototipo a producción.

Los dos prototipos permanecen fuera de V2. La generación de funciones grandes
se archiva por el coste de compilación observado; el residual queda sin
evidencia suficiente de ahorro general. No hay una nueva comparación con
Cantera ni intervalos estadísticos basados en estas parejas únicas.
