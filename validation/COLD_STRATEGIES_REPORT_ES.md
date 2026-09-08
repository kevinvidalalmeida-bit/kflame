# Evaluación de estrategias para el arranque frío de V2

Fecha: 6 de septiembre de 2026. Las pruebas son ablaciones internas de V2,
no una nueva comparación temporal con Cantera. No se cambió el solver productivo.

## Resultado principal confirmado

Reutilizar los factores exclusivamente térmicos durante la construcción del
Jacobiano conserva exactamente las soluciones guardadas y reduce el tiempo en
los dos casos medidos a 1 atm. La confirmación a 10 atm no demuestra una ventaja
temporal general. Por ello no se activa automáticamente ni se introduce una
regla especial para una presión concreta.

| Caso GRI30, phi=1, 300 K | V2 de referencia, mediana [s] | Reutilización térmica, mediana [s] | IC pareado 95% del cociente candidato/referencia | Decisión |
|---|---:|---:|---|---|
| CH4/aire, 1 atm, sin Soret | 4.1501 | 3.8242 | [0.8641, 0.9468] | Mejora demostrada en este ensayo |
| H2/aire, 1 atm, multicomponente y Soret | 6.0027 | 5.4141 | [0.9012, 0.9562] | Mejora demostrada en este ensayo |
| CH4/aire, 10 atm, sin Soret | 23.6311 | 24.6731 | [0.8894, 1.1200] | No concluyente; la mediana candidata es mayor |
| H2/aire, 10 atm, multicomponente y Soret | 51.4042 | 47.4205 | [0.9046, 1.0169] | No concluyente pese a la menor mediana |

Son siete parejas por caso, con orden alternado y calentamiento completo
separado. No se seleccionó la mejor ejecución ni se excluyeron parejas
desfavorables. La variación temporal fue considerable: no se sustituye el
estimador declarado por otro que resulte más favorable después de ver los datos.
El dominio inicial es 0.03 m, con la expansión habitual. Los criterios
slope/curve son 0.04/0.08 a 1 atm y 0.01/0.02 a 10 atm; se mantienen idénticos
entre variantes dentro de cada caso. Por ello, la diferencia temporal entre
filas no debe atribuirse exclusivamente al cambio de presión.

En estas 56 llamas medidas de la comparación térmica (28 parejas), los campos
espaciales guardados son idénticos bit a bit entre candidato y referencia.
Esto verifica la equivalencia en los casos ensayados, no la independencia de
malla del modelo ni su validación experimental.

## Qué se probó matemáticamente

### Reutilización térmica exacta

Para un estado de composición perturbada a temperatura fija, NASA, Arrhenius,
las constantes de equilibrio y la parte térmica de Troe pueden reutilizarse.
La evaluación de fuentes se separa conceptualmente en:

\[
\dot{\boldsymbol\omega}(T,\boldsymbol Y,p)
=\mathcal R(\boldsymbol Y,p;\mathcal K(T)).
\]

Se actualizan siempre densidad, calor específico de mezcla, concentraciones,
productos de acción de masas, terceros cuerpos y falloff dependiente de
composición. No se congelan estas dependencias ni se modifica el residual.
Tampoco se reutiliza un Jacobiano envejecido o una llama guardada de otra corrida.
La construcción de las tablas térmicas y el agrupamiento se pagan dentro del
tiempo medido. La complejidad total de la química no desaparece: se ahorran
operaciones repetidas, no todas las evaluaciones de reacción.

### Memoria y potencias compartidas

Se ensayó limitar los lotes de perturbaciones T/Y a unos 2 MiB y, por separado,
compartir `(p_ref/(R*T))**delta_nu` entre reacciones con el mismo exponente.
Las siete parejas a 1 atm no demostraron ahorro general en ninguna opción.
Los cribados a 10 atm tampoco justifican su adopción general.

La variante adicional `streamed` escribe directamente las contribuciones de
cada lote en los bloques del Jacobiano, sin concatenar todos los resultados
termoquímicos perturbados. Su cribado terminó con perfiles idénticos: CH4 a
1 atm 4.7280/4.7232 s; H2/Soret a 1 atm 6.5971/6.7007 s; CH4 a 10 atm
28.2187/28.2378 s; H2/Soret a 10 atm 43.0147/54.0387 s (baseline/candidato).
No demuestra ahorro general. Son parejas aisladas; no se confunde con el primer
experimento de copias temporales acotadas ni se afirma una medición de memoria RSS.

### Granularidad del paralelismo

Para GRI30 se midieron lotes de 1, 2, 4, 8, 16, 32, 64 y 256 estados, con
nueve muestras alternadas por tamaño y 50 llamadas por muestra. La calibración
eligió un umbral serial cero: conservar el kernel paralelo en todos esos tamaños.
Un despachador que acaba llamando siempre al mismo kernel no constituye una
mejora algorítmica, aunque una ejecución aislada resulte más corta.

### Presupuesto no lineal ligado a la malla

Se probó usar la discrepancia de restricción/interpolación para ampliar el
umbral de la corrección de Newton únicamente en mallas todavía marcadas por el
refinador. La corrección final recupera norma ponderada <=1 y residual <=1e4.
El indicador no es una cota demostrada del error de discretización. El cribado
conservó la precisión, pero no mostró ahorro consistente en los cuatro casos.

### Corrección FAS de dos mallas

El prototipo resuelve aproximadamente:

\[
F_H(x_H)=F_H(Rx_h)-R F_h(x_h),
\]

conservando frontera y anclaje, y acepta la corrección interpolada solo si
disminuye el residual fino completo. Después ejecuta el corrector fino habitual.
Se trata de un prototipo de dos mallas, no de una implementación multigrid
recursiva completa. Hubo correcciones efectivas y ninguna excepción registrada
en los cribados; aun así el tiempo medido aumentó en los cuatro casos. Por
ejemplo, H2/Soret a 10 atm pasó de 48.64 a 66.81 s en una pareja. Se descarta
la adopción de este prototipo, no toda la familia FAS.

## Verificación y límites

- 60 tests aprobados, con pruebas de guardas, signos admisibles en cinética,
  restricciones y propiedades. Las variantes de kernel coincidieron exactamente
  con la referencia en pruebas locales de GRI30/h2o2 a 1 y 10 atm.
- Las comparaciones de perfiles usan alineación térmica e integrales sobre una
  malla común: límites de 0.1% en velocidad, 0.5% en T, 2% en especies activas y
  5% en liberación de calor. Las fuentes de calor del análisis se posprocesan
  con Cantera, fuera de los tiempos de resolución.
- Se conserva el guardián de residual 1e4, la norma final <=1 y el refinamiento
  requerido. No se impuso un umbral nuevo para hacer parecer peor a la referencia.
- Frío significa sin perfiles guardados de otras llamas. No significa compilación
  JIT completamente fría. El bootstrap interno de transporte de cada llama con
  Soret pertenece al tiempo medido de esa misma llama.
- La conservación sigue registrada. La igualdad entre variantes no cierra por
  sí sola la convergencia de malla/dominio pendiente del artículo.
- Los prototipos permanecen en `validation/`. El código productivo de V2 no carga
  estas variantes y conserva las mejoras verificadas de campañas anteriores.

Evidencia: `cold_strategy_audit_20260906.json`,
`kinetics_granularity_calibration.json` y campañas enlazadas en
`COLD_STRATEGIES_STATUS.md`. La integridad de los 122 perfiles medidos fue
comprobada en `cold_artifact_integrity_20260906.json`. Esta ronda terminó sin
promoción global. La segunda ronda independiente se describe en
`COLD_FOLLOWUP_20260906.md`.
