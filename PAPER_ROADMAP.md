# Hoja de ruta para artículo y cierre de robustez

## Estado de implementación: predictor--corrector adaptativo

La versión actual incorpora una continuación exterior en
\(\lambda=\log\phi\), separada del corrector interno Newton--PTC--BE. El
generador ofrece tres políticas reproducibles:

- `cold`: no reutiliza perfiles vecinos;
- `fixed`: reproduce la región de confianza multiplicativa histórica;
- `adaptive-pc`: predictor copia/secante, corrección completa certificada,
  adaptación de paso, puntos puente, reintentos y recuperación fría explícita.

Cada corrida `adaptive-pc` escribe una tabla con marcas de filas solicitadas y
puente, además de una traza JSON con el defecto del predictor, decisiones de
paso, malla, dominio, tiempos y aceptación. Esta implementación es una
capacidad experimental; no debe presentarse como una mejora medida hasta
completar la ablación predefinida abajo.

La corrida nativa escribe también continuation_schedule.json. El generador de
referencia puede recibir este archivo para repetir exactamente la misma secuencia
final de \(\phi\), incluidos los puentes. El análisis pareado compara únicamente
las filas solicitadas, por lo que los puentes aumentan la resolución de la tabla
sin alterar silenciosamente la comparación física solicitada.

### Criterio de promoción a ruta productiva

Comparar, en al menos siete réplicas pareadas, `cold`, `fixed`, puentes de paso
fijo y `adaptive-pc`. Promover `adaptive-pc` solo si el intervalo pareado de
coste favorece la reducción, no disminuye la tasa de certificación y, frente al
baseline V2 en la misma configuración, satisface
\(\Delta S_u\le0.1\%\), \(E_2(T)\le0.5\%\),
\(E_2(Y_k)\le2\%\) para especies activas y
\(E_2(\dot q)\le5\%\). Si falla cualquiera, conservar `fixed` y publicar el
resultado negativo.

## Afirmación central defendible

El resultado no es un nuevo Newton, una nueva LU ni un nuevo concepto FGM por
separado. La contribución demostrada es una estrategia de **aceleración
certificada de familias de llamas** que integra:

1. continuación paramétrica local con reinicio fuera de una región de confianza;
2. aceptación conjunta no lineal, espacial, conservativa y física;
3. factorización pivotada por bloques y sustitución espacial fusionada;
4. evaluación extremo a extremo sobre barridos completos, no solo sobre
   microbenchmarks.

La evidencia versionada está en `evidence/tfm_20260903/`. Los perfiles y tablas
NPZ completos permanecen fuera de Git por tamaño, pero deben conservarse con sus
metadatos y hashes durante la preparación del artículo.

## Resultados mínimos antes de enviar un artículo

### Prioridad P0: sostener las afirmaciones actuales

- Ejecutar al menos siete pares alternados V2--Cantera y siete pares
  sustitución-LAPACK--sustitución-fusionada.
- Reportar mediana, rango intercuartílico e intervalo de confianza pareado por
  bootstrap; publicar también todos los tiempos individuales.
- Separar ejecución fría, calentamiento JIT y ejecución caliente.
- Repetir convergencia de malla, dominio y tolerancias para los dos solvers.
- Publicar conservación de masa, suma de especies y balances elementales.
- Comparar perfiles alineados de temperatura, liberación de calor y especies
  principales y radicalarias.
- Registrar CPU, memoria, sistema operativo, versiones, hilos, revisión Git,
  comando literal y hashes de los resultados.

### Prioridad P1: demostrar generalidad

- Barrer metano--aire entre los límites numéricos pobre y rico con una malla de
  parámetros más densa que los cinco puntos demostrativos.
- Añadir temperatura de entrada y presión como factores experimentales.
- Incorporar un segundo mecanismo o combustible. Hidrógeno con y sin Soret es
  especialmente informativo; requiere transporte coherente en ambos solvers.
- Medir escalabilidad frente al número de nodos y especies, memoria máxima,
  reconstrucciones de Jacobiano, factorizaciones y lados derechos resueltos.
- Publicar también fallos, reinicios y dominios expandidos para evitar sesgo de
  supervivencia.
- Ejecutar las cuatro estrategias sobre cada caso: arranque frío, región fija,
  puentes de paso fijo y predictor--corrector adaptativo. La instrumentación
  opcional de perfil registra contadores internos fuera de la medición temporal
  principal.

### Prioridad P2: convertir la demostración FGM en tabla validada

- Insertar flamelets adaptativamente en `Z` hasta que la validación dejando uno
  fuera satisfaga una tolerancia predefinida.
- Reservar flamelets de prueba que nunca participen en la selección de la malla.
- Verificar positividad, suma de masa, conservación elemental y error de los
  términos fuente después de interpolar.
- Medir por separado generación fría, regeneración cacheada y consulta de tabla.
- Para toda comparación V2--Cantera, congelar y repetir la misma secuencia final
  de \(\phi\), incluidas las filas puente.
- No presentar la tabla de cinco filas como cierre CFD: el error observado de
  liberación de calor muestra que su resolución paramétrica es insuficiente.

## Extensiones matemáticas con potencial

### 1. Región de confianza adaptativa basada en defecto predictor--corrector

Esta política ya está implementada y debe evaluarse como hipótesis, no como un
resultado. Para el predictor secante y la solución corregida, el controlador
mide un defecto ponderado en la malla final y adapta \(\Delta\log\phi\) con una
regla limitada, calibrada con la mediana de los tres primeros defectos secantes
aceptados. Las copias del primer vecino no calibran esa referencia. Un rechazo
reduce el paso; al agotarse el presupuesto se ejecuta un arranque frío explícito.

La política se mantiene solo si reduce el trabajo hasta una llama certificada
con los criterios de promoción declarados arriba. La posible novedad publicable
es su integración trazable con malla adaptativa y certificación FGM, no el
concepto general de predictor--corrector.

### 2. Continuación como grafo de coste

Para una tabla multidimensional, cada estado certificado puede ser un nodo y
cada continuación admisible una arista. Un modelo actualizado con tiempos,
reinicios y defecto del predictor estimaría el coste de cada arista. Resolver la
siguiente frontera de menor coste permitiría combinar localidad y paralelismo.
La comparación correcta sería contra orden lexicográfico, vecino más cercano y
arranque frío.

### 3. Muestreo paramétrico con estimador a posteriori

En cada intervalo de `Z`, resolver un flamelet de control y comparar sus campos
con la interpolación de los extremos. Un indicador útil debe combinar error
normalizado de temperatura, especies y fuentes:

```text
eta_I = max_q ||q_mid - I(q_left, q_right)|| / (atol_q + rtol_q ||q_mid||).
```

Se divide únicamente el intervalo con mayor `eta_I`. Esta línea ataca el número
de llamas necesarias, que puede producir una ganancia mayor que optimizar otro
porcentaje de un solve individual.

### 4. Criterio de trabajo, no solo de residual

Las decisiones de Jacobiano y globalización deberían optimizar una métrica como

```text
W = c_J N_J + c_F N_F + c_S N_rhs + c_R N_refine,
```

donde los coeficientes se estiman del perfil real. Este modelo permite decidir
cuándo reconstruir el Jacobiano o reiniciar una continuación. Debe validarse
fuera de la campaña usada para ajustar los coeficientes.

## Estructura sugerida del artículo

1. Problema físico, alcance y certificado de aceptación.
2. Solver estructurado y estrategia PTC--SER/BE.
3. Predictor--corrector adaptativo en \(\log\phi\).
4. Diseño experimental justo y reproducible.
5. Verificación numérica y robustez.
6. Ablación causal: bloque LU, PTC--SER, predictor y control adaptativo.
7. Aplicación FGM: coste, error y densidad paramétrica.
8. Limitaciones del método y del dominio de validación.

Título de trabajo:

> Certification-aware continuation and hybrid block solves for accelerated
> premixed-flame manifold generation

La afirmación debe limitarse al dominio ensayado hasta completar P0 y P1. Las
extensiones distintas del predictor--corrector ya implementado no deben
presentarse como implementadas o descubiertas antes de medirlas.
