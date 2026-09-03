# Hoja de ruta para artículo y cierre de robustez

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

### Prioridad P2: convertir la demostración FGM en tabla validada

- Insertar flamelets adaptativamente en `Z` hasta que la validación dejando uno
  fuera satisfaga una tolerancia predefinida.
- Reservar flamelets de prueba que nunca participen en la selección de la malla.
- Verificar positividad, suma de masa, conservación elemental y error de los
  términos fuente después de interpolar.
- Medir por separado generación fría, regeneración cacheada y consulta de tabla.
- No presentar la tabla de cinco filas como cierre CFD: el error observado de
  liberación de calor muestra que su resolución paramétrica es insuficiente.

## Extensiones matemáticas con potencial

### 1. Región de confianza adaptativa basada en defecto predictor--corrector

La cota fija de razón `1.15` está demostrada empíricamente, pero todavía es una
heurística. Para un predictor secante `x_pred` y la solución corregida `x*`, se
puede medir

```text
eta_i = ||x* - x_pred||_W / max(|Delta log(phi)|, epsilon).
```

El radio siguiente puede contraerse o expandirse con una regla limitada,

```text
r_(i+1) = clip(r_i * (eta_target / eta_i)^alpha, r_min, r_max).
```

Debe aceptarse solo si reduce el trabajo total hasta una llama certificada. La
novedad potencial sería su integración con malla adaptativa y certificación FGM,
no el concepto general de predictor--corrector.

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

1. Problema de llama libre y criterio de certificación.
2. Solver estructurado y sustitución por bloques fusionada.
3. Continuación protegida para familias de flamelets.
4. Protocolo pareado y estudios de ablación.
5. Fidelidad frente a Cantera, escalabilidad y mapa de fallos.
6. Demostración FGM con validación fuera de muestra.

Título de trabajo:

> Certification-aware continuation and hybrid block solves for accelerated
> premixed-flame manifold generation

La afirmación debe limitarse al dominio ensayado hasta completar P0 y P1. No se
debe presentar ninguna de las extensiones de esta hoja como implementada o
descubierta antes de medirla.
