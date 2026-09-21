# Hoja de ruta para artículo y cierre de robustez

## Actualización Soret: química consistente durante Newton (2026-09-05)

Se corrigió el recorte de concentraciones negativas admisibles durante la
iteración. La corrección es nativa, mantiene las tasas de estados físicos
positivos y está cubierta por fuentes/derivadas de dos mecanismos en cuatro
rutas de evaluación. La revisión tiene 49 pruebas automáticas satisfactorias.
La evidencia inicial recupera el arranque H2/GRI30 a 10 atm con PTC-SER normal
y reduce los tiempos de los controles a 1 atm, sin relajar la aceptación.

Esta corrección no es un nuevo modelo de Soret: elimina una incoherencia
entre la globalización no lineal y la extensión numérica de la química.
Debe documentarse como verificación de implementación, sin atribuirle
originalidad teórica. Las repeticiones y el estudio de error espacial se
registran por revisión, sin mezclar con los siete pares anteriores.
Véase `validation/SORET_SIGNED_KINETICS_DIAGNOSIS.md`.

Para el artículo siguen pendientes: ampliar independencia de malla/dominio,
incertidumbre experimental, rango paramétrico FGM/Soret y ablaciones causales.
Los coeficientes, flujos, química, mezcla y equilibrio de inicialización se
evalúan nativamente. Cantera queda como referencia opcional; la independencia
de software no sustituye la validación de malla, dominio o datos físicos.

## Estado de implementación: predictor--corrector y refresco local certificado

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

El corrector de continuación incorpora además un rescate distinto del
controlador exterior: si una prueba de amortiguamiento no contrae, mide el
defecto de linealización \(F(x+\alpha s)-F(x)-\alpha Js\) por bloque espacial.
Solo reevalúa el vecindario afectado si su extensión es como máximo el 35\% de
la malla; después factoriza el sistema actualizado completo y exige el mismo
test de contracción y certificado. No modifica el predictor ni el paso en
\(\log\phi\), y una linealización parcialmente renovada nunca se exporta como
tangente exacta al siguiente flamelet.

Esta segunda política ya tiene evidencia controlada: siete pares alternados en
CH4/aire, GRI-Mech 3.0, 300 K y 10 atm para
\(\phi=1.00\rightarrow1.02\rightarrow1.04\) produjeron una mediana de
\(2.023\times\) en las transiciones, con IC bootstrap pareado 95\%
\([1.943,2.056]\), sin pérdida de certificación. Los máximos frente al baseline
fueron \(\Delta S_u=1.36\times10^{-8}\), \(E_2(T)=2.10\times10^{-9}\),
\(E_2(Y)=3.94\times10^{-8}\) y \(E_2(\dot q)=6.23\times10^{-8}\). A 1 atm
no se activó. Es una mejora demostrada del corrector de continuación, no una
afirmación de superioridad global ni una comparación fría frente a Cantera.

La corrida nativa escribe también continuation_schedule.json. El generador de
referencia puede recibir este archivo para repetir exactamente la misma secuencia
final de \(\phi\), incluidos los puentes. El análisis pareado compara únicamente
las filas solicitadas, por lo que los puentes aumentan la resolución de la tabla
sin alterar silenciosamente la comparación física solicitada.

### Criterio de promoción a ruta productiva

Comparar, en al menos siete réplicas pareadas, `cold`, `fixed`, puentes de paso
fijo y `adaptive-pc`. Promover `adaptive-pc` solo si el intervalo pareado de
coste favorece la reducción, no disminuye la tasa de certificación y, frente al
baseline KFLAME en la misma configuración, satisface
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

## Revisión de preparación editorial: qué está listo y qué falta

### Ya defendible dentro del alcance actual

- Formulación, certificado de aceptación y verificación espacial/dominio para
  CH4--aire a 1 atm; la tesis ya informa la convergencia hacia una referencia
  ultra, conservación y perfiles alineados frente a Cantera.
- Dos ablaciones causales separadas: región de confianza fija y sustitución
  fusionada por bloques; el refresco local certificado añade una tercera
  ablación específica del corrector.
- Una conclusión honesta sobre FGM: la tabla de cinco filas conserva masa pero
  no tiene resolución paramétrica suficiente, pues el error leave-one-out de
  \(\dot q\) llega a 27.9\%.

### Bloqueadores antes de enviar

1. **Estadística temporal:** aún faltan, en un paquete versionado y citado, los
   siete pares alternados KFLAME--Cantera para los casos principales, con tiempos
   individuales, IQR e intervalo bootstrap. Una única corrida de barrido no es
   resultado de revista.
2. **Generalidad P1:** documentar toda la matriz de \(T_{\rm in}\), presión,
   \(\phi\) y al menos un combustible/mecanismo adicional. Los resultados a
   alta presión deben aparecer también cuando KFLAME no gane: el refresco local
   mejora transiciones, pero no prueba una ventaja fría a 10 atm.
3. **Tabla FGM P2:** insertar filas hasta superar una tolerancia fijada antes
   de mirar los resultados, y validarla con flamelets de retención. La actual
   no es todavía una tabla apta para reclamar uso CFD.
4. **Trazabilidad de publicación:** depositar entradas, scripts de campaña,
   perfiles, trazas, entorno y hashes en un archivo estable con DOI; el artículo
   debe citar ese material suplementario y no rutas o listados del repositorio.
5. **Posicionamiento y validación física:** dejar explícito que Cantera es una
   referencia numérica. Para afirmar precisión del modelo físico hacen falta
   datos experimentales o una sección de limitaciones que restrinja el artículo
   a verificación numérica.

## Resultados mínimos antes de enviar un artículo

### Prioridad P0: sostener las afirmaciones actuales

- Ejecutar al menos siete pares alternados KFLAME--Cantera y siete pares
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
- Para transiciones locales aceptadas, repetir además la ablación del refresco
  por defecto de linealización con y sin la política. Reportar activaciones,
  fracción de bloques, razones temporales pareadas y casos donde permanece
  inactivo; no extrapolar el resultado actual de 10 atm a toda la matriz.

### Extensión Soret nativa: implementada y con validación inicial

El transporte multicomponente/Soret ya se prepara y evalúa sin importar Cantera.
Los ajustes de viscosidad, difusión binaria y A*, B*, C* se calculan desde los
parámetros moleculares y las tablas universales Monchick--Mason incluidas con su
licencia. El cierre implementa el sistema Dixon--Lewis y conserva el término
`-D^T grad(log T)`. La inicialización global de mezcla/equilibrio también es
nativa; Cantera solo se importa en las rutas de referencia explícitas.

Optimización implementada y probada:
- eliminación exacta mediante complemento de Schur: sistema térmico K en vez de 3K;
- reutilización del bloque de difusión ordinaria en esa eliminación;
- actualización racional exacta de fracciones molares en las columnas del Jacobiano,
  reduciendo el producto de transporte por perturbación de O(K²) a O(K);
- ensamblado compilado y paralelismo sobre caras con BLAS de un hilo;
- arranque nativo promediado por mezcla en una malla preliminar, seguido de
  corrección multicomponente/Soret con los criterios finales originales;
- reutilización opcional de coeficientes entre Jacobianos, con transporte completo
  reevaluado antes de aceptar. No basta con aceptar el problema congelado.

La ruta de referencia sigue disponible para auditoría. Las 42 pruebas pasan,
incluidas construcción/evaluación con importaciones y llamadas a Cantera bloqueadas,
comparación Schur--sistema completo, coeficientes en estados reactivos a 1/10 atm,
y columnas del Jacobiano compiladas frente a evaluación escalar y Python.

Evidencia medida el 2026-09-04 (siete pares alternados, H2/aire, GRI30, 300 K,
1 atm; sin semillas persistentes): mediana KFLAME 8.1535 s, Cantera 10.4920 s.
Reducción 22.29%; IC bootstrap pareado del cociente KFLAME/Cantera [0.7463, 0.9501].
Todas las corridas fueron aceptadas, con 207/203 nodos y 0.03 m respectivamente.
Diferencia de Su 0.01071%; E2(T) 0.03850%, E2(qdot) 0.08465%, máximo E2
entre especies activas 0.54055%. El residual exacto KFLAME fue 6.7233, cierre
de composición 1.01e-12 y variación relativa de flujo másico 2.15e-10.
Los E2 usan perfiles alineados e integración espacial. El postprocesado de calor
usa Cantera sobre ambos perfiles y no forma parte del tiempo de resolución.

Datos: `resultados/soret_native_validation/benchmark_20260904_180651/`.
La primera corrida se conserva; esta campaña usa cachés JIT ya preparadas por las
pruebas y no mide la instalación ni una compilación inicial sin caché.
El tiempo KFLAME incluye la etapa preliminar y el corrector final; la preparación
externa se registra aparte. Estos resultados no demuestran superioridad para
otros mecanismos, presiones o toda la cadena FGM.

Pendiente antes de cerrar el artículo: independencia adicional de malla/dominio,
balances elementales discretos, mecanismos alternativos, ablaciones temporales
pareadas de cada optimización y validación FGM. Ver
`SORET_NATIVE_VALIDATION.md` para comandos y limitaciones.

### Prioridad P2: convertir la demostración FGM en tabla validada

- Insertar flamelets adaptativamente en `Z` hasta que la validación dejando uno
  fuera satisfaga una tolerancia predefinida.
- Reservar flamelets de prueba que nunca participen en la selección de la malla.
- Verificar positividad, suma de masa, conservación elemental y error de los
  términos fuente después de interpolar.
- Medir por separado generación fría, regeneración cacheada y consulta de tabla.
- Para toda comparación KFLAME--Cantera, congelar y repetir la misma secuencia final
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
