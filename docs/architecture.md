# Pipeline productivo de V2

El núcleo V2 se distribuye ahora como `kava.flame` y `kava.chemistry`; la
generación FGM reside en `kava.fgm`. Entrada: `python -m kava fgm`.
Las equivalencias de rutas anteriores están en [migration.md](migration.md).

Este documento describe únicamente la ruta mantenida. Las campañas, prototipos
y decisiones históricas permanecen recuperables en el commit de respaldo
anterior a la limpieza.

1. Se lee el mecanismo YAML local, se construye la mezcla fresca y se resuelve
   el equilibrio HP nativo (NASA-7, potenciales elementales y balance de entalpía)
   para inicializar una llama libre con anclaje térmico.
2. El residual estacionario combina continuidad, especies y energía con
   convección estrictamente aguas arriba, difusión conservativa y el transporte
   seleccionado.
3. La termoquímica y la cinética se evalúan con el backend nativo compilado.
   Los estados admisibles con una concentración negativa transitoria conservan
   su signo en la cinética; las cotas físicas se aplican al aceptar un paso.
   Viscosidad, conductividad y difusión se ajustan desde parámetros moleculares,
   NASA y tablas Monchick--Mason locales, sin coeficientes preexportados de Cantera.
4. El corrector intenta Newton amortiguado. Si un intento no es aceptable,
   aplica una corrección PTC--SER linealmente implícita; si esta falla, usa
   Euler implícito completamente convergido como rescate.
5. Si la etapa con energía activa no progresa, se inmoviliza temporalmente la
   temperatura, se corrigen continuidad y especies y se reactiva la energía.
   Solo la solución reactivada puede pasar a dominio, malla y aceptación.
6. El Jacobiano por diferencias finitas conserva un stencil de bloques
   tridiagonal. Los sistemas se resuelven con LU directa SciPy/LAPACK y
   sustituciones por bloques compiladas.
7. Se expande el dominio cuando el borde no es asintótico y se adapta la malla
   mediante ratio, pendiente, curvatura y poda. El resultado exige estado
   finito, norma ponderada, guarda de residual, malla estabilizada y dominio
   suficiente.

En la precomputación nativa del Jacobiano, las perturbaciones de un nodo
comparten los factores que dependen únicamente de T: se evalúan para la
temperatura base y su perturbación, no para cada especie. La dependencia
completa de composición, terceros cuerpos y falloff se recalcula en cada
columna. La reutilización es exacta y local al lote; no envejece coeficientes
entre iteraciones. Los lotes sin ese patrón conservan el evaluador general.
Véase `validation/OPTIMIZATION_STAGES_20260920.md` para costes y validación.

## Continuación FGM

La generación de tablas procesa los valores de equivalencia en orden creciente.
La primera llama usa la inicialización física; las siguientes usan una copia o
un predictor secante en \(\log\phi\), limitado por una razón multiplicativa
máxima de 1.15. La semilla se proyecta sobre límites físicos y sobre la mezcla
fresca objetivo antes del corrector completo. Las soluciones fuera de esa
vecindad se reinician físicamente.

En el corrector de continuación con transporte promediado por mezcla, un defecto
de linealización localizado puede activar el refresco de los bloques afectados
del Jacobiano y una LU exacta nueva. No se aplica a arranques fríos ni a la ruta
multicomponente. Las semillas persistentes se reutilizan solo si coinciden todos
los datos físicos y numéricos; el paralelismo se habilita únicamente cuando las
semillas exactas ya existen. La clave persistente incluye un SHA-256 del
contenido del mecanismo; cambiar un YAML en la misma ruta invalida sus aciertos
de caché anteriores, sin borrar las semillas históricas.

## Transporte Soret

El caso Soret usa transporte multicomponente nativo. Para el arranque frío, una
etapa preliminar promediada por mezcla construye una semilla y el corrector final
reactiva transporte multicomponente y termodifusión con los criterios estrictos.
La aceptación reevalúa el transporte exacto; no acepta coeficientes de cara
congelados de una linealización previa. Los ajustes moleculares inmutables se
reutilizan entre reconstrucciones de malla; la clave incluye sus entradas
numéricas y nunca almacena coeficientes dependientes del estado de la llama.
La inicialización, los campos tabulados y la ejecución con `--native-only`
funcionan sin Cantera. Los comparadores lo importan explícitamente.
Los parámetros de pares moleculares y los ajustes de conductividad también se
reutilizan en cachés inmutables de hasta ocho entradas. Nunca se cachea aquí un
estado de llama ni se congelan sus propiedades dependientes de T, Y o presión.
La auditoría sin instalación de Cantera y con polinomios históricos bloqueados
está en `validation/NATIVE_ALL_STAGES_20260920.md`.
