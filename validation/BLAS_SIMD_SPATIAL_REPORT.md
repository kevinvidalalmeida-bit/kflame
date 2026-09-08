# Tres pruebas aisladas: BLAS, SIMD y discretización espacial

Fecha: 2026-09-07. Ninguna modificación del solver productivo en esta ronda.
Los cambios anteriores del árbol de trabajo se conservan.

## 1. BLAS/LAPACK

Entorno aislado `resultados/blas_env_20260907`, con acceso de lectura a los
paquetes existentes y MKL 2025.3.1 instalado solamente en ese entorno. El pip
antiguo del entorno falló al verificar certificados; se usó el pip actual con
su almacén de confianza normal, sin deshabilitar TLS. No se reemplazaron los
binarios globales de NumPy/SciPy ni se modificó PATH de usuario/sistema.

La ablación `mkl-blocks` usa LAPACKE LP64 por ctypes para GETRF y GETRS de
los bloques del Jacobiano de V2. Conserva copia column-major, pivotes
convertidos al contrato base cero de SciPy y las sustituciones compiladas
productivas. Incluye copias y coste de despacho en tiempo de solve.

Alcance limitado: NO sustituye las llamadas BLAS de NumPy ni las resoluciones
de transporte multicomponente emitidas por Numba. No permite concluir cómo
rendiría una distribución íntegramente enlazada con MKL. MKL opera con un
hilo; OpenBLAS con uno; química Numba con cuatro.

Pruebas locales: matrices de 3, 12 y 55 incógnitas, pivoteo necesario,
múltiples RHS, sistemas transpuestos, no mutación del input y singularidad.
Dos tests aprobados antes del cribado.

## 2. SIMD NASA entre estados

`simd-nasa` cambia la evaluación NASA a orden especie/estado, manteniendo el
estado como índice contiguo para vectorizar. Evalúa TODAS las temperaturas en
CADA llamada: no reutiliza temperaturas únicas ni resultados de otra llamada.
Conserva química, orden de sumas por especie, trazas negativas y ausencia de
fastmath. Las tablas temporales, llamadas adicionales y sincronización cuentan.

Prueba local h2o2/GRI30, 1/10 atm, temperaturas a ambos lados de NASA y trazas
negativas: rho, cp, producción y entalpías idénticos a producción. LLVM mostró
36 operaciones FP64 empaquetadas, con vectores de ocho elementos. Esto prueba
vectorización del subkernel NASA, no de toda la química ni ahorro de llama.

## 3. Reconstrucción espacial limitada

Prototipo `limited-spatial`: reconstrucción lineal a caras con limitador MC en
malla no uniforme. Un limitador común para las especies conserva su suma en
las caras si los nodos están normalizados. Se usa flujo convectivo de cara
común y resta de la divergencia de masa para recuperar forma advectiva fuera
de continuidad; con flujo másico constante coincide con forma conservativa.
La contribución difusiva de entalpía usa derivada nodal centrada de segundo
orden. Conducción, difusión de especies, fuentes y fronteras se mantienen.

Limitaciones explícitas:
- Mayor orden se verifica en el interior suave; el limitador y los bordes
  pueden reducirlo. No se afirma conservación discreta exacta de entalpía total.
- El residual no lineal tiene dependencias de hasta cinco nodos. Se conserva
  inicialmente el Jacobiano upwind como aproximación quasi-Newton; NO es el
  Jacobiano exacto de la nueva discretización. Puede perjudicar globalización.
- `limited-spatial-adapt` añade marcas por diferencia reconstrucción
  lineal/cuadrática, normalizada por rango. Es un monitor de refinamiento,
  NO un estimador adjunto ni una cota certificada del error físico.
- Menos tiempo frente a la malla baseline no demostraría ahorro a igual error.
  Se requieren referencias refinadas y estudios de dominio antes de promoción.

## Protocolo

Primer cribado BLAS/SIMD: `resultados/cold_strategies/20260907_073004`.
CH4 sin Soret y H2 multicomponente/Soret, GRI30, phi=1, 300 K, 1/10 atm.
Sin perfiles previos, calentamiento completo separado por variante. Una pareja
por caso solo orienta: no prueba significación estadística ni superioridad a
Cantera. El umbral Finf<=1e4 no sustituye norma ponderada<=1 ni malla/dominio.

Fuentes:
- https://docs.scipy.org/doc/scipy/building/blas_lapack.html
- https://numba.readthedocs.io/en/stable/user/faq.html

Resultados completos y estudio espacial pendientes de cierre en este informe.

## Cribado BLAS/SIMD terminado

Doce perfiles y 36 fuentes auditados; todos pasan los filtros de precisión.
Informes `blas_simd_screening.json` y `blas_simd_integrity.json`.

| Caso | V2 actual [s] | MKL bloques [s] | SIMD NASA [s] |
|---|---:|---:|---:|
| CH4 1 atm | 3.9626 | 3.9263 | 3.8335 |
| H2/Soret 1 atm | 5.2333 | 5.0736 | 4.8774 |
| CH4 10 atm | 29.0668 | 28.0503 | 26.7473 |
| H2/Soret 10 atm | 50.2085 | 51.8898 | 57.7927 |

SIMD termina con perfiles bit a bit idénticos en los cuatro casos. En H2/10
mantiene 536 Jacobianos y 2773 LU globales; su coste de precomputación
termoquímica registrado subió de 12.95 a 16.12 s. No se atribuye todo el
cambio temporal al kernel: otras regiones también variaron y solo hay una pareja.

MKL cambia ligeramente el redondeo y la trayectoria: en H2/10 se registran
573 frente a 536 Jacobianos y 2986 frente a 2773 LU, aunque los perfiles
finales pasan precisión. No es válido atribuir el tiempo exclusivamente a
la velocidad de una LU aislada. Ninguno se promueve con este cribado.

## Incidencia detectada antes de validar la prueba espacial

La campaña `20260907_074018` se interrumpió explícitamente y NO es evidencia
de mayor orden: la traza mostró cero ensamblados nuevos completados. El
reshape de T no contigua fallaba en Numba, y el residual de producción
activaba su respaldo Python upwind. Se conserva la campaña para auditoría.

Corrección: copia contigua explícita de T (su coste cuenta), detección de
excepciones de ensamblado sin aceptar respaldo silencioso, prueba integrada
con estado real strided y guarda de ejecución antes de aceptar el candidato.
Además se aísla el ensamblado del Jacobiano para que tanto su residual base
como las filas perturbadas sean upwind: mezclar residuales introduciría un
término espurio defecto/delta en diferencias finitas. La prueba comprueba
igualdad de los tres bloques con el Jacobiano productivo y que el residual
de mayor orden sí sea distinto. Siete tests espaciales aprobaron después.

La campaña corregida es `20260907_074338`, con límite de solve de 60 s por
llama/candidato. Un candidato que no supera el calentamiento/certificado no
se cronometra como solución exitosa: se conservan el fallo, su coste y el
perfil. No se presenta un timeout como convergencia lenta pero válida.

## Puerta pendiente para afirmar ahorro espacial a igual error

Aunque un caso con mayor orden converja, se necesita una referencia independiente
refinada para cada combustible/presión/modelo de transporte y una expansión de
dominio. Comparar los candidatos en una secuencia de mallas, alineando perfiles,
con E2(T), E2(Y activas), E2(qdot), Su, espesor y balances. La diferencia entre
dos mallas de la MISMA formulación es un estimador de convergencia; comparar
upwind y mayor orden no lo sustituye. Incluir en el coste del candidato todas
las soluciones auxiliares que use para decidir su malla. Conservar también
el coste de referencias de validación, reportado por separado.

Este prototipo no implementa un adjunto ni un estimador de error garantizado.
No se declarará completada esa línea matemática a partir del monitor de caras.

## Cierre del cribado y decisión

Campaña espacial corregida terminada: las ocho combinaciones de candidato/caso
agotaron el presupuesto de 60 s durante el calentamiento sin certificado completo.
No se generó una medición exitosa de esos candidatos. Los cuatro controles sí
convergieron: CH4/1 atm 4.4344 s; H2/Soret/1 atm 5.8730 s; CH4/10 atm
26.5124 s; H2/Soret/10 atm 50.4603 s. Son controles de esta campaña, no pares
con Cantera. El residual inferior a 1e4 en algunos fallos no reemplaza la norma
ponderada ni los criterios espaciales restantes.

La auditoría verificó 4 perfiles medidos, 12 perfiles de calentamiento y 37
fuentes en `spatial_integrity.json`. `spatial_screening.json` conserva los
rechazos; ninguna traza corregida registró errores de ensamblado ocultados.
La prueba SIMD adicional confirmó 36 instrucciones empaquetadas FP64 en código
máquina (`simd_machine_code.json`), además de la evidencia LLVM. Las pruebas de
propiedades de ambos mecanismos a 1/10 atm dieron diferencias exactamente nulas.
La suite final completa pasó 87 tests en 14.157 s.

Ninguna de las tres líneas se promueve a producción con esta evidencia. MKL y
SIMD tienen un cribado de una pareja, no una conclusión estadística definitiva.
Los dos prototipos espaciales quedan fuera de producción; no se interpreta el
timeout como imposibilidad matemática de convergencia. El estudio coste/error
espacial y el estimador de error riguroso permanecen pendientes. La tesis se
actualiza con esta distinción y con los resultados Soret ya verificados, sin
mezclar cronologías o atribuir estos nuevos tiempos a comparaciones con Cantera.

### Separación editorial de la tesis

Por indicación posterior del autor, la tesis no incluye el catálogo de
experimentos no adoptados ni su historial de depuración. La documentación
anterior permanece en este registro de validación. El texto académico conserva
el método adoptado, sus comparaciones verificadas y las limitaciones de malla,
dominio y alcance físico; eliminar el historial experimental no modifica los
datos originales ni convierte los candidatos en mejoras productivas.
