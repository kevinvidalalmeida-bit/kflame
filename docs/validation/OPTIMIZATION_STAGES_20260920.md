# Optimización general por etapas — 20/09/2026

Se midió el código nativo actual en CH4/aire y H2/aire con GRI30, a 1 y
10 atm, phi=1 y 300 K. H2 utiliza multicomponente/Soret con bootstrap nativo;
CH4 utiliza transporte promediado por mezcla. Cantera estuvo bloqueado durante
los perfiles y las comparaciones de llamas completas.

## Dónde se gasta el tiempo

`native_stage_profile_20260920.json` contiene un calentamiento excluido y una
corrida con cProfile por caso. Estos tiempos incluyen instrumentación: sirven
para localizar costes, no como benchmark de aceleración. Las cuatro soluciones
instrumentadas coinciden bit a bit con sus calentamientos.

| Coste exclusivo, porcentaje del tiempo total instrumentado | CH4/1 atm | CH4/10 atm | H2-Soret/1 atm | H2-Soret/10 atm |
|---|---:|---:|---:|---:|
| Kernel termoquímico/cinético, residual + perturbaciones | 23.9% | 26.0% | 21.8% | 25.4% |
| Cuerpo de factorización por bloques | 14.3% | 17.2% | 8.8% | 15.2% |
| Kernel de ensamblado del residual local | 8.2% | 8.2% | 6.5% | 6.7% |
| Preparación Python del lote local | 7.4% | 7.4% | 7.1% | 6.6% |
| Kernel de transporte promediado | 5.8% | 7.1% | 2.7% | 5.6% |
| Kernel de transporte multicomponente | — | — | 12.6% | 5.2% |

No son todas las funciones: el resto incluye sustituciones, malla,
inicialización, copias y otras tareas. Los tiempos **inclusivos** del Jacobiano
son 41–47% del solve e incluyen parte de las primeras filas; no se deben sumar
a esta tabla.

La estrategia importa tanto como los kernels: H2/Soret a 10 atm consume
17.87 s en el bootstrap promediado y 3.19 s en el corrector multicomponente
instrumentados. El bootstrap construye 558 Jacobianos y hace 2965
factorizaciones; el corrector, 8 y 8. Acelerar únicamente Soret deja intacta
la mayor parte de ese caso. No confundir este bootstrap no lineal con el
equilibrio HP de inicialización, que es mucho más pequeño.

## Cambio matemático implementado

El bloque local de GRI30 tiene 55 columnas: u, T y 53 fracciones másicas.
Al formar diferencias finitas, 54 columnas usan exactamente la misma T y
solo la perturbación de T usa otra. Antes se repetían NASA, Arrhenius,
equilibrio químico y el factor central térmico de Troe en cada columna.

Ahora `thermochemical_jacobian_native.py` calcula esos factores dos veces por
nodo y los reutiliza dentro del bloque. Sigue recalculando para cada columna:
densidad, concentraciones, productos de acción de masas, terceros cuerpos,
presión reducida y la dependencia composicional completa de falloff/Troe.
Se conserva la extensión de concentraciones negativas de prueba de Newton.
No hay redondeo de temperaturas, fastmath, eliminación de especies ni caché
entre iteraciones. Reducir 55 evaluaciones térmicas a 2 **no** significa hacer
el solver 27.5 veces más rápido: el resto del trabajo permanece.

La selección es automática en la precomputación del Jacobiano nativo. Se
comprueba el patrón exacto de temperaturas del lote; si no cumple el contrato,
no hay Numba o se desactiva el kernel fusionado, se conserva el evaluador
general. Los otros backends mantienen su interfaz anterior. No cambian
tolerancias, criterios de aceptación, malla, mecanismo ni modelo de transporte.

## Resultado de la comparación integrada

Evidencia: `thermal_reuse_production_20260920.json`. Tres parejas alternadas
por caso, con un calentamiento excluido por variante; cuatro hilos Numba y
un hilo en **ambas** bibliotecas BLAS, comprobados y registrados. Se comparan
llamas completas desde inicialización física, sin semillas de perfiles. Los
mecanismos, ajustes moleculares y compilaciones están calientes. El tiempo
incluye construcción del problema y solve, no importaciones ni exportación FGM.

| Caso | Mediana sin reutilización | Mediana con reutilización | Reducción |
|---|---:|---:|---:|
| CH4, 1 atm | 3.7497 s | 3.3141 s | 11.6% |
| CH4, 10 atm | 10.9391 s | 9.9839 s | 8.7% |
| H2/Soret, 1 atm | 3.8401 s | 3.5863 s | 6.6% |
| H2/Soret, 10 atm | 19.7433 s | 18.1776 s | 7.9% |

Las 24 corridas medidas fueron aceptadas y las 12 parejas conservaron
exactamente z, u, T e Y; también conservaron velocidad, malla y residual.
Las 12 parejas fueron más rápidas con la reutilización. Tres parejas no
constituyen una garantía de rendimiento para todos los combustibles o equipos.
La primera ejecución necesita compilación Numba adicional, excluida de estas
medianas y no cuantificada como mejora de arranque de proceso.

El cribado previo, `thermal_reuse_20260920.json`, también conservó todos los
perfiles y dio reducciones de 3.4–9.2%. No se mezcla estadísticamente con la
tabla: allí NumPy se importaba antes de fijar OPENBLAS_NUM_THREADS. El script
actual configura los hilos antes de importar NumPy y registra los pools reales.
El prototipo original queda en `thermal_reuse_candidate.py`, fuera de producción.

La batería completa pasó: **55 tests y 112 subtests**, con cuatro avisos
preexistentes sobre `np.row_stack`. Los tests nuevos incluyen GRI30 y H2/O2,
1/10/100 atm, estados con especies nulas o negativas de prueba, cruce NASA,
buffers no contiguos, lotes arbitrarios/vacíos, ruta no fusionada y comparación
bit a bit de un Jacobiano real. La nueva prueba también se ejecuta en el job
CI que instala exclusivamente dependencias nativas.

También se ejecutó el generador FGM con phi=[0.7, 0.9, 1, 1.1, 1.4], sin
caché de semillas y con Cantera bloqueado. Las cinco filas fueron aceptadas.
En `tmp/optimization_stages_20260920/thermal_reuse/`, los campos físicos de la
tabla (T, u, Y, rho, cp, conductividad, qdot, omega_c, beta), las coordenadas,
velocidades y certificados coinciden bit a bit con
`tmp/audit_cantera_20260919/native_cold/`. Se contrastaron además los cinco
perfiles sin tabular. Los 16.08 s de esta corrida son una comprobación de
integración, no una nueva comparación pareada de velocidad FGM.

## Próximas prioridades, sin atribuirles ganancias aún no medidas

1. **Arranque no lineal y control del Jacobiano.** Investigar un refresco
   condicionado por la calidad de la linealización y el progreso por coste,
   conservando rescate BE y aceptación final. La extensión del refresco local
   a un arranque frío necesita validación propia; no basta activar el control
   usado en continuación. Aumentar globalmente la edad del Jacobiano o el
   crecimiento de SER ya tuvo regresiones documentadas.
2. **Actualizaciones estructuradas de concentraciones.** A T y P fijas,
   si se perturba solo Y_s en delta, definir
   `a = sum(Y/W) / (sum(Y/W) + delta/W_s)`. Entonces
   `C'_k = a*C_k` para k distinto de s y
   `C'_s = a*(C_s + rho*delta/W_s)`.
   Esto permite estudiar actualizaciones de productos químicos por
   estequiometría en lugar de reconstruir todos sus factores. No se ha
   integrado: ceros, concentraciones negativas, falloff y protecciones de
   desbordamiento exigen rutas seguras; una identidad algebraica no garantiza
   las mismas diferencias finitas en coma flotante. No basta reactivar el
   prototipo histórico de dependencias, cuyo ahorro no era general.
3. **Ensamblado local y movimiento de memoria.** Evitar construir/copiar
   temporales de tres nodos por cada columna y estudiar escritura directa de
   los bloques. El prototipo histórico `streamed` no demostró mejora general:
   una nueva versión debe justificarse por las copias medidas y por pruebas
   completas, no solo por parecer más compilada.
4. **Transporte multicomponente.** La eliminación exacta de Schur ya existe.
   Aún se monta una matriz 3N x 3N en `thermal_system` incluso cuando se usa
   Schur; se puede investigar evitarla, conservar L22 como diagonal y explotar
   simetrías de coeficientes binarios. Es una oportunidad secundaria en el
   coste total actual; no sustituir el modelo por mezcla promediada.
5. **Barridos FGM.** Continuación con pasos elegidos por defecto/coste, semillas
   certificadas y reparto de hilos/procesos. El código ya tiene secante,
   región de confianza, refresco local y caché de semillas. Hay que medir
   filas solicitadas, puentes y reintentos juntos. Comparar una regeneración
   sembrada con un barrido frío no cuantifica una aceleración del algoritmo.

No se proponen como ganancias nuevas las alternativas rechazadas en
`DECISIONES_DESCARTADAS.md`: GPU para estos tamaños, Jacobiano analítico
experimental, BDF2, GMRES reciclado, condensación inestable, FAS ensayado,
homotopía probada o tolerancias de malla más permisivas. Un mecanismo reducido
es una investigación física aparte, no una optimización general equivalente.

El fundamento del control adaptativo del Jacobiano es consistente con la
[estrategia de actualización de KINSOL](https://sundials.readthedocs.io/en/v7.0.0/kinsol/Mathematics_link.html#jacobian-information-update-strategy);
PTC tiene su propia [base de convergencia, Kelley y Keyes](https://doi.org/10.1137/S0036142996304796).
Estas referencias justifican líneas de investigación, no prueban una mejora
de tiempo en este solver. No se añadió dependencia de KINSOL ni de Cantera.

## Reproducción

```powershell
python benchmarks/profile_native_stages.py --output tmp/stages.json
python benchmarks/benchmark_thermal_reuse.py --output tmp/thermal_pairs.json --pairs 3
python -m pytest tests -q
```

El perfil guardado corresponde al baseline previo a este cambio. Al reproducir
el primer comando con el código nuevo se medirá la ruta optimizada. El segundo
compara el evaluador general y el nuevo dentro del mismo proceso, alternando
el orden y sin reutilizar perfiles de llama; no modifica los archivos fuente.
