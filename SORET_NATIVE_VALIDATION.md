# Soret nativo: verificación y reproducción

Estado: extensión implementada; ventaja temporal demostrada en la campaña
indicada abajo, pendiente de generalización. La variante con transporte
promediado por mezcla permanece por defecto. Soret se solicita explícitamente
con multicomponente y gradientes molares.

Esta es una limitación de la implementación V2, no de Cantera 3.2: la referencia
también dispone de un cierre Soret promediado por mezcla. Su formulación es
distinta de Dixon--Lewis y no se mezclan los modelos al comparar tiempos.
Fuente: [MixTransport.cpp, Cantera 3.2](https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MixTransport.cpp).

## Qué es nativo

La preparación de coeficientes y el cálculo de transporte multicomponente,
termodifusión, cinética y termoquímica son nativos. Las tablas universales de
colisión se distribuyen con atribución a Cantera (BSD-3-Clause); el código V2
ajusta y evalúa esas tablas sin importar Cantera. Utilizar datos moleculares y
tablas bibliográficas no equivale a consultar el solver de referencia.

La infraestructura `FreeFlameProblem` aún utiliza Cantera para construir la mezcla
y el equilibrio HP inicial; los comparadores y ciertos diagnósticos también lo
usan. Por ello la independencia se atribuye al cierre Soret y a los kernels,
no a toda la aplicación. No se utiliza una llama Cantera como semilla de V2.

## Método

El sistema Dixon--Lewis se resuelve por eliminación exacta de bloques. Si
`L=[[A,B,0],[B.T,C,D],[0,D.T,E]]`, con E diagonal, la parte térmica resuelve
`(C-B.T A^-1 B-D E^-1 D.T)b = r1-D E^-1 r2`. A ya se necesita para la difusión
ordinaria. La ruta densa 3K queda como referencia de verificación.

Para construir el Jacobiano, una perturbación de Y_i cambia X mediante
`X_new = X + d/(s+d)*(e_i-X)`, con `s=sum(Y/W)` y `d=delta_Y_i/W_i`.
Al almacenar H X en cada cara, se actualiza H X_new en O(K) por columna.
La identidad es exacta para los coeficientes congelados y no es una
aproximación de bajo rango del Jacobiano global.

El arranque multicomponente opcional resuelve primero una llama nativa
promediada por mezcla con slope/curve multiplicados por dos. Después activa
el modelo físico solicitado y restaura los umbrales finales. Ambas etapas
consumen el mismo presupuesto global de tiempo. Una falla del problema
intermedio nunca se devuelve como llama Soret aceptada.

La reutilización opcional de transporte afecta a las iteraciones. La
certificación reevalúa el transporte completo; se exige además convergencia
espacial cuando se solicita `require_grid_convergence`. El umbral residual
de 10000 no es la tolerancia relativa de Newton, que sigue siendo 1e-4.

## Comandos

Desde la raíz, con `OPENBLAS_NUM_THREADS=1` y `NUMBA_NUM_THREADS=4`:

```powershell
python -m unittest discover -s tests -v
python V2/benchmark_soret_native.py --bootstrap --bootstrap-mesh-factor 2 --lag --repeat 7
python V2/analyze_soret_validation.py resultados/soret_native_validation/benchmark_20260904_180651
```

El benchmark alterna el orden V2/Cantera. Guarda cada resumen y los perfiles,
también si la ejecución falla. No se tabula un estado rechazado. El modo
`--native-only` es diagnóstico y no establece una comparación temporal.
Se registran por separado preparación, resolución y extremo a extremo. La
primera corrida incluye carga/JIT; un JIT completamente frío debe medirse en
una campaña separada y etiquetada. No se eliminan corridas lentas del resumen.

Opciones de integración FGM:

```text
--transport-model multicomponent --soret-enabled --transport-backend native
--multicomponent-bootstrap --bootstrap-mesh-factor 2
--lag-multicomponent-transport --require-grid-convergence --max-residual-inf 10000
```

## Evidencia de siete pares del 2026-09-04

H2/aire, GRI30, 300 K, 1 atm; dominio 0.03 m; ratio 2.5, slope 0.04,
curve 0.08, prune 0.003. Aceptación por corrección ponderada y guarda residual
10000, con refinamiento obligatorio. Caché de perfiles ausente; JIT ya preparado.

| Magnitud | V2 | Cantera |
|---|---:|---:|
| Mediana de resolución | 8.1535 s | 10.4920 s |
| Cuartiles Q1/Q3 | 7.8790 / 9.2849 s | 10.2683 / 10.7651 s |
| Nodos | 207 | 203 |
| Velocidad | 2.10656035 m/s | 2.10633474 m/s |
| Temperatura quemada | 2372.518 K | 2371.884 K |

Reducción temporal: 22.29%. IC bootstrap pareado 95% del cociente de medianas
V2/Cantera: [0.7463, 0.9501], 20000 remuestreos, semilla fija. Todos los siete
pares se conservaron. Estos intervalos describen la campaña de esta máquina;
no son una garantía de rendimiento universal.

Perfiles alineados en una temperatura interior común e integrados sobre la unión
de las mallas: E2(T)=0.03850%, E2(qdot)=0.08465%; máximo E2 de especies
activas (pico Y >= 1e-5): 0.54055%, para H2O2. Diferencia de Su=0.01071%.
Residual exacto V2=6.7233; error de suma de Y=1.01e-12; variación relativa de
flujo másico=2.15e-10. El calor se reconstruyó con Cantera sobre ambos perfiles
exclusivamente para diagnóstico, fuera del tiempo medido.

## Validación que continúa pendiente

### Sensibilidad espacial independiente

Se ejecutaron dos comprobaciones adicionales, de una pareja cada una (no una
campaña estadística). Con slope/curve=0.02/0.04, V2 obtuvo 390 nodos,
Su=2.10921829 m/s y 14.2893 s; Cantera, 386 nodos, Su=2.10995877 m/s y
20.6451 s (`benchmark_20260904_182321`). La discrepancia entre solvers fue
0.03509%; E2(T)=0.00699% y E2(qdot)=0.05449%. Frente a la malla base, Su
cambió aproximadamente 0.126% en V2 y 0.172% en Cantera: dos niveles no bastan
para declarar independencia espacial a una tolerancia arbitrariamente pequeña.

Al iniciar con 0.06 m (`benchmark_20260904_182234`), V2 expandió a 0.12 m y
tardó 29.7251 s, frente a 14.5869 s y 0.06 m de Cantera. Las velocidades fueron
2.10655660 y 2.10633448 m/s, respectivamente. La sensibilidad de Su al dominio
fue muy pequeña, pero la temperatura quemada V2 cambió de 2372.518 a 2376.778 K.
Esta prueba NO acredita una ventaja temporal: la expansión y su coste adicional
se conservan como límite observado.

El diagnóstico elemental ahora reconstruye en caras el flujo total convectivo
más difusivo. Su mayor desviación relativa respecto a la entrada correspondió
al hidrógeno: V2 1.644% y Cantera 1.617% en la malla base; V2 0.958% y Cantera
0.903% en la fina. La reconstrucción convectiva de punto medio no es el
operador upwind del solver: es una prueba del balance continuo sensible a malla,
no un residual algebraico. La suma del flujo difusivo base fue inferior a
8e-13 kg/(m2 s) en ambas rutas. No se exige composición elemental constante
localmente cuando hay difusión preferencial.

### Correcciones de certificación

La revisión encontró una rama antigua que restauraba la malla anterior tras
fallar el solve refinado y devolvía éxito si no había timeout. Se corrigió:
restaurar no certifica malla y las métricas de la malla rechazada se invalidan.
Los catorce estados (intermedio y final de siete repeticiones) de la campaña
principal finalizaron con `grid_converged`, sin restauración: esa evidencia no
usó la rama defectuosa. También se añadieron pruebas para impedir aceptación
de NaN/inf y para forzar transporte completo en el certificado.

El control de dominio ya no se aplica durante el subproblema con temperatura
prescrita. Se mantiene con energía resuelta y tras refinamiento, usando el mismo
umbral. Un perfil térmico impuesto no demuestra que una llama física esté
truncada; escalarlo repetidamente conservaba las pendientes normalizadas y
provocaba expansión sin progreso. Esta elección se distingue de la llamada
automática de Cantera, no se presenta como identidad de flujo de control.

Después de las correcciones, tres nuevas parejas H2/Soret
(`benchmark_20260904_195104`) conservaron exactamente Su, 207 nodos y residual
6.7233; tiempos V2 [10.0207, 9.3404, 7.5285] s y Cantera
[10.3937, 11.3022, 9.9384] s. En tres parejas CH4 sin Soret
(`benchmark_20260904_195205`) se conservaron 261 nodos y Su=0.3788377359;
V2 [8.9134, 9.0090, 8.8620] s y Cantera [21.0367, 20.2327, 19.3277] s.
No hay regresión de solución en estos controles; la variabilidad impide
atribuir cambios pequeños de tiempo a la corrección de control.

Las 42 pruebas automáticas incluyen ahora química y termodinámica H2/aire
con GRI30 y h2o2.yaml a 1 y 10 atm, de 300 a 2400 K, además de coeficientes
Soret, Jacobiano, suma de flujos y guardas de aceptación. Las pruebas locales
pasan; no sustituyen una llama completa a presión alta.

El control adicional de GRI30 a 10 atm con recuperación de mallas
12/24/48 (`benchmark_20260904_195405`) alcanzó el límite de 120 s y se
rechazó, frente a 34.0172 s de Cantera. Con h2o2.yaml hubo una convergencia
diagnóstica usando esa secuencia (22.5595 s, 196 nodos, Su=1.34730378 m/s;
`benchmark_20260904_194922`), pero no una ganancia temporal ni fidelidad
suficiente para adoptarla: la referencia previa dio Su=1.33332860 m/s.
Reinicializar físicamente cada malla de recuperación, como hace la ruta
automática de Cantera, volvió a fallar en 60 s
(`benchmark_20260904_195737`); ese cambio se retiró. No se activa ninguna
política específica para una presión ni se reduce el certificado.

Comprobación de control sin Soret: CH4/aire, 1 atm, tres pares con los mismos
umbrales estrictos, ambos dominios finales de 0.06 m. V2: mediana 9.5051 s,
261 nodos, Su=0.37883774 m/s; Cantera: mediana 21.3492 s, 275 nodos,
Su=0.37881368 m/s. V2 sigue siendo más rápido en este control. Es una
comparación actual frente a Cantera; no demuestra igualdad temporal frente a
una revisión anterior de V2. Datos: `benchmark_20260904_180933`.

Ensayo adicional H2 a 10 atm (`benchmark_20260904_181152`): el arranque intermedio
promediado por mezcla agotó 180 s sin convergencia espacial y expandió el dominio
hasta 7.68 m. Se rechazó la salida; no constituye una solución Soret aceptada.
Cantera resolvió el caso en 34.9249 s, Su=1.33287505 m/s, 243 nodos y 0.06 m.
Es un fallo de robustez del arranque que debe resolverse antes de recomendar
esta estrategia para todo el rango de presión. No se incorpora a la mediana de
1 atm ni se omite del registro de resultados negativos.

Dos comprobaciones adicionales tampoco certificaron el arranque de 10 atm:
transporte completo directo (78.64 s, 9 nodos, sin convergencia espacial) y
arranque intermedio con 12 puntos iniciales (límite de 90 s, 129 nodos). No se
adoptó una regla específica de presión ni se rebajó el certificado para
aceptarlos. El problema abierto está en la trayectoria de arranque/refinamiento.

La integración FGM pasó una prueba de una fila H2/Soret a 1 atm, 207 nodos y
241 puntos de progreso, sin semillas persistentes. Archivo:
`resultados/soret_fgm_smoke/run_20260904_181747_fgm_native/fgm_table.npz`.
Se trata de una prueba de integración, no de validación paramétrica de una tabla.
La evidencia compacta de los siete pares se conserva en
`validation/soret_h2_gri30_20260904.json`; las trazas y perfiles regenerables
permanecen en los directorios de ejecución.

- Ampliar los dos niveles espaciales ya medidos y repetir el control de dominio.
- Balance de flujo elemental total, incluyendo difusión; no confundirlo con
  exigir fracción elemental local constante cuando hay difusión preferencial.
- Comparación completa a otras presiones, combustibles y mecanismos.
- Ablación causal pareada (denso/Schur, ensamblado, reutilización, malla preliminar).
- Regeneración FGM, aceptación de tabla y precisión de interpolación.

### Actualización 2026-09-05: causa del estancamiento identificada

Los fallos a 10 atm descritos arriba corresponden a revisiones anteriores.
La traza permitió localizar una incoherencia entre el margen negativo de
Newton y la química: se recortaban concentraciones negativas a cero, suprimiendo
la dependencia restauradora de las reacciones elementales. Se corrigió en las
cuatro rutas nativas, conservando la química de los estados positivos y los
criterios finales. Ahora pasan 49 pruebas, incluidas fuentes y derivadas cerca
de cero para GRI30/h2o2 a 1/10 atm.

La primera pareja de diagnóstico a 10 atm con PTC-SER normal convergió en
27,7425 s (V2) frente a 32,1764 s (Cantera). Esto reemplaza la afirmación de
que ese caso no puede converger, pero no demuestra todavía superioridad
estadística ni independencia espacial: la discrepancia de Su en esa malla
sigue siendo 1,1711%. Se exige comprobar el coste a menor error.

La campaña posterior de siete pares a 10 atm con slope=0.01 y curve=0.02
aceptó las 14 soluciones: mediana V2 45.8734 s frente a Cantera 78.6656 s,
reducción 41.69%; IC95 bootstrap pareado de la razón V2/Cantera
[0.56495,0.66745]. Las mallas propias tienen 854/868 nodos, dominio 0.06 m
y diferencia de Su=0.09605%. E2(T)=0.004055%, E2(calor)=0.14283% y máximo
E2 de especies activas=0.43261%. El residual final V2 es 0.35918.
No se cambió la guarda de 1e4. Son arranques sin perfiles guardados, con
caché de compilación conservada; no se publican como JIT completamente frío.

Los controles de tres pares a 1 atm dan medianas V2/Cantera de
5.1935/9.5995 s con H2/Soret y 3.7951/18.9234 s con CH4 sin Soret. Las
soluciones V2 conservan su malla y velocidad anteriores. La comparación
entre revisiones no fue alternada y no sustituye una ablación causal.

Advertencia espacial: respecto a la malla intermedia, Su cambia todavía
aproximadamente 1.18% en V2 y 0.84% en Cantera. La concordancia entre solvers
al 0.096% NO demuestra independencia de malla al 0.1%. El siguiente control
debe cerrar esa sensibilidad y el dominio antes de ampliar conclusiones.

El registro completo de causa, fuentes oficiales, variantes descartadas y
trazas está en `validation/SORET_SIGNED_KINETICS_DIAGNOSIS.md`. Las nuevas
medidas no deben mezclarse con las siete parejas de la revisión anterior.

Las mejoras matemáticas emplean técnicas conocidas. La contribución posible
reside en su combinación verificada y en la reducción reproducible del coste
hasta una llama aceptada. No se presenta Soret ni Schur como descubrimientos nuevos.

## Limpieza posterior del flujo ejecutable

Se eliminaron PTC-auto/PTC-rescue, BE exclusivo, pesos Newton congelados,
su selección desde CLI y la mezcla de convección centrada/upwind. Se conservan
PTC-SER con BE interno, LU exacta, transporte nativo y verificación independiente.
Las trazas y resultados históricos no se borraron. Detalles en `V2/PIPELINE.md`.
La limpieza no cambia la guarda residual de 1e4 ni relaja el refinamiento.

## Optimización adicional de productos químicos y despacho LU

El kernel fusionado especializa los productos de concentraciones de orden
entero total hasta tres, conservando la extensión con signo y una evaluación
logarítmica para órdenes generales o pérdida de representabilidad intermedia.
La LU accede directamente a GETRF/GETRS sin modificar su pivotado. Se retiró
el ensayo de crear el bloque de Schur directamente en orden Fortran porque
fue más lento, así como restos sin consumidores del escalado y del Jacobiano
analítico anterior.

La ablación con el orden anterior de evaluación química restituido midió
7 pares por caso a 1 atm y 3 pares a 10 atm, todos aceptados y con mallas
idénticas. Medianas V2 anterior/combinación nueva: CH4 sin Soret a 1 atm,
4.9522/4.5983 s; H2/Soret a 1 atm, 6.9245/6.5162 s; H2/Soret a 10 atm con
slope=.01 y curve=.02, 53.9952/48.0511 s. Reducciones: 7.15%, 5.90% y
11.01%. No se suman a las aceleraciones de otras revisiones ni se interpretan
como tiempos frente a Cantera. El calentamiento se registra por separado;
ninguna resolución medida usa perfiles de ese calentamiento.

Los 55 tests pasan. La evidencia con intervalos pareados, diferencias de
perfiles, huellas y política de calentamiento está en
`validation/kernel_ablation_confirmed_20260905.json`. El cribado anterior
con otra agrupación de productos logarítmicos permanece separado y no se
usa como confirmación de la revisión previa exacta. La repetición a 10 atm
sigue siendo exploratoria: faltan más pares y cerrar independencia espacial.

Control directo posterior de esa combinación: H2/Soret a 10 atm, V2
46.8486 s frente a Cantera 81.7868 s; ambos aceptados, 854/868 nodos y
diferencia de Su=0.09605%. Una sola pareja, sin inferencia estadística nueva,
guardada en `validation/kernel_optimized_cantera_20260905.json`.
