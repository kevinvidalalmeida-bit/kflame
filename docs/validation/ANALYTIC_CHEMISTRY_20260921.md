# Jacobiano quimico analitico: revision del 21/09/2026

Se implemento una nueva ruta **hibrida**, integrada en el solver real y
seleccionable con `--analytic-chemistry` en `kflame.fgm.generate`, o con
`SolveOptions(analytic_chemistry=True)`. No requiere Cantera en ejecucion.
La opcion permanece explicita; el valor predeterminado es `False`.

## Que aporta Cantera 4.0 y que aporta pyJac

La documentacion consultada identifica Cantera **4.0.0a2, en desarrollo**;
el entorno local tiene Cantera **3.2.0**. No se actualizo esa dependencia.
Cantera 4.0 introduce modos `auto`, `analytic`, `finite-difference` para
llamas 1D. Deriva las columnas de especies interiores y conserva diferencias
finitas en temperatura, velocidad y fronteras. Congela los coeficientes de
transporte en ambos caminos. Su ruta analitica 1D no cubre multicomponente.
Las notas anuncian aproximadamente 2.4-3 veces menos coste de construccion
del Jacobiano para mecanismos de 53-173 especies: no es el tiempo total de llama.

Fuentes primarias:

- https://cantera.org/dev/reference/releasenotes/v4.0.html
- https://cantera.org/dev/reference/onedim/nonlinear-solver.html
- https://doi.org/10.1016/j.cpc.2017.02.004
- https://slackha.github.io/pyJac/overview.html

Se reviso el PDF local `niemeyer2017.pdf`, especialmente la formulacion de
estado y la seccion 5.2, figura 1 y tabla 4 (pagina 12). La tabla da 6.98x
para GRI-Mech 3.0 al evaluar Jacobianos quimicos en CPU. Son evaluaciones
aisladas, monohilo, comparadas con diferencias finitas de primer orden;
no una aceleracion de 6.98x de una llama 1D completa ni de KFlame actual.

pyJac 1.x elimina la ultima especie usando la suma de fracciones igual a uno.
KFlame mantiene todas las Y independientes durante Newton. Por eso no se
puede insertar directamente su matriz ni reutilizar sin transformar su
derivada respecto de composicion.

## Implementacion y alcance

`chemistry/analytic.py` deriva las tasas masicas respecto de cada Y a T y P
constantes. Incluye Arrhenius reversible, terceros cuerpos y falloff de
Lindemann/Troe, con sus eficiencias y dependencia composicional. Se usa
NASA-7 para equilibrio y calor especifico. Para `s = sum(Y/W)`:

```
rho = P/(R*T*s)
C_k = rho*Y_k/W_k
dC_k/dY_j = rho*delta_kj/W_k - C_k/(W_j*s)
domega_k/dY_j = W_k/W_j * (rho*Jc_kj - sum_l(Jc_kl*C_l)/s)
```

La derivada de los productos no divide por concentraciones: conserva el
limite polinomico en especies nulas y la extension firmada del kernel nativo
para ensayos Newton negativos. Requiere ordenes enteros de molecularidad
menor o igual a tres, temperatura positiva y densidad sin clipping. Las
configuraciones no soportadas fallan explicitamente al solicitar esta ruta;
un error no se presenta silenciosamente como evaluacion analitica exitosa.

La integracion evita recalcular las reacciones para cada perturbacion de Y:
usa `omega(Y+dY) = omega(Y) + J_Y*dY`. Densidad, cp y entalpia conservan sus
valores termodinamicos perturbados. Solo el estado base y la perturbacion de
temperatura requieren evaluaciones completas de tasas. El residual espacial
y el cociente incremental final siguen siendo numericos: **esta ruta no es
un Jacobiano 1D completamente analitico**, y conserva errores de truncamiento
y cancelacion de esa ultima etapa. No es una integracion de la libreria pyJac.

El transporte multicomponente/Soret sigue por el ensamblador existente con
coeficientes congelados; esto permite usar la nueva quimica con esos modelos
sin afirmar que se han derivado sus coeficientes.

Tambien se corrigio la cache de propiedades: una suma de tres entradas del
estado podia identificar erroneamente dos composiciones distintas. Ahora
comprueba copia completa del estado, identidad del backend, presion y tamanos.
Ambos lados de los benchmarks usan esa correccion. No se modificaron edad
del Jacobiano, criterios de aceptacion, tolerancias ni politica de Newton.

## Tiempos de llamas completas

Tres parejas alternadas por caso, con perfil inicial frio cada vez; un
calentamiento por modo excluido, cuatro hilos Numba y uno BLAS. El tiempo
incluye construccion y solucion; mecanismos y compilacion permanecen calientes.
No representa el primer arranque sin cache JIT. GRI30 (53 especies) en los
cuatro casos. CH4 usa mezcla promediada; H2 usa multicomponente y Soret,
incluido el bootstrap en el coste. Phi=1, entrada=300 K.

| Caso | FD optimizado (s) | Hibrido (s) | Reduccion |
| --- | ---: | ---: | ---: |
| CH4, 1 atm | 3.6544 | 3.3296 | 8.89% |
| CH4, 10 atm | 10.0731 | 8.9910 | 10.74% |
| H2/Soret, 1 atm | 3.6963 | 3.2825 | 11.20% |
| H2/Soret, 10 atm | 19.2725 | 15.7012 | 18.53% |

Las **24 ejecuciones medidas fueron aceptadas**. Las dos rutas terminaron en
las mismas mallas de 261, 288, 207 y 241 nodos. Maximo entre parejas:
diferencia relativa de velocidad 2.94e-10; diferencia de temperatura 1.03e-7 K;
diferencia absoluta de fracciones 1.03e-11. No son perfiles bit a bit iguales.
Los porcentajes son medianas de tres muestras; no intervalos de confianza ni
garantias para otros mecanismos, composiciones o equipos.

Evidencia reproducible (directorios locales ignorados por Git):

- `runs/analytic_chemistry_20260921/summary.json`, CH4/1 atm y perfiles NPZ.
- `runs/analytic_chemistry_20260921_extended/summary.json`, otros tres casos.
- `benchmarks/benchmark_analytic_chemistry.py`, protocolo y registro de versiones
  de fuente mediante hashes; las comprobaciones de forma posteriores no
  cambian las operaciones numericas medidas.

## Que falta para hacerlo completamente analitico

No hay una prohibicion teorica. Hay dos objetivos diferentes:

1. **Derivadas de todas las ecuaciones con transporte congelado.** Derivar
   directamente continuidad, energia, conveccion, conversion Y->X, correccion
   del flujo difusivo, Soret explicito, fronteras y anclaje. Es el siguiente
   candidato principal. Permite llenar los tres bloques por nodo sin los
   lotes de residuos perturbados ni sus copias de arrays. Tambien elimina
   cancelacion en el cociente incremental de la parte derivada.
2. **Jacobiano del residual completo, actualizando tambien transporte.**
   Ademas requiere derivadas de conductividad, difusividades, densidad de cara,
   mezcla y coeficientes de difusion termica. Para multicomponente se puede
   diferenciar `A*x=b` como `A*dx = db - dA*x` y reutilizar la factorizacion;
   el numero de segundos miembros y el almacenamiento pueden ser caros.
   Es un Jacobiano mas fiel, pero su menor numero de iteraciones no garantiza
   compensar el coste adicional de construccion.

La columna de temperatura requiere `dC/dT=-C/T`, derivadas Arrhenius,
equilibrio, Fcent/Troe, `dh/dT=cp` y `dcp/dT` de NASA. Es viable, pero solo
quita una columna termica y deja el principal ensamblado espacial intacto.

En cambios de intervalo NASA, signo de velocidad upwind, seleccion de especie
de cierre en fronteras, clipping y reglas firmadas de concentraciones no
siempre hay una derivada clasica unica. Debe mantenerse una derivada por
rama o una alternativa numerica explicita. El Jacobiano se define sobre una
malla fija; no se diferencia a traves del refinamiento de malla.

## Prioridad respaldada por perfiles

Medianas del tiempo acumulado por llama con el candidato. Las categorias
listadas no se suman a un perfil exhaustivo; se evita sumar tiempos anidados.

| Parte | CH4/1 atm (s) | H2/Soret/10 atm (s) |
| --- | ---: | ---: |
| Precomputacion termoquimica del Jacobiano | 0.296 | 1.215 |
| Ensamblado de residuos espaciales perturbados | 0.769 | 2.911 |
| Factorizacion lineal LAPACK | 0.730 | 3.533 |
| Soluciones triangulares | 0.248 | 0.931 |
| Residual completo | 0.577 | 4.842 |

La precomputacion quimica ya baja aproximadamente 2.17-2.40x. El siguiente
experimento prioritario es derivar y ensamblar directamente los bloques
espaciales. Si, hipoteticamente, solo se eliminara todo el coste medido de
ese ensamblado en CH4/1 atm, el limite seria unos 0.77 s (23% del total actual).
Una implementacion real cuesta tiempo, por lo que esa cifra no es una
prediccion de ahorro. En H2/Soret pesa especialmente el residual completo;
se necesita perfilar por separado su transporte antes de atribuirle todo
ese coste. La LU ya usa LAPACK y sus reemplazos previos empeoraron los tiempos:
no se reintroduce uno sin una nueva evidencia independiente.

## Validacion

`python -m pytest -q`: **85 tests y 158 subtests pasan**. Se incluyen derivadas
contrastadas por diferencias centrales en h2o2/GRI30 a 1, 10 y 100 atm,
estados negativos y ceros, comparacion independiente con `ddCi` de Cantera,
bloques completos con/sin energia, bases masica/molar, Soret y multicomponente,
integridad de cache y propagacion de errores del modo explicito.

Comandos de reproduccion:

```powershell
$env:NUMBA_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
python benchmarks/benchmark_analytic_chemistry.py --output runs/analytic_repeat/summary.json --pairs 3
python benchmarks/benchmark_analytic_fgm.py --output runs/analytic_fgm_repeat --pairs 3
python -m kflame.fgm.generate --analytic-chemistry --phi-values 0.7,0.9,1,1.1,1.4 --disable-seed-cache
```

## Barrido FGM de la primera etapa

Tres parejas de cinco llamas (phi=0.7,0.9,1,1.1,1.4), procesos nuevos,
semillas persistentes desactivadas y bloqueo de imports de Cantera. Medianas
FD=16.0463 s; hibrido=14.4425 s: **9.99% menos tiempo**. Las seis tablas
medidas fueron validas y todas sus llamas aceptadas; misma malla de phi y
mismos numeros de nodos. Maxima diferencia relativa de Su=1.56e-9. La
tercera pareja fue mas lenta en ambos modos, por lo que se conservan las
muestras individuales, no solo sus medianas. Evidencia:
`runs/analytic_fgm_20260921/summary.json` y sus perfiles crudos.

La segunda etapa solicitada posteriormente deriva tambien temperatura y
ensambla directamente los bloques espaciales. Su evaluacion se registra
por separado; las cifras anteriores pertenecen a la primera etapa hibrida.
