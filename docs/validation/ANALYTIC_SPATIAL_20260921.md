# Segunda etapa: bloques analiticos con transporte congelado

La ruta nueva `--analytic-spatial` deriva todas las columnas del sistema
1D: velocidad, temperatura y especies, incluidas las fronteras. Ya no
construye lotes de estados perturbados ni divide diferencias de residuos.
**Conserva los coeficientes de transporte congelados**, como aproximacion
del Jacobiano, igual que la ruta numerica existente. No debe confundirse
con derivar los coeficientes del transporte completo.

Es una implementacion nativa, no requiere instalar Cantera 4.0 ni pyJac.
La investigacion y primera etapa hibrida estan documentadas en
[ANALYTIC_CHEMISTRY_20260921.md](ANALYTIC_CHEMISTRY_20260921.md).

## Cambios concretos

- Derivadas termicas de Arrhenius, equilibrio, terceros cuerpos,
  Lindemann y Troe, incluida Fcent y la variacion de concentraciones con T.
- Derivadas NASA-7 de cp y entalpia y de densidad ideal respecto de T/Y.
- Derivadas directas de flujos en base masica/molar y conversion Y->X,
  correccion de suma del flujo y terminos explicitos de Soret.
- Transporte multicomponente: se deriva `D*(X_R-X_L)` respecto de todas
  las composiciones manteniendo D fijo. Se reutilizan productos por cara.
- Continuidad, conveccion upwind por rama, conduccion, fuentes y transporte
  de entalpia, fronteras, especie de cierre y anclaje termico.
- Ensamblado paralelo por nodo de los tres bloques y solucion mediante
  la LU LAPACK existente. No cambia la edad del Jacobiano ni el criterio
  de aceptacion ni las tolerancias.

Fuentes: `src/kflame/chemistry/analytic.py` y
`src/kflame/flame/analytic_jacobian.py`. El caso base del cache permanece
sin perturbaciones; el sistema transitorio usa el mismo desplazamiento
diagonal posterior que antes.

A peticion del usuario, esta ruta es ahora el **valor predeterminado**:
`SolveOptions(analytic_spatial=True, jacobian_mode='block_tridiag')`.
La CLI y las funciones publicas tambien la usan. Para reproducir el hibrido,
usar `--no-analytic-spatial --analytic-chemistry`; para diferencias finitas,
`--no-analytic-spatial --no-analytic-chemistry`.
La ruta exige cinetica nativa compatible (ordenes enteros de molecularidad
<=3 y densidad sin clipping). El refresco parcial de columnas de la continuacion tambien es analitico:
solo deriva la quimica de los nodos seleccionados, ensambla sus columnas
y conserva exactamente los otros bloques. Su politica de activacion
permanece igual. La identidad de semillas distingue los modos de Jacobiano.

## Medicion contra el hibrido de la primera etapa

GRI30, phi=1, entrada=300 K, mismos criterios y perfiles iniciales frios;
tres parejas alternadas por caso, calentamiento por modo excluido, cuatro
hilos Numba y uno BLAS. Los tiempos incluyen configuracion y solucion, con
caches JIT/mecanismo calientes. Los H2 incluyen el bootstrap y transporte
multicomponente con Soret.

| Caso | Hibrido (s) | Bloques analiticos (s) | Reduccion adicional |
| --- | ---: | ---: | ---: |
| CH4, 1 atm | 3.1448 | 2.1792 | 30.70% |
| CH4, 10 atm | 9.0331 | 5.8344 | 35.41% |
| H2/Soret, 1 atm | 3.0647 | 2.2275 | 27.32% |
| H2/Soret, 10 atm | 19.1307 | 14.5549 | 23.92% |

Las **24 ejecuciones medidas fueron aceptadas**. Las mallas coinciden
exactamente entre modos; no solo su numero de nodos. Las trayectorias de
Newton cambian al quitar las aproximaciones por diferencias finitas.
La maxima diferencia entre soluciones en esta campana fue:

- Velocidad relativa: 1.776e-5 (0.001776%).
- Temperatura: 0.02064 K.
- Fraccion masica absoluta: 2.170e-6.

No se presentan como soluciones bit a bit iguales. La condicion original
de aceptacion se mantuvo para ambos modos. Los maximos corresponden a
CH4/1 atm. En H2/10 atm el residual final disminuyo de aproximadamente
1416 a 336, pero en CH4/10 atm aumento de 81.9 a 152.5: no se afirma que
la norma residual siempre mejore.

Hay variacion de carga/velocidad del equipo. En H2/10 atm las tres muestras
hibridas fueron 19.8551, 19.1307 y 15.7601 s; las analiticas fueron 14.5549,
15.0950 y 11.6220 s. Los tres cocientes emparejados son favorables, pero
las medianas no son intervalos de confianza ni garantias para otros equipos.
No se comparan tiempos absolutos de campanas distintas para atribuir una
ganancia acumulada: las reducciones de esta tabla son contra su propio
baseline hibrido medido en parejas.

Evidencia: `runs/analytic_spatial_20260921/summary.json`, perfiles NPZ y
`benchmarks/benchmark_analytic_chemistry.py`. Las validaciones posteriores
de opciones y umbral no alteran las operaciones del caso medido (umbral 0).

## Coste restante y siguientes candidatos

Medianas acumuladas, sin sumar categorias anidadas:

| Parte | CH4/1 atm (s) | H2/Soret/10 atm (s) |
| --- | ---: | ---: |
| Jacobiano completo | 0.300 | 1.927 |
| Dentro del Jacobiano: termoquimica | 0.108 | 0.603 |
| Dentro del Jacobiano: flujos y bloques | 0.159 | 0.889 |
| Factorizacion | 0.710 | 4.468 |
| Soluciones triangulares | 0.226 | 1.041 |
| Residual completo | 0.524 | 5.633 |

En CH4/1 atm, el Jacobiano baja de 1.281 a 0.300 s (4.28x en coste
acumulado por llama; 84 evaluaciones frente a 83). El ensamblado directo
ha dejado de ser el principal coste. La LU y el residual completo merecen
la siguiente investigacion; no se reintroduce la LU Numba/C++ descartada
ni se alteran tolerancias o edad del Jacobiano para mejorar artificialmente
los tiempos.

Se pueden derivar tambien conductividad, difusividades y coeficientes de
Soret respecto de composicion y temperatura. En multicomponente, diferenciar
`A*x=b` exige resolver `A*dx=db-dA*x`; se puede reutilizar la LU de A, pero
hay muchos segundos miembros. Eso cambia la aproximacion del Jacobiano y
debe demostrar una mejora del tiempo completo, no solo mas fidelidad.
Antes de implementarlo conviene medir por separado el transporte dentro
del residual: los 5.633 s incluyen mas trabajo que transporte.

Las ramas no suaves siguen existiendo: cambio de intervalo NASA, velocidad
upwind nula, cambio de especie dominante en frontera y continuation firmada
de concentraciones. Se deriva la rama activa; no hay una derivada clasica
unica en todos esos puntos. La densidad con clipping se rechaza en esta
ruta y la malla se mantiene fija durante cada construccion del Jacobiano.

## Reproduccion

```powershell
$env:NUMBA_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
python -m pytest -q
python benchmarks/benchmark_analytic_chemistry.py --output runs/spatial_repeat/summary.json --baseline analytic --candidate spatial --pairs 3
python benchmarks/benchmark_analytic_fgm.py --output runs/spatial_fgm_repeat --baseline analytic --candidate spatial --pairs 3
python -m kflame.fgm.generate --analytic-spatial --phi-values 0.7,0.9,1,1.1,1.4 --disable-seed-cache
```

La bateria completa anterior a cambiar el valor predeterminado paso
100 tests y 158 subtests, incluidos 15 tests de derivadas termicas,
columnas espaciales, umbrales y ausencia de llamadas a residuos perturbados.

Durante la primera prueba FGM de esta etapa se detecto la falta de soporte
del refresco parcial usado por la continuacion. Esas ejecuciones fallidas
se conservan en `runs/analytic_spatial_fgm_20260921/` y no se cuentan como
resultados aceptados. Se implemento el refresco analitico selectivo y se
comprobo contra las columnas de una reconstruccion completa antes de repetir
el FGM. No se desactivo el mecanismo de rescate ni se cambio su politica.

## FGM final con refresco analitico y valores predeterminados nuevos

Tres parejas alternadas de barridos phi=0.7,0.9,1,1.1,1.4, procesos nuevos,
sin semillas persistentes y con imports de Cantera bloqueados:
**14.1667 s hibrido frente a 10.2068 s analitico**, una reduccion adicional
del **27.95%**. Las seis tablas medidas fueron validas, todas las llamas
fueron aceptadas y conservaron exactamente las mallas y sus numeros de
nodos. La maxima diferencia relativa de Su es aproximadamente 0.0211%;
la maxima diferencia de temperatura es 0.06469 K y de Y es 8.20e-6.
Estas diferencias son de las soluciones aceptadas con los mismos criterios;
no se afirma igualdad bit a bit ni error de discretizacion nulo.

Evidencia final: `runs/analytic_spatial_fgm_20260921_v2/summary.json`,
`profile_comparison.json` y perfiles crudos de cada corrida. La ejecucion
fallida anterior queda documentada arriba y separada de estas mediciones.

Los ejemplos y las funciones publicas no necesitan nuevas opciones:
usan la ruta analitica por defecto. Los benchmarks historicos de
perturbaciones y reutilizacion termica desactivan explicitamente esta ruta
para seguir midiendo sus kernels originales.

## Verificacion final

Con los valores predeterminados finales y el refresco parcial integrado:
`python -m pytest -q` termina con **102 tests y 158 subtests pasados**
(17.53 s en esta ejecucion). Las pruebas confirman tambien las opciones
predeterminadas de SolveOptions/CLI y la seleccion explicita de alternativas.
`git diff --check` no reporta errores de espacios.
