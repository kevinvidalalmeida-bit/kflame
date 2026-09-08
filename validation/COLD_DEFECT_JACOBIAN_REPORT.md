# Jacobiano adaptativo por defecto: extensión acotada al arranque frío

## Hipótesis y alcance

Se ensaya extender al arranque frío la selección de columnas por defecto de
linealización ya validada en el corrector FGM. No se presenta esa idea como
una invención nueva ni como una mejora de Soret previamente demostrada.
La ruta productiva y su certificado no se modifican.

Para el residual del subproblema actual G (estacionario o BE), la sonda usa
`d = G(x + alpha*s) - G(x) - alpha*J_G*s`. En BE la parte lineal de masa
se cancela en el numerador; la escala del indicador sí depende del subproblema.
Se conserva el indicador por bloque existente: máximo absoluto del defecto
dividido por el máximo del residual inicial, cambio predicho y suelo de redondeo.
No es una norma física por especie ni una estimación del error de discretización.

Tras fallar la contracción ordinaria se seleccionan bloques con indicador
>=0.20 y sus vecinos. Si abarcan más del 35% de la malla no se hace refresco
local. Se permite como máximo una reconstrucción local y LU nueva por llamada
a Newton; las sondas sin selección no consumen ese presupuesto. El candidato
debe superar la misma contracción original. En caso contrario continúa el
amortiguamiento/rescate normal. No se cambia PTC-SER ni se recicla una LU
aproximada. El límite evita pagar varios refrescos para el mismo llamado Newton.

Se actualizan columnas quasi-Newton con transporte congelado. NO se implementan
derivadas exactas del transporte. El corrector multicomponente queda excluido,
porque el refresco de su caché de transporte necesita una auditoría de reversión
ante rechazo antes de habilitarlo. En H2/Soret se modifica exclusivamente el
arranque nativo promediado por mezcla; la solución final sigue usando Soret.

## Reproducción

Prototipo aislado: `validation/cold_defect_jacobian.py`. El contexto restaura
funciones y bandera del problema incluso ante excepciones. No se importa desde
V2 ni FGM. `defect-control` instrumenta Newton sin activar la estrategia;
`defect-cold` activa el candidato. El benchmark sigue usando `baseline` por defecto.

Comando inicial:

```powershell
python validation/benchmark_cold_strategies.py --cases ch4_1 h2_1 ch4_10 h2_10 --variants baseline defect-cold --repeat 1 --max-seconds 90
```

GRI30, phi=1, 300 K, 1/10 atm; CH4 promediado por mezcla sin Soret y H2
multicomponente/Soret. Mismos criterios que la campaña fría anterior: ratio=2.5,
prune=.003, slope/curve=.04/.08 a 1 atm y .01/.02 a 10 atm. Estos criterios por
presión son configuración preexistente del benchmark, no reglas del candidato.
Se separa calentamiento completo de cada variante del solve frío sin perfiles
previos. Se mantienen Finf<=1e4, norma ponderada<=1, malla, dominio y física.

Las sondas, rechazos por defecto global, refrescos, columnas seleccionadas,
aceptaciones y coste adicional se registran por llamado Newton. Se conservan
perfiles y snapshots con hashes. La comparación a posteriori alinea perfiles
y verifica Su, T, especies activas y qdot con el protocolo previo.

## Estado final: retirado por solicitud del autor

Se detuvo la campaña y se retiraron el prototipo, su analizador, sus tests y
los selectores `defect-cold`/`defect-control` del benchmark. Las descripciones
y el comando anteriores documentan el experimento histórico, no una ruta activa.
No se modificó el solver productivo ni el refresco ya validado de FGM.

Evidencia parcial en `resultados/cold_strategies/20260907_184940`: una pareja
medida por caso en CH4/1 atm (4.1801/4.1487 s), H2/Soret/1 atm
(5.3662/5.3801 s) y CH4/10 atm (23.0670/20.6011 s), baseline/candidato.
Los seis solves fueron aceptados. H2/Soret/10 atm quedó sin comparación
completa. No se concluye mejora general, significación estadística ni fallo
matemático del método a partir de esta interrupción solicitada.

Los snapshots, perfiles y tiempos parciales se conservan. Los tres archivos
retirados tienen copia recuperable en `retired_support` dentro de esa campaña.
Las cuatro pruebas específicas y las nueve pruebas de optimizaciones existentes
habían aprobado antes de la retirada; no se atribuye a ello validación física
completa ni mejora frente a Cantera.
