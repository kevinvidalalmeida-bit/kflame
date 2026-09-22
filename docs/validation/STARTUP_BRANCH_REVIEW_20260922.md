# Cierre de la rama de arranque no lineal

La limpieza final elimina de la distribuci?n todas las rutas de arranque que
no demostraron una mejora global: correcciones Newton durante PTC, refinamiento
anticipado, su combinaci?n y los modos de Euler impl?cito completos o
persistentes. Tambi?n se retiraron sus selectores de `SolveOptions`, sus
pruebas y sus benchmarks.

La ?nica trayectoria instalada es PTC-SER con el rescate Backward Euler ya
existente tras un rechazo. Se mantiene `max_time_step_count=500`: tres parejas
alternadas mostraron una ganancia en H2/10, pero una p?rdida de 9.36 % en
CH4/10 al reducirlo a 250; no es una pol?tica global.

Los ensayos de inicializaci?n f?sica tampoco se promocionaron. La continuaci?n
mezcla -> multicomponente sin Soret -> Soret fue m?s lenta. El perfil de
mesetas lineales de Cantera redujo una corrida H2/10, pero termin? en un perfil
no equivalente (`Delta Su=0.64 %`), por lo que no es un speedup v?lido.

La evidencia hist?rica queda resumida en `DECISIONES_DESCARTADAS.md`. La
siguiente investigaci?n de alto impacto ser?a un FAS no lineal real con
restricci?n de residual y prolongaci?n de correcci?n; repetir solves completos
en mallas gruesas no es multigrid y ya qued? descartado.
