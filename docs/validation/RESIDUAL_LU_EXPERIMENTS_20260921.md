# Experimentos de residual y LU despu?s del Jacobiano anal?tico

Esta p?gina conserva la decisi?n de la campa?a del 21/09/2026. Con cuatro
casos y tres parejas alternadas, conservar el residual, reutilizar espacios de
trabajo LU y forzar qu?mica serial no dieron una reducci?n de tiempo estable
para CH4 y H2. Las funciones de sustituci?n y sus benchmarks se retiraron del
repositorio durante la limpieza final.

La ruta de producci?n mantiene el Jacobiano espacial anal?tico, el solve por
bloques con LU pivotada y la reutilizaci?n validada del residual de un paso de
damping aceptado. Los datos hist?ricos justifican no reintroducir rutas de
parcheo din?mico de fuente o de factorizaci?n.
