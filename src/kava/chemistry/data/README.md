# Mecanismos locales

`gri30.yaml` y `h2o2.yaml` son las copias distribuidas con Cantera 3.2.0,
incorporadas sin modificar sus datos el 19 de septiembre de 2026. Conservan
las descripciones, generador, fecha y atribuciones incluidas en cada YAML.
El primero corresponde a GRI-Mech 3.0; el segundo es su submecanismo H2/O2
con N2. El lector usa la primera fase ideal-gas con termodinámica NASA-7.

Son datos de entrada locales, no llamadas a Cantera ni resultados de llamas.
La licencia de distribución de Cantera se conserva en
`CANTERA_TRANSPORT_LICENSE.txt`. La procedencia original de GRI-Mech se
indica en la cabecera de `gri30.yaml`.

Para otro mecanismo, suministrar una ruta explícita a un YAML compatible.
Una ruta inexistente produce un error; no se sustituye por otro archivo con
el mismo nombre ni se busca dentro de la instalación de Cantera.
