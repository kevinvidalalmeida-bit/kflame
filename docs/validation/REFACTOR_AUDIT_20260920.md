# Auditoría de simplificación y empaquetado — 20 septiembre 2026

## Alcance y criterio

Revisión estática de todos los módulos de `src/kflame` y benchmarks, búsqueda de
referencias en producción, tests y scripts locales de investigación, y revisión
manual de los candidatos. Se prioriza preservar comportamiento e interfaces;
una coincidencia de código «sin uso» no autoriza borrar una API pública.

La reorganización previa, solicitada por separado, sustituye los antiguos
scripts sueltos por el paquete `kflame`; sus equivalencias están en
`../migration.md`. Esta simplificación conserva las 184 firmas de funciones,
métodos públicos y constructores comprobadas contra el paquete anterior.

## Cambios realizados

- Eliminados imports y variables locales sin uso; tres helpers privados muertos
  (`_normalize_Y_col`, `_concentrations`, `_arrhenius`).
- Eliminados wrappers internos de copia, conversión a CPU y conversión Y→X;
  sus llamadas ahora realizan directamente la misma operación.
- Unificada la interpolación de perfiles nativos/de referencia en el módulo
  común existente. Se conservan las funciones públicas `build_tables`, sus
  metadatos, orden de campos, tipos e interpolaciones por especie.
- Unificadas las dos métricas de error de validación/refinamiento y los
  contadores de perfilado del solver; sin módulos ni clases adicionales.
- Eliminada una copia redundante antes del saneamiento de especies. El buffer
  de trabajo sigue siendo independiente del estado de entrada.
- Simplificados empaquetado de estado y máscara transitoria mediante asignación
  a vistas, sin aritmética nueva ni cambios de tipo.
- Defaults de hilos centralizados al importar `kflame`, manteniendo los valores
  explícitos del usuario y la aplicación del tamaño de pool de Numba.
- `row_stack` sustituido por su equivalente `vstack`: causa de los cuatro
  avisos de deprecación corregida, no ocultada.
- Avisos de especie ausente agrupados por panel. Una semilla rechazada en la
  comparación opcional ya no se descarta silenciosamente.
- Excepciones acotadas para imports opcionales, caché de semillas, configuración
  de hilos y diagnóstico del predictor; errores inesperados dejan de confundirse
  con esos fallos recuperables. Se conservan las recuperaciones numéricas.

Python de producción: **14 917 → 14 817 líneas** respecto al paquete antes de
esta auditoría (100 menos). Es una reducción conservadora, no una garantía de
que toda abstracción restante sea mínima ni una medida de velocidad.

## Verificación local

- Ruff, reglas F: sin imports/locales sin uso ni nombres indefinidos detectados
  en producción y benchmarks. Vulture se usó para seleccionar candidatos, no
  como autorización automática de borrado.
- Regresión con referencia opcional instalada: **63 tests y 136 subtests pasan**.
- Wheel instalado sin Cantera, bloqueando además imports de Cantera y aperturas
  del archivo retirado de ajustes: **48 tests pasan; 15 comparaciones opcionales
  omitidas**. Las pruebas de CLI se ejecutan también fuera del checkout.
- Comparación puntual antes/después: 30 casos de empaquetado y 30 máscaras
  idénticos en bytes y tipos; incluye vacíos, ceros con signo, NaN, float32/64
  de entrada y vistas no contiguas.
- Tabulación sobre cinco perfiles guardados: 24 campos nativos y 18 de
  referencia idénticos, incluidos tipos, formas y orden de claves.
- FGM completo CH4/GRI30, phi=[0.7,0.9,1,1.1,1.4], sin semillas persistentes,
  desde el wheel nativo: **5/5 aceptadas y 28 campos exactamente iguales** al
  resultado anterior; únicamente se excluye el tiempo de resolución.
- Paralelo con dos workers: comparación antes/después usando copias idénticas
  de las cinco semillas nativas y el mismo mecanismo; 5/5 aceptadas en cada
  variante y los 28 campos no temporales coinciden exactamente. Comparar contra
  una corrida que ya actualizó la caché no compara la misma entrada: esa primera
  comparación no coincidió y se sustituyó por el ensayo con semillas congeladas.
- H2/GRI30 multicomponente/Soret a 10 atm con bootstrap nativo: aceptado,
  241 nodos, ancho 0.06 m, Su=1.3484847717269999 m/s; perfiles z/T/Y/u
  idénticos en bytes a los anteriores. Exportación FGM completa, 53 especies,
  cinco flamelets individuales y CSV verificada con Cantera bloqueado.
- Ayuda de los ocho comandos públicos y seis benchmarks ejecutada correctamente.
- Wheel inspeccionado: 47 archivos; incluye mecanismos, datos moleculares y
  atribuciones, sin manuscritos, figuras, resultados, caches ni ajustes retirados.

Las comprobaciones puntuales se ejecutaron con el snapshot local
`.local/maintenance/before-simplification/`; no se añadió otra batería de tests
redundantes al repositorio. Los outputs crudos están en `tmp/` y no se publican.

## Decisiones conservadoras y límites

Se mantienen las clases que contienen estado físico, configuración, cachés o
factorizaciones; métodos públicos de propiedades aunque no aparezcan en los
call sites internos; y el alias de compatibilidad `_sanitize_Y_full`.
Se mantienen los modelos nativo/referencia explícitos, los formatos de Jacobiano
seleccionables y los rescates Newton/PTC/BE: no son duplicación intercambiable.

No se tocaron ecuaciones, tolerancias, criterios de aceptación, refinamiento,
tipos ni secuencias aritméticas de los kernels. No se han desactivado warnings.
En compilación limpia, la prueba de transporte Schur/denso emite un aviso de
rendimiento de Numba sobre contigüidad en `l00_inverse @ l01`; se conserva visible.
Cambiar la disposición para forzar otra ruta BLAS queda fuera de esta limpieza
conservadora. No es un certificado de precisión física ni convergencia de malla.

La mejora FGM del **13.8%** pertenece al experimento anterior de perturbaciones
vectorizadas, con tres parejas y una de ellas más lenta; no se atribuye a esta
limpieza ni se suma a otras mejoras. Ver `FGM_BATCH_VECTORIZATION_20260920.md`.

## Conservación del material local

Tesis, bibliografía, PDFs, figuras, datos y sus scripts se conservan en
`.local/research/`, excluidos de Git y del wheel. Los originales previos a la
migración están respaldados en `.local/maintenance/before-layout/`.
`DECISIONES_DESCARTADAS.md` permanece público. No se borra el historial Git:
los commits antiguos pueden seguir conteniendo material de tesis.
