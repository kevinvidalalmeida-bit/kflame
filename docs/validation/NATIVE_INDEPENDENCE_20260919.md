# Auditoría de independencia y optimización — 19/09/2026

Nota histórica: la revisión del [20/09/2026](NATIVE_ALL_STAGES_20260920.md)
eliminó además la lectura de polinomios de transporte preexportados y la
dependencia de Cantera del posprocesado de figuras. El texto siguiente conserva
el estado y las mediciones del día 19, no describe esas dependencias como actuales.

La ruta productiva V2/FGM funciona sin instalar ni importar Cantera con los
mecanismos incluidos. Es la ruta por defecto. La independencia es de ejecución:
se conservan datos locales y formulaciones con atribución a sus fuentes.
Los programas destinados a comparar con Cantera siguen requiriéndolo.

## Hallazgos y correcciones

La revisión anterior solo había retirado la dependencia de `FreeFlameProblem`.
El generador FGM todavía importaba Cantera directamente e indirectamente desde
`fgm_common`, `run_saved_comparison` y `species_backend`. Construía fases para
contar especies, obtener Bilger y evaluar punto a punto las propiedades tabuladas.
El benchmark Soret también importaba Cantera aun solicitando `--native-only`.

Ahora el mecanismo se resuelve desde una ruta local o los YAML incluidos;
`FlameCase` se importa desde `config`, y la composición y Bilger se calculan
nativamente. Los campos `rho`, `cp_mass`, `conductivity`, `qdot` y `omega_c`
se obtienen del backend solicitado. La conductividad multicomponente usa ese
modelo, no el cierre promediado por mezcla. Se mantienen los mismos criterios
de aceptación, tolerancias, ecuaciones y políticas de refinamiento.

La inicialización anterior fallaba para algunas mezclas pobres de H2 y excluía
equilibrios inferiores a 400 K; además exigía carbono incluso en mecanismos de
hidrógeno. Se añadieron balances C/H/O/S que toleran elementos ausentes,
residuos logarítmicos estables, Jacobiano analítico y una estimación de potenciales
mediante programación lineal cuando el primer intento resulta degenerado.
La temperatura se busca dentro del intervalo común del mecanismo. Una falta de
convergencia se comunica como error, sin recurrir a Cantera.

El lector respeta el orden de especies de la primera fase ideal-gas. Una ruta
explícita inexistente falla, sin sustituir el mecanismo por otro archivo homónimo.
Se incluyen GRI-Mech 3.0 y el submecanismo H2/O2; su procedencia consta en
`V2/materiales/data/README.md`. El mecanismo ya cargado se reutiliza en la
construcción del problema y del backend para evitar copias repetidas.

## Opciones por defecto y dependencias restantes

| Entrada | Comportamiento |
|---|---|
| `FreeFlameProblem` | Mezcla y equilibrio HP nativos |
| `solver._make_backend` | `NativeSpeciesBackend` si no se suministra otro |
| `generate_fgm_tables_native.py` | `--transport-backend native`; inicialización, Bilger, solución y posprocesado nativos |
| `benchmark_soret_native.py` | `native_only=True`; `--no-native-only` solicita la comparación |
| `run_saved_comparison.py`, generador Cantera, diagnósticos y figuras de referencia | Requieren Cantera por su propósito explícito |
| Tests contra Cantera | Dependencia opcional de verificación en `requirements-test.txt` |

`acceptance_criterion="cantera"`, `cantera_seed_grid` y `cantera_local` son
nombres históricos de algoritmos implementados localmente. No importan Cantera.
El generador usa por defecto Jacobiano por bloques y sustitución compilada;
la API de bajo nivel `SolveOptions()` conserva sus opciones históricas de
Jacobiano y aceptación, aunque su backend por defecto es nativo.

Soret sigue siendo una opción física explícita. El transporte predeterminado
del generador es promediado por mezcla. `requirements-native.txt` contiene
NumPy, SciPy, Numba y PyYAML, sin Cantera.

El transporte promediado por mezcla usa `cantera_transport_poly_coeffs.json`,
un archivo de coeficientes exportados previamente. El transporte multicomponente
ajusta sus polinomios con los parámetros moleculares y las tablas locales de
colisión. Esto permite ejecutar sin Cantera, pero no significa que todos los
datos o algoritmos tengan origen exclusivo en V2. Los mecanismos nuevos o con
parámetros de transporte modificados necesitan validación específica de esos
coeficientes; la auditoría no certifica mecanismos arbitrarios ni fases no ideales.

## Comprobación con Cantera bloqueado

`validation/no_cantera/sitecustomize.py` intercepta las importaciones en el
proceso principal y en los procesos hijos. Se activó mediante `PYTHONPATH`;
el test comprueba además que intentar importar Cantera produce un error.
Los resultados están en `tmp/audit_cantera_20260919/`.

| Prueba | Resultado |
|---|---|
| Construcción y backend por defecto; importación del generador | Correctos con Cantera bloqueado |
| CH4, phi = 0.7, 0.9, 1.0, 1.1, 1.4, caché inicialmente vacía | 5/5 aceptadas; tabla y perfiles generados |
| Regeneración de esas cinco llamas, dos procesos | 5/5 aceptadas; tabla válida |
| Malla Z con dos objetivos | 2/2 aceptadas; inversión nativa y tabla válidas |
| FGM H2/Soret, 1 atm, arranque nativo de dos etapas | Llama aceptada; tabla válida con conductividad multicomponente |
| H2/Soret, 10 atm, arranque nativo de dos etapas | Aceptada: 241 nodos, 0.06 m, Su = 1.3484847782 m/s |

El barrido frío de cinco mezclas tardó 17.70 s y su regeneración 3.33 s en esta
comprobación. Son escenarios diferentes y ejecuciones individuales; esos
tiempos no se presentan como una ablación ni como una nueva comparación con Cantera.

La suite completa final pasó **50 tests y 103 subtests**. La inicialización se
comparó en 36 combinaciones de mecanismo, combustible, phi, presión y temperatura:
GRI30/CH4, GRI30/H2 y H2O2/H2; phi 0.7/1/1.4; 1/10 atm; 300/700 K. Se exigieron
diferencias inferiores a 2e-5 K y 2e-9 en fracciones másicas, además de balances
elementales y de entalpía. Se probaron por separado mezclas muy pobres y corrientes
diluidas. Los campos nativos de tabla se contrastaron con ambos modelos de
transporte. La compilación Python y `git diff --check` pasaron.

Se añadió un trabajo CI que instala solo dependencias nativas y genera una llama
fría y una tabla FGM; su ejecución remota queda pendiente del siguiente ciclo CI.

## Optimización medida

`NativeMulticomponentTransport` reconstruía los mismos ajustes moleculares al
cambiar de malla. Ahora reutiliza arrays inmutables en una caché limitada a ocho
entradas, identificadas por todas las entradas numéricas del ajuste. La presión
y la composición se aplican durante la evaluación y no se congelan en la caché.
Los tests comprueban reutilización, inmutabilidad e invalidación al cambiar
parámetros moleculares.

La ablación se reproduce con:

```powershell
python validation/benchmark_native_dependencies.py --pairs 3 --output validation/native_fit_ablation_20260919.json
```

H2/aire, GRI30, phi=1, 300 K, 1 atm, multicomponente con Soret; arranque por
mezcla con factor 2 y refinamiento final ratio=2.5, slope=0.04, curve=0.08,
prune=0.003. Se excluyeron dos calentamientos completos y se alternó el orden
de tres pares. Cada resolución comenzó sin perfil previo y con caché molecular
vacía, de modo que también se incluye el primer ajuste de coeficientes.

| Mediana, tres pares | Sin reutilización | Con reutilización | Reducción |
|---|---:|---:|---:|
| Tiempo completo medido | 4.4064 s | 3.7053 s | 15.91 % |
| Resolución | 4.2055 s | 3.4975 s | 16.83 % |

Las seis llamas fueron aceptadas y sus arrays `z`, `T`, `Y` y `u` son
**idénticos elemento a elemento**. En todas: 207 nodos y Su = 2.1065603050 m/s.
La reducción corresponde a esta optimización y este caso; tres pares no
establecen una aceleración universal. Los registros están en
`validation/native_fit_ablation_20260919.json`.

También se detectó una compilación Numba adicional de 27.74 s durante la primera
exportación Soret: las columnas del estado entregaban un vector T no contiguo.
El posprocesado ahora entrega T contiguo, reutilizando la especialización del
solver. En la comprobación posterior, el posprocesado tardó 0.133 s y el recorrido
FGM completo 4.565 s. Los campos tabulados resultaron idénticos. Esta observación
describe un coste de compilación evitado, no una reducción porcentual sostenida.

## Trazabilidad y cuestiones separadas

La generación FGM existente construye las corrientes en base molar pero normaliza
Bilger contra las mismas cadenas interpretadas en base másica. Se conservó esa
convención para reproducir las coordenadas de las tablas existentes, y ahora se
declara en los metadatos. Para CH4/aire estequiométrico produce Z=0.05003057.
Unificar ambas bases requiere regenerar las coordenadas y las comparaciones de
interpolación en una campaña separada. Véase la definición de `basis` en la
[documentación de Cantera](https://www.cantera.org/3.2/python/thermo.html).

Los benchmarks y resultados históricos del TFM conservan la implementación con
la que se obtuvieron. Esta auditoría no los convierte retrospectivamente en
ejecuciones sin Cantera ni sustituye sus pruebas de independencia espacial.
La optimización nueva no relaja tolerancias ni cambia el modelo de llama.
