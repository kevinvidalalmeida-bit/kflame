# Evidencia compacta de la campaña 2026-09-03

Este directorio conserva los resultados numéricos compactos citados por
`TESIS/TFM_FGM_FINAL.tex`. Las tablas NPZ y los perfiles completos se generan en
`FGM/resultados/`, que permanece ignorado por Git debido a su tamaño.

## Contenido

- `verification_summary.json`: métricas de perfiles, conservación, convergencia
  de malla, ablación de continuación y comparación estricta V2--Cantera.
- `profile_metrics.csv`: métricas por valor de `phi` después de alinear frentes.
- `mesh_metrics.csv`: estudio de cuatro niveles de refinamiento para `phi=1`.
- `linear_solver_ablation.json`: dos pares completos de barridos con sustitución
  por bloque y sustitución fusionada, además de la comparación intercalada con
  Cantera.
- `linear_solver_ablation.csv`: forma tabular de los dos pares de ablación.

## Resultados que pueden citarse

- Barrido estricto: 28.3165 s en V2 y 61.3822 s en Cantera, razón 2.168x.
- Continuación sin región de confianza frente a radio 1.15: 153.8411 s frente a
  28.3165 s, razón 5.43x.
- Sustitución fusionada: reducción mediana de 12.02 % en dos pares completos.
- Caso `phi=1` en nivel ultra: diferencia de velocidad de llama de 0.0407 %.
- Error máximo de perfiles del barrido: 0.145 % para temperatura y 2.52 % para
  liberación de calor en norma L2.
- La validación dejando una fila fuera alcanza 27.9 % en liberación de calor;
  por ello la tabla de cinco mezclas es demostrativa y no una tabla CFD final.

## Regeneración

```powershell
python .\FGM\scripts\analyze_tfm_evidence.py
python .\FGM\scripts\analyze_linear_solver_ablation.py
```

Los analizadores requieren las campañas completas en las rutas predeterminadas.
Una repetición destinada a publicación debe conservar además los comandos
literales, metadatos, hashes de NPZ, revisión Git y configuración de hardware.
