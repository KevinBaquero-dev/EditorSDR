# TODO.md
> Dueño: SH Studios — aprueba Director

## Estado del proyecto
Fase actual: v0.2 — Refinamiento
Módulo en desarrollo: clip_candidate_generator, audio_analysis
Bloqueos: Ninguno

## Critico
- [x] Ajustar ventana dinámica en clip_candidate_generator (intensidad → ventana variable)
- [x] Extender finales de clip al cierre del segmento de transcript

## Importante
- [x] Filtrar ruido constante en audio_analysis con derivada de energía
- Revisar clips v0.2 vs v0.1 — comparar calidad de cortes
- Agregar vod_url y vod_title a metadata.json en exporter

## Futuro
- Subtítulos opcionales (subtitle_engine.py — módulo separado, activable por config)
- Formato vertical (9:16)
- Scoring IA (semantic_analysis module)
- Modo verbose/quiet en main.py via argumento
- Parámetros de ventana configurables sin tocar el código
