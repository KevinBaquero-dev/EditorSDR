# TODO.md
> Dueño: SH Studios — aprueba Director

## Estado del proyecto
Fase actual: v0.3 — Scoring Engine
Módulo en desarrollo: scoring_engine
Bloqueos: Ninguno

## Critico
- [x] Ajustar ventana dinámica en clip_candidate_generator (intensidad → ventana variable)
- [x] Extender finales de clip al cierre del segmento de transcript
- [x] Scoring engine — score compuesto por features de audio, transcript y duración

## Importante
- [x] Filtrar ruido constante en audio_analysis con derivada de energía
- [x] Revisar clips v0.2 vs v0.1 — QG completado, mejora confirmada
- Agregar vod_url y vod_title a metadata.json en exporter
- Validar ranking de scoring con VODs adicionales — confirmar que top clips son realmente los mejores
- Ajustar pesos de scoring con feedback real de clips

## Futuro
- Subtítulos opcionales (subtitle_engine.py — módulo separado, activable por config)
- Formato vertical (9:16)
- Feedback humano → ajuste de pesos en scoring_engine
- Modo verbose/quiet en main.py via argumento
- Parámetros de ventana configurables sin tocar el código
