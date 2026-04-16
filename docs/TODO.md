# TODO.md
> Dueño: Claude Code — aprueba Director

## Estado del proyecto
Fase actual: MVP — Implementación
Módulo en desarrollo: clipper
Bloqueos: Ninguno

## Critico
_Sin bloqueos._

## Importante
- Validar clip_candidate_generator con transcript.json + peaks.json reales
- Verificar que no hay clips duplicados ni solapados en output
- Confirmar que nearest_text en candidatos tiene sentido con el momento del video

## Futuro
- Ajustar ventana WINDOW_BEFORE/AFTER según feedback de clips reales (ahora: -10s/+15s)
- Filtrar picos en audio_analysis con derivada de energía para reducir ASMR accidental (teclado, ruido ambiente)
- Ignorar frecuencias bajas constantes (música de fondo) en audio_analysis v0.2
- Modelo configurable en transcription via parámetro
- SR y threshold configurable en audio_analysis via parámetros
