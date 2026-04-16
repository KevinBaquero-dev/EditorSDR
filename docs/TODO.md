# TODO.md
> Dueño: Claude Code — aprueba Director

## Estado del proyecto
Fase actual: MVP — Implementación
Módulo en desarrollo: clip_candidates
Bloqueos: Ninguno

## Critico
_Sin bloqueos._

## Importante
- Validar audio_analysis con vod.mp4 real (test: peaks.json no vacío, timestamps en rango válido)
- Validar transcription actualizada con QG: sin segmentos "...", logs de segmentos largos
- Confirmar instalaciones: yt-dlp, faster-whisper, librosa, scipy

## Futuro
- Re-segmentación de segmentos >30s en clip_candidates (ya documentado en DECISIONS.md)
- Modelo configurable en transcription via parámetro
- SR y threshold configurable en audio_analysis via parámetros
- Output dir configurable en transcription y audio_analysis
- Progress logging con tqdm para audio largo
