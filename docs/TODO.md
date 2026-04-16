# TODO.md
> Dueño: Claude Code — aprueba Director

## Estado del proyecto
Fase actual: MVP — Implementación
Módulo en desarrollo: audio_analysis
Bloqueos: Ninguno

## Critico
_Sin bloqueos._

## Importante
- Validar ingestion con URL real de Twitch (test: descarga + size check + ruta correcta)
- Validar transcription con vod.mp4 real (test: JSON válido + timestamps + segmentos no vacíos)
- Confirmar que yt-dlp y faster-whisper están instalados en el entorno

## Futuro
- Soporte para archivos locales en ingestion (skip yt-dlp si ya es ruta local)
- Modelo configurable en transcription via argumento (no hardcodeado)
- Progress logging con tqdm para descargas largas
- Output dir configurable en transcription (actualmente fijo en output/transcripts)
