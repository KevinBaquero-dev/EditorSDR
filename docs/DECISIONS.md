# DECISIONS.md
> Registro de decisiones técnicas

## Decisión — 2026-04-16
Decisión: Usar yt-dlp para descarga de VODs
Opciones consideradas: yt-dlp, streamlink, API oficial de Twitch
Razón: open-source, mantenido activamente, soporta Twitch y múltiples formatos sin API key
Impacto: ingestion no requiere credenciales; compatible con cualquier VOD público
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Forzar formato .mp4 en ingestion
Opciones consideradas: mp4, mkv, mantener formato original (ts)
Razón: FFmpeg (clipper) y faster-whisper procesan mp4 sin configuración extra; evita conversiones downstream
Impacto: todos los módulos reciben un formato predecible
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Re-descarga → skip si el archivo ya existe y tiene tamaño > 0
Opciones consideradas: siempre sobrescribir, skip si existe, preguntar al usuario
Razón: VODs de 3h son costosos de re-descargar; si se necesita forzar, se borra el archivo manualmente
Impacto: descarga idempotente; no bloquea workflow en reinicios
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Centralizar salidas en /output/raw/ y /output/transcripts/
Opciones consideradas: rutas configurables por módulo, directorio plano, estructura por fecha
Razón: predecible para módulos downstream; fácil de inspeccionar manualmente
Impacto: todos los módulos asumen esta estructura de salida
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Usar faster-whisper modelo "small" con float16 en CUDA para transcription
Opciones consideradas: tiny (rápido/impreciso), small (balance), medium (lento/preciso), large-v2 (puede saturar VRAM)
Razón: "small" cabe con margen en 8GB VRAM con float16; precisión suficiente para detectar picos de contenido en MVP
Impacto: velocidad aceptable en RTX 4060; si la precisión falla, escalar a "medium" en v0.2
Versión: v0.1
