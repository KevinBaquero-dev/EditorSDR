# DECISIONS.md
> Registro de decisiones técnicas

## Decisión — 2026-04-16
Decisión: Usar yt-dlp para descarga de VODs
Opciones consideradas: yt-dlp, streamlink, API oficial de Twitch
Razón: open-source, mantenido activamente, soporta Twitch sin API key
Impacto: ingestion no requiere credenciales; compatible con VODs públicos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Forzar formato .mp4 en ingestion
Opciones consideradas: mp4, mkv, ts (original)
Razón: FFmpeg y faster-whisper procesan mp4 sin configuración extra; evita conversiones downstream
Impacto: todos los módulos reciben formato predecible
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Re-descarga → skip si archivo ya existe con tamaño > 0
Opciones consideradas: siempre sobrescribir, skip si existe, preguntar
Razón: VODs de 3h+ son costosos de re-descargar; para forzar se borra el archivo
Impacto: descarga idempotente; no bloquea reinicios del pipeline
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Centralizar salidas en /output/raw/, /output/transcripts/, /output/analysis/
Opciones consideradas: rutas por módulo, directorio plano, estructura por fecha
Razón: predecible para módulos downstream; fácil de inspeccionar manualmente
Impacto: todos los módulos asumen esta estructura
Versión: v0.1

## Decisión — 2026-04-16
Decisión: faster-whisper modelo "small" con float16 en CUDA para transcription
Opciones consideradas: tiny (rápido/impreciso), small (balance), medium (lento), large-v2 (puede saturar VRAM)
Razón: "small" float16 cabe con margen en RTX 4060 8GB; precisión suficiente para MVP
Impacto: velocidad aceptable; si falla precisión → escalar a "medium" en v0.2
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Mantener segmentación original de Whisper; re-segmentar en clip_candidates
Opciones consideradas: re-segmentar en transcription, mantener original, truncar en transcription
Razón: transcription no debe mezclar responsabilidades; segmentos >30s se manejan downstream
Impacto: clip_candidates debe subdividir segmentos largos antes de generar candidatos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Filtrar segmentos vacíos/basura de Whisper en transcription
Opciones consideradas: mantener todo, filtrar vacíos, filtrar por confianza
Razón: Whisper genera "...", ".", " " en silencio — contaminan clip_candidates
Impacto: transcript más limpio; menos falsos candidatos downstream
Versión: v0.1

## Decisión — 2026-04-16
Decisión: audio_analysis usa librosa con SR=16000 mono
Opciones consideradas: pydub (más simple), librosa SR original, librosa SR=16000
Razón: 16000 Hz es suficiente para análisis de energía RMS; reduce RAM de ~2GB a ~690MB en VOD 3h
Impacto: carga en RAM manejable con 16GB; no afecta precisión de detección de picos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Umbral dinámico en audio_analysis (mean + 1 std dev)
Opciones consideradas: umbral fijo, percentil 90, mean + N*std
Razón: umbral fijo no escala entre streams tranquilos y streams ruidosos; mean+std se adapta al contenido
Impacto: menos falsos positivos en streams de baja energía; más consistencia entre videos distintos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Mínimo 2s de distancia entre picos en audio_analysis
Opciones consideradas: 0.5s, 1s, 2s, 5s
Razón: gritos y reacciones duran >1s; 2s evita que un solo evento genere 10 picos solapados
Impacto: peaks.json más limpio; menos candidatos redundantes en clip_candidates
Versión: v0.1
