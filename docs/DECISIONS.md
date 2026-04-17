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
Decisión: Centralizar salidas en /output/raw/, /output/transcripts/, /output/analysis/, /output/candidates/
Opciones consideradas: rutas por módulo, directorio plano, estructura por fecha
Razón: predecible para módulos downstream; fácil de inspeccionar manualmente
Impacto: todos los módulos asumen esta estructura
Versión: v0.1

## Decisión — 2026-04-16
Decisión: faster-whisper modelo "small" con float16 en CUDA para transcription
Opciones consideradas: tiny, small, medium, large-v2
Razón: "small" float16 cabe con margen en RTX 4060 8GB; precisión suficiente para MVP
Impacto: velocidad aceptable; si falla precisión → escalar a "medium" en v0.2
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Mantener segmentación original de Whisper; re-segmentar en clip_candidates si necesario
Opciones consideradas: re-segmentar en transcription, mantener original, truncar
Razón: transcription no debe mezclar responsabilidades
Impacto: clip_candidates es responsable de manejar segmentos >30s si afectan las ventanas
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Filtrar segmentos vacíos/basura de Whisper en transcription
Opciones consideradas: mantener todo, filtrar vacíos, filtrar por confianza
Razón: "...", ".", " " en silencio contaminan clip_candidates
Impacto: transcript más limpio; menos falsos candidatos downstream
Versión: v0.1

## Decisión — 2026-04-16
Decisión: audio_analysis usa librosa con SR=16000 mono
Opciones consideradas: pydub, librosa SR original, librosa SR=16000
Razón: 16000 Hz suficiente para energía RMS; reduce RAM a ~690MB en VOD 3h (dentro de 16GB)
Impacto: carga manejable; no afecta precisión de detección de picos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Umbral dinámico en audio_analysis (mean + 1 std dev)
Opciones consideradas: umbral fijo, percentil 90, mean + N*std
Razón: se adapta al contenido; streams tranquilos y ruidosos no requieren ajuste manual
Impacto: menos falsos positivos; consistencia entre videos distintos
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Mínimo 2s de distancia entre picos en audio_analysis
Opciones consideradas: 0.5s, 1s, 2s, 5s
Razón: evita que un solo grito genere 10 picos solapados
Impacto: peaks.json más limpio; menos candidatos redundantes
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Ventana de clip -10s / +15s alrededor del pico en clip_candidates
Opciones consideradas: -5/+10, -10/+15, -15/+20, simétrica
Razón: +15s da más contexto post-pico porque las reacciones se extienden después del momento; -10s da contexto previo suficiente para entender qué pasó
Impacto: clips con contexto; ajustable en v0.2 con feedback de clips reales
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Merge de candidatos con solapamiento >50% del clip más corto
Opciones consideradas: no merge, merge siempre, merge por porcentaje
Razón: picos cercanos en el tiempo forman parte del mismo momento; merge evita clips duplicados del mismo evento
Impacto: menos candidatos redundantes; clips más coherentes
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Incluir nearest_text en clips_candidates.json
Opciones consideradas: solo start/end/peak, incluir texto más cercano
Razón: permite validación visual rápida de si el candidato tiene sentido sin abrir el video
Impacto: JSON más grande pero invaluable para debugging y QG del Director
Versión: v0.1

## Decisión — 2026-04-16
Decisión: Ventana dinámica en clip_candidates según intensidad del pico
Opciones consideradas: ventana fija (-10/+15), ventana por percentil, dinámica por intensidad
Razón: picos de alta intensidad representan momentos más importantes — merecen más contexto previo y posterior; ventana fija recortaba momentos clave
Impacto: clips de alta intensidad (>0.8) tienen ventana -15s/+20s; intensidad media -12s/+18s; baja -10s/+15s
Versión: v0.2

## Decisión — 2026-04-16
Decisión: Extender final de clip al cierre del segmento de transcript en curso
Opciones consideradas: end fijo por ventana, extender al segmento, silencio detectado
Razón: clips que cortan frases a la mitad no son usables; el segmento de transcript marca el cierre natural de una idea
Impacto: clips terminan en puntos naturales; +0.3s de buffer de cierre para evitar corte brusco
Versión: v0.2

## Decisión — 2026-04-16
Decisión: Añadir derivada de RMS al score de audio_analysis (peso 50%)
Opciones consideradas: RMS puro, RMS + derivada, onset detection de librosa
Razón: RMS puro detecta ruido constante (música, teclado, ambiente) como picos; la derivada solo se activa en cambios bruscos que corresponden a reacciones reales
Impacto: score = RMS_norm + 0.5 * diff_norm — reduce falsos positivos de fondo; el peso 50% es ajustable
Versión: v0.2

## Decisión — 2026-04-16
Decisión: subtitle_engine como módulo opcional, no integrado en pipeline por defecto
Opciones consideradas: integrar en exporter, módulo independiente activable, parámetro en main.py
Razón: subtítulos requieren re-encode completo — costoso en tiempo; MVP no lo necesita; mejor mantenerlo separado hasta que el pipeline básico esté estable
Impacto: burn_subtitles() existe y es funcional pero no se llama desde main.py; activable manualmente
Versión: v0.2
