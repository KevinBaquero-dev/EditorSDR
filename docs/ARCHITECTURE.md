# ARCHITECTURE.md — Stream Content Pipeline
> Dueño: GPT (Arquitecto) | v0.4

## Sistema
Pipeline automatizado para convertir streams en clips cortos, seleccionados, refinados y formateados para redes sociales.

## Módulos

| # | Módulo | Input | Output | Tool |
|---|--------|-------|--------|------|
| 1 | ingestion | URL Twitch | output/raw/vod.mp4 | yt-dlp |
| 2 | transcription | vod.mp4 | output/transcripts/transcript.json | faster-whisper (CUDA) |
| 3 | audio_analysis | vod.mp4 | output/analysis/peaks.json | librosa RMS + derivada |
| 4 | clip_candidate_generator | transcript + peaks | output/candidates/clips_candidates.json | ventana dinámica por intensidad |
| 5 | clipper | vod.mp4 + candidates | output/clips/*.mp4 | FFmpeg stream copy |
| 6 | scoring_engine | candidates + transcript + peaks | output/ranked/clips_ranked.json | score compuesto 4 features |
| 7 | selector | clips_ranked.json | output/selected/selected_clips.json | top N + score threshold + diversidad temporal |
| 8 | start_refiner | selected + transcript | output/refined/refined_clips.json | phrase_align + silence_skip |
| 9 | vertical_formatter | refined + vod.mp4 | output/vertical/vertical_NNN.mp4 | FFmpeg crop 9:16 + scale |
| 10* | subtitle_builder | refined + transcript | output/subtitles/clip_NNN.{json,srt,meta} | chunking + highlight |
| 11* | subtitle_renderer | refined + vertical | output/subtitled/subtitled_NNN.mp4 | FFmpeg subtitles= force_style |
| 12 | exporter | output/ | output/YYYY-MM-DD/ + metadata.json | organización final |

*Pasos 10–11 opcionales — activados con `--subtitles` o `--review`

## Flujo principal (10 pasos)
```
URL
→ ingestion
→ transcription
→ audio_analysis
→ clip_candidate_generator
→ clipper
→ scoring_engine
→ selector
→ start_refiner
→ vertical_formatter
→ exporter
```

## Flujo con subtítulos (12 pasos)
```
... (mismos 9 pasos)
→ subtitle_builder   (genera JSON+SRT editables, protegidos si editados)
→ subtitle_renderer  (quema SRT sobre clips verticales)
→ exporter
```

## Modo revisión (--review)
```
... pasos 1–9 normales
→ subtitle_builder   (genera archivos, para aquí)
[edición manual de output/subtitles/]
→ re-run sin --review → subtitle_renderer → exporter
```

## Diseño de scoring
```
audio_analysis  → detecta (peaks + intensity)
clip_candidates → propone (ventana + transcript)
scoring_engine  → decide  (score = intensity*0.30 + text_density*0.25 + hook_strength*0.25 + duration_score*0.20)
selector        → filtra  (top 15, score ≥ 0.65, gap temporal ≥ 30s)
```

## Diseño de subtítulos
```
transcript → subtitle_builder → clip_NNN.json  (fuente de verdad, editable)
                             → clip_NNN.srt   (formato render)
                             → clip_NNN_meta  (subtitles_edited flag)
[edición opcional]
→ subtitle_renderer → subtitled_NNN.mp4
```

## Estilo de subtítulos (subtitle_engine.py — _SUBTITLE_STYLE)
- FontSize=18, Bold=1, FontName=Arial
- Alignment=2 (bottom center), MarginV=60px desde borde inferior
- PrimaryColour blanco, OutlineColour negro, Outline=3, Shadow=1

## Reglas
- Cada módulo tiene un único input/output explícito
- No hay lógica cruzada entre módulos
- JSON como formato estándar entre módulos
- timestamps en segundos
- scoring ≠ detección — capas separadas
- subtítulos nunca sobrescriben edición manual

## Hardware target
GPU: RTX 4060 8GB | RAM: 16GB | OS: Windows 11 | Python 3.11–3.14
