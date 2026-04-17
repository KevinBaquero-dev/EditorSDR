# ARCHITECTURE.md — Stream Content Pipeline
> Dueño: GPT (Arquitecto) | v0.5

## Sistema
Pipeline automatizado para convertir streams en clips cortos, seleccionados, refinados y formateados para redes sociales.

## Módulos

| # | Módulo | Input | Output | Tool |
|---|--------|-------|--------|------|
| 1 | ingestion | URL Twitch | output/raw/vod.mp4 | yt-dlp |
| 2 | transcription | vod.mp4 | output/transcripts/transcript.json | faster-whisper medium (CUDA), language=es |
| 3 | audio_analysis | vod.mp4 | output/analysis/peaks.json | librosa RMS + derivada |
| 4 | clip_candidate_generator | transcript + peaks | output/candidates/clips_candidates.json | ventana dinámica por intensidad |
| 5 | clipper | vod.mp4 + candidates | output/clips/*.mp4 | FFmpeg stream copy |
| 6 | scoring_engine | candidates + transcript + peaks | output/ranked/clips_ranked.json | score compuesto 4 features |
| 7 | selector | clips_ranked.json | output/selected/selected_clips.json | top N + score threshold + diversidad temporal |
| 8 | start_refiner | selected + transcript | output/refined/refined_clips.json | phrase_align + silence_skip |
| 9 | vertical_formatter | refined + vod.mp4 | output/vertical/vertical_NNN.mp4 | FFmpeg crop 9:16 + scale |
| 10* | subtitle_builder | refined + transcript | output/subtitles/clip_NNN.{json,srt,meta} | chunking + highlight + silence trim |
| 11* | timing_aligner | clips + subtitles JSON | output/subtitles/ (overwrite) | bandpass 80–3kHz + RMS dinámico + lerp |
| 12* | subtitle_renderer | refined + vertical | output/subtitled/subtitled_NNN.mp4 | FFmpeg subtitles= force_style |
| 13 | exporter | output/ | output/YYYY-MM-DD/ + metadata.json | organización final |

*Pasos 10–12 opcionales — activados con `--subtitles` o `--review`

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

## Flujo con subtítulos (13 pasos)
```
... (mismos 9 pasos)
→ subtitle_builder   (genera JSON+SRT editables, silence trim por segmento)
→ timing_aligner     (ajusta timestamps al audio real: bandpass + RMS dinámico + lerp)
→ subtitle_renderer  (quema SRT sobre clips verticales)
→ exporter
```

## Modo revisión (--review)
```
... pasos 1–9 normales
→ subtitle_builder   (genera archivos)
→ timing_aligner     (alinea timestamps)
[para aquí — edición manual de output/subtitles/]
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
           → timing_aligner  → sobreescribe JSON+SRT con timestamps alineados
                               respeta subtitles_edited=true (no toca edición manual)
                               debug en _meta: original/adjusted/delta por segmento
[edición opcional del JSON]
→ subtitle_renderer → subtitled_NNN.mp4
```

## Diseño de timing_aligner
```
clip.mp4 → librosa.load (mono, 16kHz)
         → bandpass filter (80–3000Hz) — aísla frecuencias de voz
         → RMS frame a frame (25ms/10ms)
por segmento:
  → ventana de búsqueda [start-0.5s, end+0.5s]
  → threshold dinámico: mean + 0.5*std (por ventana)
  → detección onset/offset
  → lerp factor dinámico (0.4 micro / 0.9 delta<0.2 / 0.7 / 0.5)
  → min 0.6s, max 1.0s de shift, fallback si sin voz
post-proceso:
  → split inteligente por dip de energía si >4s
  → gap cleaner semántico: solo ajusta si no hay pausa real (ratio <40% contexto)
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
