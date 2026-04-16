# ARCHITECTURE.md — Stream Content Pipeline
> Dueño: GPT (Arquitecto) | v1 MVP

## Sistema
Pipeline automatizado para convertir streams en clips cortos.

## Módulos

| Módulo | Input | Output | Tool |
|---|---|---|---|
| ingestion | URL Twitch / archivo local | video.mp4 | yt-dlp |
| transcription | video.mp4 | transcript.json (texto + timestamps) | faster-whisper |
| audio_analysis | video.mp4 | peaks.json (timestamps relevantes) | energía de señal |
| clip_candidates | transcript + peaks | clips_candidates.json | ventana alrededor de picos |
| clipper | video + clips_candidates | clips/*.mp4 | FFmpeg |
| exporter | clips + metadata | estructura organizada | — |

## Flujo
```
Usuario → URL
→ ingestion
→ transcription
→ audio_analysis
→ clip_candidates
→ clipper
→ exporter
```

## Reglas
- Cada módulo tiene input/output explícito
- No hay lógica cruzada entre módulos
- No se usa IA compleja en MVP
- Máximo 20 clips generados
- JSON como formato estándar entre módulos
- timestamps en segundos

## Riesgos detectados
- Clips irrelevantes por solo audio
- Transcripción lenta
- Mal timing en cortes
