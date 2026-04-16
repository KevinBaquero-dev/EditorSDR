# EditorSDR — Stream Content Pipeline

Pipeline automatizado para convertir streams de Twitch en clips cortos listos para redes sociales.

## ¿Qué hace?

Toma un VOD de Twitch (o archivo local), lo transcribe, detecta los momentos de mayor energía de audio y genera clips automáticamente.

## Flujo

```
URL / archivo local
  → Ingestion       (descarga el VOD como .mp4)
  → Transcription   (genera transcript con timestamps)
  → Audio Analysis  (detecta picos de audio)
  → Clip Candidates (genera segmentos candidatos)
  → Clipper         (corta los clips con FFmpeg)
  → Exporter        (organiza la salida)
```

## Requisitos

- Python 3.10+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- FFmpeg
- NVIDIA GPU recomendada (RTX 4060 8GB o similar)

## Instalación

```bash
pip install yt-dlp faster-whisper
```

FFmpeg debe estar disponible en el PATH del sistema.

## Uso

```python
from src.modules.ingestion import download_vod
from src.modules.transcription import transcribe_video

video_path = download_vod("https://www.twitch.tv/videos/123456789")
transcript_path = transcribe_video(video_path)
```

## Estructura del proyecto

```
EditorSDR/
├── docs/
│   ├── CLAUDE.md           # Contexto operativo
│   ├── ARCHITECTURE.md     # Arquitectura del sistema
│   ├── DECISIONS.md        # Decisiones técnicas
│   ├── LESSONS.md          # Errores y aprendizajes
│   ├── TODO.md             # Pendientes
│   ├── TEST_CASES.md       # Casos de prueba
│   └── PROMPTS.md          # Prompts versionados
├── src/
│   ├── modules/
│   │   ├── ingestion.py
│   │   └── transcription.py
│   └── core/
├── output/
│   ├── raw/                # VOD descargado
│   ├── transcripts/        # transcript.json
│   └── clips/              # Clips generados
└── logs/
```

## Estado actual

| Módulo | Estado |
|---|---|
| ingestion | Implementado |
| transcription | Implementado |
| audio_analysis | Pendiente |
| clip_candidates | Pendiente |
| clipper | Pendiente |
| exporter | Pendiente |

## Framework

Desarrollado bajo [SckrusH Framework v1.0](docs/ARCHITECTURE.md).
