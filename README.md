# EditorSDR

Convierte tus streams de Twitch en clips cortos automáticamente.

Le das la URL de tu VOD y el sistema lo descarga, transcribe, detecta los mejores momentos, los puntúa, refina los inicios y te entrega clips verticales listos para TikTok, Reels o Shorts.

Sin editar a mano. Sin revisar horas de video.

---

## ¿Cómo funciona?

1. Descarga el VOD desde Twitch
2. Transcribe el audio con timestamps (IA local, GPU)
3. Detecta picos de energía en el audio
4. Genera candidatos de clips con contexto ajustado por intensidad
5. Puntúa y selecciona los mejores (scoring engine)
6. Refina los inicios para evitar entradas en silencio o a mitad de frase
7. Convierte a formato vertical 9:16 (1080×1920)
8. Exporta todo organizado listo para subir

Opcional: genera subtítulos editables antes de quemarlos.

---

## Requisitos

- Windows 10/11
- Python 3.11–3.14
- GPU NVIDIA con CUDA (recomendado — también corre en CPU, más lento)
- FFmpeg instalado via winget o en PATH
- ~4GB RAM mínimo, 8GB VRAM recomendado

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/KevinBaquero-dev/EditorSDR.git
cd EditorSDR
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar FFmpeg

```bash
winget install Gyan.FFmpeg
```

> El sistema también lo busca automáticamente en la ruta de WinGet si no está en PATH.

---

## Uso

### Pipeline completo

```bash
python main.py https://www.twitch.tv/videos/TU_VOD_ID
```

### Con subtítulos (genera + renderiza directamente)

```bash
python main.py https://www.twitch.tv/videos/TU_VOD_ID --subtitles
```

### Modo revisión (genera subtítulos, pausa para editar antes de renderizar)

```bash
python main.py https://www.twitch.tv/videos/TU_VOD_ID --review
# → edita output/subtitles/clip_NNN.json o .srt
# → vuelve a ejecutar sin --review para renderizar
```

---

## Salida

```
output/
├── raw/vod.mp4                          # VOD original
├── transcripts/transcript.json          # Transcripción con timestamps
├── analysis/peaks.json                  # Picos de energía detectados
├── candidates/clips_candidates.json     # Candidatos generados
├── clips/clip_NNN.mp4                   # Clips horizontales cortados
├── ranked/clips_ranked.json             # Ranking con score compuesto
├── selected/selected_clips.json         # Top clips seleccionados
├── refined/refined_clips.json           # Clips con inicios refinados
├── vertical/vertical_NNN.mp4            # Clips 9:16 listos para subir
├── subtitles/                           # (--subtitles / --review)
│   ├── clip_NNN.json                    #   Subtítulos editables (fuente de verdad)
│   ├── clip_NNN.srt                     #   Formato SRT para FFmpeg
│   └── clip_NNN_meta.json               #   Estado de edición (subtitles_edited)
├── subtitled/subtitled_NNN.mp4          # Clips verticales con subtítulos
└── YYYY-MM-DD/                          # Exportación final organizada
    ├── vertical_NNN.mp4
    └── metadata.json
```

---

## Módulos

| Módulo | Función |
|--------|---------|
| `ingestion.py` | Descarga VOD con yt-dlp |
| `transcription.py` | Transcribe con faster-whisper (CUDA) |
| `audio_analysis.py` | Detecta picos de energía (RMS + derivada) |
| `clip_candidate_generator.py` | Genera candidatos con ventana dinámica por intensidad |
| `clipper.py` | Corta clips con FFmpeg (stream copy) |
| `scoring_engine.py` | Score compuesto: intensity + text_density + hook_strength + duration |
| `selector.py` | Selecciona top 15 con filtro de score y diversidad temporal |
| `start_refiner.py` | Refina inicios: phrase_align o silence_skip según intensidad |
| `vertical_formatter.py` | Convierte a 9:16 — crop configurable (center/left/right + offset px) |
| `subtitle_builder.py` | Genera JSON+SRT editables con chunking, highlight y offset de timing |
| `subtitle_renderer.py` | Quema SRT (editado o auto) sobre clips verticales |
| `subtitle_engine.py` | Burn directo sin etapa editable (utilidad standalone) |
| `exporter.py` | Organiza salida final con metadata |

---

## Subtítulos — flujo editable

Los subtítulos siguen un flujo de dos etapas para permitir edición humana antes del render:

```
transcript → subtitle_builder → clip_NNN.json   ← editable en cualquier editor
                              → clip_NNN.srt    ← formato render
                              → clip_NNN_meta   ← protección: subtitles_edited=true
                                                    evita sobreescribir ediciones
→ subtitle_renderer           → subtitled_NNN.mp4
```

Para regenerar el SRT después de editar el JSON:

```python
from src.modules.subtitle_builder import srt_from_json
srt_from_json("output/subtitles/clip_001.json")
# → regenera clip_001.srt y marca subtitles_edited=True en meta
```

---

## Configuración rápida

Los parámetros más útiles están en los módulos:

| Parámetro | Archivo | Default | Descripción |
|-----------|---------|---------|-------------|
| `MAX_PEAKS` | `audio_analysis.py` | 50 | Picos de audio a detectar |
| `MAX_CANDIDATES` | `clip_candidate_generator.py` | 25 | Candidatos a generar |
| `TOP_N` | `selector.py` | 15 | Clips a seleccionar |
| `SCORE_THRESHOLD` | `selector.py` | 0.65 | Score mínimo para incluir |
| `CROP_POSITION` | `vertical_formatter.py` | center | center / left / right |
| `CROP_OFFSET_PX` | `vertical_formatter.py` | 0 | Offset horizontal en píxeles |
| `SUBTITLE_OFFSET` | `subtitle_builder.py` | -0.2s | Adelantar subtítulos (sincronía) |
| `CAPITALIZE` | `subtitle_builder.py` | True | Capitalizar inicio de cada chunk |
| `FontSize` | `subtitle_engine.py` | 18 | Tamaño de fuente (1080×1920) |
| `MarginV` | `subtitle_engine.py` | 60 | Distancia en px desde el borde inferior |

---

## Estado del proyecto

| Módulo / Feature | Estado |
|-----------------|--------|
| Descarga VOD | ✅ Listo |
| Transcripción CUDA | ✅ Listo |
| Análisis de audio (RMS + derivada) | ✅ Listo v0.2 |
| Generación de candidatos (ventana dinámica) | ✅ Listo v0.2 |
| Corte de clips (stream copy) | ✅ Listo |
| Scoring engine (4 features) | ✅ Listo v0.3 |
| Selector (top N + diversidad) | ✅ Listo v0.3.1 |
| Start refiner (phrase_align + buildup) | ✅ Listo v0.4 |
| Formato vertical 9:16 (crop configurable) | ✅ Listo v0.4 |
| Subtítulos editables (JSON → SRT → render) | ✅ Listo v0.4.2 |
| Highlight keywords (! ?) | ✅ Listo |
| Ajuste de estilo de subtítulos (size, posición) | ✅ Listo |
| Feedback humano → ajuste de pesos | 🔲 Pendiente |
| Scoring antes de clipper (refactor) | 🔲 Pendiente |
| Auto-upload | 🔲 Futuro |

---

## Social Media

La carpeta `social/` contiene las plantillas de contenido para Instagram:

| Archivo | Descripción |
|---------|-------------|
| `ig-posts.html` | Plantilla HTML con los diseños de posts (1080×1080) e historias (1080×1920) |
| `screenshot.js` | Script Puppeteer — exporta cada pieza a PNG a resolución @2x |

### Exportar

```bash
# Instalar dependencias (solo la primera vez)
cd social && npm install

# Exportar todas las piezas
node screenshot.js

# Exportar una sola pieza
node screenshot.js post-01
```

Los archivos se generan en `social/exports/`. La carpeta `exports/` y `node_modules/` están en `.gitignore` — solo se versionan las plantillas fuente.

**Piezas disponibles:** post-01 al post-06 (1080×1080), story-01 al story-06 (1080×1920).

---

## Créditos

Desarrollado por **SH Studios**.
