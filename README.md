# EditorSDR

Convierte tus streams de Twitch en clips cortos automáticamente.

Le das la URL de tu VOD y el sistema lo analiza, detecta los mejores momentos y te entrega los clips listos para subir.

Sin editar a mano. Sin revisar horas de video.

---

## ¿Cómo funciona?

1. Descarga el VOD desde Twitch
2. Transcribe el audio con timestamps usando IA local
3. Analiza la energía del audio para detectar los momentos más intensos
4. Genera candidatos de clips con contexto ajustado por intensidad
5. Corta los clips directamente del video original (sin re-encode)
6. Exporta todo organizado en una carpeta lista para usar

---

## Requisitos

- Windows 10/11
- Python 3.11–3.14
- GPU NVIDIA con CUDA (recomendado — también corre en CPU, más lento)
- FFmpeg instalado via winget o en PATH
- ~4GB de RAM mínimo, 8GB VRAM recomendado para transcripción GPU

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

Si no lo tienes instalado:

```bash
winget install Gyan.FFmpeg
```

> El sistema también lo busca automáticamente en la ruta de WinGet si no está en PATH.

### 5. Verificar CUDA (opcional pero recomendado)

La transcripción usa `faster-whisper` con CUDA. Si tienes GPU NVIDIA con drivers actualizados, funciona automáticamente. Si no, el sistema cae a CPU.

---

## Uso

### Uso básico

```bash
python main.py https://www.twitch.tv/videos/TU_VOD_ID
```

Reemplaza `TU_VOD_ID` con el ID numérico de tu VOD de Twitch.

**Ejemplo:**

```bash
python main.py https://www.twitch.tv/videos/2741089850
```

### ¿Qué pasa después de ejecutarlo?

El sistema corre los 6 pasos del pipeline y genera la siguiente estructura en `output/`:

```
output/
├── raw/
│   └── vod.mp4                    # VOD descargado
├── transcripts/
│   └── transcript.json            # Transcripción con timestamps
├── analysis/
│   └── peaks.json                 # Momentos de mayor energía detectados
├── candidates/
│   └── clips_candidates.json      # Candidatos de clips generados
├── clips/
│   └── clip_001.mp4               # Clips cortados
│   └── clip_002.mp4
│   └── ...
└── 2025-01-16/                    # Exportación final organizada
    ├── clip_001.mp4
    ├── clip_002.mp4
    └── metadata.json              # Resumen de la sesión
```

### Re-ejecutar sin re-descargar

Si ya tienes el VOD descargado, el sistema detecta que el archivo existe y omite la descarga. Lo mismo con la transcripción. Solo se re-procesan los pasos que no tienen output guardado.

Para forzar re-procesar un paso, borra el archivo de salida correspondiente.

---

## Configuración avanzada

Los parámetros principales están en los módulos individuales. Los más relevantes:

| Parámetro | Archivo | Por defecto | Descripción |
|-----------|---------|-------------|-------------|
| `MAX_CLIPS` | `clipper.py` | 25 | Máximo de clips a cortar |
| `MIN_DURATION` | `clip_candidate_generator.py` | 8s | Duración mínima de un clip |
| `MAX_DURATION` | `clip_candidate_generator.py` | 60s | Duración máxima de un clip |
| `MAX_PEAKS` | `audio_analysis.py` | 50 | Picos de audio a detectar |
| `MODEL` | `transcription.py` | `small` | Modelo Whisper (tiny/small/medium) |

---

## Módulos

| Módulo | Función |
|--------|---------|
| `ingestion.py` | Descarga el VOD con yt-dlp |
| `transcription.py` | Transcribe con faster-whisper (CUDA) |
| `audio_analysis.py` | Detecta picos de energía (RMS + derivada) |
| `clip_candidate_generator.py` | Genera candidatos con ventana dinámica |
| `clipper.py` | Corta clips con FFmpeg (stream copy) |
| `exporter.py` | Organiza la salida final |
| `subtitle_engine.py` | Quema subtítulos (opcional, no activo por defecto) |

---

## Estado del proyecto

| Paso | Estado |
|------|--------|
| Descarga VOD | Listo |
| Transcripción | Listo |
| Análisis de audio | Listo (v0.2) |
| Generación de candidatos | Listo (v0.2) |
| Corte de clips | Listo |
| Exportación | Listo |
| Scoring de clips | En desarrollo (v0.3) |
| Subtítulos | Módulo listo, no activo |
| Formato vertical 9:16 | Pendiente |

---

## Créditos

Desarrollado por **SH Studios**.
