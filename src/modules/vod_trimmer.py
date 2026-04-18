"""
vod_trimmer.py — Detecta inicio y final real del VOD, recorta para YouTube.

Flujo:
  1. Detectar inicio: primeros _SCAN_START_MIN min — actividad sostenida + transcript
  2. Detectar final:  últimos  _SCAN_END_MIN   min — último momento activo antes
                      de silencio prolongado (>= _SILENCE_THRESH_S)
  3. Aplicar buffer de seguridad (_BUFFER_S) en ambos extremos
  4. Recortar con FFmpeg (-c copy, sin re-encode)
  5. Guardar metadata

Fallback: si no detecta algún límite, usa 0 o duración total — nunca rompe el video.
"""
import json
import logging
import os
import subprocess

import av
import librosa
import numpy as np

from .clipper import _find_ffmpeg

logger = logging.getLogger(__name__)

OUTPUT_DIR        = "output/long"
SAMPLE_RATE       = 16_000
HOP_LENGTH        = 512
FRAME_LENGTH      = 2048

# ── ventanas de escaneo ───────────────────────────────────────────────────────
_SCAN_START_MIN   = 10       # analizar primeros N minutos para el inicio
_SCAN_END_MIN     = 10       # analizar últimos N minutos para el final

# ── detección de inicio ───────────────────────────────────────────────────────
_ACTIVITY_WIN_S   = 8.0      # ventana de actividad sostenida requerida
_MIN_ACTIVE_RATIO = 0.55     # fracción mínima de frames activos en la ventana
_MIN_TEXT_DEN     = 0.5      # chars/s mínimo en el transcript (si disponible)

# ── detección de final ────────────────────────────────────────────────────────
_SILENCE_THRESH_S = 20.0     # silencio sostenido de esta duración = stream terminó
_MAX_SILENCE_RATIO = 0.15    # fracción máxima de frames activos para "silencio"

# ── buffer de seguridad ───────────────────────────────────────────────────────
_BUFFER_S         = 5.0      # segundos antes/después de los puntos detectados


# ─── audio helpers ────────────────────────────────────────────────────────────

def _video_duration(video_path: str) -> float:
    """Duración total del video en segundos (via PyAV, sin decodificar)."""
    with av.open(video_path) as c:
        if c.duration is None:
            raise RuntimeError(f"No se pudo obtener duración de {video_path}")
        return c.duration / 1_000_000  # AV_TIME_BASE = microsegundos


def _load_audio_window(video_path: str, sr: int,
                        offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    """
    Carga ventana de audio mono float32 usando librosa.
    Eficiente para ventanas pequeñas de VODs grandes.
    """
    audio, _ = librosa.load(video_path, sr=sr, mono=True,
                             offset=offset, duration=duration)
    return audio.astype(np.float32)


def _energy_score(audio: np.ndarray) -> tuple:
    """
    Score por frame: RMS normalizado + 0.5 * derivada normalizada.
    Mismo scoring que audio_analysis — ignora ruido constante (música, ambiente).
    Devuelve (score, frame_times, threshold).
    """
    rms  = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    diff = np.abs(np.diff(rms, prepend=rms[0]))

    rms_n  = rms  / (np.max(rms)  + 1e-9)
    diff_n = diff / (np.max(diff) + 1e-9)
    score  = rms_n + 0.5 * diff_n

    thresh = float(np.mean(score)) + float(np.std(score))
    times  = librosa.frames_to_time(
        np.arange(len(score)), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )
    return score, times.astype(np.float32), thresh


# ─── transcript helpers ───────────────────────────────────────────────────────

def _text_density(transcript: list, t_start: float, t_end: float) -> float:
    """Chars/segundo en la ventana [t_start, t_end]."""
    dur   = max(t_end - t_start, 1e-3)
    chars = sum(
        len(s["text"])
        for s in transcript
        if s["start"] < t_end and s["end"] > t_start
    )
    return chars / dur


# ─── detección de límites ─────────────────────────────────────────────────────

def _detect_start(audio: np.ndarray, transcript: list | None) -> float | None:
    """
    Primer momento con actividad sostenida:
    - al menos _MIN_ACTIVE_RATIO de frames activos en _ACTIVITY_WIN_S
    - si hay transcript, la densidad de texto supera _MIN_TEXT_DEN

    Devuelve timestamp relativo al inicio de la ventana, o None (fallback).
    """
    score, times, thresh = _energy_score(audio)
    active    = score >= thresh
    win_frames = max(1, int(_ACTIVITY_WIN_S * SAMPLE_RATE / HOP_LENGTH))

    for i in range(len(active) - win_frames):
        if np.mean(active[i : i + win_frames]) < _MIN_ACTIVE_RATIO:
            continue
        t = float(times[i])
        if transcript is not None and _text_density(transcript, t, t + _ACTIVITY_WIN_S) < _MIN_TEXT_DEN:
            continue
        return t

    return None


def _detect_end(audio: np.ndarray) -> float | None:
    """
    Último frame activo antes de un silencio prolongado (>= _SILENCE_THRESH_S).
    Escanea hacia atrás para encontrar el último momento de contenido real.

    Devuelve timestamp relativo al inicio de la ventana, o None (fallback).
    """
    score, times, thresh = _energy_score(audio)
    active         = score >= thresh
    silence_frames = max(1, int(_SILENCE_THRESH_S * SAMPLE_RATE / HOP_LENGTH))

    # Scan hacia atrás: buscar el último frame activo seguido de silencio largo
    for i in range(len(active) - silence_frames - 1, silence_frames, -1):
        if not active[i]:
            continue
        silence_after = active[i + 1 : i + 1 + silence_frames]
        if len(silence_after) == silence_frames and np.mean(silence_after) < _MAX_SILENCE_RATIO:
            return float(times[i])

    return None


# ─── corte FFmpeg ─────────────────────────────────────────────────────────────

def _cut_video(ffmpeg: str, video_path: str,
               start: float, end: float, output_path: str) -> bool:
    """
    Recorta el video entre start y end usando stream copy (sin re-encode).
    Timeout generoso (2h) para VODs largos.
    """
    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-ss", str(round(start, 3)),
            "-to", str(round(end,   3)),
            "-i", video_path,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=7200,
    )

    if result.returncode != 0:
        logger.error("FFmpeg error: %s",
                     result.stderr.decode(errors="replace")[-500:])
        return False

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 10_000:
        logger.error("Output file missing or empty: %s", output_path)
        return False

    return True


# ─── API pública ──────────────────────────────────────────────────────────────

def trim_vod(video_path: str, transcript_path: str | None = None) -> str:
    """
    Detecta inicio y final real del VOD y genera video_trimmed.mp4.

    Input:
        video_path        — output/raw/vod.mp4
        transcript_path   — output/transcripts/transcript.json (opcional)
                            Si se pasa, refuerza la detección de inicio
                            con densidad de texto.

    Output:
        output/long/video_trimmed.mp4
        output/long/video_trimmed_meta.json

    Devuelve la ruta al video recortado.
    Fallback: si no detecta algún límite, usa 0 o duración total.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_video = os.path.join(OUTPUT_DIR, "video_trimmed.mp4")
    output_meta  = os.path.join(OUTPUT_DIR, "video_trimmed_meta.json")

    # Cargar transcript si existe
    transcript = None
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
        logger.info("Transcript loaded: %d segments", len(transcript))

    # Duración total sin decodificar el video
    total_duration = _video_duration(video_path)
    logger.info("VOD duration: %.1fs (%.1f min)", total_duration, total_duration / 60)

    # Limitar ventanas a la mitad del video para streams muy cortos
    scan_start_s = min(_SCAN_START_MIN * 60.0, total_duration * 0.5)
    scan_end_s   = min(_SCAN_END_MIN   * 60.0, total_duration * 0.5)
    end_offset   = max(0.0, total_duration - scan_end_s)

    # ── Detección de inicio ───────────────────────────────────────────────
    logger.info("Scanning start window: 0s – %.0fs ...", scan_start_s)
    audio_start        = _load_audio_window(video_path, SAMPLE_RATE,
                                             offset=0.0, duration=scan_start_s)
    detected_start_rel = _detect_start(audio_start, transcript)

    if detected_start_rel is not None:
        real_start = round(max(0.0, detected_start_rel - _BUFFER_S), 3)
        logger.info("Start detected at %.1fs — with buffer: %.1fs",
                    detected_start_rel, real_start)
    else:
        real_start = 0.0
        logger.warning("Start not detected — fallback: 0s")

    # ── Detección de final ────────────────────────────────────────────────
    logger.info("Scanning end window: %.0fs – %.0fs ...", end_offset, total_duration)
    audio_end        = _load_audio_window(video_path, SAMPLE_RATE,
                                           offset=end_offset, duration=scan_end_s)
    detected_end_rel = _detect_end(audio_end)

    if detected_end_rel is not None:
        abs_detected_end = end_offset + detected_end_rel
        real_end = round(min(total_duration, abs_detected_end + _BUFFER_S), 3)
        logger.info("End detected at %.1fs (rel) / %.1fs (abs) — with buffer: %.1fs",
                    detected_end_rel, abs_detected_end, real_end)
    else:
        real_end = total_duration
        logger.warning("End not detected — fallback: full duration (%.1fs)", total_duration)

    # ── Sanity check ─────────────────────────────────────────────────────
    if real_end <= real_start + 60.0:
        logger.error(
            "Invalid bounds (start=%.1f, end=%.1f) — resetting to full duration",
            real_start, real_end
        )
        real_start, real_end = 0.0, total_duration
        detected_start_rel = detected_end_rel = None

    trimmed_duration = round(real_end - real_start, 3)
    logger.info(
        "Trimming: %.1fs → %.1fs | kept: %.1fs (%.1f min)",
        real_start, real_end, trimmed_duration, trimmed_duration / 60
    )

    # ── Corte FFmpeg ──────────────────────────────────────────────────────
    ffmpeg = _find_ffmpeg()
    ok = _cut_video(ffmpeg, video_path, real_start, real_end, output_video)
    if not ok:
        raise RuntimeError(f"FFmpeg failed to trim {video_path}")

    size_mb = os.path.getsize(output_video) / (1024 * 1024)
    logger.info("Trimmed video: %s (%.1f MB)", output_video, size_mb)

    # ── Metadata ──────────────────────────────────────────────────────────
    meta = {
        "source":             video_path,
        "total_duration_s":   round(total_duration, 3),
        "start_detected_s":   round(detected_start_rel, 3) if detected_start_rel is not None else None,
        "end_detected_s":     round(detected_end_rel,   3) if detected_end_rel   is not None else None,
        "real_start":         real_start,
        "real_end":           real_end,
        "duration_trimmed":   trimmed_duration,
        "buffer_s":           _BUFFER_S,
        "fallback_start":     detected_start_rel is None,
        "fallback_end":       detected_end_rel   is None,
        "output":             output_video,
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Metadata: %s", output_meta)
    return output_video
