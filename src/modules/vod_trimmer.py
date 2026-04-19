"""
vod_trimmer.py — Detecta inicio y final real del VOD, recorta para YouTube.

Flujo:
  1. Detectar inicio: primeros _SCAN_START_MIN min
       Prioridad: peaks validados → transcript sostenido → energía
  2. Detectar final:  últimos _SCAN_END_MIN min
       Prioridad: peaks validados → transcript → silencio relativo
  3. Safe start (#3): inicio < 30s → usar 0
  4. Buffer inteligente (#3-fin): +10s si hay texto después del último peak
  5. Consistencia final (#4): ajustar si difiere mucho del último texto
  6. Recortar con FFmpeg (-c copy, sin re-encode)
  7. Metadata con método y confidence score

Fallback: si no detecta algún límite, usa 0 o duración total.
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

OUTPUT_DIR   = "output/long"
SAMPLE_RATE  = 16_000
HOP_LENGTH   = 512
FRAME_LENGTH = 2048

# ── ventanas de escaneo ───────────────────────────────────────────────────────
_SCAN_START_MIN = 10
_SCAN_END_MIN   = 10

# ── detección de inicio ───────────────────────────────────────────────────────
_ACTIVITY_WIN_S      = 8.0
_MIN_ACTIVE_RATIO    = 0.55
_MIN_TEXT_DEN        = 0.5      # chars/s mínimo en ventana
_SAFE_START_THRESH_S = 30.0    # #3: inicio < 30s → 0

# ── validación de peaks (#1) ──────────────────────────────────────────────────
_PEAK_SUSTAINED_S    = 4.0     # actividad sostenida alrededor del peak
_TRANSCRIPT_SUST_S   = 30.0   # texto sostenido para fallback transcript (#2)

# ── detección de final ────────────────────────────────────────────────────────
_SILENCE_THRESH_S    = 20.0
_MAX_SILENCE_RATIO   = 0.15

# ── buffers (#3-fin) ──────────────────────────────────────────────────────────
_BUFFER_S            = 5.0
_BUFFER_S_EXTENDED   = 10.0   # si hay texto después del último peak

# ── consistencia final (#4) ───────────────────────────────────────────────────
_END_CONSISTENCY_S   = 30.0   # diferencia máxima antes de ajustar al transcript


# ─── audio helpers ────────────────────────────────────────────────────────────

def _video_duration(video_path: str) -> float:
    with av.open(video_path) as c:
        if c.duration is None:
            raise RuntimeError(f"No se pudo obtener duración de {video_path}")
        return c.duration / 1_000_000


def _load_audio_window(video_path: str, sr: int,
                        offset: float = 0.0, duration: float | None = None) -> np.ndarray:
    audio, _ = librosa.load(video_path, sr=sr, mono=True,
                             offset=offset, duration=duration)
    return audio.astype(np.float32)


def _energy_score(audio: np.ndarray) -> tuple:
    """RMS + derivada normalizados. Devuelve (score, times, threshold)."""
    rms  = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    diff = np.abs(np.diff(rms, prepend=rms[0]))
    rms_n  = rms  / (np.max(rms)  + 1e-9)
    diff_n = diff / (np.max(diff) + 1e-9)
    score  = rms_n + 0.5 * diff_n
    thresh = float(np.mean(score)) + float(np.std(score))
    times  = librosa.frames_to_time(
        np.arange(len(score)), sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    ).astype(np.float32)
    return score, times, thresh


# ─── transcript helpers ───────────────────────────────────────────────────────

def _text_density(transcript: list, t_start: float, t_end: float) -> float:
    dur   = max(t_end - t_start, 1e-3)
    chars = sum(
        len(s["text"])
        for s in transcript
        if s["start"] < t_end and s["end"] > t_start
    )
    return chars / dur


def _last_transcript_time(transcript: list, before: float) -> float | None:
    """Timestamp del último segmento que termina antes de `before` + margen."""
    candidates = [s["end"] for s in transcript
                  if s["end"] <= before + _END_CONSISTENCY_S and len(s["text"].strip()) > 3]
    return max(candidates) if candidates else None


# ─── validación de peaks (#1) ─────────────────────────────────────────────────

def _validate_peak(peak_t: float,
                   audio_offset: float,
                   score: np.ndarray,
                   times: np.ndarray,
                   thresh: float,
                   transcript: list | None) -> bool:
    """
    Peak válido si tiene actividad de energía sostenida ±_PEAK_SUSTAINED_S/2
    y text_density mínima en la misma ventana (si hay transcript).
    """
    active  = score >= thresh
    t_rel   = peak_t - audio_offset
    if t_rel < 0:
        return False

    half  = _PEAK_SUSTAINED_S / 2
    f_lo  = max(0,         int((t_rel - half) * SAMPLE_RATE / HOP_LENGTH))
    f_hi  = min(len(active), int((t_rel + half) * SAMPLE_RATE / HOP_LENGTH))
    if f_hi <= f_lo:
        return False

    if float(np.mean(active[f_lo:f_hi])) < _MIN_ACTIVE_RATIO:
        return False

    if transcript is not None:
        t_abs = peak_t
        if _text_density(transcript, t_abs - half, t_abs + half) < _MIN_TEXT_DEN:
            return False

    return True


# ─── candidatos por peaks ─────────────────────────────────────────────────────

def _start_from_peaks(peaks: list, scan_end: float,
                      audio_offset: float, score: np.ndarray,
                      times: np.ndarray, thresh: float,
                      transcript: list | None) -> float | None:
    """Primer peak cronológico válido dentro de la ventana de inicio."""
    candidates = sorted(
        (p for p in peaks if p["timestamp"] <= scan_end),
        key=lambda p: p["timestamp"],
    )
    for p in candidates:
        if _validate_peak(p["timestamp"], audio_offset, score, times, thresh, transcript):
            return p["timestamp"]
    return None


def _end_from_peaks(peaks: list, offset: float,
                    audio_offset: float, score: np.ndarray,
                    times: np.ndarray, thresh: float,
                    transcript: list | None) -> float | None:
    """Último peak cronológico válido dentro de la ventana de final."""
    candidates = sorted(
        (p for p in peaks if p["timestamp"] >= offset),
        key=lambda p: p["timestamp"],
        reverse=True,
    )
    for p in candidates:
        if _validate_peak(p["timestamp"], audio_offset, score, times, thresh, transcript):
            return p["timestamp"]
    return None


# ─── candidatos por transcript (#2) ──────────────────────────────────────────

def _start_from_transcript(transcript: list, scan_end: float) -> float | None:
    """
    Fallback inverso (#2): primer segmento con texto sostenido.
    Requiere text_density >= mínimo en los siguientes _TRANSCRIPT_SUST_S.
    """
    candidates = sorted(
        (s for s in transcript if s["start"] <= scan_end and len(s["text"].strip()) > 3),
        key=lambda s: s["start"],
    )
    for seg in candidates:
        t = seg["start"]
        if _text_density(transcript, t, t + _TRANSCRIPT_SUST_S) >= _MIN_TEXT_DEN:
            return t
    return None


def _end_from_transcript(transcript: list, offset: float) -> float | None:
    """Último segmento de texto como candidato de final."""
    candidates = [s["end"] for s in transcript
                  if s["end"] >= offset and len(s["text"].strip()) > 3]
    return max(candidates) if candidates else None


# ─── detección por energía ────────────────────────────────────────────────────

def _detect_start(audio: np.ndarray, transcript: list | None) -> float | None:
    score, times, thresh = _energy_score(audio)
    active     = score >= thresh
    win_frames = max(1, int(_ACTIVITY_WIN_S * SAMPLE_RATE / HOP_LENGTH))

    for i in range(len(active) - win_frames):
        if np.mean(active[i : i + win_frames]) < _MIN_ACTIVE_RATIO:
            continue
        t = float(times[i])
        if transcript is not None and _text_density(transcript, t, t + _ACTIVITY_WIN_S) < _MIN_TEXT_DEN:
            continue
        return t
    return None


def _detect_end(audio: np.ndarray,
                transcript: list | None = None,
                offset: float = 0.0) -> float | None:
    """Silencio relativo: energía baja sostenida + caída de text_density."""
    score, times, thresh = _energy_score(audio)
    active         = score >= thresh
    silence_frames = max(1, int(_SILENCE_THRESH_S * SAMPLE_RATE / HOP_LENGTH))

    for i in range(len(active) - silence_frames - 1, silence_frames, -1):
        if not active[i]:
            continue
        silence_after = active[i + 1 : i + 1 + silence_frames]
        if len(silence_after) < silence_frames:
            continue
        if np.mean(silence_after) >= _MAX_SILENCE_RATIO:
            continue
        if transcript is not None:
            t_abs = float(times[i]) + offset
            if _text_density(transcript, t_abs, t_abs + _SILENCE_THRESH_S) >= _MIN_TEXT_DEN:
                continue
        return float(times[i])
    return None


# ─── confidence score ─────────────────────────────────────────────────────────

def _compute_confidence(start_method: str, end_method: str,
                         real_start: float, real_end: float,
                         transcript: list | None) -> float:
    score = 0.0

    # Calidad del método de detección
    if start_method == "peaks":      score += 0.20
    elif start_method == "energy":   score += 0.10
    elif start_method == "transcript": score += 0.08

    if end_method == "peaks":        score += 0.20
    elif end_method == "energy":     score += 0.10
    elif end_method == "transcript": score += 0.08

    if transcript is not None:
        # Texto presente al inicio → inicio real
        den_start = _text_density(transcript, real_start, real_start + 30.0)
        if den_start >= _MIN_TEXT_DEN:     score += 0.20
        elif den_start >= _MIN_TEXT_DEN * 0.5: score += 0.10

        # Texto presente cerca del final → no cortamos la despedida
        den_end = _text_density(transcript, max(0.0, real_end - 30.0), real_end)
        if den_end >= _MIN_TEXT_DEN * 0.5: score += 0.20
        elif den_end > 0:                   score += 0.10
    else:
        score += 0.15  # sin transcript, crédito parcial

    # Sin fallbacks
    if start_method != "fallback": score += 0.10
    if end_method   != "fallback": score += 0.10

    return round(min(1.0, score), 3)


# ─── corte FFmpeg ─────────────────────────────────────────────────────────────

def _cut_video(ffmpeg: str, video_path: str,
               start: float, end: float, output_path: str) -> bool:
    result = subprocess.run(
        [ffmpeg, "-y",
         "-ss", str(round(start, 3)),
         "-to", str(round(end,   3)),
         "-i", video_path,
         "-c", "copy",
         "-avoid_negative_ts", "make_zero",
         output_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=7200,
    )
    if result.returncode != 0:
        logger.error("FFmpeg error: %s", result.stderr.decode(errors="replace")[-500:])
        return False
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 10_000:
        logger.error("Output file missing or empty: %s", output_path)
        return False
    return True


# ─── API pública ──────────────────────────────────────────────────────────────

def trim_vod(video_path: str,
             transcript_path: str | None = None,
             peaks_path: str | None = None,
             output_dir: str = None) -> str:
    """
    Detecta inicio y final real del VOD y genera video_trimmed.mp4.

    Input:
        video_path       — output/raw/vod.mp4
        transcript_path  — output/transcripts/transcript.json (opcional)
        peaks_path       — output/peaks.json de audio_analysis (opcional)

    Output:
        output/long/video_trimmed.mp4
        output/long/video_trimmed_meta.json  (incluye trim_confidence)

    Devuelve la ruta al video recortado.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    output_video = os.path.join(out, "video_trimmed.mp4")
    output_meta  = os.path.join(out, "video_trimmed_meta.json")

    transcript = None
    if transcript_path and os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
        logger.info("Transcript loaded: %d segments", len(transcript))

    peaks = None
    if peaks_path and os.path.exists(peaks_path):
        with open(peaks_path, encoding="utf-8") as f:
            peaks = json.load(f)
        logger.info("Peaks loaded: %d peaks", len(peaks))

    total_duration = _video_duration(video_path)
    logger.info("VOD duration: %.1fs (%.1f min)", total_duration, total_duration / 60)

    scan_start_s = min(_SCAN_START_MIN * 60.0, total_duration * 0.5)
    scan_end_s   = min(_SCAN_END_MIN   * 60.0, total_duration * 0.5)
    end_offset   = max(0.0, total_duration - scan_end_s)

    # ── Inicio ────────────────────────────────────────────────────────────
    logger.info("Scanning start window: 0s – %.0fs ...", scan_start_s)
    audio_start                = _load_audio_window(video_path, SAMPLE_RATE,
                                                     offset=0.0, duration=scan_start_s)
    sc_s, ti_s, th_s           = _energy_score(audio_start)
    detected_start_rel         = None
    start_method               = "fallback"

    # 1) peaks validados (#1)
    if peaks:
        t = _start_from_peaks(peaks, scan_start_s, 0.0, sc_s, ti_s, th_s, transcript)
        if t is not None:
            detected_start_rel = t
            start_method       = "peaks"
            logger.info("Start from peaks (validated): %.1fs", t)

    # 2) transcript sostenido si no hubo peak válido (#2)
    if detected_start_rel is None and transcript:
        t = _start_from_transcript(transcript, scan_start_s)
        if t is not None:
            detected_start_rel = t
            start_method       = "transcript"
            logger.info("Start from transcript fallback: %.1fs", t)

    # 3) energía + texto
    if detected_start_rel is None:
        t = _detect_start(audio_start, transcript)
        if t is not None:
            detected_start_rel = t
            start_method       = "energy"
            logger.info("Start from energy: %.1fs", t)

    # Safe start (#3)
    if detected_start_rel is not None:
        if detected_start_rel < _SAFE_START_THRESH_S:
            real_start = 0.0
            logger.info("Safe start: %.1fs < %.0fs → 0s",
                        detected_start_rel, _SAFE_START_THRESH_S)
        else:
            real_start = round(max(0.0, detected_start_rel - _BUFFER_S), 3)
            logger.info("Start [%s]: detected=%.1fs → with buffer=%.1fs",
                        start_method, detected_start_rel, real_start)
    else:
        real_start = 0.0
        logger.warning("Start not detected — fallback: 0s")

    # ── Final ─────────────────────────────────────────────────────────────
    logger.info("Scanning end window: %.0fs – %.0fs ...", end_offset, total_duration)
    audio_end                  = _load_audio_window(video_path, SAMPLE_RATE,
                                                     offset=end_offset, duration=scan_end_s)
    sc_e, ti_e, th_e           = _energy_score(audio_end)
    detected_end_abs           = None
    end_method                 = "fallback"

    # 1) peaks validados (#1)
    if peaks:
        t = _end_from_peaks(peaks, end_offset, end_offset, sc_e, ti_e, th_e, transcript)
        if t is not None:
            detected_end_abs = t
            end_method       = "peaks"
            logger.info("End from peaks (validated): %.1fs", t)

    # 2) transcript si no hubo peak válido (#2)
    if detected_end_abs is None and transcript:
        t = _end_from_transcript(transcript, end_offset)
        if t is not None:
            detected_end_abs = t
            end_method       = "transcript"
            logger.info("End from transcript fallback: %.1fs", t)

    # 3) silencio relativo
    if detected_end_abs is None:
        rel = _detect_end(audio_end, transcript=transcript, offset=end_offset)
        if rel is not None:
            detected_end_abs = end_offset + rel
            end_method       = "energy"
            logger.info("End from energy: %.1fs (rel=%.1fs)", detected_end_abs, rel)

    # Buffer inteligente (#3-fin): +10s si hay texto después del punto detectado
    if detected_end_abs is not None:
        end_buffer = _BUFFER_S
        if transcript is not None:
            text_after = _text_density(transcript, detected_end_abs,
                                        detected_end_abs + 15.0)
            if text_after >= _MIN_TEXT_DEN * 0.5:
                end_buffer = _BUFFER_S_EXTENDED
                logger.info("Extended end buffer %.0fs: text found after detected end", end_buffer)

        real_end = round(min(total_duration, detected_end_abs + end_buffer), 3)
        logger.info("End [%s]: detected=%.1fs → with buffer=%.1fs",
                    end_method, detected_end_abs, real_end)

        # Consistencia con transcript (#4): ajustar si difiere > 30s del último texto
        if transcript is not None:
            last_t = _last_transcript_time(transcript, real_end)
            if last_t is not None and abs(real_end - last_t) > _END_CONSISTENCY_S:
                adjusted = round(min(total_duration, last_t + end_buffer), 3)
                logger.info(
                    "End consistency #4: real_end=%.1fs vs last_text=%.1fs → %.1fs",
                    real_end, last_t, adjusted,
                )
                real_end = adjusted
    else:
        real_end = total_duration
        logger.warning("End not detected — fallback: full duration (%.1fs)", total_duration)

    # ── Sanity check ──────────────────────────────────────────────────────
    if real_end <= real_start + 60.0:
        logger.error(
            "Invalid bounds (start=%.1f, end=%.1f) — resetting to full duration",
            real_start, real_end,
        )
        real_start = 0.0
        real_end   = total_duration
        detected_start_rel = None
        detected_end_abs   = None
        start_method = end_method = "fallback"

    trimmed_duration = round(real_end - real_start, 3)
    logger.info("Trimming: %.1fs → %.1fs | kept %.1fs (%.1f min)",
                real_start, real_end, trimmed_duration, trimmed_duration / 60)

    # ── Corte FFmpeg ──────────────────────────────────────────────────────
    ffmpeg = _find_ffmpeg()
    if not _cut_video(ffmpeg, video_path, real_start, real_end, output_video):
        raise RuntimeError(f"FFmpeg failed to trim {video_path}")

    size_mb = os.path.getsize(output_video) / (1024 * 1024)
    logger.info("Trimmed video: %s (%.1f MB)", output_video, size_mb)

    # ── Metadata ──────────────────────────────────────────────────────────
    confidence = _compute_confidence(
        start_method, end_method, real_start, real_end, transcript
    )

    meta = {
        "source":           video_path,
        "total_duration_s": round(total_duration, 3),
        "real_start":       real_start,
        "real_end":         real_end,
        "duration_trimmed": trimmed_duration,
        "buffer_s":         _BUFFER_S,
        "start_method":     start_method,
        "end_method":       end_method,
        "fallback_start":   start_method == "fallback",
        "fallback_end":     end_method   == "fallback",
        "trim_confidence":  confidence,
        "output":           output_video,
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Metadata: %s | confidence: %.2f", output_meta, confidence)
    return output_video
