import json
import logging
import os

import numpy as np

from .subtitle_engine import _build_srt

logger = logging.getLogger(__name__)

# ── audio ──────────────────────────────────────────────────────────────────────
_TARGET_SR      = 16_000   # Hz — mismo estándar que Whisper
_RMS_WIN_S      = 0.025    # ventana de análisis RMS (25ms)
_RMS_HOP_S      = 0.010    # paso entre frames (10ms)

# ── búsqueda de voz ───────────────────────────────────────────────────────────
_SEARCH_PAD_S   = 0.5      # padding alrededor del segmento para buscar voz
_THRESH_K       = 0.5      # threshold = mean + k * std (por ventana)
_THRESH_FLOOR   = 0.005    # mínimo absoluto — evita triggers en silencio puro

# ── límites de seguridad ──────────────────────────────────────────────────────
_MAX_TRIM_S     = 1.0      # desplazamiento máximo en start o end
_MIN_DURATION_S = 0.6      # duración mínima garantizada tras ajuste
_MAX_DURATION_S = 4.0      # duración máxima — dividir si supera
_GAP_MIN_S      = 0.15     # gap mínimo entre segmentos consecutivos

# ── suavizado ─────────────────────────────────────────────────────────────────
_LERP_T         = 0.7      # 0 = sin cambio, 1 = adoptar detección pura


# ─── audio helpers ────────────────────────────────────────────────────────────

def _load_audio(path: str) -> tuple:
    """Carga audio mono float32 @ _TARGET_SR desde archivo de video o audio."""
    try:
        import librosa
        audio, sr = librosa.load(path, sr=_TARGET_SR, mono=True)
        return audio, sr
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar audio: {path} — {exc}") from exc


def _compute_rms(audio: np.ndarray, sr: int) -> tuple:
    """
    RMS frame a frame con ventana deslizante.
    Devuelve (rms_array, frame_times) donde frame_times[i] = tiempo del centro del frame i.
    """
    win  = max(1, int(_RMS_WIN_S * sr))
    hop  = max(1, int(_RMS_HOP_S * sr))
    n    = len(audio)

    frames, times = [], []
    for start in range(0, n - win + 1, hop):
        chunk = audio[start : start + win]
        frames.append(float(np.sqrt(np.mean(chunk ** 2))))
        times.append((start + win / 2) / sr)

    return np.array(frames, dtype=np.float32), np.array(times, dtype=np.float32)


def _lerp(a: float, b: float, t: float) -> float:
    return round(a + (b - a) * t, 3)


# ─── detección por segmento ───────────────────────────────────────────────────

def _detect_voice_bounds(
    rms: np.ndarray, frame_times: np.ndarray,
    seg_start: float, seg_end: float, clip_dur: float,
) -> tuple:
    """
    Busca inicio y fin de voz en la ventana [seg_start-pad, seg_end+pad].

    E — threshold dinámico: mean + k * std calculado sobre esa ventana.
    F — nunca desplaza más de _MAX_TRIM_S desde el timestamp original.
    K — si no hay voz en la ventana, devuelve found=False (fallback).

    Devuelve (detected_start, detected_end, found).
    """
    w_start = max(0.0, seg_start - _SEARCH_PAD_S)
    w_end   = min(clip_dur, seg_end + _SEARCH_PAD_S)

    mask = (frame_times >= w_start) & (frame_times <= w_end)
    if not np.any(mask):
        return seg_start, seg_end, False

    w_rms   = rms[mask]
    w_times = frame_times[mask]

    # E: umbral adaptativo por ventana
    thresh = max(_THRESH_FLOOR,
                 float(np.mean(w_rms)) + _THRESH_K * float(np.std(w_rms)))

    above = w_times[w_rms >= thresh]
    if len(above) == 0:
        return seg_start, seg_end, False  # K: fallback

    det_start = float(above[0])
    det_end   = float(above[-1]) + _RMS_WIN_S   # añadir anchura del último frame

    # F: límites de seguridad
    det_start = max(det_start, seg_start - _MAX_TRIM_S)
    det_end   = min(det_end,   seg_end   + _MAX_TRIM_S)

    # Clamp al clip
    det_start = max(0.0,      round(det_start, 3))
    det_end   = min(clip_dur, round(det_end,   3))

    return det_start, det_end, True


# ─── alineación por segmento ──────────────────────────────────────────────────

def _align_segment(
    seg: dict,
    rms: np.ndarray, frame_times: np.ndarray,
    clip_dur: float,
) -> dict:
    """
    Alinea un segmento. Devuelve copia con timestamps ajustados y clave _debug.
    No modifica el texto ni el highlight.
    """
    old_start = seg["start"]
    old_end   = seg["end"]

    det_start, det_end, found = _detect_voice_bounds(
        rms, frame_times, old_start, old_end, clip_dur
    )

    if not found:
        return {
            **seg,
            "_debug": {
                "original_start": old_start, "adjusted_start": old_start,
                "original_end":   old_end,   "adjusted_end":   old_end,
                "delta_start": 0.0, "delta_end": 0.0, "fallback": True,
            },
        }

    # J: suavizado — no adoptar la detección de golpe
    new_start = _lerp(old_start, det_start, _LERP_T)
    new_end   = _lerp(old_end,   det_end,   _LERP_T)

    # G: duración mínima garantizada
    dur = new_end - new_start
    if dur < _MIN_DURATION_S:
        pad       = (_MIN_DURATION_S - dur) / 2
        new_start = round(max(0.0,      new_start - pad), 3)
        new_end   = round(min(clip_dur, new_end   + pad), 3)

    new_start = round(new_start, 3)
    new_end   = round(new_end,   3)

    return {
        **seg,
        "start": new_start,
        "end":   new_end,
        "_debug": {
            "original_start": old_start, "adjusted_start": new_start,
            "original_end":   old_end,   "adjusted_end":   new_end,
            "delta_start": round(new_start - old_start, 3),
            "delta_end":   round(new_end   - old_end,   3),
            "fallback": False,
        },
    }


# ─── post-proceso ─────────────────────────────────────────────────────────────

def _enforce_max_duration(segments: list, clip_dur: float) -> list:
    """I: Divide por mitad cualquier segmento > _MAX_DURATION_S."""
    result = []
    for seg in segments:
        if seg["end"] - seg["start"] <= _MAX_DURATION_S:
            result.append(seg)
        else:
            mid   = round((seg["start"] + seg["end"]) / 2, 3)
            words = seg.get("text", "").split()
            half  = max(1, len(words) // 2)
            a = {k: v for k, v in seg.items() if k != "_debug"}
            b = {k: v for k, v in seg.items() if k != "_debug"}
            a.update({"end": mid,        "text": " ".join(words[:half])})
            b.update({"start": mid,      "text": " ".join(words[half:])})
            result.extend([a, b])
    return result


def _gap_cleaner(segments: list) -> list:
    """H: Corrige gaps < _GAP_MIN_S y solapamientos entre segmentos consecutivos."""
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap < _GAP_MIN_S:
            mid = round((segments[i]["end"] + segments[i + 1]["start"]) / 2, 3)
            segments[i]["end"]          = mid
            segments[i + 1]["start"]    = mid
    return segments


# ─── API pública ──────────────────────────────────────────────────────────────

def align_subtitles(audio_path: str, subtitles_json_path: str) -> str:
    """
    Alinea timestamps de un archivo de subtítulos con la voz real del audio.

    Lee el JSON, detecta los límites de voz por segmento con threshold dinámico,
    aplica suavizado (lerp), corrige gaps y sobreescribe el JSON y el SRT.

    No modifica el texto. No afecta el highlight. Respeta clips editados manualmente
    (si _meta.json tiene subtitles_edited=true, la función no toca el archivo).

    Devuelve la ruta al JSON actualizado.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")
    if not os.path.exists(subtitles_json_path):
        raise FileNotFoundError(f"Subtítulos no encontrados: {subtitles_json_path}")

    # Respetar ediciones manuales
    meta_path = subtitles_json_path.replace(".json", "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("subtitles_edited", False):
            logger.info(f"Skipping timing alignment — edited manually: {subtitles_json_path}")
            return subtitles_json_path
    else:
        meta = {}

    with open(subtitles_json_path, encoding="utf-8") as f:
        segments = json.load(f)

    if not segments:
        logger.warning(f"JSON vacío, nada que alinear: {subtitles_json_path}")
        return subtitles_json_path

    audio, sr  = _load_audio(audio_path)
    clip_dur   = len(audio) / sr
    rms, times = _compute_rms(audio, sr)

    # Alinear cada segmento
    aligned = [_align_segment(seg, rms, times, clip_dur) for seg in segments]

    # Post-proceso
    aligned = _enforce_max_duration(aligned, clip_dur)
    aligned = _gap_cleaner(aligned)

    # Separar debug
    debug_records = [seg.pop("_debug", {}) for seg in aligned]

    # Sobreescribir JSON
    with open(subtitles_json_path, "w", encoding="utf-8") as f:
        json.dump(aligned, f, ensure_ascii=False, indent=2)

    # Regenerar SRT
    srt_path = subtitles_json_path.replace(".json", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_build_srt(aligned))

    # L: escribir debug en meta
    fallback_count  = sum(1 for d in debug_records if d.get("fallback", False))
    n               = max(len(debug_records), 1)
    avg_delta_start = round(sum(abs(d.get("delta_start", 0)) for d in debug_records) / n, 3)
    avg_delta_end   = round(sum(abs(d.get("delta_end",   0)) for d in debug_records) / n, 3)

    meta["timing_aligned"]    = True
    meta["align_fallbacks"]   = fallback_count
    meta["avg_delta_start_s"] = avg_delta_start
    meta["avg_delta_end_s"]   = avg_delta_end
    meta["align_debug"]       = debug_records

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adjusted = len(debug_records) - fallback_count
    logger.info(
        "Timing aligned: %d/%d adjusted | avg Δstart %.3fs Δend %.3fs | "
        "%d fallbacks → %s",
        adjusted, len(debug_records),
        avg_delta_start, avg_delta_end,
        fallback_count, subtitles_json_path,
    )

    return subtitles_json_path


def align_all_subtitles(
    clips_dir: str     = "output/clips",
    subtitles_dir: str = "output/subtitles",
) -> str:
    """
    Alinea todos los clips disponibles en clips_dir con sus subtítulos en subtitles_dir.
    Omite clips sin subtítulos o sin archivo de video.
    Devuelve subtitles_dir.
    """
    import glob as _glob

    json_files = sorted(_glob.glob(os.path.join(subtitles_dir, "clip_*.json")))
    # Excluir _meta.json
    json_files = [p for p in json_files if not p.endswith("_meta.json")]

    if not json_files:
        logger.warning(f"No subtitle JSONs found in {subtitles_dir}")
        return subtitles_dir

    done, skipped = 0, 0

    for json_path in json_files:
        base     = os.path.splitext(os.path.basename(json_path))[0]  # clip_NNN
        clip_mp4 = os.path.join(clips_dir, f"{base}.mp4")

        if not os.path.exists(clip_mp4):
            logger.warning(f"{base}: clip no encontrado en {clips_dir} — omitido")
            skipped += 1
            continue

        try:
            align_subtitles(clip_mp4, json_path)
            done += 1
        except Exception as exc:
            logger.error(f"{base}: alignment failed — {exc}")
            skipped += 1

    logger.info(
        "align_all_subtitles: %d aligned | %d skipped → %s",
        done, skipped, subtitles_dir,
    )
    return subtitles_dir
