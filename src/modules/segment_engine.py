"""
segment_engine.py — Segmentación híbrida: Active Window + Merge + Semantic

Reemplaza clip_candidate_generator en el pipeline (el módulo original se mantiene
como fallback con --legacy).

Flujo:
  1. process_active_window()  — agrupa picos consecutivos en ventanas de contenido continuas
  2. SemanticAnalyzer         — anota similitud semántica entre ventanas (Phase 2, batch)
  3. merge_clips()            — une clips con gap pequeño; respeta cortes semánticos fuertes
  4. _finalize_metrics()      — intensity, energy_score, semantic_score, confidence_score
"""
import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR     = "output/candidates"
MIN_DURATION   = 8.0
MAX_DURATION   = 120.0   # más permisivo que clip_candidate_generator (60s)
MAX_CANDIDATES = 25

_STOPWORDS_ES = {
    'de', 'la', 'el', 'en', 'y', 'que', 'a', 'los', 'del', 'se', 'un', 'una',
    'con', 'es', 'no', 'lo', 'por', 'su', 'le', 'da', 'las', 'al', 'como',
    'pero', 'más', 'si', 'ya', 'hay', 'me', 'este', 'mi', 'para', 'o', 'te',
    'yo', 'muy', 'eso', 'esto', 'era', 'fue', 'han', 'ha', 'he', 'ser',
    'estar', 'bien', 'cuando', 'qué', 'porque', 'ahí', 'así', 'ese', 'tan',
    'todo', 'nada', 'nos', 'les', 'donde', 'vez', 'hacer', 'pues', 'ahora',
    'bueno', 'vamos', 'tipo', 'igual', 'entonces', 'también', 'solo',
}


@dataclass
class SegmentConfig:
    silence_duration_threshold: float = 3.0   # gap máx entre picos en misma ventana
    merge_gap_threshold:        float = 4.0   # gap máx para post-process merge
    max_clip_duration:          float = 120.0  # hard cap por clip
    min_clip_duration:          float = 8.0   # mínimo viable
    max_candidates:             int   = 25
    semantic_threshold:         float = 0.25  # similitud < threshold = cambio de tema
    min_text_activity:          float = 0.3   # chars/s para considerar gap "activo"


# ─── helpers de transcript ────────────────────────────────────────────────────

def _text_density(transcript: list, t_start: float, t_end: float) -> float:
    dur   = max(t_end - t_start, 1e-3)
    chars = sum(len(s["text"]) for s in transcript
                if s["start"] < t_end and s["end"] > t_start)
    return chars / dur


def _has_text_activity(transcript: list, t_start: float, t_end: float,
                        min_cps: float) -> bool:
    if t_end <= t_start or not transcript:
        return False
    return _text_density(transcript, t_start, t_end) >= min_cps


def _nearest_text(peak_ts: float, transcript: list) -> str | None:
    if not transcript:
        return None
    seg = min(transcript, key=lambda s: abs((s["start"] + s["end"]) / 2 - peak_ts))
    return seg["text"]


def _extend_to_segment_end(end: float, peak_ts: float,
                             transcript: list, max_end: float) -> float:
    if not transcript:
        return end
    overlapping = [s for s in transcript if s["start"] <= end and s["end"] > peak_ts]
    if not overlapping:
        return end
    latest = max(s["end"] for s in overlapping)
    return round(min(latest + 0.3, max_end), 3)


def _get_window(intensity: float) -> tuple[float, float]:
    """Ventana dinámica: misma lógica que clip_candidate_generator."""
    if intensity > 0.8:
        return 15.0, 20.0
    elif intensity > 0.6:
        return 12.0, 18.0
    return 10.0, 15.0


# ─── Phase 1: Active Window ───────────────────────────────────────────────────

def process_active_window(peaks_by_time: list, transcript: list,
                           config: SegmentConfig,
                           content_end: float = float("inf")) -> list:
    """
    Agrupa picos consecutivos en ventanas de contenido activo.

    Un pico se agrega a la ventana actual si:
      - El gap hasta la ventana activa ≤ silence_duration_threshold, O
      - Hay texto en el gap (silencio de audio, no de contenido).

    Si agregar el pico excedería max_clip_duration → cierra ventana y empieza nueva.
    """
    if not peaks_by_time:
        return []

    p0       = peaks_by_time[0]
    pre, post = _get_window(p0["intensity"])
    current   = {
        "start":       round(max(0.0, p0["timestamp"] - pre), 3),
        "end":         round(min(content_end, p0["timestamp"] + post), 3),
        "group_peaks": [p0],
    }
    windows = []

    for peak in peaks_by_time[1:]:
        pre, post     = _get_window(peak["intensity"])
        candidate_end = round(min(content_end, peak["timestamp"] + post), 3)
        gap           = peak["timestamp"] - current["end"]

        gap_active = (
            gap <= config.silence_duration_threshold
            or _has_text_activity(transcript, current["end"], peak["timestamp"],
                                  config.min_text_activity)
        )
        would_exceed = (candidate_end - current["start"]) > config.max_clip_duration

        if gap_active and not would_exceed:
            current["end"] = max(current["end"], candidate_end)
            current["group_peaks"].append(peak)
        else:
            windows.append(current)
            current = {
                "start":       round(max(0.0, peak["timestamp"] - pre), 3),
                "end":         candidate_end,
                "group_peaks": [peak],
            }

    windows.append(current)
    logger.info("Active windows: %d (from %d peaks)", len(windows), len(peaks_by_time))
    return windows


# ─── Phase 1.5: Post-process Merge ───────────────────────────────────────────

def merge_clips(clips: list, config: SegmentConfig,
                semantic_break_indices: list | None = None) -> list:
    """
    Une clips consecutivos cuyo gap sea ≤ merge_gap_threshold.
    Respeta índices marcados como corte semántico fuerte.
    """
    if not clips:
        return []

    break_set = set(semantic_break_indices or [])
    merged    = [clips[0].copy()]

    for i, clip in enumerate(clips[1:], 1):
        last             = merged[-1]
        gap              = clip["start"] - last["end"]
        merged_duration  = clip["end"] - last["start"]

        if (i not in break_set
                and gap <= config.merge_gap_threshold
                and merged_duration <= config.max_clip_duration):
            last["end"] = max(last["end"], clip["end"])
            last["group_peaks"] = last.get("group_peaks", []) + clip.get("group_peaks", [])
        else:
            merged.append(clip.copy())

    logger.info("Post-merge: %d → %d clips", len(clips), len(merged))
    return merged


# ─── Phase 2: Semantic Continuity ────────────────────────────────────────────

class SemanticAnalyzer:
    """
    Evalúa continuidad semántica entre clips usando similitud coseno
    sobre bag-of-words. Sin dependencias externas (solo numpy).

    Diseñado para batch/post-proceso — no bloquea la detección de picos.
    Fallback si transcript es None: semantic_score = 0.5 (neutro) en todos.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def _tokenize(self, text: str) -> list:
        words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
        return [w for w in words if w not in _STOPWORDS_ES and len(w) > 2]

    def _clip_tokens(self, transcript: list, t_start: float, t_end: float) -> list:
        text = " ".join(
            s["text"] for s in transcript
            if s["start"] < t_end and s["end"] > t_start
        )
        return self._tokenize(text)

    def _cosine_sim(self, tok_a: list, tok_b: list) -> float:
        if not tok_a or not tok_b:
            return 0.0
        vocab  = list(set(tok_a) | set(tok_b))
        vec_a  = np.array([tok_a.count(w) for w in vocab], dtype=float)
        vec_b  = np.array([tok_b.count(w) for w in vocab], dtype=float)
        denom  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0

    def analyze_continuity(self, transcript: list, clips: list) -> tuple[list, list]:
        """
        Anota semantic_score y topic_continuity en cada clip.

        Returns:
            clips  — lista modificada con scores añadidos
            breaks — índices donde hay corte semántico fuerte
        """
        if not transcript:
            for clip in clips:
                clip["semantic_score"]   = 0.5
                clip["topic_continuity"] = True
            return clips, []

        token_cache  = [self._clip_tokens(transcript, c["start"], c["end"]) for c in clips]
        break_indices = []

        for i, clip in enumerate(clips):
            if i == 0:
                clip["semantic_score"]   = 1.0
                clip["topic_continuity"] = True
                continue

            sim                      = self._cosine_sim(token_cache[i - 1], token_cache[i])
            clip["semantic_score"]   = round(sim, 4)
            clip["topic_continuity"] = sim >= self.threshold

            if not clip["topic_continuity"]:
                break_indices.append(i)
                logger.debug("Semantic break at [%d] (sim=%.3f): %.1f–%.1f",
                             i, sim, clip["start"], clip["end"])

        logger.info("Semantic analysis: %d clips | %d breaks", len(clips), len(break_indices))
        return clips, break_indices

    def detect_internal_breaks(self, transcript: list, clip: dict,
                                window_s: float = 30.0) -> float | None:
        """
        Detecta cambio de tema DENTRO de un clip largo (> 2 × window_s).
        Devuelve timestamp de corte si lo hay, None si el clip es coherente.
        """
        duration = clip["end"] - clip["start"]
        if duration < window_s * 2 or not transcript:
            return None

        best_break, lowest_sim = None, 1.0
        t = clip["start"] + window_s

        while t + window_s <= clip["end"]:
            sim = self._cosine_sim(
                self._clip_tokens(transcript, t - window_s, t),
                self._clip_tokens(transcript, t, t + window_s),
            )
            if sim < lowest_sim:
                lowest_sim, best_break = sim, t
            t += window_s / 2

        if best_break is not None and lowest_sim < self.threshold:
            logger.debug("Internal break [%.1f–%.1f] at %.1f (sim=%.3f)",
                         clip["start"], clip["end"], best_break, lowest_sim)
            return round(best_break, 3)

        return None


# ─── métricas finales ─────────────────────────────────────────────────────────

def _finalize_metrics(clip: dict, transcript: list | None) -> dict:
    """Calcula intensity, energy_score, peak_timestamp, nearest_text, confidence_score."""
    peaks = clip.get("group_peaks", [])

    if peaks:
        intensity      = round(max(p["intensity"] for p in peaks), 4)
        energy_score   = round(float(np.mean([p["intensity"] for p in peaks])), 4)
        peak_timestamp = round(max(peaks, key=lambda p: p["intensity"])["timestamp"], 3)
    else:
        intensity = energy_score = 0.0
        peak_timestamp = round((clip["start"] + clip["end"]) / 2, 3)

    duration  = clip["end"] - clip["start"]
    text_dens = _text_density(transcript, clip["start"], clip["end"]) if transcript else 0.0
    nearest   = _nearest_text(peak_timestamp, transcript) if transcript else None
    sem_score = clip.get("semantic_score", 0.5)
    n_peaks   = len(peaks)

    # Confidence: suma ponderada de factores normalizados ∈ [0, 1]
    f_intensity = min(intensity, 1.0)
    if duration < 8:
        f_duration = 0.2
    elif duration <= 45:
        f_duration = min(1.0, 0.6 + 0.4 * (duration - 8) / 37)
    elif duration <= 90:
        f_duration = 1.0
    else:
        f_duration = max(0.3, 1.0 - (duration - 90) / 60)

    confidence = round(
        0.30 * f_intensity
        + 0.25 * f_duration
        + 0.20 * min(text_dens / 5.0, 1.0)
        + 0.15 * sem_score
        + 0.10 * min(n_peaks / 5.0, 1.0),
        4,
    )

    clip.update({
        "peak_timestamp":   peak_timestamp,
        "intensity":        intensity,
        "energy_score":     energy_score,
        "nearest_text":     nearest,
        "confidence_score": confidence,
    })
    clip.setdefault("semantic_score",   0.5)
    clip.setdefault("topic_continuity", True)
    return clip


# ─── API pública ──────────────────────────────────────────────────────────────

def segment_video(peaks_path: str, transcript_path: str,
                  use_semantic: bool = True,
                  config: SegmentConfig | None = None) -> str:
    """
    Segmentación híbrida: Active Window + Semantic + Merge.
    Reemplaza generate_clip_candidates() en el pipeline.

    Input:
        peaks_path       — output/analysis/peaks.json
        transcript_path  — output/transcripts/transcript.json
        use_semantic     — activar análisis semántico (Phase 2, batch)
        config           — SegmentConfig opcional para ajuste fino

    Output:
        output/candidates/clips_candidates.json
        (campos extra vs legacy: energy_score, semantic_score, confidence_score)
    """
    for path in (peaks_path, transcript_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    config = config or SegmentConfig()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "clips_candidates.json")

    with open(peaks_path, encoding="utf-8") as f:
        peaks = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    if not peaks:
        logger.warning("No peaks — returning empty candidates list")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return output_path

    content_end    = transcript[-1]["end"] if transcript else float("inf")
    peaks_by_time  = sorted(peaks, key=lambda p: p["timestamp"])

    # ── Phase 1: Active Window ────────────────────────────────────────────
    windows = process_active_window(peaks_by_time, transcript, config, content_end)

    # Extender cada ventana al cierre del segmento de transcript en curso
    for w in windows:
        best_ts = max(w["group_peaks"], key=lambda p: p["intensity"])["timestamp"]
        w["end"] = _extend_to_segment_end(w["end"], best_ts, transcript, content_end)
        # Respetar max_clip_duration después de la extensión
        w["end"] = min(w["end"], w["start"] + config.max_clip_duration)

    # ── Phase 2: Semantic (batch, antes del merge para informar cortes) ───
    semantic_breaks: list[int] = []
    if use_semantic and len(windows) > 1:
        analyzer = SemanticAnalyzer(threshold=config.semantic_threshold)
        windows, semantic_breaks = analyzer.analyze_continuity(transcript, windows)

    # ── Phase 1.5: Post-process Merge ────────────────────────────────────
    clips = merge_clips(windows, config, semantic_breaks)

    # ── Métricas finales ──────────────────────────────────────────────────
    for clip in clips:
        _finalize_metrics(clip, transcript)

    # ── Filtrado y límite ─────────────────────────────────────────────────
    clips = [
        c for c in clips
        if config.min_clip_duration <= (c["end"] - c["start"]) <= config.max_clip_duration
    ]
    clips.sort(key=lambda c: c["start"])
    clips = clips[: config.max_candidates]

    # Limpiar campos internos antes de serializar
    for clip in clips:
        clip.pop("group_peaks", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    logger.info("Segments generated: %d → %s", len(clips), output_path)
    return output_path
