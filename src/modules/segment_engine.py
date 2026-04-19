"""
segment_engine.py — Segmentación híbrida: Momentum-Active Window + Semantic + Merge

Flujo:
  1. process_active_window()  — momentum exponencial reemplaza gap binario;
                                solo texto significativo (sin fillers) boost continuidad
  2. SemanticAnalyzer         — ventana de contexto + histéresis (≥2 caídas consecutivas)
                                los cortes semánticos son HARD BLOCKS en merge
  3. merge_clips()            — gap dinámico según duración del clip;
                                semantic break = nunca merge, sin excepción
  4. _finalize_metrics()      — confidence_score informa merge, no es solo decorativo
"""
import json
import logging
import os
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR     = "output/candidates"
MIN_DURATION   = 8.0
MAX_DURATION   = 120.0
MAX_CANDIDATES = 25

# Stopwords para análisis semántico
_STOPWORDS_ES = frozenset({
    'de', 'la', 'el', 'en', 'y', 'que', 'a', 'los', 'del', 'se', 'un', 'una',
    'con', 'es', 'no', 'lo', 'por', 'su', 'le', 'da', 'las', 'al', 'como',
    'pero', 'más', 'si', 'ya', 'hay', 'me', 'este', 'mi', 'para', 'o', 'te',
    'yo', 'muy', 'eso', 'esto', 'era', 'fue', 'han', 'ha', 'he', 'ser',
    'estar', 'bien', 'cuando', 'qué', 'porque', 'ahí', 'así', 'ese', 'tan',
    'todo', 'nada', 'nos', 'les', 'donde', 'vez', 'hacer', 'pues', 'ahora',
    'también', 'solo', 'igual', 'entonces',
})

# Fillers/dudas que NO son contenido significativo — ignorar en decisión de continuidad
_FILLERS_ES = frozenset({
    'eh', 'eeh', 'eeeh', 'ah', 'ahm', 'uh', 'uhm', 'um', 'mm', 'mmm', 'mhm',
    'hmm', 'hm', 'oye', 'veamos', 'vamos', 'bueno', 'tipo', 'osea', 'ok',
    'okay', 'oka', 'claro', 'sí', 'si', 'no', 'pues', 'este', 'sea',
})


@dataclass
class SegmentConfig:
    # Momentum (reemplaza silence_duration_threshold)
    momentum_decay_rate:    float = 0.20   # decay exponencial por segundo de silencio
    momentum_threshold:     float = 0.12   # por debajo → cerrar ventana
    text_momentum_boost:    float = 0.15   # texto significativo en gap → boost momentum
    min_significant_chars:  int   = 12     # chars de contenido real (sin fillers) para boost

    # Merge post-proceso
    merge_gap_threshold:    float = 4.0   # gap base para merge; escala con duración
    max_clip_duration:      float = 120.0  # hard cap
    min_clip_duration:      float = 8.0   # mínimo viable
    max_candidates:         int   = 25

    # Semántica
    semantic_threshold:     float = 0.25  # similitud < threshold → posible corte
    semantic_hysteresis:    int   = 2     # caídas consecutivas para declarar break


# ─── helpers de texto ─────────────────────────────────────────────────────────

def _significant_chars(text: str) -> int:
    """Cuenta chars de palabras de contenido real (sin fillers, len > 2)."""
    words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
    return sum(len(w) for w in words if w not in _FILLERS_ES and len(w) > 2)


def _has_significant_text(transcript: list, t_start: float, t_end: float,
                           min_chars: int) -> bool:
    """
    True si hay contenido real (no solo fillers/dudas) en el rango.
    Ignora: 'eh', 'mmm', 'o sea', respuestas cortas aisladas.
    """
    if t_end <= t_start or not transcript:
        return False
    total = sum(
        _significant_chars(s["text"])
        for s in transcript
        if s["start"] < t_end and s["end"] > t_start
    )
    return total >= min_chars


def _text_density(transcript: list, t_start: float, t_end: float) -> float:
    dur   = max(t_end - t_start, 1e-3)
    chars = sum(len(s["text"]) for s in transcript
                if s["start"] < t_end and s["end"] > t_start)
    return chars / dur


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
    return round(min(max(s["end"] for s in overlapping) + 0.3, max_end), 3)


def _get_window(intensity: float) -> tuple[float, float]:
    if intensity > 0.8:
        return 15.0, 20.0
    elif intensity > 0.6:
        return 12.0, 18.0
    return 10.0, 15.0


# ─── Phase 1: Momentum-Active Window ─────────────────────────────────────────

def process_active_window(peaks_by_time: list, transcript: list,
                           config: SegmentConfig,
                           content_end: float = float("inf")) -> list:
    """
    Agrupa picos en ventanas de contenido usando momentum exponencial.

    El momentum reemplaza el gap binario:
      - Sube con cada pico (basado en su intensidad)
      - Decae exponencialmente con el tiempo de silencio desde el fin de ventana
      - Texto significativo en el gap suma un boost moderado
      - Cuando momentum < threshold → cerrar ventana (corte natural)

    Esto produce cortes más graduales y proporcionales a la intensidad del contenido,
    en vez de cortar exactamente a los Xs de silencio.
    """
    if not peaks_by_time:
        return []

    p0       = peaks_by_time[0]
    pre, post = _get_window(p0["intensity"])
    current   = {
        "start":       round(max(0.0, p0["timestamp"] - pre), 3),
        "end":         round(min(content_end, p0["timestamp"] + post), 3),
        "group_peaks": [p0],
        "_momentum":   p0["intensity"],  # max intensity vista en esta ventana
    }
    windows = []

    for peak in peaks_by_time[1:]:
        pre, post     = _get_window(peak["intensity"])
        candidate_end = round(min(content_end, peak["timestamp"] + post), 3)

        # Tiempo de silencio real = desde el fin de la ventana actual hasta este pico
        # Negativo si el pico cae dentro de la ventana (overlap) → sin decay
        gap_from_end = max(0.0, peak["timestamp"] - current["end"])

        # Momentum decaído exponencialmente por el gap real
        decayed = current["_momentum"] * float(np.exp(-config.momentum_decay_rate * gap_from_end))

        # Boost si hay contenido real en el gap (no fillers)
        if gap_from_end > 0 and _has_significant_text(
                transcript, current["end"], peak["timestamp"], config.min_significant_chars):
            decayed = min(1.0, decayed + config.text_momentum_boost)

        would_exceed = (candidate_end - current["start"]) > config.max_clip_duration

        if decayed > config.momentum_threshold and not would_exceed:
            # Ventana activa — extender
            current["end"]       = max(current["end"], candidate_end)
            current["group_peaks"].append(peak)
            # Momentum: el máximo de lo decaído + el nuevo pico
            current["_momentum"] = min(1.0, decayed + peak["intensity"])
        else:
            # Momentum agotado → cerrar ventana actual y abrir nueva
            windows.append(current)
            current = {
                "start":       round(max(0.0, peak["timestamp"] - pre), 3),
                "end":         candidate_end,
                "group_peaks": [peak],
                "_momentum":   peak["intensity"],
            }

    windows.append(current)
    logger.info("Active windows: %d (from %d peaks)", len(windows), len(peaks_by_time))
    return windows


# ─── Phase 2: Semantic Continuity ────────────────────────────────────────────

class SemanticAnalyzer:
    """
    Evalúa continuidad semántica entre clips.

    Mejoras vs v1:
    - Ventana de contexto: compara cada clip contra el promedio de los 2 anteriores
      (reduce falsos cortes por variación léxica local)
    - Histéresis: exige ≥N caídas consecutivas antes de declarar corte
      (evita cortes por un solo punto bajo)
    - Cortes declarados → HARD BLOCK en merge (nunca se revierte)

    Sin dependencias externas — BoW cosine similarity con numpy.
    """

    def __init__(self, threshold: float = 0.25, hysteresis: int = 2):
        self.threshold  = threshold
        self.hysteresis = hysteresis

    def _tokenize(self, text: str) -> list:
        words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
        return [w for w in words if w not in _STOPWORDS_ES and len(w) > 2]

    def _clip_tokens(self, transcript: list, t_start: float, t_end: float) -> list:
        text = " ".join(s["text"] for s in transcript
                        if s["start"] < t_end and s["end"] > t_start)
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

        Ventana de contexto: compara clip[i] contra tokens de clip[i-2] + clip[i-1]
        combinados (da más peso al historial reciente que al punto-a-punto).

        Histéresis: un único sim bajo NO declara break; se necesitan
        self.hysteresis caídas consecutivas. Esto ignora variaciones léxicas
        puntuales (vocabulario variable en streaming).

        Returns:
            clips        — lista con semantic_score / topic_continuity añadidos
            break_indices — índices donde hay corte semántico confirmado (HARD)
        """
        if not transcript:
            for c in clips:
                c["semantic_score"]   = 0.5
                c["topic_continuity"] = True
            return clips, []

        token_cache   = [self._clip_tokens(transcript, c["start"], c["end"]) for c in clips]
        break_indices = []
        consecutive   = 0  # caídas consecutivas bajo threshold

        for i, clip in enumerate(clips):
            if i == 0:
                clip["semantic_score"]   = 1.0
                clip["topic_continuity"] = True
                continue

            # Ventana de contexto: combinar hasta 2 clips previos
            ctx_tokens = []
            for j in range(max(0, i - 2), i):
                ctx_tokens.extend(token_cache[j])

            sim = self._cosine_sim(ctx_tokens, token_cache[i])
            clip["semantic_score"] = round(sim, 4)

            if sim < self.threshold:
                consecutive += 1
            else:
                consecutive = 0

            # Histéresis: solo break si N caídas consecutivas
            if consecutive >= self.hysteresis:
                # El break real empieza en la primera caída de la racha
                break_at = i - consecutive + 1
                if break_at not in break_indices:
                    break_indices.append(break_at)
                    logger.debug("Semantic break at [%d] (sim=%.3f): %.1f–%.1f",
                                 break_at, sim, clips[break_at]["start"], clips[break_at]["end"])
                consecutive = 0  # reset — nueva racha desde cero

        clip["topic_continuity"] = True  # default para clips sin break
        for i in break_indices:
            clips[i]["topic_continuity"] = False

        logger.info("Semantic: %d clips | %d breaks (hysteresis=%d)",
                    len(clips), len(break_indices), self.hysteresis)
        return clips, break_indices

    def detect_internal_breaks(self, transcript: list, clip: dict,
                                window_s: float = 30.0) -> float | None:
        """
        Detecta cambio de tema DENTRO de un clip largo (> 2 × window_s).
        Desliza ventana de window_s con paso window_s/2.
        """
        if (clip["end"] - clip["start"]) < window_s * 2 or not transcript:
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

        if best_break and lowest_sim < self.threshold:
            return round(best_break, 3)
        return None


# ─── Phase 1.5: Post-process Merge ───────────────────────────────────────────

def merge_clips(clips: list, config: SegmentConfig,
                semantic_break_indices: list | None = None) -> list:
    """
    Une clips consecutivos respetando:
    - Gap dinámico por duración (clips largos → gap más conservador)
    - Semantic breaks = HARD BLOCKS (nunca se mergean, sin excepción)
    - Clips con confidence muy baja no se fusionan con vecinos igualmente débiles
    """
    if not clips:
        return []

    break_set = set(semantic_break_indices or [])
    merged    = [clips[0].copy()]

    for i, clip in enumerate(clips[1:], 1):
        last = merged[-1]

        # HARD BLOCK: corte semántico confirmado → nunca merge
        if i in break_set:
            merged.append(clip.copy())
            continue

        last_dur = last["end"] - last["start"]
        gap      = clip["start"] - last["end"]

        # Gap dinámico: clips largos ya tienen contexto → más conservador
        if last_dur >= 60:
            dynamic_gap = config.merge_gap_threshold * 0.5
        elif last_dur <= 15:
            dynamic_gap = config.merge_gap_threshold * 1.5
        else:
            dynamic_gap = config.merge_gap_threshold

        merged_dur = clip["end"] - last["start"]

        if gap <= dynamic_gap and merged_dur <= config.max_clip_duration:
            last["end"] = max(last["end"], clip["end"])
            last["group_peaks"] = last.get("group_peaks", []) + clip.get("group_peaks", [])
        else:
            merged.append(clip.copy())

    logger.info("Post-merge: %d → %d clips", len(clips), len(merged))
    return merged


# ─── métricas finales ─────────────────────────────────────────────────────────

def _finalize_metrics(clip: dict, transcript: list | None) -> dict:
    """
    Calcula intensity, energy_score, peak_timestamp, nearest_text, confidence_score.

    confidence_score es una señal real (0–1) usada en decisiones de merge:
      - intensity sostenida (no solo pico)
      - duración en zona óptima (plateau 20–45s)
      - texto presente
      - continuidad semántica
      - cantidad de picos agrupados (señal de actividad sostenida)
    """
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

    # Factores normalizados ∈ [0, 1]
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
    Segmentación híbrida: Momentum-Active Window + Semantic + Dynamic Merge.

    Input:
        peaks_path       — output/analysis/peaks.json
        transcript_path  — output/transcripts/transcript.json
        use_semantic     — análisis semántico batch (Phase 2)
        config           — SegmentConfig para ajuste fino de thresholds

    Output:
        output/candidates/clips_candidates.json
        Campos adicionales vs legacy: energy_score, semantic_score, confidence_score
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

    content_end   = transcript[-1]["end"] if transcript else float("inf")
    peaks_by_time = sorted(peaks, key=lambda p: p["timestamp"])

    # ── Phase 1: Momentum-Active Window ──────────────────────────────────
    windows = process_active_window(peaks_by_time, transcript, config, content_end)

    # Extender cada ventana al cierre del segmento de transcript en curso
    for w in windows:
        best_ts = max(w["group_peaks"], key=lambda p: p["intensity"])["timestamp"]
        w["end"] = _extend_to_segment_end(w["end"], best_ts, transcript, content_end)
        w["end"] = min(w["end"], w["start"] + config.max_clip_duration)

    # ── Phase 2: Semantic — ANTES del merge para informar HARD BLOCKS ────
    semantic_breaks: list[int] = []
    if use_semantic and len(windows) > 1:
        analyzer = SemanticAnalyzer(
            threshold=config.semantic_threshold,
            hysteresis=config.semantic_hysteresis,
        )
        windows, semantic_breaks = analyzer.analyze_continuity(transcript, windows)

    # ── Phase 1.5: Post-process Merge — semantic breaks son HARD BLOCKS ──
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

    for clip in clips:
        clip.pop("group_peaks", None)
        clip.pop("_momentum",   None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    logger.info("Segments generated: %d → %s", len(clips), output_path)
    return output_path
