"""
segment_engine.py — Segmentación híbrida v3

Mejoras vs v2:
  1. _content_richness()       — penaliza repetición (unique_ratio); fillers siguen fuera
  2. Detección de transiciones — frases de cambio de tema bajan la similitud efectiva
  3. Break confidence          — HARD BLOCK solo si histéresis + avg_sim bajo + gap real
  4. Rhythm-adjusted decay     — clips densos (muchos picos) decaen más lento
  5. Confidence filtra clips   — descarta candidatos débiles; ordena por confidence
  6. Soft split por densidad   — clips largos con ≥3 picos/min → split en gap mayor
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

_STOPWORDS_ES = frozenset({
    'de', 'la', 'el', 'en', 'y', 'que', 'a', 'los', 'del', 'se', 'un', 'una',
    'con', 'es', 'no', 'lo', 'por', 'su', 'le', 'da', 'las', 'al', 'como',
    'pero', 'más', 'si', 'ya', 'hay', 'me', 'este', 'mi', 'para', 'o', 'te',
    'yo', 'muy', 'eso', 'esto', 'era', 'fue', 'han', 'ha', 'he', 'ser',
    'estar', 'bien', 'cuando', 'qué', 'porque', 'ahí', 'así', 'ese', 'tan',
    'todo', 'nada', 'nos', 'les', 'donde', 'vez', 'hacer', 'pues', 'ahora',
    'también', 'solo', 'igual', 'entonces',
})

_FILLERS_ES = frozenset({
    'eh', 'eeh', 'eeeh', 'ah', 'ahm', 'uh', 'uhm', 'um', 'mm', 'mmm', 'mhm',
    'hmm', 'hm', 'oye', 'veamos', 'vamos', 'bueno', 'tipo', 'osea', 'ok',
    'okay', 'oka', 'claro', 'sí', 'si', 'no', 'pues', 'este', 'sea',
})

# Frases que señalan cambio de tema en streaming — string matching, sin modelos
_TRANSITION_PHRASES_ES = (
    'cambiando de tema', 'otra cosa', 'por cierto', 'ahora bien', 'hablando de',
    'pasando a', 'antes de continuar', 'de hecho', 'en fin', 'dejando eso',
    'al margen', 'volviendo a', 'y bueno', 'como decía', 'a otra cosa',
    'ya que estamos', 'de todas formas', 'de todas maneras', 'dicho esto',
    'y ya', 'nada más', 'y nada',
)


@dataclass
class SegmentConfig:
    # Momentum
    momentum_decay_rate:        float = 0.20   # decay exponencial por segundo de silencio
    momentum_threshold:         float = 0.12   # por debajo → cerrar ventana
    text_momentum_boost:        float = 0.15   # texto significativo en gap → boost
    min_text_richness:          float = 8.0    # richness mínima (fillers=0, repetición penaliza)

    # Merge
    merge_gap_threshold:        float = 4.0
    max_clip_duration:          float = 120.0
    min_clip_duration:          float = 8.0
    max_candidates:             int   = 25
    min_confidence:             float = 0.15   # descarta clips por debajo de este umbral

    # Semántica
    semantic_threshold:         float = 0.25
    semantic_hysteresis:        int   = 2      # caídas consecutivas para declarar break
    break_confidence_threshold: float = 0.45  # confianza mínima para HARD BLOCK

    # Soft split
    dense_peak_rate:            float = 3.0   # picos/min para activar soft split
    dense_min_gap_s:            float = 5.0   # gap mínimo entre picos para split


# ─── helpers de texto ─────────────────────────────────────────────────────────

def _content_richness(text: str) -> float:
    """
    Riqueza de contenido real en el texto:
    - Elimina fillers (eh, mmm, bueno, ok…)
    - Elimina palabras cortas (≤2 chars)
    - Penaliza repetición: unique_ratio = palabras_únicas / total_palabras
      → "muy muy muy importante" → richness baja
      → "literal esto súper importante bro" → richness plena

    Retorna float ≥ 0. Permite comparar la misma frase con sí misma
    (unique_ratio = 1.0) vs una frase repetitiva (< 1.0).
    """
    words   = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
    content = [w for w in words if w not in _FILLERS_ES and len(w) > 2]
    if not content:
        return 0.0
    unique_ratio = len(set(content)) / len(content)
    raw_chars    = sum(len(w) for w in content)
    return raw_chars * max(unique_ratio, 0.3)  # mínimo 30%: no destruir frases largas


def _has_significant_text(transcript: list, t_start: float, t_end: float,
                           min_richness: float) -> bool:
    """True si hay contenido real (no solo fillers, no solo repetición) en el rango."""
    if t_end <= t_start or not transcript:
        return False
    total = sum(
        _content_richness(s["text"])
        for s in transcript
        if s["start"] < t_end and s["end"] > t_start
    )
    return total >= min_richness


def _text_density(transcript: list, t_start: float, t_end: float) -> float:
    dur   = max(t_end - t_start, 1e-3)
    chars = sum(len(s["text"]) for s in transcript
                if s["start"] < t_end and s["end"] > t_start)
    return chars / dur


def _get_clip_text(transcript: list, t_start: float, t_end: float) -> str:
    return " ".join(s["text"] for s in transcript
                    if s["start"] < t_end and s["end"] > t_start)


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
    Agrupa picos en ventanas usando momentum exponencial con ajuste por ritmo.

    Momentum decay:
      - Base: exp(−decay_rate × gap_desde_fin_ventana)
      - Ajustado por densidad de picos en la ventana actual:
        más picos → decay más lento (contenido denso tiene más inercia)
      - Texto significativo (no fillers, no repetitivo) → boost moderado
      - Nuevo pico → momentum += intensity (boost proporcional)

    Cuando momentum < threshold → cierre natural (sin hard cutoff de segundos).
    """
    if not peaks_by_time:
        return []

    p0       = peaks_by_time[0]
    pre, post = _get_window(p0["intensity"])
    current   = {
        "start":       round(max(0.0, p0["timestamp"] - pre), 3),
        "end":         round(min(content_end, p0["timestamp"] + post), 3),
        "group_peaks": [p0],
        "_momentum":   p0["intensity"],
    }
    windows = []

    for peak in peaks_by_time[1:]:
        pre, post     = _get_window(peak["intensity"])
        candidate_end = round(min(content_end, peak["timestamp"] + post), 3)

        # Tiempo de silencio real (negativo si el pico cae dentro de la ventana)
        gap_from_end = max(0.0, peak["timestamp"] - current["end"])

        # Ajuste de decay por ritmo: más picos acumulados → decay más lento
        n_peaks_so_far = len(current["group_peaks"])
        rhythm_factor  = 1.0 / (1.0 + 0.15 * n_peaks_so_far)
        adj_decay      = config.momentum_decay_rate * max(0.4, rhythm_factor)

        decayed = current["_momentum"] * float(np.exp(-adj_decay * gap_from_end))

        # Boost por texto significativo real en el gap
        if gap_from_end > 0 and _has_significant_text(
                transcript, current["end"], peak["timestamp"], config.min_text_richness):
            decayed = min(1.0, decayed + config.text_momentum_boost)

        would_exceed = (candidate_end - current["start"]) > config.max_clip_duration

        if decayed > config.momentum_threshold and not would_exceed:
            current["end"]        = max(current["end"], candidate_end)
            current["group_peaks"].append(peak)
            current["_momentum"]  = min(1.0, decayed + peak["intensity"])
        else:
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
    Evalúa continuidad semántica con tres capas:

    1. BoW cosine similarity con ventana de contexto (tokens de 2 clips previos)
    2. Detección de frases de transición (string matching, sin modelos)
       → "cambiando de tema", "por cierto", "y ya"… bajan la similitud efectiva
    3. Break confidence: HARD BLOCK solo si histéresis cumplida + avg_sim bajo
       + gap real > umbral → evita que la semántica tenga poder sin respaldo

    Batch, sin dependencias externas.
    """

    def __init__(self, threshold: float = 0.25, hysteresis: int = 2,
                 break_confidence_threshold: float = 0.45):
        self.threshold               = threshold
        self.hysteresis              = hysteresis
        self.break_confidence_threshold = break_confidence_threshold

    def _tokenize(self, text: str) -> list:
        words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
        return [w for w in words if w not in _STOPWORDS_ES and len(w) > 2]

    def _clip_tokens(self, transcript: list, t_start: float, t_end: float) -> list:
        return self._tokenize(_get_clip_text(transcript, t_start, t_end))

    def _cosine_sim(self, tok_a: list, tok_b: list) -> float:
        if not tok_a or not tok_b:
            return 0.0
        vocab  = list(set(tok_a) | set(tok_b))
        vec_a  = np.array([tok_a.count(w) for w in vocab], dtype=float)
        vec_b  = np.array([tok_b.count(w) for w in vocab], dtype=float)
        denom  = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return float(np.dot(vec_a, vec_b) / denom) if denom > 0 else 0.0

    def _has_transition_phrase(self, text: str) -> bool:
        """
        Detecta frases de cambio de tema en el texto del clip.
        Son señales directas más fiables que la similitud léxica en streaming.
        """
        t = text.lower()
        return any(phrase in t for phrase in _TRANSITION_PHRASES_ES)

    def _break_confidence(self, sim: float, avg_sim: float, gap_s: float) -> float:
        """
        Confianza de que el break es real (no un falso positivo semántico).

        Factores:
          - sim_drop: cuánto cae por debajo del threshold
          - ctx_drop: el contexto acumulado también es bajo (no solo el punto actual)
          - gap_ok:   hay un gap físico real entre clips (valida el corte)
        """
        sim_drop = max(0.0, (self.threshold - sim)     / max(self.threshold, 1e-6))
        ctx_drop = max(0.0, (self.threshold - avg_sim) / max(self.threshold, 1e-6))
        gap_ok   = min(1.0, gap_s / 15.0)
        return round(0.40 * sim_drop + 0.35 * ctx_drop + 0.25 * gap_ok, 3)

    def analyze_continuity(self, transcript: list, clips: list) -> tuple[list, list]:
        """
        Anota semantic_score, topic_continuity y has_transition en cada clip.

        Flujo por clip[i]:
          1. Ventana de contexto: ctx = tokens[i-2] + tokens[i-1]
          2. sim = cosine(ctx, tokens[i])
          3. Si hay frase de transición en clip[i] → sim_efectiva *= 0.5
          4. Histéresis: contar caídas consecutivas < threshold
          5. Si cumple histéresis: break_confidence(sim, avg_recent, gap)
          6. HARD BLOCK solo si confidence >= break_confidence_threshold
        """
        if not transcript:
            for c in clips:
                c["semantic_score"]   = 0.5
                c["topic_continuity"] = True
                c["has_transition"]   = False
            return clips, []

        token_cache   = [self._clip_tokens(transcript, c["start"], c["end"]) for c in clips]
        break_indices = []
        consecutive   = 0
        sim_history   : list[float] = []

        for i, clip in enumerate(clips):
            if i == 0:
                clip["semantic_score"]   = 1.0
                clip["topic_continuity"] = True
                clip["has_transition"]   = False
                sim_history.append(1.0)
                continue

            # Ventana de contexto (hasta 2 clips previos)
            ctx_tokens = []
            for j in range(max(0, i - 2), i):
                ctx_tokens.extend(token_cache[j])

            sim = self._cosine_sim(ctx_tokens, token_cache[i])

            # Detección de frases de transición → bajar similitud efectiva
            clip_text = _get_clip_text(transcript, clip["start"], clip["end"])
            has_trans = self._has_transition_phrase(clip_text)
            if has_trans:
                sim *= 0.5  # transición confirmada = contenido diferente

            clip["semantic_score"] = round(sim, 4)
            clip["has_transition"] = has_trans
            sim_history.append(sim)

            if sim < self.threshold:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= self.hysteresis:
                # Calcular confianza del break antes de declararlo
                avg_recent = float(np.mean(sim_history[-self.hysteresis:]))
                gap_s      = clip["start"] - clips[i - 1]["end"]
                bc         = self._break_confidence(sim, avg_recent, gap_s)

                if bc >= self.break_confidence_threshold:
                    break_at = i - consecutive + 1
                    if break_at not in break_indices:
                        break_indices.append(break_at)
                        logger.debug(
                            "Semantic HARD break at [%d] (sim=%.3f, conf=%.2f): %.1f–%.1f",
                            break_at, sim, bc, clips[break_at]["start"], clips[break_at]["end"],
                        )
                else:
                    logger.debug(
                        "Semantic break rejected at [%d] (sim=%.3f, conf=%.2f < %.2f)",
                        i, sim, bc, self.break_confidence_threshold,
                    )

                consecutive = 0

        for i, clip in enumerate(clips):
            clip["topic_continuity"] = (i not in break_indices)

        logger.info("Semantic: %d clips | %d hard breaks", len(clips), len(break_indices))
        return clips, break_indices

    def detect_internal_breaks(self, transcript: list, clip: dict,
                                window_s: float = 30.0) -> float | None:
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


# ─── Soft split por densidad de picos ────────────────────────────────────────

def _soft_split_dense_clips(clips: list, config: SegmentConfig) -> list:
    """
    Divide clips largos (>60s) con alta densidad de picos (≥dense_peak_rate picos/min).

    Un clip continuo y semánticamente coherente puede ser demasiado largo para
    consumo si tiene múltiples momentos fuertes. Partirlo en el mayor gap entre
    picos consecutivos produce dos clips más concisos y manejables.

    Solo corta si el gap entre picos más largo supera dense_min_gap_s (5s defecto).
    """
    result = []
    for clip in clips:
        duration = clip["end"] - clip["start"]
        peaks    = clip.get("group_peaks", [])

        if duration < 60 or len(peaks) < 3:
            result.append(clip)
            continue

        peaks_per_min = len(peaks) / duration * 60
        if peaks_per_min < config.dense_peak_rate:
            result.append(clip)
            continue

        # Mayor gap entre picos consecutivos → punto de split
        sorted_peaks = sorted(peaks, key=lambda p: p["timestamp"])
        best_split, best_gap = None, 0.0
        for j in range(len(sorted_peaks) - 1):
            gap      = sorted_peaks[j + 1]["timestamp"] - sorted_peaks[j]["timestamp"]
            midpoint = (sorted_peaks[j]["timestamp"] + sorted_peaks[j + 1]["timestamp"]) / 2
            half_pre = midpoint - clip["start"]
            half_post = clip["end"] - midpoint
            if (gap > best_gap
                    and half_pre  >= config.min_clip_duration
                    and half_post >= config.min_clip_duration):
                best_gap, best_split = gap, round(midpoint, 3)

        if best_split and best_gap >= config.dense_min_gap_s:
            logger.info("Soft split [%.1f–%.1f] at %.1f (%.1fppm, gap=%.1fs)",
                        clip["start"], clip["end"], best_split, peaks_per_min, best_gap)
            half_a = {**clip, "end":   best_split,
                      "group_peaks": [p for p in peaks if p["timestamp"] <= best_split]}
            half_b = {**clip, "start": best_split,
                      "group_peaks": [p for p in peaks if p["timestamp"] >  best_split]}
            result.extend([half_a, half_b])
        else:
            result.append(clip)

    if len(result) != len(clips):
        logger.info("Soft split: %d → %d clips", len(clips), len(result))
    return result


# ─── Phase 1.5: Post-process Merge ───────────────────────────────────────────

def merge_clips(clips: list, config: SegmentConfig,
                semantic_break_indices: list | None = None) -> list:
    """
    Une clips con gap ≤ dynamic_gap.

    Gap dinámico según duración del clip previo:
      ≥60s → ×0.5 (conservador: ya tiene contexto)
      ≤15s → ×1.5 (permisivo: necesita más)

    Semantic breaks son HARD BLOCKS: si la semántica declaró un corte con
    suficiente confianza, no hay energía ni gap que lo revierta.
    """
    if not clips:
        return []

    break_set = set(semantic_break_indices or [])
    merged    = [clips[0].copy()]

    for i, clip in enumerate(clips[1:], 1):
        last = merged[-1]

        # HARD BLOCK: corte semántico confirmado
        if i in break_set:
            merged.append(clip.copy())
            continue

        last_dur   = last["end"] - last["start"]
        gap        = clip["start"] - last["end"]
        merged_dur = clip["end"] - last["start"]

        if last_dur >= 60:
            dynamic_gap = config.merge_gap_threshold * 0.5
        elif last_dur <= 15:
            dynamic_gap = config.merge_gap_threshold * 1.5
        else:
            dynamic_gap = config.merge_gap_threshold

        if gap <= dynamic_gap and merged_dur <= config.max_clip_duration:
            last["end"] = max(last["end"], clip["end"])
            last["group_peaks"] = last.get("group_peaks", []) + clip.get("group_peaks", [])
        else:
            merged.append(clip.copy())

    logger.info("Post-merge: %d → %d clips", len(clips), len(merged))
    return merged


# ─── métricas finales ─────────────────────────────────────────────────────────

def _finalize_metrics(clip: dict, transcript: list | None) -> dict:
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

    if duration < 8:
        f_duration = 0.2
    elif duration <= 45:
        f_duration = min(1.0, 0.6 + 0.4 * (duration - 8) / 37)
    elif duration <= 90:
        f_duration = 1.0
    else:
        f_duration = max(0.3, 1.0 - (duration - 90) / 60)

    confidence = round(
        0.30 * min(intensity, 1.0)
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
    clip.setdefault("has_transition",   False)
    return clip


# ─── API pública ──────────────────────────────────────────────────────────────

def segment_video(peaks_path: str, transcript_path: str,
                  use_semantic: bool = True,
                  config: SegmentConfig | None = None) -> str:
    """
    Segmentación híbrida v3: Momentum + Transition-aware Semantic + Dynamic Merge.

    Input:
        peaks_path       — output/analysis/peaks.json
        transcript_path  — output/transcripts/transcript.json
        use_semantic     — análisis semántico batch (Phase 2)
        config           — SegmentConfig para ajuste fino

    Output:
        output/candidates/clips_candidates.json

    Campos extra vs legacy: energy_score, semantic_score, confidence_score,
                             has_transition, topic_continuity
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

    for w in windows:
        best_ts = max(w["group_peaks"], key=lambda p: p["intensity"])["timestamp"]
        w["end"] = _extend_to_segment_end(w["end"], best_ts, transcript, content_end)
        w["end"] = min(w["end"], w["start"] + config.max_clip_duration)

    # ── Phase 2: Semantic — ANTES del merge, informa HARD BLOCKS ─────────
    semantic_breaks: list[int] = []
    if use_semantic and len(windows) > 1:
        analyzer = SemanticAnalyzer(
            threshold=config.semantic_threshold,
            hysteresis=config.semantic_hysteresis,
            break_confidence_threshold=config.break_confidence_threshold,
        )
        windows, semantic_breaks = analyzer.analyze_continuity(transcript, windows)

    # ── Soft split por densidad (antes del merge para no rehacer splits) ──
    windows = _soft_split_dense_clips(windows, config)

    # ── Phase 1.5: Merge — semantic breaks son HARD BLOCKS ───────────────
    clips = merge_clips(windows, config, semantic_breaks)

    # ── Métricas finales ──────────────────────────────────────────────────
    for clip in clips:
        _finalize_metrics(clip, transcript)

    # ── Filtrado real por confidence + duración ───────────────────────────
    before = len(clips)
    clips  = [
        c for c in clips
        if (config.min_clip_duration <= (c["end"] - c["start"]) <= config.max_clip_duration
            and c.get("confidence_score", 0) >= config.min_confidence)
    ]
    if len(clips) < before:
        logger.info("Confidence filter: %d → %d clips (threshold=%.2f)",
                    before, len(clips), config.min_confidence)

    clips.sort(key=lambda c: c["start"])
    clips = clips[: config.max_candidates]

    for clip in clips:
        clip.pop("group_peaks", None)
        clip.pop("_momentum",   None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    logger.info("Segments generated: %d → %s", len(clips), output_path)
    return output_path
