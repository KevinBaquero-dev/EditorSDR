"""
segment_engine.py — Segmentación híbrida v4

Mejoras vs v3:
  1. _content_richness()     — penaliza stopwords además de repetición;
                               longitud ya no equivale a valor
  2. Transición ≠ corte      — has_transition como factor en break_confidence,
                               no modifica sim directamente (transición sola no corta)
  3. break_confidence        — f_tra × f_sim: transición solo suma si sim también es baja
                               gap normalizado a 30s (escala real de streaming)
  4. Ritmo por intensidad    — rhythm usa mean_intensity del grupo (no count);
                               3 picos fuertes > 10 picos débiles
  5. Filtro en dos pasos     — calidad (confidence) + outliers virales (energy);
                               clips raros pero brutales no se pierden
  6. Split point con score   — gap + energía + semántica; no siempre el gap mayor
  7. moment_tier             — jerarquía de momentos: weak/good/excellent/brutal
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
    momentum_decay_rate:        float = 0.20
    momentum_threshold:         float = 0.12
    text_momentum_boost:        float = 0.15
    min_text_richness:          float = 8.0   # richness mínima (penaliza fillers + stopwords)

    # Merge
    merge_gap_threshold:        float = 4.0
    max_clip_duration:          float = 120.0
    min_clip_duration:          float = 8.0
    max_candidates:             int   = 25
    min_confidence:             float = 0.15  # umbral para filtro de calidad

    # Semántica
    semantic_threshold:         float = 0.25
    semantic_hysteresis:        int   = 2
    break_confidence_threshold: float = 0.45

    # Soft split
    dense_peak_rate:            float = 3.0   # picos/min para activar soft split
    split_score_threshold:      float = 0.30  # score mínimo para aceptar un split


# ─── helpers de texto ─────────────────────────────────────────────────────────

def _content_richness(text: str) -> float:
    """
    Riqueza de contenido real. Penaliza tres patrones:
      1. Fillers (eh, mmm, bueno…) → eliminados antes de contar
      2. Repetición (muy muy muy) → penalizada por unique_ratio
      3. Alto porcentaje de stopwords → penalizada por stopword_factor
         "esto es algo muy importante para todos" → stopword_ratio alto → richness reducida

    No necesita NLP: las tres penalizaciones son cálculos de frecuencia de palabras.
    """
    words = re.findall(r"\b[a-záéíóúñü]+\b", text.lower())
    if not words:
        return 0.0

    # Stopword ratio sobre todas las palabras (incluye fillers para el cálculo)
    n_stopwords    = sum(1 for w in words if w in _STOPWORDS_ES)
    stopword_ratio = n_stopwords / len(words)
    stopword_factor = max(0.3, 1.0 - stopword_ratio)  # nunca destroza del todo

    # Palabras de contenido real (sin fillers, sin palabras muy cortas)
    content = [w for w in words if w not in _FILLERS_ES and len(w) > 2]
    if not content:
        return 0.0

    unique_ratio = len(set(content)) / len(content)   # 1.0 = todas distintas
    raw_chars    = sum(len(w) for w in content)
    return raw_chars * max(unique_ratio, 0.3) * stopword_factor


def _has_significant_text(transcript: list, t_start: float, t_end: float,
                           min_richness: float) -> bool:
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
    Agrupa picos usando momentum exponencial ajustado por ritmo.

    Ritmo (rhythm_factor):
      - Usa mean_intensity del grupo, no count
      - 3 picos fuertes (0.9) > 10 picos débiles (0.2) en cuanto a inercia
      - adj_decay = base_decay / (1 + 1.5 × mean_intensity)
        → mean_intensity 0.9 → decay ~×0.43 más lento
        → mean_intensity 0.3 → decay ~×0.69 más lento (menos inercia)

    Texto en gap:
      - Solo el contenido real (post-richness) extiende el momentum
      - Fillers + stopwords no evitan el cierre de ventana
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
        gap_from_end  = max(0.0, peak["timestamp"] - current["end"])

        # Decay ajustado por intensidad media del grupo (no por conteo)
        grp             = current["group_peaks"]
        mean_intensity  = sum(p["intensity"] for p in grp) / len(grp)
        adj_decay       = config.momentum_decay_rate / (1.0 + 1.5 * mean_intensity)

        decayed = current["_momentum"] * float(np.exp(-adj_decay * gap_from_end))

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
    Continuidad semántica con tres señales calibradas:

    1. BoW cosine (ventana de contexto, histéresis)
    2. Frases de transición — no modifican sim; son factor en break_confidence
       → "cambiando de tema" + sim baja → más confianza en el break
       → "cambiando de tema" + sim alta → la transición no causa corte
    3. break_confidence: f_tra × f_sim — la transición solo AMPLIFICA un drop
       existente; nunca corta si el contenido sigue siendo coherente
    """

    def __init__(self, threshold: float = 0.25, hysteresis: int = 2,
                 break_confidence_threshold: float = 0.45):
        self.threshold                = threshold
        self.hysteresis               = hysteresis
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
        t = text.lower()
        return any(phrase in t for phrase in _TRANSITION_PHRASES_ES)

    def _break_confidence(self, sim: float, avg_sim: float,
                           gap_s: float, has_transition: bool) -> float:
        """
        Confianza de que el break es real. Señales en rango [0, 1]:

          f_sim — cuánto cae sim bajo el threshold
          f_ctx — el contexto acumulado también es bajo (no outlier puntual)
          f_gap — hay separación física real (normalizada a 30s para streaming)
          f_tra — frase de transición detectada × f_sim
                  → solo amplifica drops existentes; no causa breaks solos

        Pesos suman a 1.0. Threshold 0.45 calibrado para que se necesiten
        al menos dos señales activas (no basta una sola evidencia).
        """
        t     = max(self.threshold, 1e-6)
        f_sim = min(1.0, max(0.0, (t - sim)     / t))
        f_ctx = min(1.0, max(0.0, (t - avg_sim) / t))
        f_gap = min(1.0, gap_s / 30.0)
        f_tra = (1.0 if has_transition else 0.0) * f_sim  # transición × sim_drop

        return round(0.35 * f_sim + 0.30 * f_ctx + 0.20 * f_gap + 0.15 * f_tra, 3)

    def analyze_continuity(self, transcript: list, clips: list) -> tuple[list, list]:
        if not transcript:
            for c in clips:
                c["semantic_score"]   = 0.5
                c["topic_continuity"] = True
                c["has_transition"]   = False
            return clips, []

        token_cache   = [self._clip_tokens(transcript, c["start"], c["end"]) for c in clips]
        break_indices = []
        consecutive   = 0
        sim_history: list[float] = []

        for i, clip in enumerate(clips):
            if i == 0:
                clip["semantic_score"]   = 1.0
                clip["topic_continuity"] = True
                clip["has_transition"]   = False
                sim_history.append(1.0)
                continue

            ctx_tokens = []
            for j in range(max(0, i - 2), i):
                ctx_tokens.extend(token_cache[j])

            sim       = self._cosine_sim(ctx_tokens, token_cache[i])
            clip_text = _get_clip_text(transcript, clip["start"], clip["end"])
            has_trans = self._has_transition_phrase(clip_text)

            # sim NO se modifica por la transición — la transición es factor en confidence
            clip["semantic_score"] = round(sim, 4)
            clip["has_transition"] = has_trans
            sim_history.append(sim)

            consecutive = consecutive + 1 if sim < self.threshold else 0

            if consecutive >= self.hysteresis:
                avg_recent = float(np.mean(sim_history[-self.hysteresis:]))
                gap_s      = clip["start"] - clips[i - 1]["end"]
                bc         = self._break_confidence(sim, avg_recent, gap_s, has_trans)

                if bc >= self.break_confidence_threshold:
                    break_at = i - consecutive + 1
                    if break_at not in break_indices:
                        break_indices.append(break_at)
                        logger.debug("Semantic HARD break [%d] sim=%.3f conf=%.2f: %.1f–%.1f",
                                     break_at, sim, bc,
                                     clips[break_at]["start"], clips[break_at]["end"])
                else:
                    logger.debug("Break rejected [%d] sim=%.3f conf=%.2f < %.2f",
                                 i, sim, bc, self.break_confidence_threshold)

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
        return round(best_break, 3) if (best_break and lowest_sim < self.threshold) else None


# ─── Soft split por densidad — con scoring de punto de corte ─────────────────

def _score_split_point(midpoint: float, clip: dict, peaks_sorted: list,
                        transcript: list | None,
                        analyzer: "SemanticAnalyzer | None") -> float:
    """
    Puntúa un candidato a punto de corte con tres señales:

      f_gap  — tamaño del gap físico entre el último pico pre y el primero post
               (gap mayor → más natural cortar ahí)
      f_sem  — drop de similitud semántica entre las dos mitades
               (solo si analyzer disponible; 0.5 neutro si no)
      f_nrg  — desequilibrio de energía entre mitades
               (1.0 si una mitad domina; 0.0 si equilibrado)
               → un buen corte separa momentos distintos en intensidad

    No usa siempre el mayor gap — pondera las tres señales.
    """
    pre_peaks  = [p for p in peaks_sorted if p["timestamp"] <= midpoint]
    post_peaks = [p for p in peaks_sorted if p["timestamp"] >  midpoint]

    # Gap físico entre el último pico previo y el primero posterior
    pre_last   = pre_peaks[-1]["timestamp"]  if pre_peaks  else clip["start"]
    post_first = post_peaks[0]["timestamp"]  if post_peaks else clip["end"]
    gap        = post_first - pre_last
    f_gap      = min(1.0, gap / 30.0)

    # Desequilibrio de energía entre mitades
    pre_e  = float(np.mean([p["intensity"] for p in pre_peaks]))  if pre_peaks  else 0.0
    post_e = float(np.mean([p["intensity"] for p in post_peaks])) if post_peaks else 0.0
    balance = min(pre_e, post_e) / max(max(pre_e, post_e), 1e-6)
    f_nrg   = 1.0 - balance  # 0 = equilibrado (malo para cortar), 1 = desequilibrado

    # Caída semántica entre mitades
    f_sem = 0.5  # neutro por defecto
    if analyzer and transcript:
        half = (clip["end"] - clip["start"]) * 0.3
        window = min(25.0, half)
        t_pre  = analyzer._clip_tokens(transcript, midpoint - window, midpoint)
        t_post = analyzer._clip_tokens(transcript, midpoint, midpoint + window)
        sim    = analyzer._cosine_sim(t_pre, t_post)
        f_sem  = 1.0 - sim

    return round(0.40 * f_gap + 0.35 * f_sem + 0.25 * f_nrg, 4)


def _soft_split_dense_clips(clips: list, config: SegmentConfig,
                              transcript: list | None = None,
                              analyzer: "SemanticAnalyzer | None" = None) -> list:
    """
    Divide clips largos (>60s) con alta densidad de picos (≥dense_peak_rate/min).

    El punto de corte se elige por score (gap + semántica + energía),
    no solo por el mayor gap — que puede caer en medio de una idea.
    """
    result = []
    for clip in clips:
        duration = clip["end"] - clip["start"]
        peaks    = clip.get("group_peaks", [])

        if duration < 60 or len(peaks) < 3:
            result.append(clip)
            continue

        if len(peaks) / duration * 60 < config.dense_peak_rate:
            result.append(clip)
            continue

        sorted_peaks = sorted(peaks, key=lambda p: p["timestamp"])

        # Candidatos: midpoints entre picos consecutivos con ambas mitades viables
        candidates = [
            (sorted_peaks[j]["timestamp"] + sorted_peaks[j + 1]["timestamp"]) / 2
            for j in range(len(sorted_peaks) - 1)
            if ((sorted_peaks[j]["timestamp"] + sorted_peaks[j + 1]["timestamp"]) / 2 - clip["start"] >= config.min_clip_duration
                and clip["end"] - (sorted_peaks[j]["timestamp"] + sorted_peaks[j + 1]["timestamp"]) / 2 >= config.min_clip_duration)
        ]

        if not candidates:
            result.append(clip)
            continue

        best_mid, best_score = max(
            ((mid, _score_split_point(mid, clip, sorted_peaks, transcript, analyzer))
             for mid in candidates),
            key=lambda x: x[1],
        )

        if best_score >= config.split_score_threshold:
            best_mid = round(best_mid, 3)
            ppm = len(peaks) / duration * 60
            logger.info("Soft split [%.1f–%.1f] at %.1f (score=%.2f, %.1fppm)",
                        clip["start"], clip["end"], best_mid, best_score, ppm)
            half_a = {**clip, "end":   best_mid,
                      "group_peaks": [p for p in peaks if p["timestamp"] <= best_mid]}
            half_b = {**clip, "start": best_mid,
                      "group_peaks": [p for p in peaks if p["timestamp"] >  best_mid]}
            result.extend([half_a, half_b])
        else:
            result.append(clip)

    return result


# ─── Phase 1.5: Post-process Merge ───────────────────────────────────────────

def merge_clips(clips: list, config: SegmentConfig,
                semantic_break_indices: list | None = None) -> list:
    if not clips:
        return []

    break_set = set(semantic_break_indices or [])
    merged    = [clips[0].copy()]

    for i, clip in enumerate(clips[1:], 1):
        last = merged[-1]

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

    # Jerarquía de momentos: combina confidence + energy para no penalizar
    # clips con alta energía pero corta duración (outliers potencialmente virales)
    combined = 0.6 * confidence + 0.4 * energy_score
    if combined >= 0.70:
        tier = "brutal"
    elif combined >= 0.50:
        tier = "excellent"
    elif combined >= 0.30:
        tier = "good"
    else:
        tier = "weak"

    clip.update({
        "peak_timestamp":   peak_timestamp,
        "intensity":        intensity,
        "energy_score":     energy_score,
        "nearest_text":     nearest,
        "confidence_score": confidence,
        "moment_tier":      tier,
    })
    clip.setdefault("semantic_score",   0.5)
    clip.setdefault("topic_continuity", True)
    clip.setdefault("has_transition",   False)
    return clip


# ─── Filtro en dos pasos ──────────────────────────────────────────────────────

def _apply_filters(clips: list, config: SegmentConfig) -> list:
    """
    Filtro en dos pasos para no perder clips virales con baja confidence:

    Paso 1 (calidad): clips con confidence ≥ min_confidence
    Paso 2 (outliers): top ~20% por energy_score — clips raros pero brutales

    La unión de ambos preserva el orden cronológico original.
    """
    quality_ids = {(c["start"], c["end"]) for c in clips
                   if c.get("confidence_score", 0) >= config.min_confidence}

    n_viral     = max(3, config.max_candidates // 5)
    viral_ids   = {
        (c["start"], c["end"])
        for c in sorted(clips, key=lambda c: c.get("energy_score", 0), reverse=True)[:n_viral]
    }

    keep    = quality_ids | viral_ids
    result  = [c for c in clips if (c["start"], c["end"]) in keep]

    removed = len(clips) - len(result)
    if removed:
        logger.info(
            "Confidence filter: %d → %d clips (quality=%d, viral=%d, threshold=%.2f)",
            len(clips), len(result),
            len(quality_ids & keep), len(viral_ids & keep),
            config.min_confidence,
        )
    return result


# ─── API pública ──────────────────────────────────────────────────────────────

def segment_video(peaks_path: str, transcript_path: str,
                  use_semantic: bool = True,
                  config: SegmentConfig | None = None) -> str:
    """
    Segmentación híbrida v4.

    Input:
        peaks_path       — output/analysis/peaks.json
        transcript_path  — output/transcripts/transcript.json
        use_semantic     — análisis semántico batch
        config           — SegmentConfig para ajuste fino

    Output:
        output/candidates/clips_candidates.json
        Campos: start, end, peak_timestamp, intensity, nearest_text,
                energy_score, semantic_score, confidence_score,
                moment_tier, topic_continuity, has_transition
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

    # ── Phase 2: Semantic — ANTES del merge ───────────────────────────────
    analyzer: SemanticAnalyzer | None = None
    semantic_breaks: list[int] = []
    if use_semantic and len(windows) > 1:
        analyzer = SemanticAnalyzer(
            threshold=config.semantic_threshold,
            hysteresis=config.semantic_hysteresis,
            break_confidence_threshold=config.break_confidence_threshold,
        )
        windows, semantic_breaks = analyzer.analyze_continuity(transcript, windows)

    # ── Soft split — pasa el analyzer para scoring semántico del punto ────
    windows = _soft_split_dense_clips(windows, config, transcript, analyzer)

    # ── Phase 1.5: Merge — semantic breaks son HARD BLOCKS ───────────────
    clips = merge_clips(windows, config, semantic_breaks)

    # ── Métricas + tier de momento ────────────────────────────────────────
    for clip in clips:
        _finalize_metrics(clip, transcript)

    # ── Filtro en dos pasos (calidad + outliers virales) ──────────────────
    clips = _apply_filters(clips, config)
    clips = [c for c in clips
             if config.min_clip_duration <= (c["end"] - c["start"]) <= config.max_clip_duration]

    clips.sort(key=lambda c: c["start"])
    clips = clips[: config.max_candidates]

    for clip in clips:
        clip.pop("group_peaks", None)
        clip.pop("_momentum",   None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    tiers = {}
    for c in clips:
        tiers[c.get("moment_tier", "?")] = tiers.get(c.get("moment_tier", "?"), 0) + 1
    logger.info("Segments: %d → %s | tiers: %s", len(clips), output_path, tiers)
    return output_path
