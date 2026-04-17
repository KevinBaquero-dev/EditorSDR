import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/ranked"
HOOK_WINDOW_SHORT = 3.0
HOOK_WINDOW_EXTENDED = 5.0
TEXT_DENSITY_CAP = 15.0  # chars/sec — saturación para evitar sobrepremiar monólogos

W_INTENSITY = 0.30
W_TEXT_DENSITY = 0.25
W_HOOK_STRENGTH = 0.25
W_DURATION = 0.20
PHRASE_BONUS = 0.07       # bonus por clip con cierre de frase limpio
PHRASE_END_TOLERANCE = 0.5  # segundos de tolerancia para detectar cierre

_EMOTION_CHARS = "!?"


def _clip_text(start: float, end: float, transcript: list) -> str:
    segs = [s for s in transcript if s["start"] < end and s["end"] > start]
    return " ".join(s["text"].strip() for s in segs)


def _hook_text_windowed(clip_start: float, transcript: list, window: float) -> str:
    segs = [s for s in transcript if s["start"] < clip_start + window and s["end"] > clip_start]
    return " ".join(s["text"].strip() for s in segs)


def _energy_proxy(clip_start: float, peaks: list, window: float = 5.0) -> float:
    """Intensidad promedio de picos dentro del window inicial del clip."""
    nearby = [p for p in peaks if clip_start <= p["timestamp"] < clip_start + window]
    if not nearby:
        return 0.0
    return sum(p["intensity"] for p in nearby) / len(nearby)


def _duration_score(duration: float) -> float:
    """Campana suavizada: plateau en 20–45s, rampas en los extremos."""
    if duration < 8.0:
        return 0.0
    if duration < 20.0:
        return (duration - 8.0) / 12.0
    if duration <= 45.0:
        return 1.0
    if duration <= 60.0:
        return (60.0 - duration) / 15.0
    return 0.0


def _phrase_complete(end: float, transcript: list) -> bool:
    """True si el final del clip coincide con el cierre de un segmento."""
    return any(abs(s["end"] - end) < PHRASE_END_TOLERANCE for s in transcript)


def score_clips(candidates_path: str, transcript_path: str, peaks_path: str) -> str:
    for path in (candidates_path, transcript_path, peaks_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "clips_ranked.json")

    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)
    with open(peaks_path, encoding="utf-8") as f:
        peaks = json.load(f)

    if not candidates:
        logger.warning("No candidates to score — writing empty ranked list")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return output_path

    raw = []
    for c in candidates:
        start, end = c["start"], c["end"]
        duration = end - start
        peak_ts = c.get("peak_timestamp", (start + end) / 2)

        intensity = c.get("intensity")
        if intensity is None:
            nearest = min(peaks, key=lambda p: abs(p["timestamp"] - peak_ts), default=None)
            intensity = nearest["intensity"] if nearest else 0.5

        # text_density con saturación
        body = _clip_text(start, end, transcript)
        text_density_raw = min(len(body) / duration, TEXT_DENSITY_CAP) / TEXT_DENSITY_CAP if duration > 0 else 0.0

        # hook con fallback: 3s → 5s → energy proxy
        hook_short = _hook_text_windowed(start, transcript, HOOK_WINDOW_SHORT)
        if hook_short.strip():
            hook_density_raw = len(hook_short) / HOOK_WINDOW_SHORT
            emotion_bonus = sum(hook_short.count(ch) for ch in _EMOTION_CHARS)
            hook_energy_fallback = None
        else:
            hook_ext = _hook_text_windowed(start, transcript, HOOK_WINDOW_EXTENDED)
            if hook_ext.strip():
                hook_density_raw = len(hook_ext) / HOOK_WINDOW_EXTENDED
                emotion_bonus = sum(hook_ext.count(ch) for ch in _EMOTION_CHARS)
                hook_energy_fallback = None
            else:
                hook_density_raw = 0.0
                emotion_bonus = 0
                hook_energy_fallback = _energy_proxy(start, peaks, HOOK_WINDOW_EXTENDED)

        raw.append({
            "start": start,
            "end": end,
            "peak_timestamp": peak_ts,
            "duration": duration,
            "intensity": float(intensity),
            "text_density_raw": text_density_raw,
            "hook_density_raw": hook_density_raw,
            "emotion_bonus": emotion_bonus,
            "hook_energy_fallback": hook_energy_fallback,
            "duration_score": _duration_score(duration),
            "phrase_complete": _phrase_complete(end, transcript),
        })

    # text_density ya está normalizada (saturación aplicada) — max es 1.0 siempre
    max_hd = max(r["hook_density_raw"] for r in raw) or 1.0
    max_eb = max(r["emotion_bonus"] for r in raw) or 1.0

    ranked = []
    for r in raw:
        text_density = r["text_density_raw"]  # ya en [0,1] por saturación

        if r["hook_energy_fallback"] is not None:
            hook_strength = 0.5 * r["hook_energy_fallback"]
        else:
            hook_strength = 0.7 * (r["hook_density_raw"] / max_hd) + 0.3 * (r["emotion_bonus"] / max_eb)

        score = (
            r["intensity"] * W_INTENSITY
            + text_density * W_TEXT_DENSITY
            + hook_strength * W_HOOK_STRENGTH
            + r["duration_score"] * W_DURATION
        )

        if r["phrase_complete"]:
            score = min(score + PHRASE_BONUS, 1.0)

        ranked.append({
            "start": r["start"],
            "end": r["end"],
            "peak_timestamp": r["peak_timestamp"],
            "score": round(score, 4),
            "features": {
                "intensity": round(r["intensity"], 4),
                "text_density": round(text_density, 4),
                "hook_strength": round(hook_strength, 4),
                "duration_score": round(r["duration_score"], 4),
                "phrase_complete": r["phrase_complete"],
            },
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    scores = [r["score"] for r in ranked]
    spread = max(scores) - min(scores)
    logger.info(
        f"Score distribution — max: {max(scores):.3f} | min: {min(scores):.3f} | "
        f"avg: {sum(scores)/len(scores):.3f} | spread: {spread:.3f}"
    )
    if spread < 0.1:
        logger.warning("Low spread (<0.1) — features may not differentiate clips enough")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)

    logger.info(f"Clips ranked: {len(ranked)} -> {output_path}")
    return output_path
