import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/candidates"
MIN_DURATION = 8.0
MAX_DURATION = 60.0
MAX_CANDIDATES = 25
MERGE_OVERLAP_RATIO = 0.5


def _get_window(intensity: float) -> tuple[float, float]:
    """Ventana dinámica según intensidad — más contexto para picos más fuertes."""
    if intensity > 0.8:
        return 15.0, 20.0
    elif intensity > 0.6:
        return 12.0, 18.0
    return 10.0, 15.0


def _extend_to_segment_end(peak_ts: float, end: float, transcript: list, max_end: float) -> float:
    """Extiende el end hasta el cierre del segmento de transcript en curso."""
    if not transcript:
        return end
    # Segmentos que ya empezaron antes del end actual y contienen el pico
    overlapping = [s for s in transcript if s["start"] <= end and s["end"] > peak_ts]
    if not overlapping:
        return end
    latest_end = max(s["end"] for s in overlapping)
    return round(min(latest_end + 0.3, max_end), 3)  # 0.3s de buffer de cierre


def _find_nearest_segment(peak_ts: float, transcript: list) -> dict | None:
    if not transcript:
        return None
    return min(transcript, key=lambda s: abs((s["start"] + s["end"]) / 2 - peak_ts))


def _merge_overlapping(candidates: list) -> list:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda c: c["start"])
    merged = [sorted_candidates[0].copy()]

    for current in sorted_candidates[1:]:
        last = merged[-1]
        overlap = min(last["end"], current["end"]) - max(last["start"], current["start"])

        if overlap <= 0:
            merged.append(current.copy())
            continue

        len_last = last["end"] - last["start"]
        len_current = current["end"] - current["start"]

        if overlap / min(len_last, len_current) > MERGE_OVERLAP_RATIO:
            last["end"] = max(last["end"], current["end"])
            logger.debug(
                f"Merged: [{last['start']:.1f}–{last['end']:.1f}] + "
                f"[{current['start']:.1f}–{current['end']:.1f}]"
            )
        else:
            merged.append(current.copy())

    return merged


def generate_clip_candidates(transcript_path: str, peaks_path: str) -> str:
    for path in (transcript_path, peaks_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "clips_candidates.json")

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    with open(peaks_path, encoding="utf-8") as f:
        peaks = json.load(f)

    if not peaks:
        logger.warning("No peaks — returning empty candidates list")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return output_path

    content_end = transcript[-1]["end"] if transcript else float("inf")

    raw_candidates = []
    skipped = 0

    for peak in peaks:
        ts = peak["timestamp"]
        intensity = peak.get("intensity", 0.5)

        before, after = _get_window(intensity)
        start = max(0.0, ts - before)
        end = min(content_end, ts + after)

        # Extender hasta cierre de segmento de transcript
        end = _extend_to_segment_end(ts, end, transcript, content_end)

        duration = end - start

        if duration < MIN_DURATION:
            logger.debug(f"Peak {ts:.1f}s skipped — window too short ({duration:.1f}s)")
            skipped += 1
            continue

        if duration > MAX_DURATION:
            end = round(start + MAX_DURATION, 3)

        nearest = _find_nearest_segment(ts, transcript)

        raw_candidates.append({
            "start": round(start, 3),
            "end": end,
            "peak_timestamp": round(ts, 3),
            "intensity": round(intensity, 4),
            "nearest_text": nearest["text"] if nearest else None,
        })

    if skipped:
        logger.info(f"Skipped {skipped} peaks (window < {MIN_DURATION}s)")

    candidates = _merge_overlapping(raw_candidates)
    candidates.sort(key=lambda c: c["start"])
    candidates = candidates[:MAX_CANDIDATES]

    before_filter = len(candidates)
    candidates = [c for c in candidates if MIN_DURATION <= (c["end"] - c["start"]) <= MAX_DURATION]
    if len(candidates) < before_filter:
        logger.warning(f"Filtered {before_filter - len(candidates)} post-merge out of range")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    logger.info(f"Candidates generated: {len(candidates)} → {output_path}")
    return output_path
