import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/candidates"
WINDOW_BEFORE = 10.0       # segundos de contexto antes del pico
WINDOW_AFTER = 15.0        # segundos después — más contexto que antes, las reacciones se extienden
MIN_DURATION = 8.0
MAX_DURATION = 60.0
MAX_CANDIDATES = 25
MERGE_OVERLAP_RATIO = 0.5  # merge si solapamiento > 50% del clip más corto


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
                f"Merged clips: [{last['start']:.1f}–{last['end']:.1f}] + "
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

    # peaks.json viene ordenado por intensidad — iteramos de mayor a menor
    for peak in peaks:
        ts = peak["timestamp"]

        start = max(0.0, ts - WINDOW_BEFORE)
        end = min(content_end, ts + WINDOW_AFTER)
        duration = end - start

        if duration < MIN_DURATION:
            logger.debug(f"Peak {ts:.1f}s skipped — window too short ({duration:.1f}s)")
            skipped += 1
            continue

        if duration > MAX_DURATION:
            end = start + MAX_DURATION

        nearest = _find_nearest_segment(ts, transcript)

        raw_candidates.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "peak_timestamp": round(ts, 3),
            "nearest_text": nearest["text"] if nearest else None,
        })

    if skipped:
        logger.info(f"Skipped {skipped} peaks (window < {MIN_DURATION}s)")

    candidates = _merge_overlapping(raw_candidates)
    candidates.sort(key=lambda c: c["start"])
    candidates = candidates[:MAX_CANDIDATES]

    # Duración final — post-merge puede quedar alguno fuera de rango
    before_filter = len(candidates)
    candidates = [c for c in candidates if MIN_DURATION <= (c["end"] - c["start"]) <= MAX_DURATION]
    if len(candidates) < before_filter:
        logger.warning(f"Filtered {before_filter - len(candidates)} candidates out of duration range after merge")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    logger.info(f"Candidates generated: {len(candidates)} → {output_path}")
    return output_path
