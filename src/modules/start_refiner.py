import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/refined"
REFINE_MAX_ADJUST = 8.0      # máximo de segundos que podemos mover el inicio
SILENCE_GAP = 1.5            # si no hay texto en los primeros Xs → considera silencio
PRE_BUFFER = 0.15            # buffer antes del inicio de segmento para no arrancar justo en la vocal
MIN_CLIP_DURATION = 8.0


def _refined_start(clip_start: float, peak_ts: float, transcript: list) -> tuple[float, str]:
    """
    Devuelve (nuevo_start, reason).

    Casos:
    1. silence_skip   — clip empieza en silencio, mueve al inicio del siguiente texto
    2. phrase_align   — clip empieza a mitad de frase, retrocede al inicio de esa frase
    3. no_change      — inicio ya es limpio
    """
    # Segmentos activos o que empiezan dentro de la ventana de silencio
    covering = [
        s for s in transcript
        if s["start"] < clip_start + SILENCE_GAP and s["end"] > clip_start
    ]

    if not covering:
        # Caso 1: silencio al inicio — buscar siguiente segmento antes del peak
        upcoming = [s for s in transcript if clip_start < s["start"] < peak_ts]
        if upcoming:
            next_seg = min(upcoming, key=lambda s: s["start"])
            new_start = max(0.0, next_seg["start"] - PRE_BUFFER)
            # Verificar que no rompamos duración mínima
            if peak_ts - new_start < MIN_CLIP_DURATION:
                return clip_start, "no_change"
            return new_start, "silence_skip"
        return clip_start, "no_change"

    # Caso 2: clip empieza a mitad de frase (el segmento empezó antes del clip_start)
    mid_sentence = next((s for s in covering if s["start"] < clip_start - 0.3), None)
    if mid_sentence:
        candidate = max(0.0, mid_sentence["start"] - PRE_BUFFER)
        delta = clip_start - candidate
        if 0 < delta <= REFINE_MAX_ADJUST:
            return candidate, "phrase_align"

    # Caso 3: inicio ya limpio
    return clip_start, "no_change"


def refine_starts(selected_path: str, transcript_path: str) -> str:
    for path in (selected_path, transcript_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "refined_clips.json")

    with open(selected_path, encoding="utf-8") as f:
        selected = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    if not selected:
        logger.warning("No selected clips to refine")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return output_path

    refined = []
    stats = {"no_change": 0, "silence_skip": 0, "phrase_align": 0}

    for clip in selected:
        original_start = clip["start"]
        peak_ts = clip.get("peak_timestamp", (clip["start"] + clip["end"]) / 2)

        new_start, reason = _refined_start(original_start, peak_ts, transcript)

        entry = dict(clip)
        entry["start"] = round(new_start, 3)
        entry["refinement"] = {
            "original_start": original_start,
            "delta": round(original_start - new_start, 3),
            "reason": reason,
        }

        refined.append(entry)
        stats[reason] += 1

        if reason != "no_change":
            logger.debug(
                f"peak={peak_ts:.0f}s [{reason}]: {original_start:.2f}s -> {new_start:.2f}s "
                f"(delta={original_start - new_start:+.2f}s)"
            )

    logger.info(
        f"Refinement — total: {len(refined)} | "
        f"phrase_align: {stats['phrase_align']} | "
        f"silence_skip: {stats['silence_skip']} | "
        f"no_change: {stats['no_change']}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(refined, f, ensure_ascii=False, indent=2)

    logger.info(f"Refined clips: {len(refined)} -> {output_path}")
    return output_path
