import json
import logging
import os

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/selected"
TOP_N = 15
SCORE_THRESHOLD = 0.65
MIN_TEMPORAL_GAP = 30.0  # segundos mínimos entre inicios de clips seleccionados


def select_clips(ranked_path: str, output_dir: str = None) -> str:
    if not os.path.exists(ranked_path):
        raise FileNotFoundError(f"Input not found: {ranked_path}")

    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    output_path = os.path.join(out, "selected_clips.json")

    with open(ranked_path, encoding="utf-8") as f:
        ranked = json.load(f)

    if not ranked:
        logger.warning("No ranked clips to select from")
        with open(output_path, "w") as f:
            json.dump([], f, indent=2)
        return output_path

    # Fase 1: candidatos elegibles — score >= threshold o top N mínimo garantizado
    eligible = [c for c in ranked if c["score"] >= SCORE_THRESHOLD]
    if len(eligible) < min(TOP_N, len(ranked)):
        eligible = ranked[:TOP_N]

    # Fase 2: diversidad temporal — evitar ráfagas del mismo minuto
    selected = []
    for clip in eligible:
        too_close = any(
            abs(clip["start"] - sel["start"]) < MIN_TEMPORAL_GAP
            for sel in selected
        )
        if not too_close:
            selected.append(clip)
        if len(selected) >= TOP_N:
            break

    logger.info(
        f"Selection — eligible: {len(eligible)} | after temporal filter: {len(selected)} | "
        f"threshold: {SCORE_THRESHOLD} | top_n: {TOP_N}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    logger.info(f"Selected clips: {len(selected)} -> {output_path}")
    return output_path
