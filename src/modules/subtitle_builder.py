import json
import logging
import os

from .subtitle_engine import _build_srt, _sec_to_srt, _wrap_text  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/subtitles"


def _adjusted_segments(clip_start: float, clip_end: float, transcript: list) -> list:
    """Filtra y ajusta timestamps al tiempo relativo del clip."""
    return [
        {
            "start": round(max(0.0, s["start"] - clip_start), 3),
            "end": round(min(clip_end - clip_start, s["end"] - clip_start), 3),
            "text": s["text"].strip(),
        }
        for s in transcript
        if s["start"] < clip_end and s["end"] > clip_start
    ]


def _load_meta(meta_path: str) -> dict:
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_subtitles(refined_path: str, transcript_path: str) -> str:
    """
    Genera archivos de subtítulos editables por clip.

    Para cada clip en refined_clips.json produce:
      clip_NNN.json  — fuente de verdad (editable en cualquier editor)
      clip_NNN.srt   — formato render para FFmpeg
      clip_NNN_meta.json — estado de edición

    Nunca sobreescribe clips con subtitles_edited=true en su meta.
    """
    for path in (refined_path, transcript_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(refined_path, encoding="utf-8") as f:
        clips = json.load(f)
    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    if not clips:
        logger.warning("No clips to build subtitles for")
        return OUTPUT_DIR

    generated, skipped = 0, 0

    for i, clip in enumerate(clips, start=1):
        base = os.path.join(OUTPUT_DIR, f"clip_{i:03d}")
        json_path = f"{base}.json"
        srt_path  = f"{base}.srt"
        meta_path = f"{base}_meta.json"

        # Respetar edición manual
        meta = _load_meta(meta_path)
        if meta.get("subtitles_edited", False):
            logger.info(f"Clip {i:03d}: editado manualmente — sin cambios")
            skipped += 1
            continue

        segments = _adjusted_segments(clip["start"], clip["end"], transcript)

        if not segments:
            logger.warning(f"Clip {i:03d}: sin segmentos de transcript")

        # JSON — fuente de verdad
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        # SRT — formato render
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(_build_srt(segments))

        # Metadata
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "subtitles_edited": False,
                "clip_index": i,
                "clip_start": clip["start"],
                "clip_end": clip["end"],
                "segment_count": len(segments),
            }, f, indent=2)

        generated += 1
        logger.debug(f"Clip {i:03d}: {len(segments)} segments -> {json_path}")

    logger.info(
        f"Subtitles built: {generated} generated | {skipped} skipped (edited) -> {OUTPUT_DIR}"
    )
    return OUTPUT_DIR


def srt_from_json(json_path: str) -> str:
    """
    Regenera el SRT desde el JSON editado.
    Llama esto después de editar clip_NNN.json manualmente.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    srt_path = json_path.replace(".json", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_build_srt(segments))

    # Marcar como editado en meta
    meta_path = json_path.replace(".json", "_meta.json")
    meta = _load_meta(meta_path)
    meta["subtitles_edited"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SRT regenerado desde JSON editado: {srt_path}")
    return srt_path
