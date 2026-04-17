import json
import logging
import os
import re

from .subtitle_engine import _build_srt  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/subtitles"

MAX_CHUNK_DURATION = 2.5   # segundos máx por segmento antes de dividir
MAX_LINE_CHARS = 42        # caracteres máx por línea
MAX_LINES = 2              # líneas máx por segmento
SUBTITLE_OFFSET = -0.2     # offset global en segundos (negativo = aparece antes)


# ─── normalización de texto ───────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Trim, colapsa espacios múltiples, capitaliza primera letra."""
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    if text:
        text = text[0].upper() + text[1:]
    return text


def _wrap(text: str) -> str:
    """
    Divide el texto en máximo MAX_LINES líneas de máximo MAX_LINE_CHARS caracteres.
    No corta palabras. Trunca con … si no cabe en 2 líneas.
    """
    if len(text) <= MAX_LINE_CHARS:
        return text

    words = text.split()
    lines, current = [], []
    for word in words:
        current_len = sum(len(w) + 1 for w in current) - 1 if current else 0
        if current_len + (1 if current else 0) + len(word) <= MAX_LINE_CHARS:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
            if len(lines) >= MAX_LINES:
                break

    if current and len(lines) < MAX_LINES:
        lines.append(" ".join(current))

    result = "\n".join(lines)
    # Truncar si todavía hay más texto que no cupo
    remaining = words[sum(len(l.split()) for l in lines):]
    if remaining:
        last = lines[-1]
        lines[-1] = last.rstrip() + "…"
        result = "\n".join(lines)

    return result


# ─── chunking inteligente ─────────────────────────────────────────────────────

def _split_by_punctuation(text: str) -> list[str]:
    """
    Divide el texto en partes por puntuación natural (,  .  !  ?  ;).
    Agrupa partes cortas para no crear chunks de 1-2 palabras sueltas.
    """
    parts = re.split(r"(?<=[,\.!?;])\s+", text)
    chunks, current = [], ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + 1 + len(part) <= MAX_LINE_CHARS:
            current += " " + part
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks if len(chunks) > 1 else [text]


def _chunk_segment(seg: dict) -> list[dict]:
    """
    Divide un segmento largo en chunks de ~1-2s.
    Estrategia: puntuación primero, luego split temporal proporcional al texto.
    """
    text = _normalize(seg["text"])
    duration = seg["end"] - seg["start"]

    if duration <= MAX_CHUNK_DURATION and len(text) <= MAX_LINE_CHARS * MAX_LINES:
        return [{"start": seg["start"], "end": seg["end"], "text": text}]

    parts = _split_by_punctuation(text)
    if len(parts) == 1:
        # No hay puntuación — dividir a la mitad por tiempo
        mid = (seg["start"] + seg["end"]) / 2
        words = text.split()
        half = len(words) // 2
        return [
            {"start": seg["start"], "end": round(mid, 3), "text": " ".join(words[:half])},
            {"start": round(mid, 3), "end": seg["end"], "text": " ".join(words[half:])},
        ]

    # Distribuir tiempo proporcional a la longitud de cada parte
    total_chars = sum(len(p) for p in parts) or 1
    result, t = [], seg["start"]
    for part in parts:
        ratio = len(part) / total_chars
        chunk_end = round(t + duration * ratio, 3)
        result.append({"start": round(t, 3), "end": chunk_end, "text": _normalize(part)})
        t = chunk_end

    # Asegurar que el último chunk termine exactamente donde termina el segmento
    result[-1]["end"] = seg["end"]
    return result


# ─── ajuste de segmentos ──────────────────────────────────────────────────────

def _adjusted_segments(clip_start: float, clip_end: float, transcript: list) -> list[dict]:
    """
    Filtra, ajusta timestamps al tiempo relativo del clip, aplica offset,
    y divide segmentos largos en chunks.
    """
    clip_dur = clip_end - clip_start
    raw = []

    for s in transcript:
        if s["start"] >= clip_end or s["end"] <= clip_start:
            continue
        rel_start = max(0.0, s["start"] - clip_start + SUBTITLE_OFFSET)
        rel_end   = min(clip_dur, s["end"]   - clip_start + SUBTITLE_OFFSET)
        if rel_end <= rel_start:
            continue
        raw.append({"start": round(rel_start, 3), "end": round(rel_end, 3), "text": s["text"]})

    # Chunking + normalización
    segments = []
    for seg in raw:
        segments.extend(_chunk_segment(seg))

    # Aplicar wrap a texto final de cada chunk
    for seg in segments:
        seg["text"] = _wrap(seg["text"])

    return segments


# ─── protección de edición ────────────────────────────────────────────────────

def _load_meta(meta_path: str) -> dict:
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── API pública ──────────────────────────────────────────────────────────────

def build_subtitles(refined_path: str, transcript_path: str) -> str:
    """
    Genera archivos de subtítulos editables por clip.

    Para cada clip produce:
      clip_NNN.json      — fuente de verdad (editable)
      clip_NNN.srt       — formato render para FFmpeg
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
        base      = os.path.join(OUTPUT_DIR, f"clip_{i:03d}")
        json_path = f"{base}.json"
        srt_path  = f"{base}.srt"
        meta_path = f"{base}_meta.json"

        meta = _load_meta(meta_path)
        if meta.get("subtitles_edited", False):
            logger.info(f"Clip {i:03d}: editado manualmente — sin cambios")
            skipped += 1
            continue

        segments = _adjusted_segments(clip["start"], clip["end"], transcript)

        if not segments:
            logger.warning(f"Clip {i:03d}: sin segmentos en el transcript")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(_build_srt(segments))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "subtitles_edited": False,
                "clip_index": i,
                "clip_start": clip["start"],
                "clip_end": clip["end"],
                "segment_count": len(segments),
                "subtitle_offset": SUBTITLE_OFFSET,
            }, f, indent=2)

        generated += 1
        logger.debug(f"Clip {i:03d}: {len(segments)} chunks -> {json_path}")

    logger.info(
        f"Subtitles built: {generated} generated | {skipped} skipped (edited) -> {OUTPUT_DIR}"
    )
    return OUTPUT_DIR


def srt_from_json(json_path: str) -> str:
    """
    Regenera el SRT desde el JSON editado manualmente.
    Marca el clip como subtitles_edited=true en su meta.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    srt_path = json_path.replace(".json", ".srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(_build_srt(segments))

    meta_path = json_path.replace(".json", "_meta.json")
    meta = _load_meta(meta_path)
    meta["subtitles_edited"] = True
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SRT regenerado desde JSON editado: {srt_path}")
    return srt_path
