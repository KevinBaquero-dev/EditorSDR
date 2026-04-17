import json
import logging
import math
import os
import re

import numpy as np

from .subtitle_engine import _build_srt  # noqa: F401

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/subtitles"

MAX_CHUNK_DURATION    = 2.5    # segundos máx por segmento antes de dividir
MAX_SUBTITLE_DURATION = 4.0    # hard cap: ningún chunk puede superar este valor
MAX_LINE_CHARS        = 38     # caracteres máx por línea
MAX_LINES             = 2      # líneas máx por segmento
SUBTITLE_OFFSET       = -0.2   # offset global (negativo = aparece antes)
CAPITALIZE            = True   # capitalizar primera letra de cada chunk

# ── silence trim ──────────────────────────────────────────────────────────────
_SILENCE_THRESHOLD = 0.015   # RMS por debajo de este valor = silencio
_TRIM_WINDOW_S     = 0.04    # ventana de análisis RMS (40ms)
_TRIM_STEP_S       = 0.02    # paso de escaneo (20ms)
_TRIM_MAX_SHIFT_S  = 0.35    # desplazamiento máximo permitido en start/end

# ── gap cleaner ───────────────────────────────────────────────────────────────
_GAP_MAX_TAIL_S    = 0.5     # gap mínimo antes del siguiente segmento

# ── highlight tuning ──────────────────────────────────────────────────────────

# Palabras excluidas del highlight.
# "no", "sí", "ya" se excluyen intencionalmente — son cortas y muy emocionales.
_STOPWORDS = {
    "de", "la", "el", "un", "una", "y", "a", "me", "te", "se", "lo", "le",
    "que", "en", "es", "con", "por", "para", "como", "pero",
    "o", "al", "del", "las", "los", "mi", "tu", "su", "más", "hay",
    "muy", "tan", "qué", "cómo", "porque", "cuando", "donde", "esto",
    "esta", "ese", "esa", "yo", "él", "tú", "va", "ha", "está",
}

_MAX_HIGHLIGHTS        = 3     # máx palabras destacadas por chunk
MAX_HIGHLIGHTS_PER_CLIP = 7    # presupuesto global por clip (A)
_REPEAT_MULTIPLIER     = 1.5   # penalización × por palabra repetida del chunk anterior
_SEMANTIC_MULTIPLIER   = 1.3   # penalización × por grupo semántico repetido (B)
_FORCE_HIGHLIGHT_MIN   = 0.60  # factor mínimo para activar highlight sin ! o ?
_MIN_HIGHLIGHT_SPACING = 2     # mínimo de chunks entre highlights (evita clusters, A)
_LOW_DENSITY_THRESHOLD = 0.30  # text_density debajo de este valor → highlight off (J)

# Grupos semánticos para diversidad — si ya se destacó uno del grupo,
# los demás reciben penalización en el siguiente chunk (B)
_SEMANTIC_GROUPS: tuple[frozenset, ...] = (
    frozenset({"no", "nada", "nunca", "jamás"}),
    frozenset({"me", "yo", "mi", "mío"}),
    frozenset({"sí", "claro", "obvio", "exacto"}),
)

# Offsets adaptativos por duración de segmento (E)
_OFFSET_SHORT  = -0.1   # segmentos 0.8s–1.2s
_OFFSET_VSHORT =  0.0   # segmentos < 0.8s


# ─── audio helpers ───────────────────────────────────────────────────────────

def _load_clip_audio(clip_path: str) -> tuple:
    """Carga audio mono float32 del clip. Devuelve (array, sr) o (None, None)."""
    try:
        import librosa
        audio, sr = librosa.load(clip_path, sr=None, mono=True)
        return audio, sr
    except Exception as exc:
        logger.debug("Audio load failed for %s: %s", clip_path, exc)
        return None, None


def _trim_silence(audio: np.ndarray, sr: int, start: float, end: float) -> tuple:
    """
    Recorta silencio inicial y final de un segmento usando RMS.
    Devuelve (new_start, new_end) dentro de los límites originales.
    """
    win   = max(1, int(_TRIM_WINDOW_S * sr))
    step  = max(1, int(_TRIM_STEP_S   * sr))
    limit = int(_TRIM_MAX_SHIFT_S * sr)

    s = int(start * sr)
    e = int(end   * sr)
    seg = audio[s : min(e, len(audio))]

    if len(seg) < win:
        return start, end

    # onset: primer frame con voz real
    new_start = start
    for i in range(0, min(len(seg) - win, limit), step):
        if np.sqrt(np.mean(seg[i : i + win] ** 2)) >= _SILENCE_THRESHOLD:
            new_start = start + i / sr
            break

    # offset: último frame con voz real
    new_end = end
    for i in range(len(seg) - win, max(0, len(seg) - win - limit), -step):
        if np.sqrt(np.mean(seg[i : i + win] ** 2)) >= _SILENCE_THRESHOLD:
            new_end = start + (i + win) / sr
            break

    if new_end - new_start < 0.1:  # no comprimir por debajo de 100ms
        return start, end

    return round(new_start, 3), round(new_end, 3)


def _enforce_max_duration(segments: list) -> list:
    """Divide por mitad cualquier chunk que supere MAX_SUBTITLE_DURATION."""
    result = []
    for seg in segments:
        if seg["end"] - seg["start"] <= MAX_SUBTITLE_DURATION:
            result.append(seg)
        else:
            mid   = round((seg["start"] + seg["end"]) / 2, 3)
            words = seg["text"].split()
            half  = max(1, len(words) // 2)
            result.append({"start": seg["start"], "end": mid,       "text": " ".join(words[:half])})
            result.append({"start": mid,          "end": seg["end"], "text": " ".join(words[half:])})
    return result


def _gap_cleaner(segments: list) -> list:
    """
    Dos pases:
    1. Solapamientos: trunca el final del segmento anterior.
    2. Gap vacío: si el gap con el siguiente es > _GAP_MAX_TAIL_S, el
       subtítulo ya terminó antes — no hace nada (la trim de silencio lo
       resuelve). Si el gap es negativo (overlap), lo corrige.
    """
    for i in range(len(segments) - 1):
        gap = segments[i + 1]["start"] - segments[i]["end"]
        if gap < 0:
            new_end = round(segments[i + 1]["start"] - 0.05, 3)
            segments[i]["end"] = max(round(segments[i]["start"] + 0.1, 3), new_end)
    return segments


# ─── estado compartido de highlight ──────────────────────────────────────────

def _hl_state_new() -> dict:
    """Crea estado compartido entre todos los chunks de un clip."""
    return {
        "budget":            MAX_HIGHLIGHTS_PER_CLIP,
        "recent_words":      set(),
        "recent_categories": set(),
        "last_hl_chunk":     -999,
        "chunk_idx":         0,
    }


# ─── helpers de texto ─────────────────────────────────────────────────────────

def _normalize(text: str, capitalize: bool = CAPITALIZE) -> str:
    """Trim, colapsa espacios múltiples, capitaliza si corresponde."""
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    if capitalize and text and text[0].isalpha() and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _is_highlighted(word: str) -> bool:
    """True si todos los caracteres alfanuméricos del token son mayúsculas (y len > 1)."""
    clean = re.sub(r"[^\w]", "", word)
    return bool(clean) and clean == clean.upper() and len(clean) > 1


def _word_category(word: str) -> str | None:
    """Devuelve el índice (como str) del grupo semántico al que pertenece la palabra, o None."""
    for i, group in enumerate(_SEMANTIC_GROUPS):
        if word in group:
            return str(i)
    return None


# ─── scoring de highlights ────────────────────────────────────────────────────

def _highlight_score(length: int, position: int, total: int) -> float:
    """
    Score de destacado: menor = más prioritario.
    sqrt(position) suaviza sesgo al inicio sin eliminarlo.
    """
    return (
        length
        + math.sqrt(position / max(total, 1))
        + max(0, length - 4) * 0.3
    )


# ─── aplicación de highlights ────────────────────────────────────────────────

def _highlight_keywords(text: str, enabled: bool = True,
                        intensity_factor: float = 0.0,
                        hl_state: dict | None = None) -> str:
    """
    Pone en mayúsculas palabras clave en el chunk aplicando:
    A  – spacing mínimo entre chunks con highlight
    B  – penalización multiplicativa por repetición de palabra y grupo semántico
    F  – límite seguro: no más de total_words // 2 highlights por chunk
    force progresivo – permite highlight sin ! o ? según intensity_factor
    """
    if hl_state is None:
        hl_state = _hl_state_new()

    chunk_idx = hl_state["chunk_idx"]
    hl_state["chunk_idx"] += 1

    has_punct = "!" in text or "?" in text

    if not enabled or hl_state["budget"] <= 0:
        hl_state["recent_words"].clear()
        hl_state["recent_categories"].clear()
        return text

    # A: spacing — si el gap desde el último highlight es insuficiente, saltar
    # (no borramos recent para que la memoria persista al próximo chunk)
    gap = chunk_idx - hl_state["last_hl_chunk"]
    if gap < _MIN_HIGHLIGHT_SPACING:
        return text

    if has_punct:
        chunk_max = _MAX_HIGHLIGHTS
    elif intensity_factor >= _FORCE_HIGHLIGHT_MIN:
        ratio = (intensity_factor - _FORCE_HIGHLIGHT_MIN) / (1.0 - _FORCE_HIGHLIGHT_MIN)
        chunk_max = max(1, round(ratio * _MAX_HIGHLIGHTS))
    else:
        hl_state["recent_words"].clear()
        hl_state["recent_categories"].clear()
        return text

    words = text.split()
    total = len(words)

    # F: límite seguro — no destacar más de la mitad de palabras del chunk
    chunk_max = min(chunk_max, hl_state["budget"], max(1, total // 2))

    candidates = []
    for i, word in enumerate(words):
        clean = re.sub(r"[^\w]", "", word.lower())
        if not clean or clean in _STOPWORDS or len(clean) <= 1:
            continue
        score = _highlight_score(len(clean), i, total)
        # B1: repetición de palabra exacta
        if clean in hl_state["recent_words"]:
            score *= _REPEAT_MULTIPLIER
        # B2: repetición de grupo semántico
        cat = _word_category(clean)
        if cat and cat in hl_state["recent_categories"]:
            score *= _SEMANTIC_MULTIPLIER
        candidates.append((score, i, clean, cat))

    if not candidates:
        return text

    candidates.sort()
    top            = candidates[:chunk_max]
    selected_idx   = {c[1] for c in top}
    selected_words = {c[2] for c in top}
    selected_cats  = {c[3] for c in top if c[3] is not None}

    hl_state["budget"]           -= len(selected_idx)
    hl_state["last_hl_chunk"]     = chunk_idx
    hl_state["recent_words"]      = selected_words
    hl_state["recent_categories"] = selected_cats

    return " ".join(w.upper() if i in selected_idx else w for i, w in enumerate(words))


# ─── wrap y control de líneas ─────────────────────────────────────────────────

def _wrap(text: str) -> str:
    """
    Divide en máximo MAX_LINES líneas de MAX_LINE_CHARS caracteres.
    C – smart ellipsis: solo añade … si overflow > 10% del límite Y la última
        palabra no está destacada.
    D – rebalanceo: si la diferencia entre líneas supera el 40%, redistribuye.
    """
    if len(text) <= MAX_LINE_CHARS:
        return text

    words = text.split()
    lines: list[str] = []
    current: list[str] = []

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

    # C: smart ellipsis
    placed = sum(len(ln.split()) for ln in lines)
    if placed < len(words):
        last_word    = lines[-1].split()[-1] if lines[-1].split() else ""
        overflow_len = sum(len(w) + 1 for w in words[placed:])
        if overflow_len > MAX_LINE_CHARS * 0.1 and not _is_highlighted(last_word):
            lines[-1] = lines[-1].rstrip() + "…"

    # D: rebalanceo de 2 líneas si diferencia > 40%
    if len(lines) == 2:
        l1, l2 = lines
        longer = max(len(l1), len(l2))
        if longer > 0 and abs(len(l1) - len(l2)) / longer > 0.4:
            all_w       = (l1 + " " + l2).split()
            total_chars = sum(len(w) for w in all_w) + len(all_w) - 1
            target      = total_chars // 2
            acc, split_at = 0, max(1, len(all_w) // 2)
            for j, w in enumerate(all_w):
                acc += len(w) + (1 if j > 0 else 0)
                if acc >= target:
                    split_at = max(1, min(j + 1, len(all_w) - 1))
                    break
            lines = [" ".join(all_w[:split_at]), " ".join(all_w[split_at:])]

    return "\n".join(lines)


# ─── coherencia visual (post-proceso) ────────────────────────────────────────

def _coherence_check(segments: list[dict]) -> list[dict]:
    """
    G: si >50% de las palabras de un chunk están en caps, conserva solo
    los primeros 2 highlights y restaura los demás a minúsculas.
    Trabaja línea a línea para preservar capitalización al inicio de cada línea.
    """
    for seg in segments:
        flat  = seg["text"].replace("\n", " ")
        words = flat.split()
        if not words:
            continue
        hl_count = sum(1 for w in words if _is_highlighted(w))
        if hl_count == 0 or hl_count / len(words) <= 0.5:
            continue

        kept      = 0
        new_lines = []
        for line in seg["text"].split("\n"):
            new_tokens = []
            for j, token in enumerate(line.split(" ")):
                if _is_highlighted(token):
                    if kept < 2:
                        new_tokens.append(token)
                        kept += 1
                    else:
                        restored = token.lower()
                        # restaurar capitalización al inicio de línea
                        if j == 0 and restored:
                            restored = restored[0].upper() + restored[1:]
                        new_tokens.append(restored)
                else:
                    new_tokens.append(token)
            new_lines.append(" ".join(new_tokens))
        seg["text"] = "\n".join(new_lines)

    return segments


# ─── estadísticas de highlight ────────────────────────────────────────────────

def _highlight_stats(segments: list[dict]) -> tuple[int, float]:
    """Devuelve (highlight_count, highlight_density) para los segmentos del clip."""
    total_words = hl_words = 0
    for seg in segments:
        words       = seg["text"].split()
        total_words += len(words)
        hl_words    += sum(1 for w in words if _is_highlighted(w))
    density = round(hl_words / total_words, 3) if total_words else 0.0
    return hl_words, density


# ─── chunking inteligente ─────────────────────────────────────────────────────

def _split_by_punctuation(text: str) -> list[str]:
    """
    Divide por puntuación natural (, . ! ? ;).
    Agrupa partes cortas para no crear chunks de 1-2 palabras sueltas.
    """
    parts = re.split(r"(?<=[,\.!?;])\s+", text)
    chunks: list[str] = []
    current = ""
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


def _chunk_segment(seg: dict, highlight: bool = True,
                   intensity_factor: float = 0.0,
                   hl_state: dict | None = None) -> list[dict]:
    """
    Divide un segmento largo en chunks de ~1-2s.
    `hl_state` se comparte entre llamadas consecutivas del mismo clip.
    """
    if hl_state is None:
        hl_state = _hl_state_new()

    text     = _normalize(seg["text"])
    duration = seg["end"] - seg["start"]

    def hl(t: str) -> str:
        return _highlight_keywords(t, highlight, intensity_factor, hl_state)

    if duration <= MAX_CHUNK_DURATION and len(text) <= MAX_LINE_CHARS * MAX_LINES:
        return [{"start": seg["start"], "end": seg["end"], "text": hl(text)}]

    parts = _split_by_punctuation(text)
    if len(parts) == 1:
        mid   = (seg["start"] + seg["end"]) / 2
        words = text.split()
        half  = len(words) // 2
        return [
            {"start": seg["start"],   "end": round(mid, 3), "text": hl(" ".join(words[:half]))},
            {"start": round(mid, 3),  "end": seg["end"],    "text": hl(" ".join(words[half:]))},
        ]

    total_chars = sum(len(p) for p in parts) or 1
    result, t   = [], seg["start"]
    for part in parts:
        chunk_end = round(t + duration * len(part) / total_chars, 3)
        result.append({"start": round(t, 3), "end": chunk_end, "text": hl(_normalize(part))})
        t = chunk_end

    result[-1]["end"] = seg["end"]
    return result


# ─── ajuste de segmentos ──────────────────────────────────────────────────────

def _adjusted_segments(clip_start: float, clip_end: float, transcript: list,
                       highlight: bool = True, features: dict | None = None,
                       audio: "np.ndarray | None" = None, sr: int | None = None) -> list[dict]:
    """
    Filtra, ajusta timestamps al tiempo relativo del clip y divide en chunks.
    E – micro-timing: offset adaptado a duración del segmento.
    Silence trim: si se pasa audio+sr, recorta inicio/fin de cada segmento
    al primer/último frame con voz real antes de chunkear.
    """
    features         = features or {}
    intensity_factor = (
        features.get("intensity", 0.0) + features.get("text_density", 0.0)
    ) / 2.0

    clip_dur   = clip_end - clip_start
    use_trim   = audio is not None and sr is not None
    raw: list[dict] = []

    for s in transcript:
        if s["start"] >= clip_end or s["end"] <= clip_start:
            continue
        # E: offset adaptativo según duración del segmento original
        seg_dur = s["end"] - s["start"]
        if seg_dur < 0.8:
            offset = _OFFSET_VSHORT
        elif seg_dur < 1.2:
            offset = _OFFSET_SHORT
        else:
            offset = SUBTITLE_OFFSET

        rel_start = max(0.0, s["start"] - clip_start + offset)
        rel_end   = min(clip_dur, s["end"]   - clip_start + offset)
        if rel_end <= rel_start:
            continue

        # Silence trim: ajustar al audio real del clip
        if use_trim:
            rel_start, rel_end = _trim_silence(audio, sr, rel_start, rel_end)
            if rel_end <= rel_start:
                continue

        raw.append({"start": round(rel_start, 3), "end": round(rel_end, 3), "text": s["text"]})

    hl_state = _hl_state_new()
    segments: list[dict] = []
    for seg in raw:
        segments.extend(_chunk_segment(seg, highlight, intensity_factor, hl_state))

    for seg in segments:
        seg["text"] = _wrap(seg["text"])

    # Hard cap: ningún chunk > MAX_SUBTITLE_DURATION
    segments = _enforce_max_duration(segments)

    # Gap cleaner: corrige solapamientos
    segments = _gap_cleaner(segments)

    # G: coherencia visual post-proceso
    segments = _coherence_check(segments)

    return segments


# ─── protección de edición ────────────────────────────────────────────────────

def _load_meta(meta_path: str) -> dict:
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ─── API pública ──────────────────────────────────────────────────────────────

def build_subtitles(refined_path: str, transcript_path: str,
                    clips_dir: str = "output/clips") -> str:
    """
    Genera archivos de subtítulos editables por clip.

    Para cada clip produce:
      clip_NNN.json      — fuente de verdad (editable)
      clip_NNN.srt       — formato render para FFmpeg
      clip_NNN_meta.json — estado de edición + estadísticas de highlight

    Nunca sobreescribe clips con subtitles_edited=true en su meta.
    J – desactiva highlight automáticamente si text_density < _LOW_DENSITY_THRESHOLD.
    Silence trim — si clips_dir existe, carga el audio de cada clip y recorta
    los timestamps de Whisper al inicio/fin real de la voz.
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

        features    = clip.get("features", {})
        highlight   = meta.get("highlight_enabled", True)

        # J: safety fallback — transcript pobre → sin highlights
        if features.get("text_density", 1.0) < _LOW_DENSITY_THRESHOLD:
            highlight = False

        # Silence trim: cargar audio del clip si está disponible
        audio, sr = None, None
        clip_audio_path = os.path.join(clips_dir, f"clip_{i:03d}.mp4")
        if os.path.exists(clip_audio_path):
            audio, sr = _load_clip_audio(clip_audio_path)
            if audio is not None:
                logger.debug(f"Clip {i:03d}: silence trim activo ({len(audio)/sr:.1f}s @ {sr}Hz)")
            else:
                logger.debug(f"Clip {i:03d}: silence trim omitido (no se pudo cargar audio)")
        else:
            logger.debug(f"Clip {i:03d}: silence trim omitido (clip no encontrado en {clips_dir})")

        segments = _adjusted_segments(
            clip["start"], clip["end"], transcript, highlight, features, audio, sr
        )

        if not segments:
            logger.warning(f"Clip {i:03d}: sin segmentos en el transcript")

        hl_count, hl_density = _highlight_stats(segments)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(_build_srt(segments))

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "subtitles_edited":  False,
                "highlight_enabled": highlight,
                "highlight_count":   hl_count,
                "highlight_density": hl_density,
                "edited_segments":   0,
                "clip_index":        i,
                "clip_start":        clip["start"],
                "clip_end":          clip["end"],
                "segment_count":     len(segments),
                "subtitle_offset":   SUBTITLE_OFFSET,
            }, f, indent=2)

        generated += 1
        logger.debug(f"Clip {i:03d}: {len(segments)} chunks | {hl_count} highlights ({hl_density:.0%}) -> {json_path}")

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
    meta      = _load_meta(meta_path)

    hl_count, hl_density = _highlight_stats(segments)
    meta["subtitles_edited"]  = True
    meta["highlight_count"]   = hl_count
    meta["highlight_density"] = hl_density
    meta["edited_segments"]   = len(segments)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"SRT regenerado desde JSON editado: {srt_path}")
    return srt_path


def subtitle_review_summary(output_dir: str = OUTPUT_DIR) -> dict:
    """
    H: Imprime resumen de estado antes del render y bloquea si hay clips sin editar.

    Uso típico en --review mode antes de lanzar el renderer:
        subtitle_review_summary()  # lanza ValueError si hay clips sin editar
    """
    import glob as _glob

    meta_files = sorted(_glob.glob(os.path.join(output_dir, "*_meta.json")))
    total = len(meta_files)
    if total == 0:
        raise FileNotFoundError(f"No se encontraron metas en {output_dir}")

    edited        = 0
    total_hl      = 0
    total_density = 0.0

    for mf in meta_files:
        with open(mf, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("subtitles_edited", False):
            edited += 1
        total_hl      += m.get("highlight_count", 0)
        total_density += m.get("highlight_density", 0.0)

    avg_hl      = round(total_hl      / total, 1)
    avg_density = round(total_density / total, 3)
    ready       = edited == total

    logger.info(
        "\n%s\n  SUBTITLE REVIEW SUMMARY\n"
        "  Clips generados:           %d\n"
        "  Editados manualmente:      %d/%d\n"
        "  Highlights promedio/clip:  %s\n"
        "  Densidad promedio:         %.1f%%\n"
        "  Estado:                    %s\n%s",
        "─" * 50, total, edited, total, avg_hl,
        avg_density * 100,
        "LISTO PARA RENDER" if ready else "PENDIENTE EDICIÓN",
        "─" * 50,
    )

    if not ready:
        raise ValueError(
            f"Render bloqueado: {total - edited} clip(s) sin editar. "
            "Edita los subtítulos con srt_from_json() antes de renderizar."
        )

    return {"total": total, "edited": edited,
            "avg_highlights": avg_hl, "avg_density": avg_density}
