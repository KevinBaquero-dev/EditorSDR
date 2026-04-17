import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

OUTPUT_DIR_SUBTITLED = "output/subtitled"

# Estilo para formato vertical (1080x1920)
# Valores en force_style de FFmpeg: escala relativa al alto del video
_SUBTITLE_STYLE = (
    "FontName=Arial,"
    "FontSize=72,"
    "PrimaryColour=&H00FFFFFF,"   # blanco
    "OutlineColour=&H00000000,"   # negro
    "BackColour=&H80000000,"      # sombra semitransparente
    "Bold=1,"
    "BorderStyle=1,"
    "Outline=4,"
    "Shadow=2,"
    "Alignment=2,"                # bottom center
    "MarginV=160"                 # px desde el borde inferior
)

_MAX_CHARS_PER_LINE = 28  # ~2 líneas limpias en 1080px con FontSize=72


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    pattern = os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError("ffmpeg not found. Install with: winget install Gyan.FFmpeg")


def _wrap_text(text: str, max_chars: int = _MAX_CHARS_PER_LINE) -> str:
    """Divide el texto en máximo 2 líneas sin cortar palabras."""
    if len(text) <= max_chars:
        return text
    words = text.split()
    line1, line2 = [], []
    for word in words:
        if sum(len(w) + 1 for w in line1) + len(word) <= max_chars:
            line1.append(word)
        else:
            line2.append(word)
    l1 = " ".join(line1)
    l2 = " ".join(line2)
    # Si línea 2 sigue siendo muy larga, truncar con elipsis
    if len(l2) > max_chars * 2:
        l2 = l2[: max_chars * 2 - 1] + "…"
    return f"{l1}\n{l2}" if l2 else l1


def _sec_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _sec_to_srt(seg["start"])
        end = _sec_to_srt(seg["end"])
        text = _wrap_text(seg["text"].strip())
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _srt_path_escaped(path: str) -> str:
    """Escapa la ruta para el filtro subtitles= de FFmpeg en Windows."""
    escaped = path.replace("\\", "/")
    # El filtro de FFmpeg requiere escapar el ":" del drive letter en Windows
    if len(escaped) >= 2 and escaped[1] == ":":
        escaped = escaped[0] + "\\:" + escaped[2:]
    return escaped


def burn_subtitles(
    clip_path: str,
    transcript_path: str,
    clip_start: float,
    clip_end: float,
    output_path: str,
) -> str:
    """
    Quema subtítulos estilizados en un clip ya cortado. Requiere re-encode.

    clip_start / clip_end: timestamps del clip dentro del VOD original,
    usados para filtrar y ajustar timestamps del transcript.
    """
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ffmpeg = _find_ffmpeg()

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    segments = [
        {
            "start": max(0.0, s["start"] - clip_start),
            "end": min(clip_end - clip_start, s["end"] - clip_start),
            "text": s["text"],
        }
        for s in transcript
        if s["start"] < clip_end and s["end"] > clip_start
    ]

    if not segments:
        logger.warning(f"No segments for clip {clip_start:.1f}-{clip_end:.1f}s — copying as-is")
        shutil.copy2(clip_path, output_path)
        return output_path

    srt_content = _build_srt(segments)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".srt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(srt_content)
        srt_path = tmp.name

    try:
        srt_escaped = _srt_path_escaped(srt_path)
        vf = f"subtitles='{srt_escaped}':force_style='{_SUBTITLE_STYLE}'"

        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", clip_path,
                "-vf", vf,
                "-c:v", "libx264",
                "-c:a", "copy",
                "-preset", "fast",
                "-crf", "23",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg subtitle burn failed: {result.stderr.decode(errors='replace')[-400:]}"
            )
    finally:
        os.unlink(srt_path)

    logger.info(f"Subtitles burned: {output_path} ({len(segments)} segments)")
    return output_path


def burn_subtitles_batch(
    refined_path: str,
    transcript_path: str,
    vertical_dir: str,
) -> str:
    """
    Aplica subtítulos a todos los clips verticales correspondientes a refined_clips.json.
    Input:  vertical_dir/vertical_001.mp4, vertical_002.mp4, ...
    Output: output/subtitled/subtitled_001.mp4, ...
    """
    for path in (refined_path, transcript_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    os.makedirs(OUTPUT_DIR_SUBTITLED, exist_ok=True)

    with open(refined_path, encoding="utf-8") as f:
        clips = json.load(f)

    if not clips:
        logger.warning("No clips to subtitle")
        return OUTPUT_DIR_SUBTITLED

    done, failed = 0, 0
    for i, clip in enumerate(clips, start=1):
        clip_file = os.path.join(vertical_dir, f"vertical_{i:03d}.mp4")
        out_file = os.path.join(OUTPUT_DIR_SUBTITLED, f"subtitled_{i:03d}.mp4")

        if not os.path.exists(clip_file):
            logger.warning(f"Clip file not found, skipping: {clip_file}")
            failed += 1
            continue

        logger.info(f"[{i}/{len(clips)}] Subtitling {clip['start']:.1f}-{clip['end']:.1f}s")
        try:
            burn_subtitles(clip_file, transcript_path, clip["start"], clip["end"], out_file)
            done += 1
        except RuntimeError as e:
            logger.error(f"Clip {i} subtitle failed: {e}")
            failed += 1

    logger.info(f"Subtitles batch done: {done} ok / {failed} failed -> {OUTPUT_DIR_SUBTITLED}")
    return OUTPUT_DIR_SUBTITLED
