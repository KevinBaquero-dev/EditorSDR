import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)


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


def _build_srt(segments: list) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _sec_to_srt(seg["start"])
        end = _sec_to_srt(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text']}\n")
    return "\n".join(lines)


def _sec_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def burn_subtitles(
    clip_path: str,
    transcript_path: str,
    clip_start: float,
    clip_end: float,
    output_path: str,
) -> str:
    """
    Quema subtítulos en un clip. Requiere re-encode.

    clip_start / clip_end: timestamps del clip dentro del VOD original,
    usados para filtrar los segmentos del transcript.
    """
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    ffmpeg = _find_ffmpeg()

    with open(transcript_path, encoding="utf-8") as f:
        transcript = json.load(f)

    # Filtrar segmentos que caen dentro del clip y ajustar timestamps al inicio del clip
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
        logger.warning(f"No transcript segments found for clip {clip_start:.1f}–{clip_end:.1f}s")
        shutil.copy2(clip_path, output_path)
        return output_path

    srt_content = _build_srt(segments)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as tmp:
        tmp.write(srt_content)
        srt_path = tmp.name

    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", clip_path,
                "-vf", f"subtitles={srt_path}",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "fast",
                output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg subtitle burn failed: {result.stderr.decode(errors='replace')[-300:]}")
    finally:
        os.unlink(srt_path)

    logger.info(f"Subtitles burned: {output_path} ({len(segments)} segments)")
    return output_path
