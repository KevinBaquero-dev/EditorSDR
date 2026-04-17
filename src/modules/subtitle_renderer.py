import glob
import logging
import os
import shutil
import subprocess

from .subtitle_engine import _find_ffmpeg, _srt_path_escaped, _SUBTITLE_STYLE

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/subtitled"
SUBTITLES_DIR = "output/subtitles"


def render_subtitles(refined_path: str, vertical_dir: str) -> str:
    """
    Quema los SRT de output/subtitles/ sobre los clips verticales.

    Usa siempre el SRT presente en disco — que puede haber sido editado manualmente.
    Requiere que subtitle_builder haya corrido primero.

    Input:  refined_clips.json (para el conteo) + vertical_dir/vertical_NNN.mp4
    Output: output/subtitled/subtitled_NNN.mp4
    """
    if not os.path.exists(refined_path):
        raise FileNotFoundError(f"Input not found: {refined_path}")

    ffmpeg = _find_ffmpeg()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import json
    with open(refined_path, encoding="utf-8") as f:
        clips = json.load(f)

    if not clips:
        logger.warning("No clips to render subtitles for")
        return OUTPUT_DIR

    done, failed, skipped = 0, 0, 0

    for i, clip in enumerate(clips, start=1):
        srt_path  = os.path.join(SUBTITLES_DIR, f"clip_{i:03d}.srt")
        clip_file = os.path.join(vertical_dir, f"vertical_{i:03d}.mp4")
        out_file  = os.path.join(OUTPUT_DIR, f"subtitled_{i:03d}.mp4")

        if not os.path.exists(srt_path):
            logger.warning(f"Clip {i:03d}: SRT no encontrado — ejecuta subtitle_builder primero")
            skipped += 1
            continue

        if not os.path.exists(clip_file):
            logger.warning(f"Clip {i:03d}: video no encontrado: {clip_file}")
            failed += 1
            continue

        logger.info(f"[{i}/{len(clips)}] Rendering subtitles -> {out_file}")

        try:
            _burn(ffmpeg, clip_file, srt_path, out_file)
            done += 1
        except RuntimeError as e:
            logger.error(f"Clip {i:03d} render failed: {e}")
            failed += 1

    logger.info(
        f"Subtitle render done: {done} ok | {failed} failed | {skipped} skipped -> {OUTPUT_DIR}"
    )
    return OUTPUT_DIR


def _burn(ffmpeg: str, clip_path: str, srt_path: str, output_path: str) -> None:
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
            f"ffmpeg subtitle render failed: {result.stderr.decode(errors='replace')[-400:]}"
        )
