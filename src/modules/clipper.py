import glob
import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/clips"


def _find_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    # Fallback a winget install path
    pattern = os.path.expanduser(
        r"~\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg-*\bin\ffmpeg.exe"
    )
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "ffmpeg not found. Install with: winget install Gyan.FFmpeg"
    )


def _run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
    )


def _cut_clip(ffmpeg: str, video_path: str, start: float, duration: float, output_path: str) -> bool:
    """Intenta -c copy primero; si el archivo queda vacío o corrupto, re-encode."""
    args_copy = [
        ffmpeg,
        "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_path,
    ]
    result = _run_ffmpeg(args_copy)

    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
        return True

    # -c copy falló o produjo archivo sospechoso → re-encode
    logger.warning(f"stream copy failed for {output_path}, retrying with re-encode")
    if os.path.exists(output_path):
        os.remove(output_path)

    args_encode = [
        ffmpeg,
        "-y",
        "-i", video_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        output_path,
    ]
    result = _run_ffmpeg(args_encode)

    if result.returncode != 0:
        logger.error(f"re-encode failed: {result.stderr.decode(errors='replace')[-300:]}")
        return False

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        logger.error(f"re-encode produced empty file: {output_path}")
        return False

    return True


def generate_clips(video_path: str, candidates_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates not found: {candidates_path}")

    ffmpeg = _find_ffmpeg()
    logger.info(f"Using ffmpeg: {ffmpeg}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)

    if not candidates:
        logger.warning("No candidates — no clips to generate")
        return OUTPUT_DIR

    success = 0
    failed = 0

    for i, candidate in enumerate(candidates, start=1):
        start = candidate["start"]
        end = candidate["end"]
        duration = round(end - start, 3)
        output_path = os.path.join(OUTPUT_DIR, f"clip_{i:02d}.mp4")

        if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
            logger.info(f"clip_{i:02d} already exists, skipping")
            success += 1
            continue

        logger.info(f"Cutting clip_{i:02d}: {start:.1f}s → {end:.1f}s ({duration:.1f}s)")

        ok = _cut_clip(ffmpeg, video_path, start, duration, output_path)
        if ok:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"clip_{i:02d} OK ({size_mb:.1f} MB)")
            success += 1
        else:
            logger.error(f"clip_{i:02d} FAILED — skipping")
            failed += 1

    logger.info(f"Done: {success} clips OK | {failed} failed → {OUTPUT_DIR}")
    return OUTPUT_DIR
