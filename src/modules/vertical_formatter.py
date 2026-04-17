import glob
import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/vertical"
OUTPUT_W = 1080
OUTPUT_H = 1920

# center | left | right
CROP_POSITION = "center"
# Offset en píxeles desde la posición base (+ = derecha, - = izquierda)
# Útil para streams donde el personaje está desplazado del centro
CROP_OFFSET_PX = 0


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


def _crop_x_expr(position: str, offset_px: int) -> str:
    """
    Expresión FFmpeg para la coordenada X del crop.
    Usa iw/ih como variables del filtro — funciona con cualquier resolución.
    Nota: evitar min()/max() porque las comas dentro de funciones rompen el filtergraph.
    """
    base = {
        "center": "(iw-ih*9/16)/2",
        "left":   "0",
        "right":  "iw-ih*9/16",
    }.get(position, "(iw-ih*9/16)/2")

    if offset_px == 0:
        return base
    return f"({base})+({offset_px})"


def _cut_and_crop(
    ffmpeg: str,
    video_path: str,
    start: float,
    duration: float,
    output_path: str,
    position: str = CROP_POSITION,
    offset_px: int = CROP_OFFSET_PX,
) -> None:
    x_expr = _crop_x_expr(position, offset_px)
    crop_filter = f"crop=ih*9/16:ih:{x_expr}:0,scale={OUTPUT_W}:{OUTPUT_H}"

    result = subprocess.run(
        [
            ffmpeg, "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-vf", crop_filter,
            "-c:v", "libx264",
            "-c:a", "aac",
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
            f"ffmpeg vertical encode failed: {result.stderr.decode(errors='replace')[-400:]}"
        )


def format_vertical(
    refined_path: str,
    video_path: str,
    position: str = CROP_POSITION,
    offset_px: int = CROP_OFFSET_PX,
) -> str:
    for path in (refined_path, video_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input not found: {path}")

    ffmpeg = _find_ffmpeg()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(refined_path, encoding="utf-8") as f:
        clips = json.load(f)

    if not clips:
        logger.warning("No refined clips to format")
        return OUTPUT_DIR

    if position != "center" or offset_px != 0:
        logger.info(f"Crop config: position={position} offset={offset_px:+d}px")

    output_paths = []
    for i, clip in enumerate(clips, start=1):
        start = clip["start"]
        end = clip["end"]
        duration = round(end - start, 3)
        score = clip.get("score", 0)
        out_file = os.path.join(OUTPUT_DIR, f"vertical_{i:03d}.mp4")

        logger.info(f"[{i}/{len(clips)}] {start:.1f}-{end:.1f}s ({duration:.0f}s) score={score:.3f}")

        try:
            _cut_and_crop(ffmpeg, video_path, start, duration, out_file, position, offset_px)
            output_paths.append(out_file)
        except RuntimeError as e:
            logger.error(f"Clip {i} failed: {e}")

    logger.info(f"Vertical clips done: {len(output_paths)}/{len(clips)} -> {OUTPUT_DIR}")
    return OUTPUT_DIR
