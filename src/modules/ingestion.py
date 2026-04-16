import logging
import os

import yt_dlp

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "output/raw"


def download_vod(url: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "vod.mp4")

    # Skip if already downloaded — VODs de 3h son costosos de re-descargar
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"VOD already exists, skipping download: {output_path}")
        return output_path

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, "vod.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "no_part": True,  # evita archivos .part sueltos en caso de interrupción
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError("Could not retrieve video info — VOD may be private or unavailable")
            logger.info(f"Downloading: {info.get('title', 'unknown')} ({info.get('duration', '?')}s)")
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"Download failed: {e}") from e

    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Expected output not found after download: {output_path}. "
            "Check that the VOD is public and the URL is correct."
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    if size_mb < 1:
        raise RuntimeError(
            f"Output file is suspiciously small ({size_mb:.1f} MB) — "
            "download may have failed silently. VOD might be private."
        )

    logger.info(f"Download complete: {output_path} ({size_mb:.1f} MB)")
    return output_path
