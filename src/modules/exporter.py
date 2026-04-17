import glob
import json
import logging
import os
import shutil
from datetime import date

import av

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: str) -> float:
    try:
        with av.open(video_path) as container:
            return float(container.duration / av.time_base)
    except Exception:
        return 0.0


def _resolve_export_dir(base: str, date_str: str) -> str:
    candidate = os.path.join(base, date_str)
    if not os.path.exists(candidate):
        return candidate
    suffix = 2
    while os.path.exists(f"{candidate}_{suffix}"):
        suffix += 1
    return f"{candidate}_{suffix}"


def _copy_dir(src: str, dst: str) -> int:
    """Copia todos los archivos de src a dst. Retorna cantidad copiada."""
    if not os.path.isdir(src):
        return 0
    os.makedirs(dst, exist_ok=True)
    copied = 0
    for filename in os.listdir(src):
        src_file = os.path.join(src, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(dst, filename))
            copied += 1
    return copied


def export_pipeline(output_base_path: str) -> str:
    output_base_path = os.path.abspath(output_base_path)
    date_str = date.today().isoformat()
    export_dir = _resolve_export_dir(output_base_path, date_str)

    logger.info(f"Exporting to: {export_dir}")
    os.makedirs(export_dir, exist_ok=True)

    folders = ["raw", "transcripts", "analysis", "candidates", "clips"]
    total_copied = 0

    for folder in folders:
        src = os.path.join(output_base_path, folder)
        dst = os.path.join(export_dir, folder)
        n = _copy_dir(src, dst)
        if n:
            logger.info(f"  {folder}/: {n} archivo(s) copiado(s)")
        total_copied += n

    # Metadata
    video_path = os.path.join(output_base_path, "raw", "vod.mp4")
    duration_sec = round(_get_video_duration(video_path)) if os.path.exists(video_path) else 0

    clips_dir = os.path.join(output_base_path, "clips")
    total_clips = len(glob.glob(os.path.join(clips_dir, "clip_*.mp4"))) if os.path.isdir(clips_dir) else 0

    metadata = {
        "date": date_str,
        "total_clips": total_clips,
        "duration_original_sec": duration_sec,
        "notes": "Pipeline MVP run",
    }

    metadata_path = os.path.join(export_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"metadata.json: {metadata}")
    logger.info(f"Export complete: {total_copied} archivos → {export_dir}")

    return export_dir
