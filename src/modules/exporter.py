import glob
import json
import logging
import os

import av

logger = logging.getLogger(__name__)


def _get_video_duration(video_path: str) -> float:
    try:
        with av.open(video_path) as container:
            return float(container.duration / av.time_base)
    except Exception:
        return 0.0


def export_pipeline(vod_dir: str, vod_url: str = "", vod_title: str = "") -> str:
    """
    Finaliza el pipeline escribiendo metadata.json en vod_dir.
    No copia archivos — ya están organizados en vod_dir desde el inicio.
    """
    vod_dir = os.path.abspath(vod_dir)
    os.makedirs(vod_dir, exist_ok=True)

    video_path = os.path.join(vod_dir, "raw", "vod.mp4")
    duration_sec = round(_get_video_duration(video_path)) if os.path.exists(video_path) else 0

    clips_dir = os.path.join(vod_dir, "clips")
    total_clips = len(glob.glob(os.path.join(clips_dir, "clip_*.mp4"))) if os.path.isdir(clips_dir) else 0

    vertical_dir = os.path.join(vod_dir, "vertical")
    total_vertical = len(glob.glob(os.path.join(vertical_dir, "vertical_*.mp4"))) if os.path.isdir(vertical_dir) else 0

    from datetime import date
    metadata = {
        "date": date.today().isoformat(),
        "vod_url": vod_url,
        "vod_title": vod_title,
        "total_clips": total_clips,
        "total_vertical": total_vertical,
        "duration_original_sec": duration_sec,
    }

    metadata_path = os.path.join(vod_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Export complete: {metadata}")
    return vod_dir
