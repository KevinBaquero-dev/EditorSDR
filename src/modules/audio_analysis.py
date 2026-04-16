import json
import logging
import os

import librosa
import numpy as np
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/analysis"
SAMPLE_RATE = 16000      # suficiente para energía — conserva RAM en videos de 3h+
HOP_LENGTH = 512
FRAME_LENGTH = 2048
MAX_PEAKS = 50
MIN_PEAK_DISTANCE_SEC = 2.0  # evita picos agrupados en el mismo grito


def analyze_audio(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "peaks.json")

    logger.info(f"Loading audio: {video_path}")
    y, sr = librosa.load(video_path, sr=SAMPLE_RATE, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    logger.info(f"Duration: {duration:.1f}s | Sample rate: {sr}Hz")

    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    mean_energy = float(np.mean(rms))
    std_energy = float(np.std(rms))
    threshold = mean_energy + std_energy
    logger.info(f"Energy — mean: {mean_energy:.6f} | std: {std_energy:.6f} | threshold: {threshold:.6f}")

    min_distance_frames = max(1, int(MIN_PEAK_DISTANCE_SEC * sr / HOP_LENGTH))
    peak_indices, _ = find_peaks(rms, height=threshold, distance=min_distance_frames)

    if len(peak_indices) == 0:
        logger.warning("No peaks detected above threshold — returning empty peaks list")
        peaks = []
    else:
        times = librosa.frames_to_time(peak_indices, sr=sr, hop_length=HOP_LENGTH)
        intensities = rms[peak_indices]

        max_intensity = float(np.max(intensities))
        norm_intensities = (intensities / max_intensity) if max_intensity > 0 else intensities

        peaks = [
            {"timestamp": round(float(t), 3), "intensity": round(float(i), 4)}
            for t, i in zip(times, norm_intensities)
        ]

        peaks.sort(key=lambda x: x["intensity"], reverse=True)
        peaks = peaks[:MAX_PEAKS]

        logger.info(f"Peaks detected: {len(peak_indices)} total | keeping top {len(peaks)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(peaks, f, indent=2)

    logger.info(f"Peaks saved: {output_path}")
    return output_path
