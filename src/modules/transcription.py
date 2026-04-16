import json
import logging
import os

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/transcripts"
MODEL_SIZE = "small"  # balance velocidad/precisión para RTX 4060 8GB en MVP


def transcribe_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "transcript.json")

    device = "cuda"
    compute_type = "float16"

    try:
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        logger.info(f"Model loaded: {MODEL_SIZE} on {device}")
    except Exception:
        logger.warning("CUDA unavailable, falling back to CPU (int8)")
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    logger.info(f"Transcribing: {video_path}")
    segments_iter, info = model.transcribe(video_path, beam_size=5)

    logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    transcript = [
        {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": seg.text.strip()}
        for seg in segments_iter
        if seg.text.strip()
    ]

    if not transcript:
        raise RuntimeError("Transcription produced no segments — video may have no audio or be silent")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    logger.info(f"Transcript saved: {output_path} ({len(transcript)} segments)")
    return output_path
