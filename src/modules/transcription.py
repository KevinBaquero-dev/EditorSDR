import json
import logging
import os

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/transcripts"
MODEL_SIZE = "small"  # balance velocidad/precisión para RTX 4060 8GB en MVP

# Segmentos que Whisper genera cuando no hay voz clara — no aportan nada
_JUNK_TEXT = {"", "...", "…", ".", ",", "- ", "[ Música ]", "[Música]", "[Music]", "[Applause]"}

LONG_SEGMENT_WARN_SEC = 30.0  # segmentos >30s rompen el clipper — se re-segmentan en clip_candidates


def transcribe_video(video_path: str) -> str:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "transcript.json")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logger.info(f"Transcript already exists, skipping: {output_path}")
        return output_path

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

    transcript = []
    long_segments = 0

    for seg in segments_iter:
        text = seg.text.strip()
        if not text or text in _JUNK_TEXT:
            continue

        duration = seg.end - seg.start
        if duration > LONG_SEGMENT_WARN_SEC:
            long_segments += 1
            logger.warning(f"Long segment ({duration:.1f}s) at {seg.start:.1f}s — will need re-segmentation in clip_candidates")

        transcript.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})

    if not transcript:
        raise RuntimeError("Transcription produced no segments — video may have no audio or be silent")

    if long_segments > 0:
        logger.warning(f"{long_segments} segments exceed {LONG_SEGMENT_WARN_SEC}s — clip_candidates must handle re-segmentation")

    last_timestamp = transcript[-1]["end"]
    logger.info(f"Transcript saved: {output_path} | {len(transcript)} segments | coverage: 0s – {last_timestamp:.1f}s")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)

    return output_path
