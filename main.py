import glob
import logging
import os
import re
import sys

# Registrar DLLs de NVIDIA y VC++ runtime antes de importar ctranslate2/faster-whisper
# ctranslate2 carga cublas/cudnn tanto via add_dll_directory como via PATH en runtime
if sys.platform == "win32":
    _site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages")
    _nvidia_dirs = glob.glob(os.path.join(_site_packages, "nvidia", "*", "bin"))
    for _dll_dir in _nvidia_dirs:
        os.add_dll_directory(_dll_dir)
    # Agregar también a PATH para que ctranslate2 encuentre las libs en runtime
    if _nvidia_dirs:
        os.environ["PATH"] = os.pathsep.join(_nvidia_dirs) + os.pathsep + os.environ.get("PATH", "")
    # msvcp140.dll no está en System32 pero sí en Edge-WebView (Windows 11)
    _edge_webview = r"C:\Windows\System32\Microsoft-Edge-WebView"
    if os.path.isdir(_edge_webview):
        os.add_dll_directory(_edge_webview)
        os.environ["PATH"] = _edge_webview + os.pathsep + os.environ.get("PATH", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.modules.ingestion import download_vod
from src.modules.transcription import transcribe_video
from src.modules.audio_analysis import analyze_audio
from src.modules.segment_engine import segment_video
from src.modules.clip_candidate_generator import generate_clip_candidates  # legacy --legacy
from src.modules.clipper import generate_clips
from src.modules.scoring_engine import score_clips
from src.modules.selector import select_clips
from src.modules.start_refiner import refine_starts
from src.modules.vertical_formatter import format_vertical
from src.modules.subtitle_builder import build_subtitles
from src.modules.timing_aligner import align_all_subtitles
from src.modules.subtitle_renderer import render_subtitles
from src.modules.vod_trimmer import trim_vod
from src.modules.exporter import export_pipeline


def _extract_vod_id(url: str) -> str:
    """Extrae un ID único de la URL para nombrar la carpeta del VOD."""
    # Twitch: twitch.tv/videos/2750604736
    m = re.search(r'/videos/(\d+)', url)
    if m:
        return m.group(1)
    # YouTube: youtube.com/watch?v=ABC123 o youtu.be/ABC123
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    if m:
        return m.group(1)
    # Fallback: fecha de hoy
    from datetime import date
    return date.today().isoformat()


def run(url: str, subtitles: bool = False, review: bool = False,
        trim: bool = False, legacy: bool = False) -> None:
    vod_id   = _extract_vod_id(url)
    vod_base = os.path.join("output", vod_id)

    # Subdirectorios del VOD
    d_raw        = os.path.join(vod_base, "raw")
    d_transcripts = os.path.join(vod_base, "transcripts")
    d_analysis   = os.path.join(vod_base, "analysis")
    d_candidates = os.path.join(vod_base, "candidates")
    d_clips      = os.path.join(vod_base, "clips")
    d_ranked     = os.path.join(vod_base, "ranked")
    d_selected   = os.path.join(vod_base, "selected")
    d_refined    = os.path.join(vod_base, "refined")
    d_vertical   = os.path.join(vod_base, "vertical")
    d_subtitles  = os.path.join(vod_base, "subtitles")
    d_subtitled  = os.path.join(vod_base, "subtitled")
    d_long       = os.path.join(vod_base, "long")

    total = 11 if (subtitles and review) else 13 if subtitles else 10
    print(f"\n=== Stream Content Pipeline ===")
    print(f"VOD: {vod_id}  →  {vod_base}\n")

    print(f"1/{total} Descargando VOD...")
    video_path = download_vod(url, output_dir=d_raw)
    print(f"    {video_path}\n")

    if trim:
        print(f"   [trim] Detectando y recortando VOD para YouTube...")
        trimmed_path = trim_vod(video_path, output_dir=d_long)
        print(f"   [trim] {trimmed_path}\n")

    print(f"2/{total} Transcribiendo...")
    transcript_path = transcribe_video(video_path, output_dir=d_transcripts)
    print(f"    {transcript_path}\n")

    print(f"3/{total} Analizando audio...")
    peaks_path = analyze_audio(video_path, output_dir=d_analysis)
    print(f"    {peaks_path}\n")

    print(f"4/{total} Segmentando clips{'  [legacy]' if legacy else ''}...")
    if legacy:
        candidates_path = generate_clip_candidates(transcript_path, peaks_path,
                                                   output_dir=d_candidates)
    else:
        candidates_path = segment_video(peaks_path, transcript_path,
                                        output_dir=d_candidates)
    print(f"    {candidates_path}\n")

    print(f"5/{total} Cortando clips...")
    clips_dir = generate_clips(video_path, candidates_path, output_dir=d_clips)
    print(f"    {clips_dir}\n")

    print(f"6/{total} Scoring...")
    ranked_path = score_clips(candidates_path, transcript_path, peaks_path,
                              output_dir=d_ranked)
    print(f"    {ranked_path}\n")

    print(f"7/{total} Seleccionando top clips...")
    selected_path = select_clips(ranked_path, output_dir=d_selected)
    print(f"    {selected_path}\n")

    print(f"8/{total} Refinando inicios...")
    refined_path = refine_starts(selected_path, transcript_path, output_dir=d_refined)
    print(f"    {refined_path}\n")

    print(f"9/{total} Formato vertical (9:16)...")
    vertical_dir = format_vertical(refined_path, video_path, output_dir=d_vertical)
    print(f"    {vertical_dir}\n")

    if subtitles:
        print(f"10/{total} Generando archivos de subtítulos...")
        subs_dir = build_subtitles(refined_path, transcript_path,
                                   clips_dir=clips_dir, output_dir=d_subtitles)
        print(f"    {subs_dir}\n")

        print(f"11/{total} Alineando timing con audio real...")
        subs_dir = align_all_subtitles(clips_dir=clips_dir, subtitles_dir=subs_dir)
        print(f"    {subs_dir}\n")

        if review:
            print("─" * 50)
            print("MODO REVISIÓN — render pausado.")
            print(f"Edita los archivos en {subs_dir}")
            print("Luego ejecuta con --subtitles (sin --review) para renderizar.")
            print("─" * 50)
        else:
            print(f"12/{total} Renderizando subtítulos...")
            subtitled_dir = render_subtitles(refined_path, vertical_dir,
                                             subtitles_dir=subs_dir,
                                             output_dir=d_subtitled)
            print(f"    {subtitled_dir}\n")

    print(f"{total}/{total} Exportando...")
    export_dir = export_pipeline(vod_base, vod_url=url)
    print(f"    {export_dir}\n")

    print("=== Listo ===")
    print(f"Export en: {export_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL> [--subtitles] [--review] [--trim] [--legacy]")
        print("Ejemplo: python main.py https://www.twitch.tv/videos/123456789")
        print("         python main.py <URL> --subtitles    # genera + renderiza subtítulos")
        print("         python main.py <URL> --review       # genera SRT, pausa para editar")
        print("         python main.py <URL> --trim         # recorta VOD completo para YouTube")
        print("         python main.py <URL> --legacy       # usa segmentación por picos (v1)")
        sys.exit(1)

    _url       = sys.argv[1]
    _subtitles = "--subtitles" in sys.argv or "--review" in sys.argv
    _review    = "--review" in sys.argv
    _trim      = "--trim" in sys.argv
    _legacy    = "--legacy" in sys.argv
    run(_url, subtitles=_subtitles, review=_review, trim=_trim, legacy=_legacy)
