import glob
import logging
import os
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
from src.modules.clip_candidate_generator import generate_clip_candidates
from src.modules.clipper import generate_clips
from src.modules.scoring_engine import score_clips
from src.modules.selector import select_clips
from src.modules.start_refiner import refine_starts
from src.modules.vertical_formatter import format_vertical
from src.modules.subtitle_builder import build_subtitles
from src.modules.subtitle_renderer import render_subtitles
from src.modules.exporter import export_pipeline


def run(url: str, subtitles: bool = False, review: bool = False) -> None:
    total = 11 if (subtitles and review) else 12 if subtitles else 10
    print(f"\n=== Stream Content Pipeline ===\n")

    print(f"1/{total} Descargando VOD...")
    video_path = download_vod(url)
    print(f"    {video_path}\n")

    print(f"2/{total} Transcribiendo...")
    transcript_path = transcribe_video(video_path)
    print(f"    {transcript_path}\n")

    print(f"3/{total} Analizando audio...")
    peaks_path = analyze_audio(video_path)
    print(f"    {peaks_path}\n")

    print(f"4/{total} Generando candidatos...")
    candidates_path = generate_clip_candidates(transcript_path, peaks_path)
    print(f"    {candidates_path}\n")

    print(f"5/{total} Cortando clips...")
    clips_dir = generate_clips(video_path, candidates_path)
    print(f"    {clips_dir}\n")

    print(f"6/{total} Scoring...")
    ranked_path = score_clips(candidates_path, transcript_path, peaks_path)
    print(f"    {ranked_path}\n")

    print(f"7/{total} Seleccionando top clips...")
    selected_path = select_clips(ranked_path)
    print(f"    {selected_path}\n")

    print(f"8/{total} Refinando inicios...")
    refined_path = refine_starts(selected_path, transcript_path)
    print(f"    {refined_path}\n")

    print(f"9/{total} Formato vertical (9:16)...")
    vertical_dir = format_vertical(refined_path, video_path)
    print(f"    {vertical_dir}\n")

    if subtitles:
        print(f"10/{total} Generando archivos de subtítulos...")
        subs_dir = build_subtitles(refined_path, transcript_path)
        print(f"    {subs_dir}\n")

        if review:
            print("─" * 50)
            print("MODO REVISIÓN — render pausado.")
            print(f"Edita los archivos en {subs_dir}")
            print("Luego ejecuta con --subtitles (sin --review) para renderizar.")
            print("─" * 50)
        else:
            print(f"11/{total} Renderizando subtítulos...")
            subtitled_dir = render_subtitles(refined_path, vertical_dir)
            print(f"    {subtitled_dir}\n")

    print(f"{total}/{total} Exportando...")
    export_dir = export_pipeline("output")
    print(f"    {export_dir}\n")

    print("=== Listo ===")
    print(f"Export en: {export_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main.py <URL> [--subtitles] [--review]")
        print("Ejemplo: python main.py https://www.twitch.tv/videos/123456789")
        print("         python main.py <URL> --subtitles    # genera + renderiza")
        print("         python main.py <URL> --review       # genera SRT, pausa para editar")
        sys.exit(1)

    _url = sys.argv[1]
    _subtitles = "--subtitles" in sys.argv or "--review" in sys.argv
    _review = "--review" in sys.argv
    run(_url, subtitles=_subtitles, review=_review)
