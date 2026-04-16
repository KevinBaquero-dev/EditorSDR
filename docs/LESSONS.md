# LESSONS.md
> Dueño: Claude Code (Ejecutor)

## Lección — 2026-04-16
Módulo: setup / transcription
Problema: ctranslate2.dll no cargaba en Windows 11 con Python 3.14
Causa raíz: MSVCP140.dll no estaba en System32; está en Windows\System32\Microsoft-Edge-WebView
Solución: os.add_dll_directory() + os.environ['PATH'] apuntando a esa carpeta antes de importar ctranslate2
Lección: En Windows 11, MSVCP140.dll puede estar en Edge-WebView en lugar de System32 — agregar esa ruta al DLL search path antes de cualquier import de ctranslate2.

## Lección — 2026-04-16
Módulo: setup / transcription
Problema: RuntimeError "cublas64_12.dll is not found" al ejecutar transcripción
Causa raíz: ctranslate2 carga las libs CUDA en runtime vía PATH del sistema, no vía add_dll_directory
Solución: agregar los directorios nvidia/*/bin también a os.environ['PATH'] antes del import
Lección: add_dll_directory no es suficiente para ctranslate2 en Windows — también necesita PATH.

## Lección — 2026-04-16
Módulo: audio_analysis
Problema: librosa.load() falla con .mp4 — NoBackendError
Causa raíz: soundfile no soporta mp4; audioread necesita ffmpeg en PATH del sistema (no disponible en la sesión actual)
Solución: usar PyAV (av) directamente para decodificar audio del video sin depender de ffmpeg en PATH
Lección: Para cargar audio de video (.mp4) en Windows sin ffmpeg en PATH, usar PyAV que bundlea su propio ffmpeg.
