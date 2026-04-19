# LESSONS.md
> Dueño: SH Studios (Ejecutor)

## Lección — 2026-04-16
Módulo: setup / transcription
Problema: ctranslate2.dll no cargaba en Windows 11 con Python 3.14
Causa raíz: MSVCP140.dll no estaba en System32; está en Windows\System32\Microsoft-Edge-WebView
Solución: os.add_dll_directory() + os.environ['PATH'] apuntando a esa carpeta antes de importar ctranslate2
Lección: En Windows 11, MSVCP140.dll puede estar en Edge-WebView — agregar esa ruta al DLL search path antes de cualquier import de ctranslate2.

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

## Lección — 2026-04-16
Módulo: clip_candidate_generator
Problema: algunos clips empiezan demasiado tarde en momentos de alta intensidad
Causa raíz: ventana fija -10s no da suficiente contexto previo para picos muy intensos
Solución: ventana dinámica según intensidad (intensity > 0.8 → -15s/+20s)
Lección: ventana fija funciona para intensidad media pero recorta contexto en momentos clave. Escalar ventana con intensidad.

## Lección — 2026-04-16
Módulo: clip_candidate_generator
Problema: finales de clip cortan frases a la mitad
Causa raíz: el end del candidato se calcula solo desde el pico, sin considerar si hay una frase en curso
Solución: extender end hasta el final del segmento de transcript más cercano
Lección: los clips que cortan frases no son usables — siempre extender al cierre del segmento de transcript.

## Lección — 2026-04-16
Módulo: audio_analysis
Problema: picos detectados en ruido constante (música de fondo, teclado, ambiente)
Causa raíz: RMS puro no distingue entre ruido sostenido y reacciones reales
Solución: combinar RMS con derivada de energía — el ruido constante tiene derivada baja, las reacciones la tienen alta
Lección: RMS solo es ciego a la "sorpresa". La derivada del RMS detecta cambios bruscos y filtra ruido estático.

## Lección — 2026-04-19
Módulo: clip_candidate_generator → segment_engine
Problema: múltiples picos cercanos (ej. segundo 10, 25, 40 con gaps de 1–2s) generaban 3 clips separados en vez de uno continuo
Causa raíz: el modelo de ventana estática crea un candidato por pico independientemente; el merge solo actúa si el overlap supera 50% del clip más corto, lo cual no ocurre con picos distanciados > post_buffer
Solución: Active Window — mantener ventana abierta mientras gap < 3s O hay texto entre picos; solo cerrar si hay silencio sostenido real
Lección: la fragmentación de clips es invisible en los logs pero destructiva para la calidad del contenido — el merge por overlap no es suficiente cuando los picos tienen distancias que superan el post-buffer individual.

## Lección — 2026-04-19
Módulo: segment_engine / SemanticAnalyzer
Problema: análisis semántico con modelos de embeddings (sentence-transformers) requiere descarga, GPU warmup y dependencias extra que no están en requirements.txt
Causa raíz: querer precisión semántica alta con modelos modernos
Solución: BoW cosine similarity con numpy — ya disponible, cero setup, suficiente para detectar cambios de tema en vocabulario de streaming
Lección: para análisis semántico de streaming en español, BoW + cosine similarity da resultados prácticos sin overhead. Guardar sentence-transformers para cuando la precisión sea el cuello de botella real.
