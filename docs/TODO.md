# TODO.md
> Dueño: SH Studios — aprueba Director

## Estado del proyecto
Fase actual: v0.6 — Segmentación híbrida (Active Window + Semantic)
Módulos activos: todos (15 módulos, pipeline de 10–12 pasos)
Bloqueos: Ninguno

## Completado — Segmentación híbrida (v0.6)
- [x] segment_engine.py — Active Window + Semantic + Merge
- [x] process_active_window() — agrupa picos con gap < 3s O texto en el gap
- [x] SemanticAnalyzer — BoW cosine similarity entre clips, batch post-proceso
- [x] analyze_continuity() — semantic_score + topic_continuity por clip
- [x] detect_internal_breaks() — detecta cambio de tema dentro de clips largos
- [x] merge_clips() — post-process merge con respeto a cortes semánticos
- [x] confidence_score en metadata de cada clip (ponderado: intensity + duration + text + semantic + peaks)
- [x] energy_score (media de intensidades del grupo de picos vs max) por clip
- [x] max_clip_duration = 120s (vs 60s legacy) — clips más contextualizados
- [x] --legacy flag en main.py para fallback a clip_candidate_generator
- [x] clip_candidate_generator.py preservado sin cambios
- [x] ARCHITECTURE.md, DECISIONS.md, LESSONS.md, TODO.md actualizados

## Completado — Timing Aligner (v0.5)
- [x] timing_aligner.py — alineación de timestamps con voz real (RMS dinámico)
- [x] threshold dinámico por segmento: mean + 0.5 * std (no valor fijo)
- [x] padding de búsqueda ±0.5s alrededor de cada segmento
- [x] suavizado lerp(0.7) — evita saltos bruscos en timestamps
- [x] límites de seguridad: max shift 1.0s, min duración 0.6s, max duración 4.0s
- [x] gap cleaner: corrige gaps < 0.15s entre segmentos consecutivos
- [x] fallback si no se detecta voz — mantiene timestamps originales sin modificar
- [x] debug en meta: original/adjusted/delta por segmento + estadísticas
- [x] respeta subtitles_edited=true — no sobreescribe ediciones manuales
- [x] integrado en main.py como paso 11 (subtitle_builder → timing_aligner → renderer)

## Completado — Social Media
- [x] Plantilla HTML de posts e historias para Instagram (`social/ig-posts.html`)
- [x] 6 posts (1080×1080): identidad, AUTO SUBS, social proof, velocidad, multiplataforma, CTA
- [x] 6 historias (1080×1920): pipeline, antes/después, scoring, tres datos, FAQ, CTA final
- [x] Script de exportación PNG con Puppeteer @2x (`social/screenshot.js`)
- [x] `.gitignore` actualizado — solo se versiona la plantilla y el script

## Completado
- [x] Ventana dinámica en clip_candidate_generator (intensidad → ventana variable)
- [x] Extender finales de clip al cierre del segmento de transcript
- [x] Filtrar ruido constante en audio_analysis con derivada de energía
- [x] QG v0.2 vs v0.1 — mejora confirmada (spread 0.368 en scoring)
- [x] Scoring engine — score compuesto (intensity, text_density, hook_strength, duration_score)
- [x] Hook fallback con energy proxy para clips sin texto inicial
- [x] Text density con saturación (cap 15 chars/s)
- [x] Duration plateau 20–45s
- [x] Bonus phrase_complete (+0.07)
- [x] Selector con top 15 + score ≥ 0.65 + diversidad temporal
- [x] Start refiner — phrase_align y silence_skip con protección de buildup por intensidad
- [x] Formato vertical 9:16 (crop centrado + scale 1080x1920)
- [x] Crop offset configurable (center | left | right + px)
- [x] subtitle_builder — JSON+SRT editables, chunking inteligente, highlight ! ?, offset -0.2s
- [x] subtitle_renderer — quema SRT (editado o auto) sobre clips verticales
- [x] Flag --subtitles y --review en main.py
- [x] Protección de edición manual (subtitles_edited en meta)
- [x] Ajuste de estilo de subtítulos (FontSize 18, bottom center, MarginV 60)

## Importante
- Agregar vod_url y vod_title a metadata.json en exporter
- Validar ranking de scoring con 2–3 VODs adicionales
- Ajustar pesos de scoring con feedback real de clips
- Mover scoring antes de clipper (refactor de pipeline — cortar solo top N)

## Futuro
- Feedback humano → ajuste de pesos en scoring_engine
- Modo verbose/quiet en main.py via argumento
- Parámetros de ventana configurables sin tocar el código
- Zoom inteligente / tracking (post-MVP)
- Auto-upload
