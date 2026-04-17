# TODO.md
> Dueño: SH Studios — aprueba Director

## Estado del proyecto
Fase actual: v0.4 — Producción vertical + subtítulos editables
Módulos activos: todos (13 módulos, pipeline de 10–12 pasos)
Bloqueos: Ninguno

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
