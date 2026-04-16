# TEST_CASES.md
> Define: GPT | Ejecuta: Claude Code | Valida: Director

## Test 1 — Flujo completo
Input: URL de Twitch (3h)
Output esperado: video descargado + transcript generado + 10–20 clips creados
Resultado actual: —
Estado: ❌

## Test 2 — Video corto
Input: video de 10 min
Output esperado: 3–5 clips
Resultado actual: —
Estado: ❌

## Test 3 — Sin picos de audio
Input: video silencioso
Output esperado: 0 clips o fallback básico
Resultado actual: —
Estado: ❌

## Edge Case — archivo corrupto
Input: video dañado
Output esperado: error controlado
Resultado actual: —
Estado: ❌

## Test: ingestion — descarga básica
Input: URL de Twitch válida
Output esperado: /output/raw/vod.mp4 existe, tamaño > 1MB
Resultado actual: —
Estado: ❌

## Test: ingestion — URL inválida
Input: URL malformada o inexistente
Output esperado: RuntimeError con mensaje descriptivo
Resultado actual: —
Estado: ❌

## Test: ingestion — skip re-descarga
Input: /output/raw/vod.mp4 ya existe
Output esperado: retorna ruta sin volver a descargar
Resultado actual: —
Estado: ❌

## Test: transcription — VOD 30–60 min
Input: /output/raw/vod.mp4 real
Output esperado: JSON con segmentos, sin nulls, coverage ≈ duración del video
Resultado actual: —
Estado: ❌

## Test: transcription — filtro de basura
Input: video con silencios prolongados
Output esperado: sin segmentos "...", " ", vacíos
Resultado actual: —
Estado: ❌

## Test: transcription — CPU fallback
Input: cualquier video, CUDA deshabilitado
Output esperado: transcripción completa, sin excepción
Resultado actual: —
Estado: ❌

## Test: audio_analysis — picos reales
Input: /output/raw/vod.mp4
Output esperado: peaks.json con 1–50 entradas, timestamps en rango [0, duración]
Resultado actual: —
Estado: ❌

## Test: audio_analysis — video silencioso
Input: video sin audio o muy baja energía
Output esperado: peaks.json vacío [], sin excepción
Resultado actual: —
Estado: ❌
