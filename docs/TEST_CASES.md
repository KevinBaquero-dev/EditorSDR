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
Output esperado: /output/raw/vod.mp4 existe y tiene tamaño > 0
Resultado actual: —
Estado: ❌

## Test: ingestion — URL inválida
Input: URL malformada o inexistente
Output esperado: RuntimeError con mensaje descriptivo
Resultado actual: —
Estado: ❌
