# TODO.md
> Dueño: Claude Code — aprueba Director

## Estado del proyecto
Fase actual: MVP — Completo
Módulo en desarrollo: Ninguno (todos implementados)
Bloqueos: Ninguno

## Critico
_Sin bloqueos._

## Importante
- Revisar clips generados manualmente: ¿los cortes tienen sentido?
- Validar que metadata.json sea útil para comparar runs futuros

## Futuro
- Ajustar ventana WINDOW_BEFORE/AFTER en clip_candidates con feedback de clips reales
- Filtrar frecuencias bajas constantes (ruido/música) en audio_analysis v0.2
- Derivada de energía para mejor detección de picos bruscos
- Modelo configurable en transcription via parámetro (tiny/small/medium)
- Campo "vod_url" y "vod_title" en metadata.json
- Modo verbose/quiet en main.py via argumento
