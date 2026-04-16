# CLAUDE.md — Stream Content Pipeline

## Proyecto
Pipeline automatizado para convertir streams en clips cortos.

## Hardware
GPU: RTX 4060 8GB | CPU: Ryzen 5 5600X | RAM: 16GB | OS: Windows 11

## Módulos activos
- ingestion: descarga VOD
- transcription: genera transcript con timestamps
- audio_analysis: detecta picos de audio
- clip_candidates: genera segmentos
- clipper: corta clips con FFmpeg
- exporter: organiza salida

## Reglas críticas
- Cada módulo tiene una sola responsabilidad
- Inputs/outputs deben ser archivos JSON o rutas claras
- No usar IA avanzada en MVP
- No mezclar lógica entre módulos
- Máximo 20 clips

## Convenciones
- JSON como formato estándar
- timestamps en segundos
- nombres descriptivos de archivos

## Referencias
Arquitectura: ARCHITECTURE.md
Prompts: PROMPTS.md
