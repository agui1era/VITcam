# VTCam – Sistema de Monitoreo con Análisis de Video e IA

Sistema de vigilancia inteligente que captura video desde webcam o RTSP, genera captions con modelo BLIP y analiza comportamientos sospechosos usando GPT-4. Incluye control de alertas repetidas por similitud y envío automático a Telegram.

## 🧠 Características

- Captura desde webcam o stream RTSP
- Descripción automática de escenas con modelo BLIP
- Análisis LLM con GPT-4 (configurable)
- Filtros de alerta por similitud de mensajes recientes
- Envío automático de alertas con video por Telegram
- Logs de captions y análisis en archivos `.jsonl`
- Variables completamente configurables vía `.env`

## 📦 Requisitos

- Python 3.9+
- Dependencias instalables vía `pip` o `requirements.txt`
- Acceso a GPU recomendado para BLIP
- Cuenta de OpenAI con API Key válida
- Token y Chat ID de Telegram Bot

