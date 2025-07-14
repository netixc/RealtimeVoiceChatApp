"""
Application constants and default values
"""

# Default TTS Engines
TTS_ENGINES = {
    "kokoro": "Kokoro TTS Engine"
}

# Default LLM Providers
LLM_PROVIDERS = {
    "openai": "OpenAI API"
}

# Audio Constants
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "int16"

# WebSocket Message Types
WS_MESSAGE_TYPES = {
    "audio_chunk": "audio_chunk",
    "tts_chunk": "tts_chunk",
    "partial_user_request": "partial_user_request",
    "final_user_request": "final_user_request", 
    "partial_assistant_answer": "partial_assistant_answer",
    "final_assistant_answer": "final_assistant_answer",
    "status": "status"
}

# Default Models
DEFAULT_MODELS = {
    "openai": "openai/gpt-4o-mini"
}

# System Prompts Path
SYSTEM_PROMPT_FILE = "/app/src/data/system_prompt.txt"
REFERENCE_AUDIO_FILE = "/app/src/data/reference_audio.json"