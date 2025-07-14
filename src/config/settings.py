"""
Centralized configuration for Real-Time Voice Chat Application
"""
import os
from typing import Optional

class Settings:
    """Application configuration settings"""
    
    # SSL Configuration
    USE_SSL: bool = os.getenv("USE_SSL", "false").lower() == "true"
    SSL_CERT_PATH: Optional[str] = os.getenv("SSL_CERT_PATH")
    SSL_KEY_PATH: Optional[str] = os.getenv("SSL_KEY_PATH")
    
    # TTS Configuration
    TTS_START_ENGINE: str = os.getenv("TTS_START_ENGINE", "kokoro")
    
    # LLM Configuration
    LLM_START_PROVIDER: str = os.getenv("LLM_START_PROVIDER", "openai")
    LLM_START_MODEL: str = os.getenv("LLM_START_MODEL", "openai/gpt-4o-mini")
    
    # Processing Configuration
    NO_THINK: bool = os.getenv("NO_THINK", "false").lower() == "true"
    DIRECT_STREAM: bool = os.getenv("DIRECT_STREAM", "false").lower() == "true"
    
    # Audio Configuration
    MAX_AUDIO_QUEUE_SIZE: int = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", "50"))
    
    # Server Configuration
    SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# Create a global settings instance
settings = Settings()