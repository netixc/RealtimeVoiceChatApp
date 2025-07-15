
# Real-Time AI Voice Chat 🎤💬🧠🔊

**Have a natural, spoken conversation with an AI!**  

This project lets you chat with a Large Language Model (LLM) using just your voice, receiving spoken responses in near real-time. Think of it as your own digital conversation partner.

## What's Under the Hood?

A sophisticated client-server system built for low-latency interaction:

1.  🎙️ **Capture:** Your voice is captured by your browser.
2.  ➡️ **Stream:** Audio chunks are whisked away via WebSockets to a Python backend.
3.  ✍️ **Transcribe:** `RealtimeSTT` rapidly converts your speech to text.
4.  🤔 **Think:** The text is sent to an LLM for processing.
5.  🗣️ **Synthesize:** The AI's text response is turned back into speech using `RealtimeTTS`.
6.  ⬅️ **Return:** The generated audio is streamed back to your browser for playback.
7.  🔄 **Interrupt:** Jump in anytime! The system handles interruptions gracefully.

## Key Features ✨

*   **Fluid Conversation:** Speak and listen, just like a real chat.
*   **Real-Time Feedback:** See partial transcriptions and AI responses as they happen.
*   **Low Latency Focus:** Optimized architecture using audio chunk streaming.
*   **Smart Turn-Taking:** Dynamic silence detection (`turndetect.py`) adapts to the conversation pace.
*   **Visual Voice Activity Indicator:** Multi-state indicator shows exactly what's happening:
    *   🔘 Grey - Inactive/disconnected
    *   ⚪ Silver (pulsing) - Activating microphone
    *   🔴 Red - Listening for speech
    *   🟢 Green - Recording your voice
    *   🟦 Teal - Processing speech end
    *   🔵 Dark Cyan (pulsing) - AI processing response
*   **Web Interface:** Clean and simple UI using Vanilla JS and the Web Audio API.
*   **Toggle Controls:** Separate TTS (Text-to-Speech) and STT (Speech-to-Text) toggles for privacy and resource control.
*   **Dockerized Deployment:** Recommended setup using Docker Compose for easier dependency management.

## Technology Stack 🛠️

*   **Backend:** Python < 3.13, FastAPI
*   **Frontend:** HTML, CSS, JavaScript (Vanilla JS, Web Audio API, AudioWorklets)
*   **Communication:** WebSockets
*   **Containerization:** Docker, Docker Compose
*   **Core AI/ML Libraries:**
    *   `RealtimeSTT` (Speech-to-Text)
    *   `RealtimeTTS` (Text-to-Speech)
    *   `transformers` (Turn detection, Tokenization)
    *   `torch` / `torchaudio` (ML Framework)
    *   `openai` (LLM Client)
*   **Audio Processing:** `numpy`, `scipy`

## Before You Dive In: Prerequisites 🏊‍♀️

This project leverages powerful AI models, which have some requirements:

*   **Operating System:**
    *   **Docker:** Linux is recommended for the best GPU integration with Docker.
*   **🐍 Python:** 3.9 or higher (if setting up manually).
    *   The setup assumes **CUDA 12.1**. Adjust PyTorch installation if you have a different CUDA version.
    *   **Docker (Linux):** Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
*   **🐳 Docker (Optional but Recommended):** Docker Engine and Docker Compose v2+ for the containerized setup.

---

## Getting Started: Installation & Setup ⚙️

**Clone the repository first:**

```bash
git clone https://github.com/netixc/RealtimeVoiceChatApp.git
cd RealtimeVoiceChatApp
```
*(If you want to customize models/settings in `src/*.py`, do it **before** this step!)*

Copy `.env.example` to `.env` and edit it with your API keys and configuration
```bash
docker compose up --build
```
or
```bash
docker compose build --no-cache
```

---



**Accessing the Client (Both Methods):**

1.  Open your web browser to `https://localhost:8000` (or your server's IP if running remotely/in Docker on another machine).
2.  **Grant microphone permissions** when prompted.
3.  Click **"Start"** to begin chatting! Watch the voice activity indicator to see the system status.
4.  Use "Stop" to end and "Reset" to clear the conversation.
5.  Use the **speaker icon** to toggle TTS (Text-to-Speech) on/off.
6.  Use the **microphone icon** to toggle STT (Speech-to-Text) on/off.

---

## Configuration Deep Dive 🔧

Want to tweak the AI's voice, brain, or how it listens? Configuration is managed through environment variables in the `.env` file (created from `.env.example`).

**⚠️ Important Docker Note:** If using Docker, make any configuration changes to the `.env` file *before* running `docker compose up`.

*   **TTS Engine & Voice:**
    *   Set `TTS_START_ENGINE` in `.env` to `"kokoro"` (currently the only supported engine).
    *   For advanced engine-specific settings (e.g., voice, speed), modify `AudioProcessor.__init__` in `audio_module.py`.
*   **LLM Backend & Model:**
    *   Set `LLM_START_PROVIDER` (e.g., `"openai"`) and `LLM_START_MODEL` (e.g., `"openai/gpt-4o-mini"`) in `.env`.
    *   Customize the AI's personality by editing `system_prompt.txt`.
*   **STT Settings (`transcribe.py`):**
    *   Modify `DEFAULT_RECORDER_CONFIG` to change the Whisper model (`model`), language (`language`), silence thresholds (`silence_limit_seconds`), etc. The default `base.en` model is pre-downloaded during the Docker build.
*   **Turn Detection Sensitivity (`turndetect.py`):**
    *   Adjust pause duration constants within the `TurnDetector.update_settings` method.
*   **SSL/HTTPS:**
    *   Set `USE_SSL=true` in `.env` file 

---

## License 📜

The core codebase of this project is released under the **MIT License** (see the [LICENSE](./LICENSE) file for details).

This project relies on external TTS engines and LLM providers which have their **own licensing terms**. Please ensure you comply with the licenses of all components you use.
