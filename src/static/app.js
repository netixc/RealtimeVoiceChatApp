(function() {
  const originalLog = console.log.bind(console);
  console.log = (...args) => {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    originalLog(
      `[${hh}:${mm}:${ss}.${ms}]`,
      ...args
    );
  };
})();

const statusDiv = document.getElementById("status");
const messagesDiv = document.getElementById("messages");
const speedSlider = document.getElementById("speedSlider");
speedSlider.disabled = true;  // start disabled


// VAD indicator element
const vadIndicator = document.getElementById("vadIndicator");

let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;

let isTTSPlaying = false;
let ignoreIncomingTTS = false;

let chatHistory = [];
let typingUser = "";
let typingAssistant = "";
let ttsEnabled = true;
let sttEnabled = true;

// --- batching + fixed 8‑byte header setup ---
const BATCH_SAMPLES = 2048;
const HEADER_BYTES  = 8;
const FRAME_BYTES   = BATCH_SAMPLES * 2;
const MESSAGE_BYTES = HEADER_BYTES + FRAME_BYTES;

const bufferPool = [];
let batchBuffer = null;
let batchView = null;
let batchInt16 = null;
let batchOffset = 0;

function initBatch() {
  if (!batchBuffer) {
    batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
    batchView   = new DataView(batchBuffer);
    batchInt16  = new Int16Array(batchBuffer, HEADER_BYTES);
    batchOffset = 0;
  }
}

function flushBatch() {
  const ts = Date.now() & 0xFFFFFFFF;
  batchView.setUint32(0, ts, false);
  const flags = isTTSPlaying ? 1 : 0;
  batchView.setUint32(4, flags, false);

  socket.send(batchBuffer);

  bufferPool.push(batchBuffer);
  batchBuffer = null;
}

function flushRemainder() {
  if (batchOffset > 0) {
    for (let i = batchOffset; i < BATCH_SAMPLES; i++) {
      batchInt16[i] = 0;
    }
    flushBatch();
  }
}

function initAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) {
    view[i] = raw.charCodeAt(i);
  }
  return new Int16Array(buf);
}

async function startRawPcmCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 24000 },
        channelCount: 1,
        echoCancellation: true,
        // autoGainControl: true,
        noiseSuppression: true
      }
    });
    mediaStream = stream;
    initAudioContext();
    await audioContext.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');

    micWorkletNode.port.onmessage = ({ data }) => {
      if (!sttEnabled) return;
      const incoming = new Int16Array(data);
      let read = 0;
      while (read < incoming.length) {
        initBatch();
        const toCopy = Math.min(
          incoming.length - read,
          BATCH_SAMPLES - batchOffset
        );
        batchInt16.set(
          incoming.subarray(read, read + toCopy),
          batchOffset
        );
        batchOffset += toCopy;
        read       += toCopy;
        if (batchOffset === BATCH_SAMPLES) {
          flushBatch();
        }
      }
    };

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(micWorkletNode);
    statusDiv.textContent = "Recording...";
  } catch (err) {
    statusDiv.textContent = "Mic access denied.";
    console.error(err);
  }
}

async function setupTTSPlayback() {
  await audioContext.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
  ttsWorkletNode = new AudioWorkletNode(
    audioContext,
    'tts-playback-processor'
  );

  ttsWorkletNode.port.onmessage = (event) => {
    const { type } = event.data;
    if (type === 'ttsPlaybackStarted') {
      if (!isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = true;
        console.log(
          "TTS playback started. Reason: ttsWorkletNode Event ttsPlaybackStarted."
        );
        socket.send(JSON.stringify({ type: 'tts_start' }));
      }
    } else if (type === 'ttsPlaybackStopped') {
      if (isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = false;
        console.log(
          "TTS playback stopped. Reason: ttsWorkletNode Event ttsPlaybackStopped."
        );
        socket.send(JSON.stringify({ type: 'tts_stop' }));
      }
    }
  };
  ttsWorkletNode.connect(audioContext.destination);
}

function cleanupAudio() {
  if (micWorkletNode) {
    micWorkletNode.disconnect();
    micWorkletNode = null;
  }
  if (ttsWorkletNode) {
    ttsWorkletNode.disconnect();
    ttsWorkletNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

function renderMessages() {
  messagesDiv.innerHTML = "";
  chatHistory.forEach(msg => {
    const bubble = document.createElement("div");
    bubble.className = `bubble ${msg.role}`;
    bubble.textContent = msg.content;
    messagesDiv.appendChild(bubble);
  });
  if (typingUser) {
    const typing = document.createElement("div");
    typing.className = "bubble user typing";
    typing.innerHTML = typingUser + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  if (typingAssistant) {
    const typing = document.createElement("div");
    typing.className = "bubble assistant typing";
    typing.innerHTML = typingAssistant + '<span style="opacity:.6;">✏️</span>';
    messagesDiv.appendChild(typing);
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function handleJSONMessage(msg) {
  const { type, content, level, is_speaking } = msg;
  if (type === "partial_user_request") {
    typingUser = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    // User is speaking, ensure we're in recording state
    if (content?.trim()) {
      setVADState('recording');
    }
    return;
  }
  if (type === "final_user_request") {
    if (content?.trim()) {
      chatHistory.push({ role: "user", content, type: "final" });
    }
    typingUser = "";
    renderMessages();
    // User finished speaking, move to processing
    setVADState('processing');
    return;
  }
  if (type === "partial_assistant_answer") {
    typingAssistant = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    // Assistant is responding, back to listening
    if (vadState === 'processing') {
      setVADState('listening');
    }
    return;
  }
  if (type === "final_assistant_answer") {
    if (content?.trim()) {
      chatHistory.push({ role: "assistant", content, type: "final" });
    }
    typingAssistant = "";
    renderMessages();
    // Assistant finished, ensure we're listening
    setVADState('listening');
    return;
  }
  if (type === "tts_chunk") {
    if (ignoreIncomingTTS || !ttsEnabled) return;
    const int16Data = base64ToInt16Array(content);
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage(int16Data);
    }
    return;
  }
  if (type === "tts_interruption") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = false;
    return;
  }
  if (type === "stop_tts") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = true;
    console.log("TTS playback stopped. Reason: tts_interruption.");
    socket.send(JSON.stringify({ type: 'tts_stop' }));
    return;
  }
  if (type === "vad_state") {
    updateVADIndicator(is_speaking);
    return;
  }
}

function escapeHtml(str) {
  return (str ?? '')
    .replace(/&/g, "&amp;")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, "&quot;");
}

// UI Controls

document.getElementById("clearBtn").onclick = () => {
  chatHistory = [];
  typingUser = typingAssistant = "";
  renderMessages();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'clear_history' }));
  }
};

speedSlider.addEventListener("input", (e) => {
  const speedValue = parseInt(e.target.value);
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: 'set_speed',
      speed: speedValue
    }));
  }
  console.log("Speed setting changed to:", speedValue);
});

document.getElementById("startBtn").onclick = async () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    statusDiv.textContent = "Already recording.";
    return;
  }
  statusDiv.textContent = "Initializing connection...";
  setVADState('activating');

  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${wsProto}//${location.host}/ws`);

  socket.onopen = async () => {
    statusDiv.textContent = "Connected. Activating mic and TTS…";
    await startRawPcmCapture();
    await setupTTSPlayback();
    speedSlider.disabled = false;
    setVADState('listening');
  };

  socket.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        handleJSONMessage(msg);
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    }
  };

  socket.onclose = () => {
    statusDiv.textContent = "Connection closed.";
    flushRemainder();
    cleanupAudio();
    speedSlider.disabled = true;
    setVADState('inactive');
  };

  socket.onerror = (err) => {
    statusDiv.textContent = "Connection error.";
    cleanupAudio();
    console.error(err);
    speedSlider.disabled = true; 
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    flushRemainder();
    socket.close();
  }
  cleanupAudio();
  statusDiv.textContent = "Stopped.";
  setVADState('inactive');
};

document.getElementById("copyBtn").onclick = () => {
  const text = chatHistory
    .map(msg => `${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}: ${msg.content}`)
    .join('\n');
  
  navigator.clipboard.writeText(text)
    .then(() => console.log("Conversation copied to clipboard"))
    .catch(err => console.error("Copy failed:", err));
};

// Text input handlers
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");

function sendTextMessage() {
  const text = textInput.value.trim();
  if (!text) return;
  
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    console.error("WebSocket not connected. Please start the voice chat first.");
    return;
  }
  
  // Clear the input
  textInput.value = "";
  
  // Send the text message via WebSocket
  socket.send(JSON.stringify({
    type: 'text_user_request',
    text: text
  }));
  
  console.log("Sent text message:", text);
}

sendBtn.onclick = sendTextMessage;

// Send on Enter key
textInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendTextMessage();
  }
});

// TTS Toggle button handler
const ttsToggleBtn = document.getElementById("ttsToggle");
ttsToggleBtn.onclick = () => {
  ttsEnabled = !ttsEnabled;
  
  // Update button visual state
  const enabledPath = ttsToggleBtn.querySelector('.tts-enabled');
  const disabledPath = ttsToggleBtn.querySelector('.tts-disabled');
  
  if (ttsEnabled) {
    enabledPath.style.display = 'block';
    disabledPath.style.display = 'none';
    console.log("TTS enabled");
  } else {
    enabledPath.style.display = 'none';
    disabledPath.style.display = 'block';
    console.log("TTS disabled");
    
    // Clear any playing TTS when disabled
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
  }
};

// STT Toggle button handler
const sttToggleBtn = document.getElementById("sttToggle");
sttToggleBtn.onclick = () => {
  sttEnabled = !sttEnabled;
  
  // Update button visual state
  const micDisabledPath = sttToggleBtn.querySelector('.stt-disabled');
  
  if (sttEnabled) {
    micDisabledPath.style.display = 'none';
    console.log("STT enabled - microphone active");
  } else {
    micDisabledPath.style.display = 'block';
    console.log("STT disabled - microphone muted");
    
    // Flush any remaining audio when disabled
    flushRemainder();
  }
};

// First render
renderMessages();


// VAD indicator states
let vadState = 'inactive';
let vadStateTimer = null;

// VAD indicator function
function setVADState(state) {
  // Remove all state classes
  vadIndicator.classList.remove('inactive', 'activating', 'listening', 'recording', 'waiting', 'processing');
  
  // Add the new state class
  vadIndicator.classList.add(state);
  vadState = state;
  
  // Clear any existing timer
  if (vadStateTimer) {
    clearTimeout(vadStateTimer);
    vadStateTimer = null;
  }
}

// Legacy function for VAD state updates
function updateVADIndicator(isSpeaking) {
  if (isSpeaking) {
    setVADState('recording');
  } else {
    // When speech stops, go to waiting state briefly
    setVADState('waiting');
    vadStateTimer = setTimeout(() => {
      setVADState('listening');
    }, 1000);
  }
}

