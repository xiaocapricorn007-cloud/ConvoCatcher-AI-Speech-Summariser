# ConvoCatcher AI Speech Summariser

ConvoCatcher is a completely local, privacy-focused Python Command Line Interface (CLI) application that transcribes audio files and summarizes them using on-device AI.

It uses **Faster-Whisper** for efficient, highly accurate local speech-to-text transcription, and **Ollama** (e.g., Llama 3) for generating clear, bulleted summaries—all without sending your data to the cloud.

## Features
- **100% Local & Private:** No API keys required, no data leaves your machine.
- **Efficient Transcription:** Uses `faster-whisper` for optimized inference.
- **Smart Summarization:** Powered by Ollama, providing a short summary paragraph and bulleted key takeaways.
- **Clean CLI Interface:** Easy to use directly from your terminal.

## Prerequisites

1. **Python 3.8+** installed.
2. **Ollama** installed and running on your system. 
   - Download from [Ollama's official website](https://ollama.com/).
   - Pull a model to use for summarization (e.g., Llama 3):
     ```bash
     ollama run llama3
     ```
3. **FFmpeg** installed (required by Whisper for audio processing).
   - Windows: `winget install ffmpeg`

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/xiaocapricorn007-cloud/ConvoCatcher-AI-Speech-Summariser.git
   cd ConvoCatcher-AI-Speech-Summariser
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the CLI by pointing it to an audio or video file:

```bash
python convocatcher.py path/to/your/audio_file.mp3
```

### Options:
- `--model_size`: Whisper model size (default: `base`). Options: `tiny`, `base`, `small`, `medium`, `large-v3`.
- `--ollama_model`: The Ollama model to use for summarization (default: `llama3`).
- `--output`: Path to save the summary to a Markdown file.

Example:
```bash
python convocatcher.py meeting.mp4 --model_size small --ollama_model llama3 --output summary.md
```
