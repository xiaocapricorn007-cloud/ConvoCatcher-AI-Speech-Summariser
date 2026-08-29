# ConvoCatcher AI Speech Summariser

ConvoCatcher is a fast Python Command Line Interface (CLI) application that transcribes audio files locally and summarizes them using the Gemini API.

It uses **Faster-Whisper** for efficient, highly accurate local speech-to-text transcription (keeping the heavy audio processing on your machine), and **Google Gemini** for generating clear, bulleted summaries blazingly fast without needing to download massive AI models.

## Features
- **Efficient Local Transcription:** Uses `faster-whisper` for optimized inference so you don't have to upload large audio files.
- **Fast Cloud Summarization:** Powered by the Google Gemini API, providing a short summary paragraph and bulleted key takeaways without the heavy compute requirements of local LLMs.
- **Clean CLI Interface:** Easy to use directly from your terminal.

## Prerequisites

1. **Python 3.8+** installed.
2. **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
   - Set the API key as an environment variable:
     ```bash
     # Windows (Command Prompt)
     set GEMINI_API_KEY=your_api_key_here

     # Windows (PowerShell)
     $env:GEMINI_API_KEY="your_api_key_here"
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

### 1. Process an existing file
Run the CLI by pointing it to an audio or video file:
```bash
python convocatcher.py path/to/your/audio_file.mp3
```

### 2. Live Microphone Mode
Speak directly to your computer! It listens until you stop speaking, then immediately transcribes and summarizes what you just said.
```bash
python convocatcher.py --live
```

### 3. GUI Mode
Run the live microphone transcription in a simple Graphical User Interface (GUI) where you can easily read the transcripts and summaries.
```bash
python convocatcher.py --gui
```

### Options:
- `--live`: Enable live microphone mode (ignores file input).
- `--gui`: Enable GUI live mode.
- `--model_size`: Whisper model size (default: `base`). Options: `tiny`, `base`, `small`, `medium`, `large-v3`.
- `--gemini_model`: The Gemini model to use for summarization (default: `gemini-3.7-flash`).
- `--output`: Path to save the summary to a Markdown file.

Example:
```bash
python convocatcher.py --gui --model_size small --gemini_model gemini-3.7-pro
```
