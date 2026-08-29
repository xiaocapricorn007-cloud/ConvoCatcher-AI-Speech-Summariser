# ConvoCatcher - Development Log

## Initialization (Project Setup)
**Action:** Cloned the empty repository and initialized basic project files.
**Reason:** To set up the foundation for the ConvoCatcher project, which will be a fully local Python CLI app for transcribing and summarizing audio.

## Tool Selection and Design Decisions
**Action:** Decided on the tech stack based on our interactive `/grill-me` session:
- **Interface:** Python CLI for ease of use from the terminal.
- **Transcription:** Local Whisper model (specifically `faster-whisper` for optimized performance and lower memory usage).
- **Summarization:** Local LLM via Ollama (to keep all processing local, private, and free).
- **Output format:** Paragraph + Bulleted key takeaways.
**Reason:** Ensures the tool is private (no cloud API dependencies), efficient (using `faster-whisper`), and provides clear, actionable summaries.

## Creating Core Files
**Action:** Created `README.md`, `requirements.txt`, and the main script `convocatcher.py`.
**Reason:** 
- `README.md` documents how to install and use the tool. 
- `requirements.txt` manages dependencies (`faster-whisper`, `ollama`, `colorama`). 
- `convocatcher.py` contains the main application logic.

## Implementing Application Logic
**Action:** Wrote `convocatcher.py` using `argparse` for CLI arguments, `faster-whisper` for transcription, and the `ollama` Python library for local LLM requests.
**Reason:** 
- **`argparse`**: Native and clean way to handle flags like `--model_size`, `--ollama_model`, and `--output`.
- **`faster-whisper` implementation**: Used `device="auto"` and `compute_type="int8"` to maximize compatibility across systems (using GPU if available, but staying memory efficient).
- **Ollama Prompt Engineering**: Engineered a specific prompt instructing the LLM to output exactly what was requested: a short summary paragraph followed by a bulleted list of key takeaways.
- **`colorama`**: Added colors to terminal output to make status messages clearly distinguishable from the actual transcript and summary output.

## Switch to Gemini API
**Action:** Replaced the ollama local summarization with the google-generativeai package to use the Gemini API.
**Reason:** The user did not have Ollama installed and preferred a lightweight, fast alternative using cloud-based LLM (Gemini) instead of downloading large local LLMs.


## Live Microphone Mode
**Action:** Implemented a continuous live microphone listening mode triggered by the --live flag. Added SpeechRecognition and pyaudio dependencies.
**Reason:** The user requested the ability to speak to the app directly and have it summarize automatically every time they stop speaking. The SpeechRecognition library inherently handles silence detection (VAD), making it perfectly suited for this loop.

