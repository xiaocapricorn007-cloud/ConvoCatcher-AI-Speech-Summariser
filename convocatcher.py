import argparse
import os
import sys
from faster_whisper import WhisperModel
import ollama
from colorama import init, Fore, Style

# Initialize colorama for cross-platform terminal colors
init(autoreset=True)

def transcribe_audio(audio_path, model_size="base"):
    """Transcribes audio using faster-whisper."""
    print(f"{Fore.CYAN}[*] Loading Whisper model '{model_size}'...{Style.RESET_ALL}")
    try:
        # Run on GPU if available (requires CUDA), otherwise fallback to CPU
        # compute_type "int8" is used for faster CPU performance and less memory, change to "float16" for GPU if preferred
        model = WhisperModel(model_size, device="auto", compute_type="int8")
    except Exception as e:
        print(f"{Fore.RED}[!] Failed to load Whisper model: {e}{Style.RESET_ALL}")
        sys.exit(1)

    print(f"{Fore.CYAN}[*] Transcribing audio: '{audio_path}'...{Style.RESET_ALL}")
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        print(f"{Fore.GREEN}[+] Detected language '{info.language}' with probability {info.language_probability:.2f}{Style.RESET_ALL}")
        
        full_transcript = []
        for segment in segments:
            full_transcript.append(segment.text)
            # Optional: Print progress
            print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            
        transcript_text = " ".join(full_transcript)
        return transcript_text
    except Exception as e:
        print(f"{Fore.RED}[!] Transcription failed: {e}{Style.RESET_ALL}")
        sys.exit(1)


def summarize_text(text, model_name="llama3"):
    """Summarizes text using Ollama."""
    print(f"\n{Fore.CYAN}[*] Generating summary using Ollama model '{model_name}'...{Style.RESET_ALL}")
    
    prompt = (
        "You are an expert summarizer. I will provide you with a transcript of an audio recording. "
        "Please provide your response in the following exact format:\n\n"
        "1. A short summary paragraph (3-5 sentences) capturing the main essence of the transcript.\n"
        "2. A bulleted list of the key takeaways or action items.\n\n"
        "Transcript:\n"
        f"{text}"
    )

    try:
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        print(f"{Fore.RED}[!] Summarization failed: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[!] Ensure Ollama is running and the model '{model_name}' is installed (run: 'ollama run {model_name}').{Style.RESET_ALL}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="ConvoCatcher - AI Speech Summariser (Fully Local)")
    parser.add_argument("audio_file", help="Path to the audio or video file to process.")
    parser.add_argument("--model_size", default="base", help="Whisper model size (tiny, base, small, medium, large-v3). Default is 'base'.")
    parser.add_argument("--ollama_model", default="llama3", help="Ollama model to use for summarization. Default is 'llama3'.")
    parser.add_argument("--output", help="Optional path to save the summary to a file (e.g., summary.md).")
    
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"{Fore.RED}[!] Error: File '{args.audio_file}' not found.{Style.RESET_ALL}")
        sys.exit(1)

    # 1. Transcribe
    transcript = transcribe_audio(args.audio_file, args.model_size)
    
    if not transcript.strip():
        print(f"{Fore.YELLOW}[!] No speech detected in the audio file.{Style.RESET_ALL}")
        sys.exit(0)
        
    print(f"\n{Fore.GREEN}[+] Transcription complete. ({len(transcript.split())} words){Style.RESET_ALL}")

    # 2. Summarize
    summary = summarize_text(transcript, args.ollama_model)
    
    print(f"\n{Fore.MAGENTA}--- SUMMARY ---{Style.RESET_ALL}\n")
    print(summary)
    print(f"\n{Fore.MAGENTA}---------------{Style.RESET_ALL}\n")

    # 3. Save Output
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"# Transcript Summary\n\n**Source File:** {os.path.basename(args.audio_file)}\n\n")
                f.write(summary)
            print(f"{Fore.GREEN}[+] Summary saved to '{args.output}'{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}[!] Failed to save output to file: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
