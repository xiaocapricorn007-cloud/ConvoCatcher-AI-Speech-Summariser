import argparse
import os
import sys
import tempfile
from faster_whisper import WhisperModel
from google import genai
from colorama import init, Fore, Style

# Initialize colorama for cross-platform terminal colors
init(autoreset=True)

def transcribe_audio(audio_path, model_size="base", quiet=False):
    """Transcribes audio using faster-whisper."""
    if not quiet:
        print(f"{Fore.CYAN}[*] Loading Whisper model '{model_size}' (CPU mode)...{Style.RESET_ALL}")
    try:
        # Forcing device="cpu" to avoid missing CUDA library (cublas64_12.dll) errors
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"{Fore.RED}[!] Failed to load Whisper model: {e}{Style.RESET_ALL}")
        sys.exit(1)

    if not quiet:
        print(f"{Fore.CYAN}[*] Transcribing audio...{Style.RESET_ALL}")
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        full_transcript = []
        for segment in segments:
            full_transcript.append(segment.text)
            if not quiet:
                print(f"  [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            
        transcript_text = " ".join(full_transcript)
        return transcript_text
    except Exception as e:
        print(f"{Fore.RED}[!] Transcription failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

def summarize_text(text, model_name="gemini-3.7-flash", quiet=False):
    """Summarizes text using Gemini API."""
    if not quiet:
        print(f"\n{Fore.CYAN}[*] Generating summary using Gemini model '{model_name}'...{Style.RESET_ALL}")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"{Fore.RED}[!] Missing GEMINI_API_KEY environment variable.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[!] Please set it by running: set GEMINI_API_KEY=your_api_key{Style.RESET_ALL}")
        sys.exit(1)
        
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "You are an expert summarizer. I will provide you with a transcript of an audio recording. "
            "Please provide your response in the following exact format:\n\n"
            "1. A short summary paragraph (3-5 sentences) capturing the main essence of the transcript.\n"
            "2. A bulleted list of the key takeaways or action items.\n\n"
            "Transcript:\n"
            f"{text}"
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"{Fore.RED}[!] Summarization failed: {e}{Style.RESET_ALL}")
        sys.exit(1)

def live_mode(model_size, gemini_model):
    try:
        import speech_recognition as sr
    except ImportError:
        print(f"{Fore.RED}[!] SpeechRecognition or PyAudio not installed.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[!] Please install them using: pip install SpeechRecognition pyaudio{Style.RESET_ALL}")
        sys.exit(1)

    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print(f"{Fore.YELLOW}[*] Adjusting for ambient noise... Please wait.{Style.RESET_ALL}")
            r.adjust_for_ambient_noise(source, duration=2)
            print(f"{Fore.GREEN}[+] Ready! Start speaking. (Press Ctrl+C to stop){Style.RESET_ALL}")
            
            while True:
                print(f"\n{Fore.CYAN}[*] Listening...{Style.RESET_ALL}")
                audio = r.listen(source)
                
                print(f"{Fore.CYAN}[*] Processing speech...{Style.RESET_ALL}")
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio.get_wav_data())
                    temp_path = f.name
                
                # Transcribe quietly
                transcript = transcribe_audio(temp_path, model_size, quiet=True)
                os.remove(temp_path)
                
                if transcript.strip():
                    print(f"{Fore.GREEN}[+] You said:{Style.RESET_ALL} {transcript}")
                    # Summarize
                    summary = summarize_text(transcript, gemini_model, quiet=True)
                    print(f"\n{Fore.MAGENTA}--- SUMMARY ---{Style.RESET_ALL}")
                    print(summary)
                    print(f"{Fore.MAGENTA}---------------{Style.RESET_ALL}\n")
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[*] Stopping live mode.{Style.RESET_ALL}")
        sys.exit(0)

def gui_mode(model_size, gemini_model):
    import tkinter as tk
    from tkinter import scrolledtext
    import threading
    import tempfile
    try:
        import speech_recognition as sr
    except ImportError:
        print(f"{Fore.RED}[!] SpeechRecognition or PyAudio not installed.{Style.RESET_ALL}")
        sys.exit(1)

    root = tk.Tk()
    root.title("ConvoCatcher - Live Summarizer GUI")
    root.geometry("700x600")

    status_label = tk.Label(root, text="Initializing...", font=("Arial", 12, "bold"), fg="blue")
    status_label.pack(pady=10)

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Arial", 11))
    text_area.pack(expand=True, fill='both', padx=10, pady=10)

    is_running = True

    def update_status(text, color):
        if is_running:
            root.after(0, lambda: status_label.config(text=text, fg=color))

    def log_gui(message):
        if is_running:
            def _append():
                text_area.insert(tk.END, message + "\n")
                text_area.see(tk.END)
            root.after(0, _append)

    def live_thread():
        r = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                update_status("Adjusting for ambient noise... Please wait.", "orange")
                r.adjust_for_ambient_noise(source, duration=2)
                
                update_status("Ready! Start speaking.", "green")
                
                while is_running:
                    update_status("Listening...", "blue")
                    try:
                        # timeout helps us check is_running flag periodically without blocking forever
                        audio = r.listen(source, timeout=1, phrase_time_limit=None)
                    except sr.WaitTimeoutError:
                        continue
                    
                    if not is_running:
                        break

                    update_status("Processing speech...", "purple")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio.get_wav_data())
                        temp_path = f.name
                    
                    transcript = transcribe_audio(temp_path, model_size, quiet=True)
                    os.remove(temp_path)
                    
                    if transcript.strip():
                        log_gui(f"\n[You said]: {transcript}")
                        update_status("Summarizing...", "purple")
                        summary = summarize_text(transcript, gemini_model, quiet=True)
                        log_gui(f"\n--- SUMMARY ---\n{summary}\n---------------\n")
                        
        except Exception as e:
            if is_running:
                log_gui(f"Error: {e}")
                update_status("Error occurred.", "red")

    t = threading.Thread(target=live_thread, daemon=True)
    t.start()

    def on_closing():
        nonlocal is_running
        is_running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="ConvoCatcher - AI Speech Summariser (Whisper + Gemini)")
    parser.add_argument("audio_file", nargs='?', help="Path to the audio or video file to process.")
    parser.add_argument("--live", action="store_true", help="Enable live microphone transcription and summarization.")
    parser.add_argument("--gui", action="store_true", help="Launch the live microphone transcription in a simple GUI.")
    parser.add_argument("--model_size", default="base", help="Whisper model size (tiny, base, small, medium, large-v3). Default is 'base'.")
    parser.add_argument("--gemini_model", default="gemini-3.7-flash", help="Gemini model to use for summarization. Default is 'gemini-3.7-flash'.")
    parser.add_argument("--output", help="Optional path to save the summary to a file (e.g., summary.md).")
    
    args = parser.parse_args()

    if args.gui:
        gui_mode(args.model_size, args.gemini_model)
    elif args.live:
        live_mode(args.model_size, args.gemini_model)
    else:
        if not args.audio_file:
            print(f"{Fore.RED}[!] Error: Please provide an audio file or use --live or --gui.{Style.RESET_ALL}")
            sys.exit(1)
            
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
        summary = summarize_text(transcript, args.gemini_model)
        
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
