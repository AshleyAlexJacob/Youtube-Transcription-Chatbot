import os
import whisper
import subprocess
import sys

class STTService:
    def __init__(self, output_directory='artifacts/transcriptions', model='base'):
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        self._check_ffmpeg()
        self.model = whisper.load_model(model)
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available and provide installation guidance if not."""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("✓ ffmpeg is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ ffmpeg is not installed or not in PATH")
            print("\nTo fix this issue, please install ffmpeg:")
            print("1. Download ffmpeg from: https://ffmpeg.org/download.html")
            print("2. Extract the archive and add the 'bin' folder to your system PATH")
            print("3. Alternatively, use chocolatey: choco install ffmpeg")
            print("4. Or use winget: winget install ffmpeg")
            print("\nAfter installation, restart your terminal and try again.")
            raise RuntimeError("ffmpeg is required but not found. Please install it and restart.")

    def convert_audio_to_text(self, filepath, filename=None, output_directory=None):
        print(f'Audio for Transcribing {filepath}')
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        try:
            print(os.listdir(os.path.join("artifacts", "audios")))
        except FileNotFoundError:
            print("Warning: artifacts/audios directory not found")
        
        try:
            result = self.model.transcribe(filepath)
            print(f"Transcribe Successfully {result['text'][:50]}")
        except Exception as e:
            if "ffmpeg" in str(e).lower() or "subprocess" in str(e).lower():
                print("❌ Error: ffmpeg is required but not working properly")
                print("Please ensure ffmpeg is installed and accessible from command line")
                raise RuntimeError("ffmpeg error during transcription. Please install ffmpeg and restart.")
            else:
                raise e

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        else:
            output_directory = self.output_directory

        if filename is not None:
            with open(os.path.join(output_directory, f"{filename}.txt"), "w") as fp:
                fp.write(result['text'])
        
        return result


# if __name__=="__main__":
#     service = STTService()
#     service.convert_audio_to_text(
#         filepath="./test.m4a",
#        filename="What_is_Data_Science")