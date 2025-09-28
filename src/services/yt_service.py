from pytubefix import YouTube
from pytubefix.cli import on_progress
import os

class YtService:
    def __init__(self ,output_directory='artifacts'):
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        
    def download_audio(self, video_url, output_directory=None):
        if output_directory:
            os.makedirs(output_directory, exist_ok=True)
        else:
            output_directory = self.output_directory

        
        yt = YouTube(video_url, on_progress_callback=on_progress,
                    #  use_oauth=True, 
                    #  allow_oauth_cache=True
                     )
        print(f"Downloading Video {yt.title}")

        
        ys = yt.streams.get_audio_only()
        output_path = os.path.join(output_directory,"audios")
        file_path = f'{yt.title.replace(" ", "_").replace("|", "")}.m4a'
        ys.download(output_path= output_path, filename=file_path)
        return os.path.join(output_path, file_path)

# if __name__=="__main__":
#     service = YtService()
#     service.download_audio(video_url="https://youtu.be/81glyreIXPk")