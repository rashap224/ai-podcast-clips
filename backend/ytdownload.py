from pytubefix import YouTube
from pytubefix.cli import on_progress

url1="https://youtu.be/wehrI5AAxWk?feature=shared"
url2="https://youtu.be/4rbncveNjN8?feature=shared"

yt = YouTube(url1, on_progress_callback=on_progress)
print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()