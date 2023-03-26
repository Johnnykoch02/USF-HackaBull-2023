import ffmpy
import subprocess as sp

class AudioFileReader(object):

    def Read():
        ffmpegPath = "C:/Contents/Projects/Python/Libraries/ffmpeg-essentials/ffmpeg-2023-03-23-git-30cea1d39b-essentials_build/bin/ffmpeg.exe"

        pipe = sp.Popen([ffmpegPath,"-i", 'mySong.mp3', "-"],
        stdin=sp.PIPE, stdout=sp.PIPE,  stderr=sp.PIPE)
        pipe.stdout.readline()
        pipe.terminate()
        infos = proc.stderr.read()
    pass