import ffmpy
import subprocess as sp
import numpy
from datetime import timedelta
import librosa

class AudioIO(object):
    PLAYREAD_SPEED = 48000
    FFMPEG_PATH = "Z:/ffmpeg/ffmpeg.exe"
    FFMPROBE_PATH =  "Z:/ffmpeg/ffprobe.exe"

    def GetAudioData(filename, secondSampleSize):
        command = [ AudioIO.FFMPEG_PATH,
                '-i', filename,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(AudioIO.PLAYREAD_SPEED), # ouput will have 44100 Hz
                '-ac', '2', # stereo (set to '1' for mono)
                '-']

        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
        lengthCalculation = (44100 * secondSampleSize) * 4
        return pipe.stdout.read(int(lengthCalculation))        #The raw audio

    def GetAudioDataSegment(filename, currentTime, sampleSize):

        command = [ AudioIO.FFMPEG_PATH,
                '-i', filename,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', str(AudioIO.PLAYREAD_SPEED), # ouput will have 44100 Hz
                '-ac', '2', # stereo (set to '1' for mono)
                '-ss', str(timedelta(seconds=currentTime)),
                '-to', str(timedelta(seconds=currentTime+sampleSize)),
                '-']

        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

        lengthCalculation = ((44100 * sampleSize) * 4)

        return pipe.stdout.read(int(lengthCalculation)) # The raw audio

    def get_length(filename):
        result = sp.run([AudioIO.FFMPROBE_PATH, "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", filename],
            stdout=sp.PIPE,
            stderr=sp.STDOUT)
        return float(result.stdout)

    def SpliceAudioData(fn, sampleSize):
        allSegments = []
        print(fn)
        master_time = int(AudioIO.get_length(fn) / sampleSize)
        print(master_time)
        for i in range(0, master_time):
            raw_audio = AudioIO.GetAudioDataSegment(fn, i * sampleSize, sampleSize)
            allSegments.append(numpy.frombuffer(raw_audio, dtype="int32"))

        return allSegments

    def GetAudioSplice(audioData, dataSamplesPerSplice, index):
        return audioData[index * dataSamplesPerSplice : (index + 1) * dataSamplesPerSplice]

    def ExportAudioFromMP4(filename, outputName, extension = "wav"):
        ffmpy.FFmpeg(executable=AudioIO.FFMPEG_PATH ,inputs={f'{filename}.mp4': None}, outputs={outputName: None}).run()