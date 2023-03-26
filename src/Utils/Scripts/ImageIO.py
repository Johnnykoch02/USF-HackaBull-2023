from random import sample
import ffmpy
import subprocess as sp
import numpy
import cv2
from datetime import timedelta

class ImageIO(object):
    FFMPEG_PATH = "Z:/ffmpeg/ffmpeg.exe"
    FFMPROBE_PATH =  "Z:/ffmpeg/ffprobe.exe"
   
    def format_timedelta(time):
        """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
        omitting microseconds and retaining milliseconds"""
        result = str(time)
        try:
            result, ms = result.split(".")
        except ValueError:
            return (result + ".00").replace(":", "-")
        ms = int(ms)
        ms = round(ms / 1e4)
        return f"{result}.{ms:02}".replace(":", "-")


    def GetFrames(video_file, rate):
        allFrames = []
        # read the video file    
        capture = cv2.VideoCapture(video_file)
        # get the FPS of the video
        fps = capture.get(cv2.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, 1/rate)
        # get the list of duration spots to save
        nfps = round(fps)
        saving_frames_durations = nfps / saving_frames_per_second
        offset =  round(saving_frames_durations / 2)

        # start the loop
        loopCount = 0
        last_frame = None
        while True:
            is_read, frame = capture.read()
            g = (loopCount + offset) % saving_frames_durations
            c = g == 0 or g == int(saving_frames_durations)
            if not is_read:
                if c:
                    allFrames.append(last_frame)
                break
            if c:
                allFrames.append(frame)
            # increment the frame count
            loopCount += 1
            last_frame = frame
        return allFrames


    def GetImageDimensions(filename):
        command = [ ImageIO.FFMPROBE_PATH,
                '-i', filename,
                '-v', 'error', '-show_entries',
                'stream=width,height',
                '-of', 'default=noprint_wrappers=1:nokey=1'
                ]

        result = sp.run(command, stdout=sp.PIPE, stderr=sp.STDOUT)
        stringRes = str(result.stdout)
        currDim = 0
        dimensions = {"width" : 0, "height" : 0}
        readingHeight = False
        for i in stringRes:
            if i.isnumeric():
                currDim *= 10
                currDim += int(i)
            else:
                if (readingHeight):
                    dimensions["height"] = currDim
                else:
                    dimensions["width"] = currDim
                readingHeight = True 
            
        return dimensions        #Returned as a  width, height dictionary

    def ExportImagesFromMP4(filename, outputName, extension = "wav"):
        ffmpy.FFmpeg(executable=ImageIO.ImageIO.FFMPEG_PATH ,inputs={f'{filename}.mp4': None}, outputs={f'{outputName}.{extension}': None}).run()