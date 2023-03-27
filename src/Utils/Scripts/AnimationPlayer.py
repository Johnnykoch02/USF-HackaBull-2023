from time import sleep
import ffmpy
import subprocess as sp
import numpy
import AudioFileReader
import sounddevice
from AudioIO import AudioIO
from ImageIO import ImageIO
import cv2
import os
from PIL import Image

# Colours
BACKGROUND = (255, 255, 255)
  
TESTFILE_1 = "../Videos/Just Dance 2015 - Me and My Broken Heart - Rixton, 5_ GamePlay - 1080p HD"
TESTFILE_2 = os.path.dirname(os.path.realpath(__file__)) + "/input.mp4"
currentTestFile = TESTFILE_2

#raw_audio = samples = AudioIO.AudioIO.SpliceAudioData(sampleCount, 0.250, 20)
#audio_array = numpy.frombuffer(raw_audio, dtype="int32")
#def pilImageToSurface(pilImage):
#    return pygame.image.fromstring(
#        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()
# size = ImageIO.GetImageDimensions(currentTestFile)

rawFrameData = ImageIO.GetFrames(currentTestFile, 0.250)
samples = AudioIO.SpliceAudioData(currentTestFile, 0.250)

image = rawFrameData[0]

for i in range(len(samples)):
    image = rawFrameData[i]
    cv2.imshow("Win", image)
    # sounddevice.play(samples[i], AudioIO.PLAYREAD_SPEED)
    cv2.waitKey(250)