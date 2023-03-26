from os import listdir
from os.path import isfile, join
import numpy
from AudioIO import AudioIO
from ImageIO import ImageIO
import os

FILENAME = "input.mp4"
RATE = 0.250
OFFSET = 20
data_dir = "Z:\\TraingingData\\"

index = 0
onlyfiles = [f for f in listdir("Z:/Videos/") if isfile(join("Z:/Videos/", f))]
for file in onlyfiles:
    __f = f"Z:/Videos/{file}"
    rawFrameData = ImageIO.GetFrames(__f, RATE, OFFSET)
    samples = AudioIO.SpliceAudioData(__f, RATE, OFFSET)
    audioPath = os.path.join(data_dir,f"{index}\\AudioData.npy")
    videoPath = os.path.join(data_dir,f"{index}\\FrameData.npy")
    if (not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    index += 1
