import os
import json
from src.Utils.Skeleton import Skeleton
import time as t

FILE_PATH = ""

def write_file(data_out):
    count = 1
    pn = True

    while True:
        if count % 10:
            pn = not pn
            count = 1
        try:
            f = open(os.path.join(os.getcwd(), "interop/unity.json"), "r")
            k = json.loads(f.read())
            f.close()
            sk = Skeleton(0, elbow_r=k[0], ankle_r=k[1], wrist_r=k[2], shoulder_r=k[3], hip_r=k[4], knee_r=k[5],elbow_l=k[6], ankle_l=k[7], wrist_l=k[8], shoulder_l=k[9], hip_l=k[10], knee_l=k[11])
        except Exception as e:
            print(e)
            continue

        f = open(os.path.join(os.getcwd(), "interop/python.json"), "w")
        f.write(json.dumps(data_out))
        f.close()
        break
    return sk

