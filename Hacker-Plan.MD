#### USF HACKER PLAN FOR TEAM ####


Hackathon we will be using a human pose estimation model on our collected data and well have to match the timing of the music and the sampling frequency for the song to the frequency of the collection of estimations.
After that we are going to perform the Angle Overlay Algorithm which will get the appropriate angles between the joints and store those indicies in an array of shape (song index x num samples x num joints). We will then go through each song index in that file, take the current reading and compute the dTheta between that index and the previous index and save those values in a npy array of shape [ song index x num samples x num joints + num joint_dthetas (num joints + num joint_dthetas should be equal to 2 x num joints ).

Then we will take all of that data, including our song data preprocessed using MFCC, and pretrain our model to output the correct joint angle d_thetas for the music and the joint data. Given the algorithm above, we will have to ensure the correct dthetas are given for each audio sample, which means we will have to offset the provided data index by 1 when doin the d_theta training.
[12:33 AM]
After the model is pretrained and it can perform basic motions/dances, we will then run the reinforcement learning, guided by a User Inputted score

JayDawg — 03/22/2023 12:35 AM
Our goal is to have the update to the model occur once every two and a half songs, which we will control via t.sleep() and model sampling frequency. This means that we will have to keep track of the FPS during the training loops, and control the amount of time each loop takes.

@TheGuy920
What’s the sample rate?

JayDawg — 03/22/2023 12:37 AM
our sound image will be a snippet in time, usually ranging from 100 milliseconds to 2 seconds. Our goal should be to output four dtheta updates a second, therfore our snippet time would be 250 milliseconds
[12:38 AM]
sampling frequency of 4, four actions per second
[12:42 AM]
four forward passes per second, one update i would say every 64 seconds, the number of steps between every update would then be 256 steps per update. Were gonna want at least a thousand updates for it to be any good, that means we would need to train it for 16.6666667 hours