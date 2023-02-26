"""
FishSeg
Fish Tracking

Copyright (c) 2023 ETH Zurich
Written by Fan Yang and Martin Detert
BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
License under the 3-clause BSD License (see LICENSE for details)
"""

# "Tracking" consists of 5 main scripts:
# (1) backgroundSubtraction.py: Do MOG2 background subtraction for experimental videos
# (2) FishSeg_training.py: Training FishSeg model and test video tracking based on the model;
# (3) FishSeg_tracking.py: Do video tracking based on the established model;
# (4) mask2tracks.py: Turn masks predicted by FishSeg into tracks;
# (5) ReadTensorboard.py: Read loss functions produced in log folder under C:\FishSeg 
    # check if the model get good performance;
    
#%% Read before you start
# This script is written for doing background subtraction for all experimental videos.
# Three main sections are included in the script.
# (1) For specified video tests with start time
# (2) For single video
# (3) For all videos under specific folder
# Select to run one of the sections according to your needs. 

# Things to check before you start:
    # (1) Specify original video path and folder path where you want to save videos after BGS.
    # (2) Check the desired framerate for output videos and frequency for frame selection. 
        # They are all annotated in each section.
    
#%% For specified video tests with start time

import os
import numpy as np
import cv2

videoinpath = 'Y:\\'
videooutpath = 'S:\\vaw_public\\yanfan\\FishTracking\\BGS Videos'
xlsrID = 'FishGroup_all.txt'

f = open(os.path.join(videooutpath,xlsrID))
lines = f.readlines()
runIDs = []
starts = []
folder_name = []
for i in range(len(lines)):
    runIDs.append(lines[i][5:25])
    starts.append(lines[i][27:32])
    folder_name.append(os.path.join(videoinpath,runIDs[i]))

class TempStore:
    pass

outpath = [TempStore()]*len(folder_name)

for i in range(len(folder_name)):
    outpath[i] = os.path.join(videooutpath,folder_name[i][3:]+'_mog2')
    
    if not os.path.exists(outpath[i]):
        os.makedirs(outpath[i])
        
    for root, dirs, names in os.walk(folder_name[i]):
        j = 0
        for name in names:
            if name[11:15] == '.avi': # [11:15] for "Cam_0_final.avi"
                j +=1
        writepath = [TempStore()]*j    
        
        j = -1
        for name in names:
            ext = os.path.splitext(name)[1].replace('\\','/')
            if ext == '.avi':
                video_dir = os.path.join(root,name)
                cap = cv2.VideoCapture(video_dir)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS) # 25 fps for original video
                size = (int(width),int(height))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                
                start_time = starts[i]
                t = start_time.split(':')
                start_frame = int(t[0])*60*fps+int(t[1])*fps
                fps = fps-5 # framerate of the output video (20fps)
                freq = 2 # Select one frame from every 2 frames
                fr = 1
                
                j += 1
                writepath[j] = os.path.join(outpath[i],name.replace('final','mog2'))
                vidout = cv2.VideoWriter(writepath[j],fourcc,fps,size,0)                
                fgbg = cv2.createBackgroundSubtractorMOG2()
                
                while cap.isOpened():
                    ret,frame = cap.read()
                    if not ret:
                        cap.release()
                        vidout.release()
                        break
                    if fr >= start_frame:
                        if (start_frame % freq == 0):   
                            fgmask = fgbg.apply(frame)
                            vidout.write(fgmask)
                        start_frame += 1
                    fr += 1 

#%% For single video

import numpy as np
import cv2

video_input = 'Y:\\386_2020_05_11_08_18\\Cam_0_final.avi'
video_output = 'S:\\vaw_public\\yanfan\\FishTracking\\BGS Videos\\Cam_0_mog2.avi'
start_time = '18:20'

cap = cv2.VideoCapture(video_input)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)  # 25fps
size = (int(width),int(height))    
fourcc = cv2.VideoWriter_fourcc(*'XVID')

t = start_time.split(':')
start_frame = int(t[0])*60*fps+int(t[1])*fps
fps = fps-5 # framerate of the output video (20fps)
freq = 2 # Select one frame from every 2 frames
fr = 1

fgbg = cv2.createBackgroundSubtractorMOG2()
vidout = cv2.VideoWriter(video_output,fourcc,fps,size,0)

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        cap.release()
        vidout.release()
        break
    if fr >= start_frame:
        if (start_frame % freq == 0): 
            fgmask = fgbg.apply(frame)
            vidout.write(fgmask)
        start_frame += 1
    fr += 1

#%% For all videos under specific folder

import os
import cv2

videoinpath = 'Y:\\342_2020_27_10_16_26'
videooutpath = 'S:\\vaw_public\\yanfan\\FishTracking\\BGS Videos'
start_time = '02:00'

outpath = os.path.join(videooutpath,videoinpath[3:]+'_mog2')

if not os.path.exists(outpath):
    os.makedirs(outpath)

for root, dirs, names in os.walk(videoinpath):
    for name in names:
        ext = os.path.splitext(name)[1].replace('\\','/')
        if ext == '.avi':
            video_dir = os.path.join(root,name)
            cap = cv2.VideoCapture(video_dir)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS) # 25fps
            size = (int(width),int(height))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            
            t = start_time.split(':')
            start_frame = int(t[0])*60*fps+int(t[1])*fps
            fps = fps-5 # framerate of the output video (20fps)
            freq = 2 # Select one frame from every 2 frames
            fr = 1
            
            writepath = os.path.join(outpath,name.replace('final','mog2'))
            vidout = cv2.VideoWriter(writepath,fourcc,fps,size,0)            
            fgbg = cv2.createBackgroundSubtractorMOG2()
            
            while cap.isOpened():
                ret,frame = cap.read()
                if not ret:
                    cap.release()
                    vidout.release()
                    break
                if fr >= start_frame:
                    if (start_frame % freq == 0): 
                        fgmask = fgbg.apply(frame)
                        vidout.write(fgmask)
                    start_frame += 1
                fr += 1 
