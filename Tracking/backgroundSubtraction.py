# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 16:52:23 2022

@author: yanfan
""" 
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
freq = 2 # Select one frame from every 5 frames
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
