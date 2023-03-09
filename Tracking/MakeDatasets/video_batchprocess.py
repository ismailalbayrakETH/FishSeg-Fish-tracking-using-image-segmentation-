"""
FishSeg
Make Datasets

Copyright (c) 2023 ETH Zurich
BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
License under the 3-clause BSD License (see LICENSE for details)
"""

# This script is intended for applying backgroundSubtractorMOG2 for all video clips

import os
import cv2

class TempStore:
    pass

videoinpath = 'D:\\FishSeg_Results\\VideoClips\\Temp clips_trout'
videooutpath = 'D:\\FishSeg_Results\\VideoClips\\BGS clips_trout'

folder_name = os.listdir(videoinpath)
inpath = [TempStore()]*len(folder_name)
outpath = [TempStore()]*len(folder_name)

for i in range(len(folder_name)):
    inpath[i] = os.path.join(videoinpath,folder_name[i]) 
    outpath[i] = os.path.join(videooutpath,folder_name[i].replace('temp','mog2'))
    
    if not os.path.exists(outpath[i]):
        os.makedirs(outpath[i])
        
    for root, dirs, names in os.walk(inpath[i]):
        j = 0
        for name in names:
            ext = os.path.splitext(name)[1].replace('\\','/')
            if ext == '.avi':
                j+=1
        writepath = [TempStore()]*j       
        
        j = -1
        for name in names:
            ext = os.path.splitext(name)[1].replace('\\','/')
            if ext == '.avi':
                video_dir = os.path.join(root,name)
                cap = cv2.VideoCapture(video_dir)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS) # stay at 25 fps
                size = (int(width),int(height))
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                
                j += 1
                writepath[j] = os.path.join(outpath[i],name.replace('temp','mog2'))
                out = cv2.VideoWriter(writepath[j],fourcc,fps,size,0)
                fgbg = cv2.createBackgroundSubtractorMOG2()
                
                while cap.isOpened():
                    ret,frame = cap.read()
                    if not ret:
                        print('Finishing background-subtraction of video' + video_dir)
                        cap.release()
                        out.release()
                        continue                         
                    fgmask = fgbg.apply(frame)
                    out.write(fgmask) 
