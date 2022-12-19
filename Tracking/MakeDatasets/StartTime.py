# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:23:40 2022

@author: yanfan
"""
#%% FishSeg tracking
# Part 1 Make datasets
# Fan Yang, Aug 2022
# "MakeDatasets" consists of 4 main scripts:
# （1） StartTime.py: Correct the manually recorded start time (from 25 fps to 20 fps),
# specially setup for VAW fisheye camera system;
# （2）get_videoclip.py: get video clips where fish actually show up and do MOG2
# background subtraction in the meantime;
# （3）concatenate_clips.py: concatenate all the video clips together;
# （4）datasets.py: check annotation datasets and reorganize datasets;  

#%% Read before you start
# This script is intended for fixing the bug of mismatched framerates.
# Note that videos have fps of 25, while time recording is aimed at 20 fps.

# You may need to modify following parts:
    # (1) path for "TrackingGroups" and "StartTime".
    # (2) check the demo folder for the organization of txt files of "TrackingGroup" and "StartTime"
        # "TrackingGroup": folder name and start time recorded manually during experiments;
        # "StartTime": Write the correct start time of 20 fps in the sequence as "TrackingGroup"
    # (3) Check line 51: [27:32] may be changed if you have organize folder name in a different way.
        # Just make sure that "min" and "sec" are obtained correctly.
    
#%%
import os
import numpy as np

TrackingGroups = 'D:\FishSeg_Results\TrackingPlan\TrackingGroup_2.txt'
StartTime = 'D:\FishSeg_Results\TrackingPlan\StartTime_2.txt'    

f = open(TrackingGroups)
lines = f.readlines()
start_time = []

def new_time(timecode):
    t = timecode.split(':')
    min = int(t[0])
    sec = int(t[1])
    tt = (min*60 + sec)*0.8-30 # start tracking 30s ahead of the entry time
    minx = int(tt//60)
    secx = int(tt%60)
    return minx, secx

for i in range(len(lines)):
    min, sec = new_time(lines[i][27:32]) # [27:32] may be changed if you have organize folder name in a different way.
    if min <= 0:
        start_time.append(['00:00'])
    if min<10 and min>0 and sec<10:
        start_time.append(['0'+str(min)+':'+'0'+str(sec)])
    if min<10 and min>0 and sec>=10:
        start_time.append(['0'+str(min)+':'+str(sec)])
    if min>=10 and sec<10:
        start_time.append([str(min)+':'+'0'+str(sec)])
    if min>=10 and sec>=10:
        start_time.append([str(min)+':'+str(sec)])

 
with open(StartTime,"w") as g:
    for i in start_time:
        g.write(str(i)[2:7] + '\n')
g.close()
    
