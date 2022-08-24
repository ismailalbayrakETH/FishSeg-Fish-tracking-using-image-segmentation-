# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 08:23:40 2022

@author: yanfan
"""

# This script is intended for fixing the bug of mismatched framerates.
# Note that videos have fps of 25, while time recording is aimed at 20 fps.

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
    min, sec = new_time(lines[i][27:32]) # [23:28] for 1 / [27:32] for 2:5
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
    
