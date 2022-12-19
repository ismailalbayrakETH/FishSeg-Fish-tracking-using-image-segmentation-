# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:13:32 2022

@author: 26369
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
# This script is intended for concatenating the video clips produced from 'get_videoclip.py'.
# Herein, video clips collected from different setups and cameras are put together to make the annotation easier.
# Specify the folder path for video clips in "clip path" before you start. 
# The final video is named as "final.avi" under the path that you have specified. 
# You may change the name of final video at the last line of the script.

#%% 
clip_path = 'D:\\CNNTest\\Exp\\BGS clips_eel' # path where BGS video clips are located
collect_path = clip_path # path where final video for annotation is located

import os
from moviepy.editor import *

class Temp:
    pass

def get_videolist(clip_path,collect_path):
    folder_name = os.listdir(clip_path)
    path = [Temp()]*len(folder_name)
    video_list = []
    
    if not os.path.exists(collect_path):
        os.mkdir(collect_path)
    
    for i in range(len(folder_name)):
        path[i] = os.path.join(clip_path,folder_name[i])
        for root, dirs, names in os.walk(path[i]):
            for name in names:
                ext = os.path.splitext(name)[1].replace('\\','/')
                video_dir = os.path.join(root, name)
                video_list.append(video_dir)
    return video_list
    
video_list = get_videolist(clip_path,collect_path)

L = []
for video_path in video_list:
    video = VideoFileClip(video_path)
    L.append(video)
    
final_clip = concatenate_videoclips(L)
final_clip.write_videofile(collect_path+'\\final.avi',codec="libx264") # specify the video name (e.g., final.avi)

