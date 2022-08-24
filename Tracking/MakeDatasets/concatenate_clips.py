# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:13:32 2022

@author: 26369
"""

# This script is intended for concatenating the video clips produced from 'get_videoclip.py'.
# Herein, video clips collected from different setups and cameras are put together to make the annotation easier.
# This is only necessary if you have multiple video clips.

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
    

clip_path = 'D:\\CNNTest\\Exp\\BGS clips_eel' # path where BGS video clips are located
collect_path = clip_path # path where final video for annotation is located

video_list = get_videolist(clip_path,collect_path)

L = []
for video_path in video_list:
    video = VideoFileClip(video_path)
    L.append(video)
    
final_clip = concatenate_videoclips(L)
final_clip.write_videofile(collect_path+'\\final.avi',codec="libx264") # specify the video name (e.g., final.avi)

