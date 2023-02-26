"""
FishSeg
Make Datasets

Copyright (c) 2023 ETH Zurich
Written by Fan Yang and Martin Detert
BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
License under the 3-clause BSD License (see LICENSE for details)
"""

# "MakeDatasets" consists of 4 main scripts:
# （1） StartTime.py: Correct the manually recorded start time (from 25 fps to 20 fps),
# specially setup for VAW fisheye camera system;
# （2）get_videoclip.py: get video clips where fish actually show up and do MOG2
# background subtraction in the meantime;
# （3）concatenate_clips.py: concatenate all the video clips together;
# （4）datasets.py: check annotation datasets and reorganize datasets;  

#%% Read before you start
# Two main sections are included in the script.
# (1) check annotation datasets
# (2) reorganize datasets
# Make sure that you specify the right path and video names before you start.
# In section(2), places where you need to check video names are annotated within the section. 

#%% check annotation datasets
    # Aim: how many images are contained in each dataset
    
import numpy as np
import h5py as h5

dataset_path = 'D:\\FishSeg_Results\\Datasets\\newtraining304.h5' # path for the dataset that you want to check
file = h5.File(dataset_path,"r")
kys = [key for key in file.keys()]
print([key for key in file.keys()])
with h5.File(dataset_path, 'r') as h5_file:
    array_images = h5_file['annotations'][:].astype(np.str)
    
#%% reorganize datasets 
    # Aim: check video names for annotations at corresponding place (3 places in total)
    
import h5py
import numpy as np

train = "D:\\FishSeg_Results\\Datasets\\newtraining304.h5" # path for training datasets
val = "D:\\FishSeg_Results\\Datasets\\newvalidation115.h5" # path for validation datasets

with h5py.File(val, 'r') as f:
    val_images = f['annotations'][:].astype(np.str)
val_extra_decodes = np.random.choice(val_images,100) 
    # 100 is set based on images in validation datasets
    # Herein, 100 images from validation datasets will be checked. 
    # Any images not repeated in training datasets will be moved to training dataset, and delete from validation dataset.


## TRAINING DATASET ##
train_keys = []
train_names = []
with h5py.File(val,'r') as v:
    with h5py.File(train,'a') as t:
        for val_extra_decode in val_extra_decodes:
            if t.__contains__(val_extra_decode):
                print ('Object:', val_extra_decode, 'also in training datasets. Skipping...\n')
            else:
                print ('Object:', val_extra_decode, 'NOT in training datasets. Copying...\n')
                v.copy(val_extra_decode,t['final.avi']) # check video name for annotations
                val_extra_encode = val_extra_decode.encode('utf8')
                t['annotations'].resize((t['annotations'].shape[0]+1),axis=0)
                t['annotations'][-1:] = val_extra_encode
        for key in t.keys():
            print(t[key],key,t[key].name)
        train_images = t['annotations'][:].astype(np.str)
        train_final = t['final.avi'] # check video name for annotations
        for key in train_final.keys():
            train_key = train_final[key]
            train_name = train_final[key].name
            train_keys.append(train_key)
            train_names.append(train_name)
     
## VALIDATION DATASET ##
val_keys = []
val_names = []
with h5py.File(train,'r') as t:
    train_images = t['annotations'][:].astype(np.str)
    with h5py.File(val,'a') as v:
        val_images = v['annotations'][:].astype(np.str)
        for train_image in train_images:
            if v.__contains__(train_image):
                del v[train_image]
        del v['annotations']
        val_final = v['final.avi'] # check video name for annotations
        for key in val_final.keys():
            val_key = val_final[key]
            val_name = val_final[key].name
            val_keys.append(val_key)
            val_names.append(val_name)
        val_names = np.array(val_names)
        arr=[]
        for val_name in val_names:
            arr = np.append(arr,val_name.encode('utf8'))
        v.create_dataset("annotations",data=arr)
