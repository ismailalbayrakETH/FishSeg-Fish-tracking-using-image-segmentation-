"""
FishSeg
Fish Tracking

Copyright (c) 2023 ETH Zurich
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
# This script is written for reading and visualizing the losses functions
# Two main sections are included in the script.
# (1) Plots from single log file
# (2) Plots from multiple log files (continue training)
# Always check the annotated block first whether you want to use section (1) or (2).

#%% Plots from single log file
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#### CHECK THIE BLOCK FIRST ####
os.chdir('C:\\FishSeg\\logs\\') # set current working directory
log = 'trout20220731T0840' # Specify the log name
suffix = '.VAWSRVI02' # name of your computing device
folder = '\\'+log 
path = os.path.join(os.getcwd()+folder) 
save_path = os.path.join(os.getcwd()+'\\LogFiles') 
save_dir = os.path.join(os.getcwd()+'\\LogFiles'+'\\'+log) # Path where the excel files and images are saved
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Load log data
event_file = []
for root, dirs, names in os.walk(path):           
    for name in names:
        ext = os.path.splitext(name)[1].replace('\\','/')
        if ext == suffix:
            event_dir = os.path.join(root,name)
            event_file.append(event_dir)

ea = event_accumulator.EventAccumulator(event_file[0])
ea.Reload()
ea_x = event_accumulator.EventAccumulator(event_file[1])
ea_x.Reload()
variables = ea.scalars.Keys()

# Extract data
dct = {}
for i in range(len(variables)):
    items = ea.scalars.Items(variables[i])
    data = [k.value for k in items]
    func_values = np.zeros(shape=len(data))
    for j in range(len(data)):
        func_values[j] =  data[j]    
    dct[variables[i]] = func_values
    
    items_x = ea_x.scalars.Items(variables[i])
    data_x = [k.value for k in items_x] 
    func_values_x = np.zeros(shape=len(data_x))
    for j in range(len(data_x)):
        func_values_x[j] =  data_x[j]    
    dct[variables[i]] = np.append(dct[variables[i]],func_values_x)   
    
dct['step'] = np.arange(len(data)+len(data_x))+1    

# Export data in excel files
excel_files = save_dir+'.xlsx'
dd = pd.DataFrame(dct)
dd.to_excel(excel_files)

# Read data from excel files
loss = pd.read_excel(excel_files,sheet_name='Sheet1')
step = loss.loc[:,'step']
epoch_loss = loss.loc[:,'epoch_loss']
epoch_rpn_class_loss = loss.loc[:,'epoch_rpn_class_loss']
epoch_rpn_bbox_loss = loss.loc[:,'epoch_rpn_bbox_loss']
epoch_mrcnn_class_loss = loss.loc[:,'epoch_mrcnn_class_loss']
epoch_mrcnn_bbox_loss = loss.loc[:,'epoch_mrcnn_bbox_loss']
epoch_mrcnn_mask_loss = loss.loc[:,'epoch_mrcnn_mask_loss']
epoch_val_loss = loss.loc[:,'epoch_val_loss']
epoch_val_rpn_class_loss = loss.loc[:,'epoch_val_rpn_class_loss']
epoch_val_rpn_bbox_loss = loss.loc[:,'epoch_val_rpn_bbox_loss']
epoch_val_mrcnn_class_loss = loss.loc[:,'epoch_val_mrcnn_class_loss']
epoch_val_mrcnn_bbox_loss = loss.loc[:,'epoch_val_mrcnn_bbox_loss']
epoch_val_mrcnn_mask_loss = loss.loc[:,'epoch_val_mrcnn_mask_loss']

# Plots of learning curves
fig, axs = plt.subplots(2,3,sharex='col',figsize=(15,8))
axs[0,0].plot(step,epoch_loss,label='Training',color='blue',linewidth=1.5)
axs[0,0].plot(step,epoch_val_loss,label='Validation',color='red',linewidth=1.5)
axs[0,0].set_xlabel('epoch')
axs[0,0].set_ylabel('loss')
axs[0,0].legend()
axs[0,1].plot(step,epoch_rpn_class_loss,label='Training',color='blue',linewidth=1.5)
axs[0,1].plot(step,epoch_val_rpn_class_loss,label='Validation',color='red',linewidth=1.5)
axs[0,1].set_xlabel('epoch')
axs[0,1].set_ylabel('rpn_class_loss')
axs[0,1].legend()
axs[0,2].plot(step,epoch_rpn_bbox_loss,label='Training',color='blue',linewidth=1.5)
axs[0,2].plot(step,epoch_val_rpn_bbox_loss,label='Validation',color='red',linewidth=1.5)
axs[0,2].set_xlabel('epoch')
axs[0,2].set_ylabel('rpn_bbox_loss')
axs[0,2].legend()
axs[1,0].plot(step,epoch_mrcnn_class_loss,label='Training',color='blue',linewidth=1.5)
axs[1,0].plot(step,epoch_val_mrcnn_class_loss,label='Validation',color='red',linewidth=1.5)
axs[1,0].set_xlabel('epoch')
axs[1,0].set_ylabel('mrcnn_class_loss')
axs[1,0].legend()
axs[1,1].plot(step,epoch_mrcnn_bbox_loss,label='Training',color='blue',linewidth=1.5)
axs[1,1].plot(step,epoch_val_mrcnn_bbox_loss,label='Validation',color='red',linewidth=1.5)
axs[1,1].set_xlabel('epoch')
axs[1,1].set_ylabel('mrcnn_bbox_loss')
axs[1,1].legend()
axs[1,2].plot(step,epoch_mrcnn_mask_loss,label='Training',color='blue',linewidth=1.5)
axs[1,2].plot(step,epoch_val_mrcnn_mask_loss,label='Validation',color='red',linewidth=1.5)
axs[1,2].set_xlabel('epoch')
axs[1,2].set_ylabel('mrcnn_mask_loss')
axs[1,2].legend()
plt.suptitle(log,fontsize=18,fontweight='bold')
fig.tight_layout()
plt.savefig(save_dir+'_sum.jpg',dpi = 300) # 保存图片
plt.show()

#%% Plots from multiple log files (continue training)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#### CHECK THIE BLOCK FIRST ####
os.chdir('C:\\FishSeg\\logs\\') 
save_path = os.path.join(os.getcwd()+'\\LogFiles')
logname = ['trout20220730T1415.xlsx','trout20220730T1818.xlsx'] # Names of log files

log = logname[0]
for i in range(len(logname)-1):
    log = log+'+'+logname[i+1]

class TempStore:
    pass

logdir = []
loss = [TempStore()]*len(logname)
epoch_loss = [TempStore()]*len(logname)
epoch_rpn_class_loss = [TempStore()]*len(logname)
epoch_rpn_bbox_loss = [TempStore()]*len(logname)
epoch_mrcnn_class_loss = [TempStore()]*len(logname)
epoch_mrcnn_bbox_loss = [TempStore()]*len(logname)
epoch_mrcnn_mask_loss = [TempStore()]*len(logname)
epoch_val_loss = [TempStore()]*len(logname)
epoch_val_rpn_class_loss = [TempStore()]*len(logname)
epoch_val_rpn_bbox_loss = [TempStore()]*len(logname)
epoch_val_mrcnn_class_loss = [TempStore()]*len(logname)
epoch_val_mrcnn_bbox_loss = [TempStore()]*len(logname)
epoch_val_mrcnn_mask_loss = [TempStore()]*len(logname)
step = [TempStore()]*len(logname)

for i in range(len(logname)):
    logdir = np.append(logdir, os.path.join(save_path+'\\'+logname[i]))
    loss[i] = pd.read_excel(logdir[i],sheet_name='Sheet1')
    epoch_loss[i] = loss[i].loc[:,'epoch_loss']
    epoch_rpn_class_loss[i] = loss[i].loc[:,'epoch_rpn_class_loss']
    epoch_rpn_bbox_loss[i] = loss[i].loc[:,'epoch_rpn_bbox_loss']
    epoch_mrcnn_class_loss[i] = loss[i].loc[:,'epoch_mrcnn_class_loss']
    epoch_mrcnn_bbox_loss[i] = loss[i].loc[:,'epoch_mrcnn_bbox_loss']
    epoch_mrcnn_mask_loss[i] = loss[i].loc[:,'epoch_mrcnn_mask_loss']
    epoch_val_loss[i] = loss[i].loc[:,'epoch_val_loss']
    epoch_val_rpn_class_loss[i] = loss[i].loc[:,'epoch_val_rpn_class_loss']
    epoch_val_rpn_bbox_loss[i] = loss[i].loc[:,'epoch_val_rpn_bbox_loss']
    epoch_val_mrcnn_class_loss[i] = loss[i].loc[:,'epoch_val_mrcnn_class_loss']
    epoch_val_mrcnn_bbox_loss[i] = loss[i].loc[:,'epoch_val_mrcnn_bbox_loss']
    epoch_val_mrcnn_mask_loss[i] = loss[i].loc[:,'epoch_val_mrcnn_mask_loss']

# Following lines should be commented based on the length of "logname"
step[0] = range(len(loss[0]))
step[1] = range(len(loss[0]),len(loss[0])+len(loss[1]))
# step[2] = range(len(loss[0])+len(loss[1]),len(loss[0])+len(loss[1])+len(loss[2])) # This line should be uncommented if there are three logfiles

fig, axs = plt.subplots(2,3,sharex='col',figsize=(15,8))
for i in range(len(logname)):
    axs[0,0].plot(step[i],epoch_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[0,0].plot(step[i],epoch_val_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[0,0].set_xlabel('epoch')
    axs[0,0].set_ylabel('loss')
    axs[0,1].plot(step[i],epoch_rpn_class_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[0,1].plot(step[i],epoch_val_rpn_class_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[0,1].set_xlabel('epoch')
    axs[0,1].set_ylabel('rpn_class_loss')
    axs[0,2].plot(step[i],epoch_rpn_bbox_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[0,2].plot(step[i],epoch_val_rpn_bbox_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[0,2].set_xlabel('epoch')
    axs[0,2].set_ylabel('rpn_bbox_loss')
    axs[1,0].plot(step[i],epoch_mrcnn_class_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[1,0].plot(step[i],epoch_val_mrcnn_class_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[1,0].set_xlabel('epoch')
    axs[1,0].set_ylabel('mrcnn_class_loss')
    axs[1,1].plot(step[i],epoch_mrcnn_bbox_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[1,1].plot(step[i],epoch_val_mrcnn_bbox_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[1,1].set_xlabel('epoch')
    axs[1,1].set_ylabel('mrcnn_bbox_loss')
    axs[1,2].plot(step[i],epoch_mrcnn_mask_loss[i],label='Training',color='blue',linewidth=1.5)
    axs[1,2].plot(step[i],epoch_val_mrcnn_mask_loss[i],label='Validation',color='red',linewidth=1.5)
    axs[1,2].set_xlabel('epoch')
    axs[1,2].set_ylabel('mrcnn_mask_loss')
plt.suptitle(log,fontsize=18,fontweight='bold')
fig.tight_layout()
plt.savefig(save_path+'\\0731connect.jpg',dpi = 300)
plt.show()
