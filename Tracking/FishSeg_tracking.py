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
# This script is written for video tracking based on established model.
# Four main sections are included in the script.
# (1) Pre-define the model framework
# (2) Open an interactive session for tracking
# (3) Define functions for video tracking
# (4) Video testing

# In section(2), you may select to start tracking based on the last trained weight (Block 1) 
    # or specified model (Block 2, e.g.,Run_1/TroutModel.h5)

#%% Check the following things before you start

GPU_num = "1" # Set the name of GPU that you want to use (0/1)
GPU_percent = 0.3 # Maximum consumption of GPU limit to 0.3 (1 image_per_gpu)

threshold = 0.96 # tracking score higher than the threshold is considered as effective tracks.
video_path = 'D:\\Tracking_Workflow\\FishTracking\\BGS Videos' # Folder path for all BGS videos
output_path = 'D:\Tracking_Workflow\\FishTracking\\test_outputs' # Output folder path

classes = {1: 'trout'} # The same as the class name that you spefify in the annotation step.
NAMEX = 'trout' # Consistent with class name specified above
FishSeg_model = "Run_1/TroutModel.h5" # established model name

# Set the number range for cases that you want to track
start_num = 300
end_num = 310

#%% Pre-define the model framework

##### import everything and set paths to coco weights, log dir ####
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid AVX Warning from using CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_num

import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import h5py
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from glob import glob

# Root directory of the project
ROOT_DIR = os.path.abspath('/FishSeg/') 

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
gpus = tf.config.experimental.list_physical_devices('GPU') # If there is any GPU available
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True) # Allocating GPU memory
config.gpu_options.per_process_gpu_memory_fraction = GPU_percent
sess = tf.compat.v1.InteractiveSession(config=config) 

# for visualization
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

#### replicate the classes used in TrackUtil annotations and create a config instance used for training ####
class Config(Config):
    NAME = NAMEX

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
    
    NUM_CLASSES = len(classes) + 1  # background + number of classes

    IMAGE_MIN_DIM = 1024 # set resolution for training (image downsampling)
    IMAGE_MAX_DIM = 1024
    
    IMAGE_RESIZE_MODE = 'square'

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 128, 256, 512) 

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 250 # 128 in the original line

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300 

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
    
    MAX_GT_INSTANCES = 5  # maximum number of instances per image(100)
    
    DETECTION_MIN_CONFIDENCE = 0.85
    
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 0.5, 'mrcnn_mask_loss': 0.3}
    
    BACKBONE = 'resnet101' 
    
config = Config()
config.display()

#### define our custom Dataset class that loads the datasets exported with TrackUtil ####
from multiprocessing import Lock, Value, Pool, cpu_count
import tqdm

def init_pool(l, c):
    global lock, count
    lock = l
    count = c

class Dataset(utils.Dataset):
    
    def load_images(self, annotation_file, classes, source, mode='full'):
        self.classes = classes
        self.annotation_file = annotation_file
        self.temp = os.path.join(os.path.join(os.path.dirname(annotation_file), os.path.split(annotation_file)[-1] + 'temp'))
        if not os.path.exists(self.temp):
            os.mkdir(self.temp)
        with h5py.File(self.annotation_file, 'r') as h5_file:
            for c in self.classes:
                self.add_class(source, c, self.classes[c])
            self.images = h5_file['annotations'][:].astype(str)
        np.random.seed(0)
        image_idx = np.arange(self.images.size)
        if mode == 'train':
            val_idx = np.random.choice(image_idx, image_idx.size // 20)
            self.images = self.images[np.invert(np.isin(image_idx, val_idx))]
        elif mode == 'val':
            val_idx = np.random.choice(image_idx, image_idx.size // 20) # increase number of validation images
            self.images = self.images[np.invert(np.isin(image_idx, val_idx))]
        else:
            print('Warning: set mode to "train" or "val", otherwise using full dataset')
        
        # Read and write image data from .h5 annotation file
        for i in range(len(self.images)):
            path = os.path.join(self.images[i],'image')
            path = path.replace('\\','/') # Essential line to move from Linux/Ubuntun system to Windows
            img_path = os.path.join(self.temp, os.path.split(path)[0].replace('/', '-').replace(':', '') + '.png')
            if os.path.exists(img_path):
                height, width = cv2.imread(img_path).shape[:2]
            else:
                with h5py.File(annotation_file, 'r') as h5_file:
                    height, width = h5_file[path][:].shape[:2]
                    image = h5_file[path][:]
                    cv2.imwrite(img_path, image[:, :, ::-1])
            self.add_image(source,image_id=i,path=path,width=width,height=height)      
    
    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        img_path = os.path.join(self.temp, os.path.split(path)[0].replace('/', '-').replace(':', '') + '.png')
        print(img_path)
        image = cv2.imread(img_path)[:, :, ::-1]
        return image
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img_path = info['path']
        path = os.path.split(img_path)[0]
        with h5py.File(self.annotation_file, 'r') as h5_file:
            mask_path = os.path.join(path, 'mask')
            mask_path = mask_path.replace('\\','/')
            mask = h5_file[mask_path][:]
            class_names_path = os.path.join(path,'class_names')
            class_names_path = class_names_path.replace('\\','/')
            classes = h5_file[class_names_path][:].astype(str)
        use = np.array([idx for idx, name in enumerate(classes) for c in self.classes if name in self.classes[c].split(',')], dtype=np.int32)
        class_ids = np.array([c for name in classes for c in self.classes if name in self.classes[c].split(',')], dtype=np.int32)
        mask = mask[:, :, use]
        non_empty = mask.sum(axis=(0, 1)) > 10
        return mask[:, :, non_empty], class_ids[non_empty]
    
    def prepare(self):
        super().prepare()
        print('{} images, classes: '.format(len(self.image_ids)), *['[{}: {}]'.format(idx, self.classes[idx]) for idx in self.classes])

#%% Open an interactive session for tracking

#### create a config instance used for inference ####
class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # 1 for visualize validation datasets; 3 for video testing
    
    IMAGE_MAX_DIM = 1024 # Formal training
    IMAGE_MIN_DIM = 1024
    
    IMAGE_RESIZE_MODE = 'square'
    
    MAX_GT_INSTANCES = 5  # Maximum number of instances per images
    
    Train_ROIs_Per_Image = 250
    DETECTION_MIN_CONFIDENCE = 0.8
    
inference_config = InferenceConfig()
batch_size = inference_config.BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

## Select Block 1 or 2 depending on your needs
#### Block 1: load the last training weight for inference ####
# model.load_weights(model.find_last(), by_name=True) 

#### Block 2: load weights from saved model ####
# test_model = os.path.join(ROOT_DIR, "trout/save_model/Run_1/", "test0731_epoch60.h5") 
test_model = os.path.join(ROOT_DIR,"trout/save_model/",FishSeg_model)
model.load_weights(test_model, by_name=True) # 72

#%% Define functions for video tracking

##### define some functions for inference and visualization ####
def generate_tiles(img, size=512, overlap=100, shifts=1, as_list=False):
    height, width = img.shape[:2]
    origins = np.mgrid[0:height:(size // shifts - overlap),0:width:(size // shifts - overlap)].T.reshape(-1,2)
    imgs = []
    for tl in origins:
        tile = img[tl[0]:(tl[0] + size), tl[1]:(tl[1] + size)]
        if tile.shape[0] < size and tile.shape[1] == size:
            tile = img[(height - size):height, tl[1]:(tl[1] + size)]
        elif tile.shape[0] == size and tile.shape[1] < size:
            tile = img[tl[0]:(tl[0] + size), (width - size):width]
        elif tile.shape[0] < size and tile.shape[1] < size:
            tile = img[(height - size):height, (width - size):width]
        imgs.append(tile)
    if as_list:
        return imgs
    return np.stack(imgs, axis=2)

def stitch_tiles(tiles, target_shape, size=512, overlap=100, shifts=1, flatten=False):
    height, width = target_shape[:2]
    origins = np.mgrid[0:height:(size // shifts - overlap),0:width:(size // shifts - overlap)].T.reshape(-1,2)
    img = np.zeros((height, width, np.sum([tile.shape[2] for tile in tiles])), dtype=np.uint8)
    idx = 0
    for tile, tl in zip(tiles, origins):
        if tl[0] + size > height:
            tl[0] = height - size
        if tl[1] + size > width:
            tl[1] = width - size
        img[tl[0]:(tl[0] + size), tl[1]:(tl[1] + size), idx:(idx + tile.shape[2])] = tile
        idx += tile.shape[2]
    if flatten:
        img = img.sum(axis=(2)) > 0
    return img

def stitch_rois(rois, target_shape, size=512, overlap=100, shifts=1):
    height, width = target_shape[:2]
    origins = np.mgrid[0:height:(size // shifts - overlap),0:width:(size // shifts - overlap)].T.reshape(-1,2)
    for idx, (roi, tl) in enumerate(zip(rois, origins)):
        if tl[0] + size > height:
            tl[0] = height - size
        if tl[1] + size > width:
            tl[1] = width - size
        rois[idx][:, ::2] += tl[0]
        rois[idx][:, 1::2] += tl[1]
    rois = np.concatenate(rois)
    return rois

def stitch_predictions(results, target_shape, size=512, overlap=100, shifts=1):
    stitched_result = {}
    stitched_result['masks'] = stitch_tiles([result['masks'] for result in results], target_shape, size, overlap, shifts)
    stitched_result['rois'] = stitch_rois([result['rois'] for result in results], target_shape, size, overlap, shifts)
    stitched_result['scores'] = np.concatenate([result['scores'] for result in results])
    stitched_result['class_ids'] = np.concatenate([result['class_ids'] for result in results])
    return stitched_result

import colorsys
from glob import glob
import h5py
import skvideo.io

def split_image(image, overlap=1/6):
    shape = image.shape[:2]
    overlap = round(min(shape) * overlap / 2)
    ul = np.concatenate([np.array([0, 0]), np.array(shape) / 2 + overlap]).astype(np.int)
    ur = np.array([0, shape[1] / 2 - overlap, shape[0] / 2 + overlap, shape[1]]).astype(np.int)
    bl = np.array([shape[0] / 2 - overlap, 0, shape[0], shape[1] / 2 + overlap]).astype(np.int)
    br = np.concatenate([np.array(shape) / 2 - overlap, shape]).astype(np.int)
    ul = image[ul[0]:ul[2], ul[1]:ul[3]]
    ur = image[ur[0]:ur[2], ur[1]:ur[3]]
    bl = image[bl[0]:bl[2], bl[1]:bl[3]]
    br = image[br[0]:br[2], br[1]:br[3]]
    return [ul, ur, bl, br]

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.25):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
        image = image.astype(np.uint8)
    return image

def draw_instances(image, boxes, masks, class_ids, class_names, scores=None):
    if len(image.shape) > 2 and image.shape[2] == 1:
        image = np.stack([image.reshape(*image.shape[:2])] * 3, axis=2)
    N = boxes.shape[0]
    if not N:
        return image
    else:
        colors = random_colors(N)
        height, width = image.shape[:2]
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            if not np.any(boxes[i]):
                continue
            y1, x1, y2, x2 = boxes[i]
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            color = [int(c) * 255 for c in color]
#             cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            cv2.putText(masked_image, caption, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return masked_image

#%% Video testing
import skvideo
import skvideo.io
import os

class TempStore:
    pass

def LoadVideofiles(video_path,output_path):
    folders = os.listdir(video_path)   
    folder_name = []
    for i in range(len(folders)-6):
        if int(folders[i][0:3])>=start_num and int(folders[i][0:3])<=end_num:
            folder_name.append(folders[i])
            
    inpath = [TempStore()]*len(folder_name)
    outpath = [TempStore()]*len(folder_name)
    video_files = [TempStore()]*len(folder_name)    
    
    for i in range(len(folder_name)):
        video_dirs = []
        inpath[i] = os.path.join(video_path,folder_name[i]) 
        outpath[i] = os.path.join(output_path,folder_name[i]+'_results')
        if not os.path.exists(outpath[i]):
            os.mkdir(outpath[i])
        for root, dirs, names in os.walk(inpath[i]):           
            for name in names:
                ext = os.path.splitext(name)[1].replace('\\','/')
                if ext == '.avi':
                    video_dir = os.path.join(root,name)
                    video_dirs.append(video_dir)
        video_files[i] = video_dirs
    return video_files,outpath,inpath

video_files,outpaths,inpaths = LoadVideofiles(video_path,output_path)

for i in range(len(video_files)):
    for j in range(len(video_files[i])):
        queue = []
        images = []
        results = []
        frame_idx = 0
        
        video_name = os.path.splitext(os.path.basename(video_files[i][j]))[0]
        with h5py.File(os.path.join(outpaths[i], 'predictions_' + video_name + '.h5'), 'w') as h5_file:
    
            predictions = h5_file.create_dataset('predictions', shape=(0, ), maxshape=(None, ), dtype='|S200')    
            cap = skvideo.io.vreader(video_files[i][j])

            for idx, frame in enumerate(cap):
                
                images.append(frame)
                tiles = [frame] 
                queue.extend(tiles)
                
                while len(queue) >= batch_size:
                    results.extend(model.detect(queue[:batch_size]))
                    queue = queue[batch_size:]
    
                while len(results) >= len(tiles):
                        
                    r = results[0] 
                    
                    group = h5_file.create_group(str(frame_idx))
                    group.create_dataset('masks', data=r['masks'], compression="gzip", compression_opts=9)
                    group.create_dataset('scores', data=r['scores'])
                    group.create_dataset('classes', data=np.array([classes[c] for c in r['class_ids']], dtype=np.bytes_))
                    
                    predictions.resize((predictions.shape[0] + 1, ))
                    predictions[-1] = np.bytes_(str(frame_idx))
                    
                    frame_idx += 1
                    images = images[1:]
                    results = results[len(tiles):]
                    
                    print('{}'.format(frame_idx), sep=' ', end='\r', flush=True)
            print('Finished tracking: '+video_files[i][j])
