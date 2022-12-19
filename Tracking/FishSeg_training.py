#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue Aug 23 14:43:36 2022

@author: yanfan
"""
#%% FishSeg tracking
# Part 2 Tracking
# Fan Yang, Aug 2022
# "Tracking" consists of 5 main scripts:
# (1) backgroundSubtraction.py: Do MOG2 background subtraction for experimental videos
# (2) FishSeg_training.py: Training FishSeg model and test video tracking based on the model;
# (3) FishSeg_tracking.py: Do video tracking based on the established model;
# (4) mask2tracks.py: Turn masks predicted by FishSeg into tracks;
# (5) ReadTensorboard.py: Read loss functions produced in log folder under C:\FishSeg 
    # check if the model get good performance;
    
#%% Read before you start
# This script is written for training FishSeg model with datasets and trying to do video tracking.

# Nine main sections are included in the script.
### Part 1: Model trainig ###
# (1) Check available GPUs
# (2) Pre-define the model framework
# (3) Load training and validation datasets
# (4) Start training
# (5) Open an interactive session for testing
# (6) Visualize the validation datasets
# (7) Save model
### Part 2: Model testing ###
# (8) Define functions for video tracking
# (9) Video testing
    # (a) For single video
    # (b) Video testing under specific folder

# In section(4), you may select to start training based on "coco/last" weights or
    # continue training based on the model that you previously saved.
# In section(5), you may select to start tracking based on the last trained weight (Block 1) 
    # or specified model (Block 2, e.g.,Run_1/TroutModel.h5)
# In section(9), you need to specify the "threshold","video_path" and "output_path" at the beginning of each part (a or b).
    
#%% Check the following things before you start

GPU_num = "1" # Set the name of GPU that you want to use (0/1)
GPU_percent = 0.6 # Maximum consumption of GPU limit to 0.6 (2 image per gpu)

classes = {1: 'trout'} # The same as the class name that you spefify in the annotation step.
NAMEX = 'trout' # Consistent with class name specified above

train_dd = '/FishSeg/trout/Dataset/newtraining304.h5' # path for training dataset
validation_dd = '/FishSeg/trout/Dataset/newvalidation115.h5' # path for validation dataset
epoch_head = 5 # epochs for training the head layer
epoch_all = 60 # epochs for fine-tuning all layers
save_model = '/FishSeg/trout/save_model/Run_1/TroutModel.h5' # path for model that you want to save

#%% Check available GPUs
import tensorflow as tf
print("Num GPU Available:",len(tf.config.experimental.list_physical_devices('GPU')))

#%% Pre-define the model framework

##### import everything and set paths to coco weights, log dir ####
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid AVX Warning from using CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Uncomment this line if you hope to use CPU only

# Use following block if you want to train and test on GPU:1
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
# classes = {1: 'trout'}

class Config(Config):
    NAME = NAMEX

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
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
            val_idx = np.random.choice(image_idx, image_idx.size // 20) 
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

#%% load training and validation datasets

#### use image augmentations via the imgaug package to avoid overfitting ####
augmentation = iaa.Sequential([   
    
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Affine(rotate=iap.Choice([0,90,180,270])),
    iaa.Affine(scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                      translate_percent={'x': (-0.2, 0.2), 'y': (-0.2, 0.2)},
                      rotate=(-45, 45))
    ], random_order=True)

#### Prepare training and validation dataset ####
# Training Dataset
dataset_train = Dataset()
dataset_train.load_images(annotation_file = train_dd,
                         classes=classes,
                         source = NAMEX,
                         mode='train')
dataset_train.prepare()

# Validation Dataset
dataset_val = Dataset()
dataset_val.load_images(annotation_file = validation_dd,
                        classes=classes,
                        source = NAMEX,
                        mode = 'val')
dataset_val.prepare()

##### Visualize some annotations #####
image_ids = dataset_val.image_ids[:10]
# image_ids = np.array([39]) # Uncomment this line if you hope to compare with a specified image
for image_id in image_ids:
    image = dataset_val.load_image(image_id)
    mask, class_ids = dataset_val.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_val.class_names, limit=1)

#%% Start training

#### create Mask_RCNN model ####
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR) # Define the model

#### Start training from pretrained weights ####
## Important: 
    # Block 1: if you want to start training from coco weights 
      # you can also use 'last' weights, but the weight files will not be continuously recorded in the same "logs" due to some internal network design
      # so we suggest you to save your model and use block 2 if you want to continue training
    # Block 2: if you want to continue training based on pretrained model
    # If you want you use block 1, you need to comment block 2 to avoid conflicts, and vice versa.

## Block 1 ##
# Which weights to start with?
init_with = 'coco'  # coco, or last
ly_names=['mrcnn_class_logits', 'mrcnn_bbox_fc', 
'mrcnn_bbox', 'mrcnn_mask']
# ly_decodes=[unicode(s) for s in ly_names]

if init_with == 'coco':
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                      exclude=ly_names)
elif init_with == 'last':
    model.load_weights(model.find_last(), by_name=True) # model.find_last()

## Block 2 ##
# MY_MODEL_PATH =os.path.join(ROOT_DIR, "trout/save_model/Run_1/", "test0731_epoch60.h5") 
# ly_names=['mrcnn_class_logits', 'mrcnn_bbox_fc', 
# 'mrcnn_bbox', 'mrcnn_mask']
# model.load_weights(MY_MODEL_PATH,by_name=True,
#                     exclude=ly_names)

#### Train the model layers ####
# Train the head layers
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE, 
            epochs=epoch_head, 
            layers='heads',
            augmentation=augmentation)

# Fine-tune all layers
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=epoch_all, 
            layers='all',
            augmentation=augmentation)

#%% Open an interactive session for testing

#### create a config instance used for inference ####
class InferenceConfig(Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1 # 1 for visualize validation datasets; 3 for video testing
    
    IMAGE_MAX_DIM = 1024 
    IMAGE_MIN_DIM = 1024
    
    IMAGE_RESIZE_MODE = 'square'
    
    MAX_GT_INSTANCES = 5
    
    Train_ROIs_Per_Image = 250
    DETECTION_MIN_CONFIDENCE = 0.8
    
inference_config = InferenceConfig()
batch_size = inference_config.BATCH_SIZE

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

## Select Block 1 or 2 depending on your needs
#### Block 1: load the last training weight for inference ####
model.load_weights(model.find_last(), by_name=True) 

#### Block 2: load weights from saved model ####
# test_model = os.path.join(ROOT_DIR, "trout/save_model/Run_1/", "test0731_epoch60.h5") 
# model.load_weights(test_model, by_name=True) # 72

#%% Visualize the validation datasets

#### visualize inference on some images of the validation set ####
threshold = 0.96

image_ids = np.random.choice(dataset_val.image_ids, 10) 
# image_ids = np.array([39]) # Uncomment this line if you want to compare with a specified image

for image_id in image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)

    fig, ax = plt.subplots(1, 2, figsize=(32, 32))

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                                dataset_train.class_names, figsize=(16, 16), ax=ax[0])

    results = model.detect([original_image], verbose=0)

    r = results[0]
    visualize.display_instances(original_image, r['rois'][r['scores'] > threshold], r['masks'][:, :, r['scores'] > threshold], r['class_ids'][r['scores'] > threshold], 
                                dataset_val.class_names, r['scores'][r['scores'] > threshold], ax=ax[1])

    plt.show()

#%% Save model
from tensorflow import keras
model.keras_model.save(save_model)

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

#### For single video testing ####

import skvideo
import skvideo.io
import os

threshold = 0.96
video_file = 'S:\\vaw_public\\yanfan\\Demo\\Videos_test\\300_2020_19_10_09_00_mog2\\Cam_0_mog2.avi'
outpath = 'S:\\vaw_public\\yanfan\\Demo\\Videos_test\\Temp_outputs'

queue = []
images = []
results = []
frame_idx = 0

video_name = os.path.splitext(os.path.basename(video_file))[0]
with h5py.File(os.path.join(outpath, 'predictions_' + video_name + '.h5'), 'w') as h5_file:

    predictions = h5_file.create_dataset('predictions', shape=(0, ), maxshape=(None, ), dtype='|S200')

    cap = skvideo.io.vreader(video_file) 
    
    # Comment following line if no image outputs are needed
    # os.makedirs(os.path.join(outpath, 'predictions_' + video_name), exist_ok=True)

    for idx, frame in enumerate(cap):
        
        images.append(frame)
        tiles = [frame] # use this line for faster and smaller outputs
        # tiles = generate_tiles(frame, size=1024, as_list=True) # use this line if images are too big to load for inference
        queue.extend(tiles)
        
        while len(queue) >= batch_size:
            results.extend(model.detect(queue[:batch_size]))
            queue = queue[batch_size:]

        while len(results) >= len(tiles):
                
            r = results[0] # use this line for faster and smaller outputs
            # r = stitch_predictions(results[:len(tiles)], images[0].shape, size=1024) # use this line if images are too big to load for inference
            
            # comment following block to prevent image output
            # masked_img = draw_instances(images[0], r['rois'][r['scores'] > threshold], r['masks'][:, :, r['scores'] > threshold], r['class_ids'][r['scores'] > threshold], 
            #             dataset_val.class_names, r['scores'][r['scores'] > threshold])
            # cv2.imwrite(os.path.join(outpath, 'predictions_' + video_name, 'pred_{:>05}.jpg').format(frame_idx), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
            
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
    print('Finished tracking: '+video_file)

#%% Video testing under specific folder
import skvideo
import skvideo.io
import os

threshold = 0.96
video_path = 'S:\\vaw_public\\yanfan\\FishTracking\\BGS Videos\\342_2020_27_10_16_26_mog2'
output_path = 'S:\\vaw_public\\yanfan\\FishTracking\\Outputs'

outpath = os.path.join(output_path,video_path[45:]+'_results') # Note:[45:] may be changed if you have different path names

if not os.path.exists(outpath):
    os.makedirs(outpath)
    
def LoadVideofiles(video_path,output_path):
    video_files = []
    for root, dirs, names in os.walk(video_path):           
        for name in names:
            ext = os.path.splitext(name)[1].replace('\\','/')
            if ext == '.avi':
                video_dir = os.path.join(root,name)
                video_files.append(video_dir)
    return video_files

video_files = LoadVideofiles(video_path,output_path)

for i in range(len(video_files)):
        queue = []
        images = []
        results = []
        frame_idx = 0
        
        video_name = os.path.splitext(os.path.basename(video_files[i]))[0]
        with h5py.File(os.path.join(outpath, 'predictions_' + video_name + '.h5'), 'w') as h5_file:
    
            predictions = h5_file.create_dataset('predictions', shape=(0, ), maxshape=(None, ), dtype='|S200')
    
            cap = skvideo.io.vreader(video_files[i])
            
            # Comment following line if no image outputs are needed
            # os.makedirs(os.path.join(output_path, 'predictions_' + video_name), exist_ok=True)
    
            for idx, frame in enumerate(cap):
                
                images.append(frame)
                tiles = [frame] # use this line for faster and smaller outputs
                # tiles = generate_tiles(frame, size=1024, as_list=True) # use this line if images are too big to load for inference
                queue.extend(tiles)
                
                while len(queue) >= batch_size:
                    results.extend(model.detect(queue[:batch_size]))
                    queue = queue[batch_size:]
    
                while len(results) >= len(tiles):
                        
                    r = results[0] # use this line for faster and smaller outputs
                    # r = stitch_predictions(results[:len(tiles)], images[0].shape, size=1024) # use this line if images are too big to load for inference
                    
                    # comment following block to prevent image output
                    # masked_img = draw_instances(images[0], r['rois'][r['scores'] > threshold], r['masks'][:, :, r['scores'] > threshold], r['class_ids'][r['scores'] > threshold], 
                    #             dataset_val.class_names, r['scores'][r['scores'] > threshold])
                    # cv2.imwrite(os.path.join(output_path, 'predictions_' + video_name, 'pred_{:>05}.jpg').format(frame_idx), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
                    
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
            print('Finished tracking: '+video_files[i])