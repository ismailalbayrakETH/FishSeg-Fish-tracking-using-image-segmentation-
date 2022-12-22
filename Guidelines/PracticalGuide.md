<!-- TOC -->

- [Practical Guide for fish tracking workflow](#practical-guide-for-fish-tracking-workflow)
- [Fisheye camera calibration](#fisheye-camera-calibration)
    - [Calibration procedures](#calibration-procedures)
    - [Check before you start](#check-before-you-start)
- [2 D fish tracking](#2-d-fish-tracking)
    - [Installation](#installation)
        - [For FishSeg](#for-fishseg)
        - [For TrackUtil](#for-trackutil)
        - [Instructions for python beginners](#instructions-for-python-beginners)
        - [Instructions for python beginners](#instructions-for-python-beginners)
    - [(Optional) Step 1: Pre-processing of videos](#optional-step-1-pre-processing-of-videos)
    - [Step 2: Make datasets](#step-2-make-datasets)
    - [Step 3: Segmentation and tracking](#step-3-segmentation-and-tracking)
    - [Step 4: Post-processing of tracks](#step-4-post-processing-of-tracks)
- [Turn 2D tracks into 3D metric space](#turn-2d-tracks-into-3d-metric-space)
- [Q&A](#qa)

<!-- /TOC -->
# Practical Guide for fish tracking workflow
Hallo! Welcome to the introduction of VAW 3D fish tracking system. Here, you will see how 3D fish tracking system works step by step. Just get yourself a computer and experience this magic 3D world! 
Before we start, let's have a general idea of how 3D tracking is done first. In our project, 3D tracking is conducted in three major steps, which are:
1. Fisheye camera calibration (matlab)
2. X-Y plane (2D) fish tracking (python)
3. Turn 2D coordinates into 3D metric space (matlab)

As you can see, the whole system is implemented using two programming language (matlab and python). So you will need to have MATLAB and Anaconda/Miniconda/Colab on your computer first to get started. Note that we have already tested our system with Matlab R2021b and Anaconda3/Colab.

# Fisheye camera calibration
## Calibration procedures
The fisheye camera calibration is done following the [fisheye camera model](https://ch.mathworks.com/help/vision/ug/fisheye-calibration-basics.html).
Herein, there are three main "DOITALL" functions that you need to run one by one. The instructions and necessary files in the Matlab path are specified in the header. 
1. Kalibing_11_DOITALL.m
    a. Search for checkerboard intersections
2. Undisting_11_DOITALL.m
    a. Undistort single camera
    b. Connect pairwise cameras using checkerboard intersections
    c. Get intrinsic and extrinsic camera parameters
3. Bring2FlumeKOS58_12_2022_DOITALL.m (some manual labor needed)
    a. Undist fisheye images, link 5 cameras together by reference points at the flume bottom
    b. Triangulate 2D points for 3D positions (flume coordinate system)

## Check before you start
Make sure you have checked the following tips before you start calibration.
1. funcID should be consistent over all used codes/functions
2. Specify correct directory for "datfir" -- Video Files for Calibration
3. Specify correct directory for "savdir" -- Path for Data Storage
4. Adapt MAINFOLDERS -- Naming *CLB_2020_12_11_14_04*, containing video and excel sheets for all cameras

An example of "datdir" and "savedir" setup for different **DOITALL** functions:
* Kalibing_DOITALL.m
    * datdir = 'Y:\'
    * savdir = 'D:\FishSeg_Results\CalibResults'
* Undisting_DOITALL.m
    * datdir = 'D:\FishSeg_Results\CalibResults'
    * savdir = 'D:\FishSeg_Results\CalibResults\CLB_2020_12_11_14_04_xxx#xxx'
* Bring2Flume_DOITALL.m
    * datdir = 'Y:\'
    * savdir = 'D:\FishSeg_Results\CalibResults\CLB_2020_12_11_14_04_xxx#xxx'

# 2D fish tracking
Basically, we apply the [Mask_RCNN](https://github.com/ErinYang96/FishSeg.git) model, which is a typical deep learning model for instance segmentation, to conduct 2D fish tracking.
Two steps are required to conduct the 2D fish tracking.
1. (Optional) Pre-processing of videos (VideoProcessing)
2. Fish segmentation and tracking (using FishSeg)
3. Post-processing of tracks (using TrackUtil)

Note that this process with deep learning model can be conducted using CPU. But with NVIDIA GPU, it is much faster. See the following *Installation* section for more instructions about how to implement the python packages.

## Installation
### Enable GPU usage on a Computer/Server using a NVIDIA GPU
In order to enable the python scripts to use your GPU and therefore run much faster than on CPU, you need to copy the NVIDIA GPU Computing Toolkit under your C folder. 
For vawsrv02 this is already done. 

Each user needs to Click start -> Type env -> select [Edit environment variables for your account] -> Select [Path] and click on [Edit]
Here you need to add the following 3 paths: 

```
C:\Apps\NVIDIA GPU Computing Toolkit\CUDA\v11.2
C:\Apps\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
C:\Apps\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64
```

### For FishSeg
**Requirements**
Python 3.8.10, Tensorflow 2.7.0, Keras 2.7.0, and other common packages listed in `requirements.txt`. The installation of these packages is included in the step-by-step guide below. 

**Step-by-step installation guide**
1. Create a virtual environment called `FishSeg`
    ```
    conda create -n FishSeg python=3.8.10
    conda activate FishSeg
    conda install git
    ```
2. Clone this repository or alternatively download and unzip and copy it to C:\
    ```
    git clone https://github.com/ErinYang96/FishSeg.git
    ```
3. Go to the repository root directory of above packages
    ```
    cd FishSeg
    ```
4. Install dependencies
    ```
    pip3 install -r requirements.txt
    ```
5. Run setup from the repository root directory
    ```
    python setup.py install
    ```
    In this step, check the root directory of site-packages in this environment (e.g. *C:\Apps\Anaconda3\envs\FishSeg\Lib\site-packages*) to see if there are two folders called *mask_rcnn_tf2-1.0.dist-info* and *mrcnn*. If not, you need to manually copy the two folders under *For_site_packages*  folder in above path.
6. Other package installation to assist in your work
    a. (Mandatory) To enable video testing
    ```
    conda install -c menpo ffmpeg
    ```
    b. (Optional) To add menu list for your jupyter notebook
    ```
    jupyter contrib nbextension install --user
    ```
    If you get the error message `ImportError: cannot import 'contextfilter' from 'jinja2'`, you need to further update your `jinja2` package by running:
    ```
    conda install jinja2==3.0.3
    ```
    and then repeat the former command again to make the jupyter extension work. To turn on the menu function for jupyter notebook, just follow: *jupyter notebook -> Nbextensions -> Table of Contents(2)*.

**How to activate the model afterwards**

Assuming everything works well, you can then use this model by excuting following commands on your Anaconda Prompt.
```
conda activate FishSeg # not necessary if still active
cd FishSeg
jupyter notebook
```
The last command `jupyter notebook` can also be replaced by `spyder` if you hope to tune the model for your specific needs.
### For TrackUtil
TrackUtil is a good python app with a purpose-oriented GUI, which has two main functions:
1. Make polygon-based video annotations for multiple classes (Get training and validation datasets)
2. Visualize, check and correct trajectories with optional video playback (Check the accuracy of tracks outputs by FishSeg)

More detailed introductions about TrackUtil can be found [here](https://gitlab.mpcdf.mpg.de/pnuehren/trackutil). For your convenience, I just repeat the installation procedures here, which is basically the same as FishSeg. You just need to run the commands one by one through Anaconda Prompt.
```
conda create -n trackutil python=3.9
conda activate trackutil
conda install git
git clone https://gitlab.mpcdf.mpg.de/pnuehren/trackutil.git
cd trackutil
pip install -r REQUIREMENTS.txt
```
Sometimes you may encounter a warning such as: *ModuleNotFoundError: No module named 'scipy'*. If this occurs, just run `conda install scipy` to fix this bug.
Now, assuming eveything works well, later you can run **TrackUtil** by excuting the following command.
```
conda activate trackutil # not necessary if still active
python TrackUtil.py
```
### Instructions for python beginners
```
conda deactivate # go back to the base environment
cd C:\ # go to the root directory of C disk
conda remove -n [envs name] --all # remove specific environment
conda info --envs # check all environments that you currently have
```

## (Optional) Step 1: Pre-processing of videos
Althrough Mask_RCNN model is powerful for instance segmentation of objects moving in various backgrounds, it may not perform well if the color contrast between objects and backgrounds is too low. We have to say that this is the case for our project, where fish, especially trout, sometimes can be barely seen with the color of flume bottom so close to the fish itself. Herein, we have developed an extra pre-processing step to help improve the subsequent model performance. If you encounter similar situations as we do, you may apply this procedure first instead of stepping directly into the deep learning model.

The *pre-processing* step is mainly based on an OpenCV [backgroundSubtractorMOG2](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) method, which is applied in the `backgroundSubtraction.py`. This function enables you to do background subtraction for single video, video sets under specific folder, and videos under multiple folders. Two features are especially designed for our tracking projects.
1. Decide the start time when fish enter the shooting area to discard the periods with no fish.
2. Set frequency of frame selection, which is quite useful if
    a. you have long-term videos;
    b. you have short-term videos with high framerates;
    c. you don't need very precise tracking information;
    d. you want to reduce tracking time;

## Step 2: Make datasets
If you have videos where the target objects are easy to find, you can directly use the *Annotation* function of [TrackUtil](https://gitlab.mpcdf.mpg.de/pnuehren/trackutil) to make datasets. Note that you can play multiple videos simultaneously using the *File* menu > *Connect*. Also, *File* menu > *Write to dataset* can extend an existing dataset by adding the annotations of the current video, which is quite a useful function if you want to build one dataset from more than one cameras.

However, if you have multiple videos where the target objects enter and leave the observation zone at random time, you may want to collect the video clips where target objects are present, which is exactly our case. To deal with this problem, we further implement two more python scripts (list below).
1. `get_videoclip.py`

    Get video clips for periods when the target objects show up, and then apply the backgroundSubtractorMOG2 method. A small number of video clips with representive or hard-to-detect configurations is enough. Note that a series of un-preprocessed video clips is also clipped as temporary files.
    
2. `concatenate_clips`

   Concatenate all pre-processed video clips obtained in the "get_videoclip" step to generate the final video for making datasets.

It is suggested that you organize your folder in the following way.
```
| -- VideoClips
    | -- BGS clips_trout
        | -- 386_2020_05_11_08_18_bgs
            | -- Cam_0_bgs.avi
            | -- Cam_1_bgs.avi
        | -- 387_2020_05_11_09_19_bgs
        | -- final.avi
    | -- Temp clips_trout
        | -- 386_2020_05_11_08_18_temp
            | -- Cam_0_temp.avi
            | -- Cam_1_temp.avi
        | -- 387_2020_05_11_09_19_temp
```



## Step 3: Training and Tracking
Currently, We have two sets of codes which are suitable for *Google Colab (Ubuntu System)* and *Windows 10*, respectively. Please select one of them according to your operating system. Note that if you hope to further develop the code for your Linus system, it is recommended that you start from the code for Colab.

For those who don't have a NVIDA GPU on your PC or have GPUs of low memory, we recommand you to use Colab, which have free GPU resources and is quite suitable for beginners who have limited knowledge about python. If your have no idea about Colab before, check [here](https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d) to get an overview of what Colab is capable of. Please pay special attention to _how to mount on Google Drive_ and _how to set free GPU_, which are essential for your start. Note that Google Drive only provides 15G for data storage, which means Colab is not a good option for long-term training, since it will produce log data which can consume your RAM quickly.

In addition, in case you may want to check your datasets or adjust the proportion between training and validation datasets for better model performance, `datasets.py` is implemented to make it easier for you with no need to make annotations over again.

**Workflow:**
If a model has already been trained, Step 3 and 6 are optional 
1) Tracking\StartTime.py - run all
2) Tracking\backgroundSubtraction.py - run only the first cell!
3) (Optional) Tracking\FishSeg_training.py
4) Tracking\FishSeg_tracking.py
5) Tracking\mask2tracks.py
6) (Optional) Tracking\ReadTensorboard.py

**StartTime.py** will takes as input a text file specifying which videos you want to track and the time when a first fish was observed in the experimental area. Due to a particularity of the camera system used at VAW, the videos are recorded using a framerate of 20fps but saved with 25fps. StartTime.py generates a text file specifying the start time for the tracking assuming 25fps. 

Example Folder Structure: 
```
TrackingGroups = 'D:\TrackingPlan\Exp_092-120.txt'
StartTime = 'D:\TrackingPlan\StartTime_Exp_092-120.txt'
```

**BackgroundSubstraction.py** generates background subtracted Videos and at determines if each frame (freq=1) or only every other frame (freq=2) should be tracked. 
Example Folder Structure: 
```
videoinpath = 'D:\\' #Location of the videos to be tracked
videooutpath = 'E:\\vaw_efigus\\3_TrackingOutput\\BGS_Videos'
xlsrID = 'FishGroup_092-120.txt'  #Should be located in videooutpath Folder!
```
**FishSeg_training.py** generates a training dataset. After you finish training the model (`FishSeg_training.py`), you need to check the model performance using _Tensorboard_ or our `ReadTensorboard.py`, which enables you to export loss metrics and images with learning curves. For details about how to diagnose model performance from learning curves, please go to Q&A for more information.

**FishSeg_tracking.py** does the actual tracking. In order to run, the pretrained model needs to be copied to 
```
C\FishSeg\trout\Datasets - eel_train.h5, eel_valid.h5, trout_train.h5 and trout_valid.h5
C\FishSeg\trout\save_model - EelModel.h5 and TroutModel.h5
```
Example Folder Structure: 
```
video_path = 'D:\\vaw_efigus\\3_TrackingOutput\\BGS_Videos' # Folder path for all BGS videos
output_path = 'D:\\vaw_efigus\\3_TrackingOutput\\FishSeg_tracking_output' # Output folder path
```
When you finish tracking, you will get tracking results (`FishSeg_tracking.py`) from the model in `.h5` format.
In order to run several trackings in parallel, open another instance of spyder and run a copy of FishSeg_tracking.py there. You can run 2 instances of spyder on each GPU. 


**mask2tracks.py** is needed to convert the masks to tracks (outputs in `.pkl` formate), and then export tracks to an excel spreadsheet. For `mask2tracks.py`, there are two main parameters which you can fine tune for better tracks, which are `min_overlap` and `max_merge_dist`. In addition, it is recommended that you delete unneccesary log files (in .h5 format under the _logs_ folder) to avoid slowing down the computer. You may keep the newest log in case it is needed when loading the _last_ weights.

Further instructions about how to use the model has been given within the code. 

**FishSeg** is basically written to be able to operate in two ways (after starting training from *COCO* weights): 
1. Load model for video testing from _last_ weights (_logs_ folder). With trained model, you may skip *Pre-definition* section, and start video testing from *From last-trained model (coco/last)* section.
2. Save current model in HD5F format (*save_model* folder), load model for video testing from pre-saved .h5 model. Same as above, in this case you can start video testing from *From saved model* section.





## Step 4: Post-processing of tracks
For post-processing of tracks, [Francisco et al. (2020)](https://movementecologyjournal.biomedcentral.com/articles/10.1186/s40462-020-00214-w) has already done great work by creating *TrackUtil*. In addition to *Annotation*, another key function, *tracks*, enables you to manually delete misidentified tracks, merge tracks, and interpolate segmented tracks. For more functions, go to the [Official Introduction Page of TrackUtil](https://gitlab.mpcdf.mpg.de/pnuehren/trackutil).
In this step, you can use `Load videos` and `Load tracks` to help with visualizing the tracks, and decide which tracks are the right ones. After finishing the tracks processing (merge, delete, and interpolate), you can further `save tracks` for later tracks connection.

# Turn 2D tracks into 3D metric space
This is done in Matlab. 
To turn 2D tracks identified from 5 fisheye cameras into flume coordinates, you need to run `XLMging_DOITALL_2D.m` for 2D tracks and `XLMging_DOITALL_3D.m` for 3D tracks. In this step, you can adjust the `minDistance` for tracks connection, and set proper threshold for accepted length of points cluster to filter out mis-identified tracks.

The folder for results (calibration and tracking) can organized in other disk (e.g. D disk) as follow.

```
| -- TrackingResults
    | -- Outputs
        | -- 300_2020_19_10_09_00_mog2_results
            | -- predictions_Cam_0_mog2.h5
            | -- tracks_Cam_0_mog2.pkl
            | -- tracks_Cam_0_mog2.xlsx
    | -- Results_initial
        | -- 300_2020_19_10_09_00_mog2_results
            | -- tracks_Cam_0_mog2.xlsx
            | -- tracks_Cam_0_doitall.mat
            | -- tracks_300_Cam_0.jpg
        | -- CLB_2020_08_10_16_51_xxx#xxx # Calibration results
    | -- Results_final
        | -- #000XXX_2020_00_00_00_01
            | -- 300_2020_19_10_09_00_mog2_results.mat
            | -- 300_2020_19_10_09_00_mog2_results_figi52tmp_58.fig
            | -- 300_2020_19_10_09_00_mog2_results_figi52tmp_58.png
        | -- 300_2020_19_10_09_00_mog2_results
```
* Outputs - Output from tracking.py
* initialresults - xlsx files 
* final results - just the plot and .m file you will need to further analyze your data. 

# Q&A
**Is there any way that enables me to train large datasets in Google Colab?**
If you have enough budget for your deep learning work, you can use *CoLab Pro* or even *CoLab Pro+*. Check [here](https://colab.research.google.com/signup?utm_source=faq&utm_medium=link&utm_campaign=how_billing_works) for more information.

**What should I do if I cannot use jupyter menu function following your instruction?**
Check [Jupyter nbextensions does not appear](https://stackoverflow.com/questions/49647705/jupyter-nbextensions-does-not-appear) to find more possible solutions. For more info, go to [documentation](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) and [its source code](https://github.com/ipython-contrib/jupyter_contrib_nbextensions).

**Why can't I use GPU on my computer?**
First, make sure that you GPU is produced by NVIDIA, not AMD or some other companies. For now, only NVIDIA GPU provides good supports for deep learning. If you find a way that enables the application of AMD GPU, we are glad to know about it.
Second, check if you have NVIDIA CUDA Toolkit and cuDNN installed on your computer. For this project, we have NVIDIA CUDA Toolkit 11.2 and cuDNN 8.1 to accelerate training process. However, the CUDA and cuDNN versions depend on your NIVIDIA Driver Version. Here I will give brief instructions about how to get the right CUDA and cuDNN installed. If it is not clear enough for you, you may google for more detailed guides.
1. Check your NVIDIA Driver Version by executing `nvidia-smi.exe` in Command Prompt. If you don't have a NVIDIA Driver yet, go to [Download NVIDIA Drivers](https://www.nvidia.cn/Download/index.aspx?lang=ch), and get the right version based on your NVIDIA product info.
2. Check which versions of CUDA and cuDNN your Driver is compatible with through [Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).
3. Download CUDA and cuDNN through [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-download). Note that cuDNN needs you to register on NVIDIA for downloads. But don't worry, it is free.
4. Install CUDA and cuDNN following [Installation Guide Windows: CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/) 
5. Check if the installation is successul by executing `nvcc -V` on Command Prompt. If it is, you will see your CUDA Toolkit version.

For other questions, go to [GPU support | Tensorflow](https://www.tensorflow.org/install/gpu) for possible answers.

**If I only have GPUs of limited memory, how can I test the model locally?**
Matterport team has given good answer for optimatizing GPU memory. See [here](https://github.com/matterport/Mask_RCNN/wiki).

**Can I run the model by other versions of tensorflow and keras?**
Well, as you may know, tensorflow and keras have strict limitations in version compatibility, otherwise strange errors may occur. Check [list of Available Environments](https://master--floydhub-docs.netlify.app/guides/environments/?msclkid=535e03adb67c11ec8e4eb4f27d020997) for compatible version matchings that have been tested. For now, in addition to the present versions, tensorflow 2.5.3 and keras 2.4.0 also works for this model. If you want to use other versions, you have to test them yourself.
For other possibilities, you can see [Which Python, Tensorflow, Keras, CUDA version for GPU usage?](https://github.com/matterport/Mask_RCNN/issues/2312) for more inspirations.
Also, there are other versions of Mask_RCNN branchs which may work for you, though they have not been formally tested yet. Please see [tomgross/Mask_RCNN for tf2.1](https://github.com/tomgross/Mask_RCNN/tree/71003370a447cbe39430622f007ea83bb1815c2b) and [ leekunhee/Mask_RCNN: Mask R-CNN modified to run on TensorFlow 2](https://github.com/leekunhee/Mask_RCNN).

**How can I get to know about the Mask R-CNN model as a beginner?**
I can provide you several examples, which have shown how the model is implemented (see below).
* [R-CNN, Fast R-CNN, Faster R-CNN, YOLO — Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)
* [Getting started with Mask R-CNN in Keras](https://gilberttanner.com/blog/getting-started-with-mask-rcnn-in-keras/)
* [A simple guide to MaskRCNN custom dataset implementation](https://medium.com/analytics-vidhya/a-simple-guide-to-maskrcnn-custom-dataset-implementation-27f7eab381f2)
* [Splash of Color: Instance Segmentation with Mask R-CNN and TensorFlow](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46)

**How can I know more internal details of current model?**
This fish tracking part is developed mainly based on two academic papers, which are [Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow](https://github.com/matterport/Mask_RCNN) and [High-resolution, non-invasive animal tracking and reconstruction of local environment in aquatic ecosystems](https://movementecologyjournal.biomedcentral.com/articles/10.1186/s40462-020-00214-w). You may find more details that are helpful for improving your model performance.

**How can I assess whether my training process is successful or not?**
1. Read learning curves
Mask R-CNN model provides learning curves to show how the model performs with epochs going on. Since this model is implemented based on tensorflow backend, it is convenient to use _Tensorboard_ to visualize these curves. Note that Tensorboard supports monitoring the training process even if it has not finished yet. You can also use `ReadTensorboard.py` to extract these metrics and export learning curves.
Tensorboard can be activated in two ways:
    a.  Jupyter notebook or Google Colab
    ``` 
    %load_ext tensorboard
    %tensorboard --logdir="/FishSeg/logs"
    ```
    b. Anaconda command
    ```
    tensorboard --log="/FishSeg/logs"
    ```
2. Diagnose model performance using learning curves
I can provide you several links that may be helpful (see below).
    * [How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)
    * [How to Diagnose Overfitting and Underfitting of LSTM Models](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/)
    * [How to Avoid Overfitting in Deep Learning Neural Networks](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)

**If my model performance is not satisfying, how should I improve it?**
To improve the model, you need to have a general idea about the hyper-parameters in this model. Go to [Taming the Hyper-Parameters of Mask RCNN](https://medium.com/analytics-vidhya/taming-the-hyper-parameters-of-mask-rcnn-3742cb3f0e1b) to get an overview.
To clarify the relationship between steps, epochs, and batch size, check below:
* [Difference Between a Batch and an Epoch in a Neural Network](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
* [What is the difference between steps and epochs in TensorFlow?](https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow)

You may also adjust the model by changing batch size and learning rate, check below for possible directions.
* [How to Control the Stability of Training Neural Networks With the Batch Size](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/)
* [How to Configure the Learning Rate When Training Deep Learning Neural Networks](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)
* [Understand the Impact of Learning Rate on Neural Network Performance](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)

However, every model is different, and there are no general rules that can predict how results may change after fine tunning the parameters. So you have to be patient and check them yourself. Good luck!

**How can I use multiple GPU to accelerate the model training and video testing?**
Well, `ParallelModel.py` is intended to enable multi-GPU usage, but it cannot work well for now due to some internal conflicts. 
Herein, we provide an alternative to use more GPUs by the following commands.
1. Open a new instance of *spyder*
    ```
    spyder --new-instance
    ```
2. Uncomment following lines in the first section
    ```
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" #just list the names of GPU that you hope to use
    ```

**What should I do if I meet problems that you don't mention here?**
I can provide you several websites that I find quite useful when I implement the workflow. Search your questions by these links, and your problems probably have already been solved by others. If not, create a question by yourself to ask for help.
* [stack overflow](https://stackoverflow.com/)
* [Issue section of Matterport Mask_RCNN](https://github.com/matterport/Mask_RCNN/issues)
* [Github](https://github.com/)
* [Tensorflow Core](https://www.tensorflow.org/guide?hl=zh-cn)
* [keras FAQ](https://keras.io/getting_started/faq/#why-is-the-training-loss-much-higher-than-the-testing-loss)
* [Google](https://www.google.com)
