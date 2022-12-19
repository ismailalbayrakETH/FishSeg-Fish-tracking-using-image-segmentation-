# FishSeg

## Aim ##
Do 2D fish tracking in a large etho-hydraulic flume and turn 2D tracks into 3D metric space.

## Functions for each folder
### 1. Calibration
All scripts needed to do fisheye camera calibration; 
Use three scripts end with _DOITALL_ for main works in sequence of *Kalibing-Undisting-bring2Flume*.
Check how to use _bring2Flume_DOITALL_ at "Instructions for bring2flume step.pdf".

### 2. Tracking
**MakeDatasets**: Scripts needed to make video clips and arrange datasets;
* StartTime.py: Correct the manually recorded start time (from 25 fps to 20 fps), specially setup for VAW fisheye camera system;
* get_videoclip.py: get video clips where fish actually show up and do MOG2 background subtraction in the meantime;
* concatenate_clips.py: concatenate all the video clips together;
* datasets.py: check annotation datasets and reorganize datasets;

**Notebooks**: Scripts for FishSeg tracking used at google Colab (FishSeg_Colab.ipynb) or jupyter notebook (FishSeg_Windows.ipynb).

**Main Scripts** : Scripts for FishSeg model training and tracking;
* backgroundSubtraction.py: Do MOG2 background subtraction for experimental videos
* FishSeg_training.py: Training FishSeg model and do video tracking based on the model;
* FishSeg_tracking.py: Do video tracking based on established model;
* mask2tracks.py: Turn masks predicted by FishSeg into tracks;
* ReadTensorboard.py: Read loss functions produced in *log* folder under C:\FishSeg to check if the model get good performance;

### 3. Turn2Flume
* XLMging_DOITALL_2D.m: Turn sperate tracks from different fisheye cameras into 2D tracks in the flume coordinate system;
* XLMging_DOITALL_3D.m: Turn sperate tracks from different fisheye cameras into 3D tracks in the flume coordinate system;
Note: Other .m files are functions files used for the two main scripts.

### 4. For_site_packages
Both folders should be copied and pasted into the site-packages path if the setup.py does not work well. Check the *Practical Guide* for more info.

### 5. trout
Listed here as an example of how to arrange datasets and saved models when trying to train the model for trout tracking.









