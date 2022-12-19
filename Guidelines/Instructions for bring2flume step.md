# Instructions for "bring2flume" of calibration

## Flume coordinates marked with reference points
![](D:/FishSeg\Figures\Flume.PNG)
Note that the red lines are division of points which indicates how the points should be selected in each pair-camera window.

## Instrcutions for manual labor
In the following part, four windows are dispalyed, which show how the manual works should look like. Four principles for calibration are listed as follow.
1. In each window, all clicked points should be in pair.
2. For the next window, click the invisible points in the grey bar, and start from the next points of the former window.
3. At least 3 rows of points should be included for each calibration.
4. All points should be in accordance with the markers displayed in the above "Flume coordinates" picture. Note that you don't need to finish all red points marked in the picture. Just stop at point 34 can obtain a good calibration.

### Cam3-4 (Points: 1-7)
![](D:/FishSeg\Figures\cam3-4.PNG)

### Cam2-3 (Points: 8-16)
![](D:/FishSeg\Figures\cam2-3.PNG)

### Cam1-2 (Points:17-25)
![](D:/FishSeg\Figures\cam1-2.PNG)

### Cam0-1 (Points: 26-34)
![](D:/FishSeg\Figures\cam0-1.PNG)

## How to decide whether it is a good calibration
**If you have done the calibration successfully, you can obtain the following two outputs.**
![](D:/FishSeg\Figures\32-1.PNG)
Note_1: All the red points are basically within the circle.
![](D:/FishSeg\Figures\32-2.PNG)
Note_2: Four errors for camera-pairs are visible in the figure.

**If your calibration fails, you may get outputs as follow.**
![](D:/FishSeg\Figures\51-1.PNG)
Note_1: In the region of *cam1-2*, red points deviate significantly from the circles.
![](D:/FishSeg\Figures\51-2.PNG)
Note_2: The cam_to_world error is missing for *cam1-2*.

**What to do with bad calibration results**
Rerun the *bring2FlumeKOS58_12_2020_12_11_14_14_DOITALL.m*, and you will see the warning windows again.
![](D:/FishSeg\Figures\warning.PNG)
Press **Yes** to repeat the points-selection procedure for cam1-2; press **No** to skip good calibrations of other camera pairs. 

