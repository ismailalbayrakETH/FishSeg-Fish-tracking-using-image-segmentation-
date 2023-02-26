%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FishSeg
%Calibration of Cameras for Video Tracking
%Part 2 Undisting

%Copyright (c) 2023 ETH Zurich
%Written by Fan Yang and Martin Detert
%D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
%License under the 3-clause BSD License (see LICENSE for details)

%%%%%%%%%%%%%%%%%%%%%%%%
%Matlab 2018b, VLC Player
%Calibration consists of 3 Work packages, each needing the output from the previous one. 
%1) - multiObjectKalibingVAWMDE58_11_DOITALL  - Search Checkerboard intersections
%2) - multiObjectUndistingVAWMDE58_11_DOITALL - Calibrate single cams (fisheye) and pairwise cameras using checkerboard intersections. 
%3) - bring2FlumeKOS58_12_2020_12_11_14_14_DOITALL - Transformation to flume coordinate system. Some manual labor required
%% Required functions/Files in Matlab Path: 
% undistSingleCam58_11
% undistDoubleCam58_11
% orderDupliFiles58_11

%% Steps for Usage: 
%0) funcID should be consistent over all used codes/functions!
%2) Specify correct directory for datdir - Video Files of calibration. 
%Naming CLB_2020_08_10_16_32 - contains avi and excel for all 5 cameras. 
%4) Specify directory for savdir - Ideally on local machine to avoid junkon NAS
%5) Within part 05.00 parameters: add current Calibration

%% 
clc
clear all
close all

%% part 01.00 main definitions
funcID      = '58';
datdir      = 'D:\FishSeg_Results\CalibResults'; % Path for video files
savdir      = 'D:\FishSeg_Results\CalibResults\CLB_2020_08_10_16_32_xxx#xxx'; % Path for calibration data storage

%% part 05.00 parameter    
squareSize      = 39.86;            %[mm]
boardsize       = [7,10];           %numbers of chessboard fields
facW            = 0.25;             %scale factor for newly plotted image
dSS_crit        = 12;               %[px]; minimal accepted dist. squareSizePx (single calibration)
pk              = 100;              %take ~pk images for parameter computation (single calibration), each@[max(dSS);ctr;{min,max}(x,y));quadrants]
maxErp01        = 1.0;              %[px]; max-avg accepted reprojection-error (single calibration)
maxErp11        = 2.0;              %[px]; max-sng accepted reprojection-error (single calibration)
maxErp9x        = 0.95;             %[px]; max-qnt accepted reprojection-error (stereo calibration)
NtkEv2c         = 100;              %take ~NtkEv2c stereo-pair images to calib (stereo calibration)
%--
MAINFOLDERS =  {'CLB_2020_08_10_16_32'};
             
%% part 10.00 sort data
for M = 1:size(MAINFOLDERS,1)
    dirname(M).X       = dir([datdir,filesep,MAINFOLDERS{M},'_xxx#xxx',filesep,'tr0toXX_*_',funcID,'_doitAll.mat']);
end

%% part 50.00 calibrate single cam
for M = 1:size(MAINFOLDERS,1)  
    for mm = 1:size(dirname(M).X,1)
            disp([dirname(M).X(mm).folder,filesep,dirname(M).X(mm).name])
            % --
            camFldrFull =[dirname(M).X(mm).folder,filesep,dirname(M).X(mm).name];
            camFldr     = dirname(M).X(mm).folder;
            camN        = dirname(M).X(mm).name(9);
            %--
            disp('undistSingleCam**_**')
            undistSingleCam58_11(camFldr,camN,funcID,squareSize,boardsize,dSS_crit,maxErp01,maxErp11,facW,pk);
            %--
    end
    close all
end

%% part 60.00 calibrate neighboring stereo cam pairs
for M = 1:size(MAINFOLDERS,1)  
    for mm = 1:size(dirname(M).X,1)
            disp([dirname(M).X(mm).folder,filesep,dirname(M).X(mm).name])
            %--
            camFldrFull =[dirname(M).X(mm).folder,filesep,dirname(M).X(mm).name];
            camFldr     = dirname(M).X(mm).folder;
            camN        = dirname(M).X(mm).name(9);
            %--
            if str2num(camN) < (size(dirname(M).X,1)-1)
            cat0        = cat(1,dirname(M).X.name);
            mmp1        = find(str2num(cat0(:,9))==str2num(camN)+1);
            camFldrFul2 =[dirname(M).X(mmp1).folder,filesep,dirname(M).X(mmp1).name];
            camFld2     = dirname(M).X(mmp1).folder;
            camNp1      = num2str(str2num(camN)+1);
            %--
            disp('orderDupliFiles**_**')
            orderDupliFiles58_11(camFldr,camFld2,camN,camNp1,funcID);
            %--
            disp('undistDoubleCam**_**')
            undistDoubleCam58_11(camFldr,camN,camNp1,funcID,maxErp9x,NtkEv2c);
            end
            %--
    end
    close all
end
