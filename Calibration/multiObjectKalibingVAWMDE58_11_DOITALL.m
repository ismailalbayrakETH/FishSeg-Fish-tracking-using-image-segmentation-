%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FishSeg
%Calibration of Cameras for Video Tracking
%Part 1 Calibration

%Copyright (c) 2023 ETH Zurich
%Written by Fan Yang and Martin Detert
%D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
%License under the 3-clause BSD License (see LICENSE for details)

%%%%%%%%%%%%%%%%%%%%%%%%
%Matlab 2018b, VLC Player
%Calibration consists of 3 Work packages, each needing the output from the previous one. 
%1) - multiObjectKalibingVAWMDE58_11_DOITALL  - Search Checkerboard intersections
%2) - multiObjectUndistingVAWMDE58_11_DOITALL - Calibrate single cams (fisheye) and pairwise cameras using checkerboard intersections. 
%3) - bring2FlumeKOS58_12_2019_07_11_14_29_DOITALL - Transformation to flume coordinate system. Some manual labor required

%% Required functions/Files in Matlab Path: 
% multiObjectKalibingVAWMDE58_11_kaliber

%% Steps for Usage: 
%1) funcID should be consistent over all used codes/functions!
%2) Specify correct directory for datdir - Video Files of calibration. 
%3) Adapt MAINFOLDERS - Naming CLB_2020_08_10_16_32 - contains avi and excel for all 5 cameras. 
%4) Specify directory for savedir - Ideally on local machine to avoid junkon NAS
%5) Within part 05.00 parameters add current Calibration

%% 
clc;
clear all;
close all;

%% part 01.00 main definitions
funcID      = '58'; 
datdir      = 'Y:\'; % Path for videos to be analyzed
savdir      = 'D:\FishSeg_Results\CalibResults';% Path for data storage

%% part 05.00 parameters
tkNfr           = 10; % take number of frames to compute average background images
rmi             = 150; % pixels radius reduction due to inaccuracies in border fields
pXthr0          = 15; % gray threshold of tracking
pXthr1          = 12; 
mXthr0          = 0.0015;
strIC0          = 7;
tkdID           = 2;
medDpnzf        = 6;
medDpnz0        = 6;
medDpnz1        = 6;
%--
boardsize       = [7,10];           % checkerboard fields
hini            = boardsize(2)-4;   
vini            = boardsize(1)-3;
hok             = boardsize(2);
vok             = boardsize(1);  
%--
MAINFOLDERS =  {'CLB_2020_08_10_16_51',100, NaN};
%{Folder with Videos, #Frames start analysis,#frames end} NaN: analyse until end of Video  
%% part 10.00 sort data
for M = 1:size(MAINFOLDERS,1)
    searchvid = 'Cam_*_final.avi'; % Track all cams 
%    searchvid = 'Cam_4_final.avi'; % Number only specific cam
    if iscell(MAINFOLDERS)
        dirname(M).X        = dir([datdir,filesep,MAINFOLDERS{M,1},filesep, searchvid]);
        dirname(M).run      =                     MAINFOLDERS{M,2};% frames start analysis
        dirname(M).fin      =                     MAINFOLDERS{M,3};% frames end
    else
        dirname(M).X        = dir([datdir,filesep,MAINFOLDERS{M,1},filesep, searchvid]);
        dirname(M).run      =                     0;
        dirname(M).fin      =                     NaN;
    end
    
    %--
    idX = zeros(size(dirname(M).X,1),1);
    for i = 1:size(dirname(M).X,1)
        if  strcmp(dirname(M).X(i).name,'Cam_5_final.avi') || ...
            strcmp(dirname(M).X(i).name,'Cam_6_final.avi') || ...
            strcmp(dirname(M).X(i).name,'Cam_7_final.avi')
            idX(i) = 1;
        end
    end

%     idX = zeros(size(dirname(M).X,1),1); 
    dirname(M).X(logical(idX)) = [];
end
%% part 50.00 work
for M = 1:size(MAINFOLDERS,1) 
    for mm = 1:size(dirname(M).X,1) 
        
        %-- diverse parameter
        avi2read    = [dirname(M).X(mm).folder,filesep,dirname(M).X(mm).name]; 
        disp(avi2read); 
        camstr      = dirname(M).X(mm).name(5); 
        dir2save    = [savdir,dirname(M).X(mm).folder(3:end),'_xxx#xxx']; % 19 is determined by the folder_name
        vidInf      = VideoReader(avi2read);
        
        if isnan(dirname(M).fin) 
            dirname(M).fin = vidInf.Duration*vidInf.FrameRate; 
        end
        
        %-- delete old data and make directories
        if isdir([dir2save,filesep,  'mf0toXX_',camstr,'_',funcID]) 
           rmdir([dir2save,filesep,  'mf0toXX_',camstr,'_',funcID],'s'); 
        end
        mkdir([dir2save,filesep,  'mf0toXX_',camstr,'_',funcID]); 
        
        %-- load an individual mask
        if exist([dir2save,filesep,'Cam_',camstr,'_final_CW.mat']) == 2 
            load([dir2save,filesep,'Cam_',camstr,'_final_CW.mat']); 
        else
           maskBW = 1;
        end
        
        %-- compute the background image
        vidFrame    = zeros(vidInf.Height,vidInf.Width,tkNfr);
        startI      = round(dirname(M).run);
        finalI      = round(dirname(M).fin);
        startS      =       dirname(M).run/vidInf.FrameRate;% Start time of analysis
        finalS      =       dirname(M).fin/vidInf.FrameRate;% End time of analysis
        for v = 1:tkNfr   
            fprintf('%3.0f ... ',tkNfr-v+1); 
            vidInf.CurrentTime  = round(startS + (v-0.5)*(finalS-startS)/tkNfr);
            vidFrame(:,:,v)     = rgb2gray(readFrame(vidInf));
        end
        fprintf('\n'); 
        
        
        %-- cut-out boundary ring of the background image
        vidFramM    = uint8(shiftdim(nanmedian(shiftdim(vidFrame,2))));% average values for background images
        vidFramB    = imclose(logical(vidFramM),strel('disk',size(vidFramM,1)/2));
        stats       = regionprops(vidFramB,'Centroid','EquivDiameter');
        maskRR      = zeros(size(vidFramM)); 
        maskRR      = insertShape(maskRR,'FilledCircle',...
            [stats.Centroid 0.5*stats.EquivDiameter-rmi],'color',[1 1 1],'LineWidth',1, 'Opacity',1);
        maskRR      = logical(maskRR(:,:,1));
        vidFramM    = uint8(maskBW).*uint8(maskRR).*vidFramM;

        %-- do the 'find-chessboard-corners' job
         eval([...
            'multiObjectKalibingVAWMDE',funcID,'_11_kaliber('''                                         ... 
                                                 avi2read,''','''                                       ...
                                                 dir2save,''','''                                       ...
                                               ['mf0toXX_',camstr,'_',funcID],''','                     ... 
                                                'maskBW,',                                              ...
                                                'hini,','vini,','hok,','vok,',                          ...
                                                'vidFramM,','pXthr0,','pXthr1,','mXthr0,',              ...
                                                'strIC0,','tkdID,'                                      ...
                                                'medDpnzf,','medDpnz0,','medDpnz1,',                    ...
                                                'startI,', 'finalI'                                     ...
                                                ');'                                                    ...
                 ]);      
         save([               dir2save,filesep, 'tr0toXX_',camstr,'_',funcID,'_doitAll','.mat'],        ... 
                                                'funcID',                                               ...
                                                'maskBW',                                               ...
                                                'hini','vini','hok','vok',                              ...
                                                'vidFramM','pXthr0','pXthr1','mXthr0',                  ...
                                                'strIC0','tkdID',                                       ...
                                                'medDpnzf','medDpnz0','medDpnz1',                       ...
                                                'startI', 'finalI'                                      ...
              );
    end
end
