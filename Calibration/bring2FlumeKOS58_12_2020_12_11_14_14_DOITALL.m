%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FishSeg
%Calibration of Cameras for Video Tracking
%Part 3 bring2Flume - Converts Calibrated Files into Flume Coordinate System

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
% Folder CLB_2020_08_10_16_32_xxx#xx1 - Output from multiObjectUndisting, needs to be located in savdir!!

%performRigidTrsform58_00
%estimateRigidTransform.m
%bring2FlumeKOS_58

%% Steps for Usage: 
%0) funcID should be consistent over all used codes/functions!
%2) Specify correct directory for datdir - Video Files of calibration. 
%Naming CLB_2020_08_10_16_32 - contains avi and excel for all 5 cameras. 
%4) Specify directory for savedir - Ideally on local machine to avoid junkon NAS
%5) Within part 05.00 parameters: add current Calibration
%6)dirname - dirnamesep anpassen auf aktuelles Jahr

%IMPORTANT/IMPORTANT!!!
%When clicking the points click same points for CAM1 in both
%CAM0-CAM1 and CAM1-CAM2!!!
%The numbers of the clicked points must match the numbers in file 20181017_KAL00.pdf!
%There should be at least 4 rows of numbers overlapping between 2 cameras in order to
% to get a good calibration. 

%% 
clear all
close all
clc

%% part 05.00 definitions
funcID      = '58';
datdir      = 'Y:\'; % Directory of Video Files of calibration.
mfldir      = pwd;
savdir      = 'D:\FishSeg_Results\CalibResults\CLB_2020_08_10_16_32_xxx#xxx';
%IMPORTANT! Also Folder CLB_2020_08_10_16_32_xxx#xx1 with all Matlab files
%resulting from Calibration needs to be located in Directory of savdir!

addpath(mfldir);
fldrcal    = [datdir,filesep,'CLB_2020_08_10_16_32'];  
xlsxnam     = '20181017_KAL00.xlsx';

FCOS        = xlsread([mfldir,filesep,xlsxnam],...                           
                       'Tabelle2','B3:E52');   % positions of marker points (101-150)
HCOS        = xlsread([mfldir,filesep,xlsxnam],...                              
                       'Tabelle2','G3:J10');   % positions of marker points (501-508)
CCOS        = xlsread([mfldir,filesep,xlsxnam],...                              
                       'Tabelle2','L3:O7');  % positions of 5 fisheye cameras (104,113,121,130,139)

camNpid       = [1,1,1,1,1,1];

%% part 10.00 digitize point pairs
for c = 0:4
    aVid = VideoReader([fldrcal,filesep,'Cam_',num2str(c),'_final.avi']);
    aVid.CurrentTime = camNpid(c+1)/aVid.FrameRate;
    aCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_',num2str(c)]);
    A    = undistortFisheyeImage(rgb2gray(readFrame(aVid)),aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
    A0   = adapthisteq(A,'Distribution','rayleigh');
    idA0 = num2str(100000+camNpid(c+1));
    idcx = num2str(   100+        c );
    imwrite(A0,        [savdir,filesep,'A0_cam',idcx(2:end),'_',idA0(2:end),'.jpg']);
end

% digitize point pairs
for c = (4-1):-1:0
    opts.Default        = 'No';
    opts.Interpreter    = 'tex';
    answer  = questdlg(['Pressing OK will start {\itbring2FlumeKOS\_58.m} to digitize ', newline, ...
                        'joint flume points of   ... {\bfcam', num2str(c),'} and {\bfcam', num2str(c+1),'}.', newline, ...
                        newline, ...
                        'In case ''{\bfYes}'':',                                                        newline, ...
                        '   > Note that point numbers will start every time at point #1.',              newline, ...
                        '   > Click gray areas to count up point numbers for non-visible points.',      newline, ...
                        '   > Then digitize the visible point pairs.',                                  newline, ...
                        '   > Finish clicking after you have digitized the last visible point pair.',   newline, ...
                        '   > Finally close the {\itcpselect-tool} by clicking the {\times}-button.',   newline, ...
                        '   > The rest will be done automatically.',                                    newline, ...
                        ], ...
                        '!! Warning !!', ...
                        'Yes','No',...
                        opts);               
    %--
    switch answer
        case 'Yes'
            [ptscamA,ptscamB] = bring2FlumeKOS_58(	datdir,...
                                                    savdir,...
                                                    camNpid(c+1),camNpid(c+2),funcID,c,c+1);
            %--
            if size(ptscamA,1) < size(FCOS,1)
                    ptscamA(size(ptscamA,1)+1:size(FCOS,1),:) = NaN; 
            end
            if size(ptscamB,1) < size(FCOS,1)
                    ptscamB(size(ptscamB,1)+1:size(FCOS,1),:) = NaN; 
            end
            %--
            IDcA = num2str(100+c+0  );
            IDcB = num2str(100+c+1);
            eval(['ptscam', IDcA(2:3) ,' = ptscamA']);
            eval(['ptscam', IDcB(2:3) ,' = ptscamB']);
            save([savdir,filesep,'pts',IDcA(2:3),'to',IDcB(2:3)],...
                 ['ptscam', IDcA(2:3)],['ptscam', IDcB(2:3)]);  
        case 'No'
    end                  
end

%% part 20.00 check points
%--00
aVid = VideoReader([fldrcal,filesep,'Cam_0_final.avi']);
aVid.CurrentTime = camNpid(0+1)/aVid.FrameRate;
aCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_0']);
A0   = undistortFisheyeImage(rgb2gray(readFrame(aVid)),aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
%--
load([savdir, filesep, 'pts00to01.mat'])
AB = ptscam00;
c00 = figure('Name','Cam00'); 
imshow(imadjust(A0));
hold on; 
plot(AB(:,1),AB(:,2),'cx','MarkerSize',6);
for i = 1:size(AB,1); if ~isnan(AB(i,1)); text(10+AB(i,1),AB(i,2),num2str(i),'Color',[0 1 1]); end; end
axis on
xy.c00 = AB;
clear AB


%--01
bVid = VideoReader([fldrcal,filesep,'Cam_1_final.avi']);
bVid.CurrentTime = camNpid(1+1)/bVid.FrameRate;
bCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_1']);
B0    = undistortFisheyeImage(rgb2gray(readFrame(bVid)),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
%--
A = load([savdir, filesep, 'pts00to01.mat']);
B = load([savdir, filesep, 'pts01to02.mat']);
AB(1:size(A.ptscam01,1),1:2) = A.ptscam01;
AB(1:size(B.ptscam01,1),3:4) = B.ptscam01;
AB( :                  ,  5) = nanmedian([AB(:,1),AB(:,3)]')'
AB( :                  ,  6) = nanmedian([AB(:,2),AB(:,4)]')'
c01 = figure('Name','Cam01'); 
imshow(imadjust(B0));
hold on;  
plot(AB(:,1),AB(:,2),'ro','MarkerSize',20);
plot(AB(:,3),AB(:,4),'go','MarkerSize',24);
plot(AB(:,5),AB(:,6),'cx','MarkerSize',6);
for i = 1:size(AB,1); if ~isnan(AB(i,1)); text(10+AB(i,1),AB(i,2),num2str(i),'Color',[0 1 1]); end; end
axis on
xy.c01 = AB;
clear AB


%--02
bVid = VideoReader([fldrcal,filesep,'Cam_2_final.avi']);
bVid.CurrentTime = camNpid(2+1)/bVid.FrameRate;
bCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_2']);
B0    = undistortFisheyeImage(rgb2gray(readFrame(bVid)),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
%--
A = load([savdir, filesep, 'pts01to02.mat']);
B = load([savdir, filesep, 'pts02to03.mat']);
AB(1:size(A.ptscam02,1),1:2) = A.ptscam02;
AB(1:size(B.ptscam02,1),3:4) = B.ptscam02;
AB( :                  ,  5) = nanmedian([AB(:,1),AB(:,3)]')'
AB( :                  ,  6) = nanmedian([AB(:,2),AB(:,4)]')'
c02 = figure('Name','Cam02'); 
imshow(imadjust(B0)); hold on;  
plot(AB(:,1),AB(:,2),'ro','MarkerSize',20);
plot(AB(:,3),AB(:,4),'go','MarkerSize',24);
plot(AB(:,5),AB(:,6),'cx','MarkerSize',6);
for i = 1:size(AB,1); if ~isnan(AB(i,1)); text(10+AB(i,1),AB(i,2),num2str(i),'Color',[0 1 1]); end; end
axis on
xy.c02 = AB;
clear AB


%--03
bVid = VideoReader([fldrcal,filesep,'Cam_3_final.avi']);
bVid.CurrentTime = camNpid(3+1)/bVid.FrameRate;
bCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_3']);
B0    = undistortFisheyeImage(rgb2gray(readFrame(bVid)),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
%--
A = load([savdir, filesep, 'pts02to03.mat'])
B = load([savdir, filesep, 'pts03to04.mat'])
AB(1:size(A.ptscam03,1),1:2) = A.ptscam03;
AB(1:size(B.ptscam03,1),3:4) = B.ptscam03;
AB( :                  ,  5) = nanmedian([AB(:,1),AB(:,3)]')'
AB( :                  ,  6) = nanmedian([AB(:,2),AB(:,4)]')'
c03 = figure('Name','Cam03'); 
imshow(imadjust(B0));
hold on; 
plot(AB(:,1),AB(:,2),'ro','MarkerSize',20); 
plot(AB(:,3),AB(:,4),'go','MarkerSize',24);
plot(AB(:,5),AB(:,6),'cx','MarkerSize',6);
for i = 1:size(AB,1); if ~isnan(AB(i,1)); text(10+AB(i,1),AB(i,2),num2str(i),'Color',[0 1 1]); end; end
axis on
xy.c03 = AB;
clear AB


%--04
bVid= VideoReader([fldrcal,filesep,'Cam_4_final.avi']);
bVid.CurrentTime= camNpid(4+1)/bVid.FrameRate;
bCam= load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_4']);
B0   = undistortFisheyeImage(rgb2gray(readFrame(bVid)),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
%--
load([savdir, filesep, 'pts03to04.mat'])
AB = ptscam04;
c04 = figure('Name','Cam04'); 
imshow(imadjust(B0)); hold on; 
plot(AB(:,1),AB(:,2),'cx','MarkerSize',6);
for i = 1:size(AB,1); if ~isnan(AB(i,1)); text(10+AB(i,1),AB(i,2),num2str(i),'Color',[0 1 1]); end; end
axis on
xy.c04 = AB;
clear AB

%% part 25.00 save figures
%--
saveas(c00,[savdir,filesep,'c00pts.fig']);
saveas(c00,[savdir,filesep,'c00pts.png']);
%--
saveas(c01,[savdir,filesep,'c01pts.fig']);
saveas(c01,[savdir,filesep,'c01pts.png']);
%--
saveas(c02,[savdir,filesep,'c02pts.fig']);
saveas(c02,[savdir,filesep,'c02pts.png']);
%--
saveas(c03,[savdir,filesep,'c03pts.fig']);
saveas(c03,[savdir,filesep,'c03pts.png']);
%--
saveas(c04,[savdir,filesep,'c04pts.fig']);
saveas(c04,[savdir,filesep,'c04pts.png']);

%% part 30.00 summarize points
%--00to01
AB = NaN*ones(10,4);
AB(1:size(xy.c00,1),1:2) = xy.c00(:,end-1:end);
AB(1:size(xy.c01,1),3:4) = xy.c01(:,end-1:end);
AB((isnan(sum(AB')')),:) = NaN;
save([savdir,filesep,'pts00to01ok.mat'],'AB','FCOS');   
clear AB

%--01to02
AB = NaN*ones(10,4);
AB(1:size(xy.c01,1),1:2) = xy.c01(:,end-1:end);
AB(1:size(xy.c02,1),3:4) = xy.c02(:,end-1:end);
AB((isnan(sum(AB')')),:) = NaN;
save([savdir,filesep,'pts01to02ok.mat'],'AB','FCOS');   
clear AB

%--02to03
AB = NaN*ones(10,4);
AB(1:size(xy.c02,1),1:2) = xy.c02(:,end-1:end);
AB(1:size(xy.c03,1),3:4) = xy.c03(:,end-1:end);
AB((isnan(sum(AB')')),:) = NaN;
save([savdir,filesep,'pts02to03ok.mat'],'AB','FCOS');   
clear AB

%--03to04
AB = NaN*ones(10,4);
AB(1:size(xy.c03,1),1:2) = xy.c03(:,end-1:end);
AB(1:size(xy.c04,1),3:4) = xy.c04(:,end-1:end);
AB((isnan(sum(AB')')),:) = NaN;
save([savdir,filesep,'pts03to04ok.mat'],'AB','FCOS');   
clear AB

hasi = 1

%% part 40.00 3D rigid transformation
for j = 3:-1:0
    [~,delta1,delta2] = performRigidTransform58_00(...
                            FCOS,NaN,...
                            savdir,...
                            j+0,...
                            funcID);
    disp(['Cam',num2str(j),'_to_Cam',num2str(j+1),' : ', num2str(delta2)]);
    delta1A(j+1).ll=delta1;
    for n=1:size(delta1,2); delta1A(j+1).nn(n)={['Cam',num2str(j),'&',num2str(j+1)]}; end
    delta2All( j+1)=delta2;
end
%--
cNatoNb12 = figure('Name','Flume')
boxplot([delta1A.ll],[delta1A.nn]);
ylabel('cams to world \surd([\Delta({\itx,y,z})]^2) (mm)');
ylim([0 50]);
%--
disp(['------------']);
disp(['Mean:   ',num2str(  mean(delta2All))]);
disp(['Median: ',num2str(median(delta2All))]);

%% part 50.00 test all
datadir     = fldrcal;
dirname     = dir([datadir,filesep,'2020*']);
for i = 1:size(dirname,1)-1
    idTmp                = strfind(dirname(i).name,'Cam');
    dirname(i).camNR   	 = str2num(dirname(i).name(idTmp+3));
end
%--
camNRsort                = cat(1,dirname.camNR);
camNRsort(:,2)           = 1:size(camNRsort,1);
camNRsort                = sortrows(camNRsort);
%--
cNatoNb02 = figure('Name','Flume','Position', [390 199 984 352]);
hold on
%--
plot3(HCOS(1:2,2),HCOS(1:2,3),HCOS(1:2,4),'k-','Color',[0.5 0.5 0.5]);
plot3(HCOS(3:6,2),HCOS(3:6,3),HCOS(3:6,4),'k-','Color',[0.5 0.5 0.5]);
plot3(HCOS(7:8,2),HCOS(7:8,3),HCOS(7:8,4),'k-','Color',[0.5 0.5 0.5]);
plot3(CCOS(:,2),CCOS(:,3),CCOS(:,4),      'ko','Color',[0.5 0.5 0.5]);
%--
cmap = colormap(hsv(14));
cmap = cmap*0.8;   
for j = flipud(camNRsort(:,2))'
    load([savdir,filesep,filesep,'camPrmsRIGEDetc_',funcID,'_c',num2str(j-1),'to',num2str(j)]);
    %--
    hold on
    plot3(k', l', m','ro','MarkerSize',3,'MarkerFaceColor','r');
    for i = 1:size(k,2)
        text(k(i)+5, l(i)+5, m(i)+5,        num2str(idOKx(i)), 'Color',cmap(j,:),'FontSize',8);
    end
    %--
    plot3(FLUMEwrldPoints(:,1),FLUMEwrldPoints(:,2),FLUMEwrldPoints(:,3),'ko');%,'MarkerSize',10);
    hold on
end
xlabel('{\itx} (mm)'); ylabel('{\ity} (mm)'); zlabel('{\itz} (mm)');
grid on;
axis equal;
xlim([   0  10000])
ylim([-200   1700])
zlim([ -50   1000])

%--
savefig(cNatoNb02,[savdir,filesep,'cNatoNb02all','.fig']);
saveas( cNatoNb02,[savdir,filesep,'cNatoNb02all','.png']);
saveas( cNatoNb12,[savdir,filesep,'cNatoNb12all','.png']);
clear cNatoNb02
save(             [savdir,filesep,'cNatoNb02all','.mat']);
