%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%FishSeg
%Equalize and apply calibration to the generated tracks

%Copyright (c) 2023 ETH Zurich
%Written by Fan Yang and Martin Detert
%D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW), Prof. Robert Boes
%License under the 3-clause BSD License (see LICENSE for details)

%%%%%%%%%%%%%%
%Matlab 2021b, VLC Player
%% Required functions/Files in Matlab Path: 
%Output from Mask R-CNN (in excel files)
%Output from CALIBRATION (CLB Folder)

% undistFisheySng.m
% registDoubleCam.m
% showItDoubleCam.m
% trigomDoubleCam.m

%% Steps for Usage: 
%1) Step 1 - main definitions
    %a) funcID should be consistent over all used codes/functions!
    %b) Specify correct directory for datdir - Output of Calibration and Tracking
    %c) Specify directory for savedir - Ideally on local machine to avoid junk on NAS
    %d) Add correct Calibration to KALIFOLDERS. Should be of the form  'CLB_2019_07_11_14_29_xxx#xx1';
    %e) Specify start_num and end_num to determine converted cases;
    %f) Adjust minDistance and lgtracks for better tracks
%2) Step 2 - order Data
    % Add which calibration should be used for which experiments
%3) Step 3 pre work and Step 3_1/2 work - 
    % Specify which tracks you want to run - M = 3 means the 3rd track in the tracking
%5) Adapt savefld 4 to laufnummer where you want final results
%6) The outputs are saved under the savdir in "3D_Tracks" folder, including
    %a) ".mat" format file with all track data (XYZ);
    %b) ".png" pictures of the final track (XOY & XOZ);
    %c) ".figure" figure to be opened in MATLAB (XOY & XOZ)

%% 
clear all
close all
clc

%% Step 0: extract output files
% This step is to extract excel files from Outputs folder into the
% Initial_result folder. It only needs to be run once for the all output
% files.

datdir = 'S:\vaw_public\yanfan\Demo\TrackingResults\Results_initial';
outdir = 'S:\vaw_public\yanfan\Demo\TrackingResults\Outputs';
folders = dir([outdir,filesep,'*_2020_*_*_*_*_mog2_results']);
for i = 1:length(folders)
    ID = str2num(folders(i).name(1:3));
    if ID>=342 && ID<=342
        new_path = [outdir,filesep,folders(i).name];
        files = dir([new_path,filesep,'tracks_Cam_*_mog2.xlsx']);
        mkdir([datdir,filesep,folders(i).name]);
        destinyfile = [datdir,filesep,folders(i).name];
        for j = 1:5
            sourcefile = [new_path,filesep,files(j).name];           
            copyfile(sourcefile,destinyfile);
        end
    end
end

%% Step 1: main definitions
funcID      = '58';
datdir = 'S:\vaw_public\yanfan\Demo\TrackingResults\Results_initial';
savdir = 'S:\vaw_public\yanfan\Demo\TrackingResults\Results_final';

% -- cases that needed to be converted
start_num = 301;% start case name
end_num = 301;% end case name

% -- threshold adjustment (IMPORTANT!)
minDistance = 1500; % minimum distance between 3D points
lgtracks = 50; % delete very short tracks

% -- parameter definitions for 'showItDoubleCam'
vidOn   	= 0;
sig         = 1.0;
alp       	= 0.8;
dnnMin      = 500; % [px] for nearest neighbor criteria
avifps      = 25;  % [fps] bugfix MDE 20181212
% avifps      = 5; 

% -- points in flume-COS
FCOS        = xlsread([datdir,filesep,'20221014_KAL01'],'Sheet1','C3:E53');
HCOS        = xlsread([datdir,filesep,'20221014_KAL01'],'Sheet1','H3:J10');
CCOS        = xlsread([datdir,filesep,'20221014_KAL01'],'Sheet1','M3:O7');

% -- data folders
fldinf = dir([datdir,filesep,'*_2020_*_*_*_*_mog2_results']); % folder name can be changed
for f = 1:size(fldinf,1)
    if ~isempty(str2num(fldinf(f).name(1:3)))
        if size(fldinf(f).name,2) == 33 % 32 for bgs/knn // 33 for mog2
        fldinf(f).meas = str2num(fldinf(f).name(1:3));
        else
        fldinf(f).meas = NaN;  
        end
    else
        fldinf(f).meas = NaN;
    end
end

meas = zeros(size(fldinf,1),1);
for i = 1:size(fldinf,1)
    meas(i) = fldinf(i).meas;
    if meas(i)<start_num || meas(i)>end_num
        meas(i) = NaN;
    end
end

% MNFLDRS = cat(1,fldinf(~isnan(cat(1,fldinf.meas))).name);
MNFLDRS = cat(1,fldinf(~isnan(cat(1,meas))).name);
%--
MAINFOLDERS = [];
%--
for mm = [1:size(MNFLDRS,1)]
    MAINFOLDERS{mm} = [MNFLDRS(mm,:)];
end
MAINFOLDERS     = MAINFOLDERS';

%-- calibration folders
KALIFOLDERS = {'CLB_2020_12_11_14_04_xxx#xx1'}; 

%% Step 2: order data
for i = 1:length(MAINFOLDERS)
%     file(i).dir = [fldinf(i).folder,filesep,fldinf(i).name]; % All folders under "Outputs"
    dirname(i).X = dir([datdir,filesep,MAINFOLDERS{i},filesep,'tracks_Cam_*_*.xlsx']); % All excel files under above folder
end

for i = 1:length(MAINFOLDERS)
    IDmf = str2num(MAINFOLDERS{i}(1:3));
    if 300<=IDmf && IDmf<=400 % changed according to video names
     kalname(i).X = dir([datdir,filesep,KALIFOLDERS{1},filesep,'tr0toXX_*_58_doitAll.mat']);
    end
end

%% Step 3: pre-work (undistort single fisheyepoints)
for i = 1:length(MAINFOLDERS)
    for j = 1:length(dirname(i).X)
        disp(['------------']);
        disp([dirname(i).X(j).folder,filesep,dirname(i).X(j).name]);
        % -- 
        kalFldr     = kalname(i).X(j).folder;
        camFldr     = dirname(i).X(j).folder;
        savID       = strfind(dirname(i).X(j).folder,'_2020');
        savFldr     = [savdir,dirname(i).X(j).folder(savID-4:end)];
        camN        = dirname(i).X(j).name(12); % from 9 to 12
        camN        = str2num(camN);        
        % -- 
        if camN <=size(dirname(i).X,1)-1 && camN >= 0
            disp(['------------']);
            disp(['undistFisheySng: undistort points from single fisheye cam'])
            if ~exist(savFldr) 
                mkdir(savFldr); 
            end
            eval([...
              'undistFisheySng(camFldr,kalFldr,savFldr,camN,funcID);']);
            disp('Frames have been single-defisheyed already.')
        end
        close all;
    end

%% Step 3_1: work
    % registDoubleCam; showItDoubleCam; trigomDoubleCam
    for j = 1:length(dirname(i).X)
        disp(['------------']);
        disp([dirname(i).X(j).folder,filesep,dirname(i).X(j).name]);
        % -- 
        kalFldr     = kalname(i).X(j).folder;
        camFldr     = dirname(i).X(j).folder;
        savID       = strfind(dirname(i).X(j).folder,'_2020');
        savFldr     = [savdir,dirname(i).X(j).folder(savID-4:end)];
        camN        = dirname(i).X(j).name(12); % from 9 to 12
        camN        = str2num(camN);        
        % -- 
        if camN <size(dirname(i).X,1)-1
            camN0=camN+100; camN0=num2str(camN0); camN0=camN0(2:3);
            camN1=camN+101; camN1=num2str(camN1); camN1=camN1(2:3);
            if ~exist(savFldr)
                mkdir(savFldr);
            end
            disp(['------------']);
            disp(['registDoubleCam: Camera registration']);
            eval([...
                  'registDoubleCam(camFldr,camFldr,kalFldr,kalFldr,savFldr,camN0,camN1,funcID);']);
            disp(['Frames have been double-show-checked already']);
            disp(['------------']);
            disp(['showItDoubleCam']);
            eval([...
                'showItDoubleCam(camFldr,camFldr,kalFldr,kalFldr,savFldr,camN0,camN1,funcID,vidOn,sig,alp,dnnMin);']);            
            disp(['Frames have been double-show-checked already']);
            disp(['------------']);
            disp(['trigomDoubleCam']);
            eval([...
                'trigomDoubleCam(camFldr,camFldr,kalFldr,kalFldr,savFldr,camN0,camN1,funcID);']);
            disp(['Frames have been trigom-doublecamed already']);
            CAM(camN+1).dat   = load([savFldr,filesep,'XYZetr_',funcID,'_c',camN0,'to',camN1]);
        end
    end
%% Step 3_2: work
    % Turn to flume coordinate
    % --------    
    XYZetc = [];
    ryx2Da = [];
    ryx2Ea = [];
    ryx2Ca = [];
    XYZetcm = [];
    XYZetcn = [];
    for k = 1:size(CAM,2)
        k;
        if ~isempty(CAM(k).dat)
        XYZetcTMP = CAM(k).dat.XYZetc;
        XYZetc = [XYZetc; XYZetcTMP];
        clear XYZetcTMP
        %--
            ryx2DaTMP = CAM(k).dat.ryx2Da;
            ryx2Da = [ryx2Da; ryx2DaTMP];
            clear ryx2DaTMP
        %--
         if CAM(k).dat.ryx2Db(1,4) == size(CAM,2)
            ryx2DaTMP = CAM(k).dat.ryx2Db;
            ryx2Da = [ryx2Da; ryx2DaTMP];
            clear ryx2DaTMP
         end
        %--
            ryx2EaTMP = CAM(k).dat.ryx2Ea;
            ryx2Ea = [ryx2Ea; ryx2EaTMP];
            clear ryx2EaTMP
        %--
         if CAM(k).dat.ryx2Eb(1,4) == size(CAM,2)
            ryx2EaTMP = CAM(k).dat.ryx2Eb;
            ryx2Ea = [ryx2Ea; ryx2EaTMP];
            clear ryx2EaTMP
         end
         %--
            ryx2CaTMP = CAM(k).dat.ryx2Ca;
            ryx2Ca = [ryx2Ca; ryx2CaTMP];
            clear ryx2CaTMP
         %--
         if ~isempty(CAM(k).dat.ryx2Cb)
         if CAM(k).dat.ryx2Cb(1,4) == size(CAM,2)
            ryx2CaTMP = CAM(k).dat.ryx2Cb;
            ryx2Ca = [ryx2Ca; ryx2CaTMP];
            clear ryx2CaTMP
         end
         end
        end
    end

    %-------- check the column number
    XYZetc = sortrows(XYZetc,[6,7]);
    
    rgb00   = 10^6*XYZetc(:,6)+XYZetc(:,7);
    idRUNr  = find(logical([1;diff(rgb00)  ]));
    idFINr  = find(logical([  diff(rgb00);1]));
    %--
    rgb01   = 10^6*XYZetc(:,6)+XYZetc(:,7);           
               
% --------
ax      = 500+round(   XYZetc(:,1)); % 500+round(   ryx2Ca(:,5))
ay      = 000+round(20*XYZetc(:,5)); % turn time to frames--(25*ryx2Ca)
% ay      = 000+round(5*ryx2Ca(:,3))
ai      = 1:size(ax,1);
ax0     = ax;
ay0     = ay;
ai0     = ai;

ABW0    = uint64(zeros(max(ay)-min(ay),10000));

idax    = find(ax<1         | ax>10000);
iday    = find(ay<min(ay)   | ay>max(ay));
ax(unique([idax; iday])) = [];
ay(unique([idax; iday])) = [];
ai(unique([idax; iday])) = [];
for k = 1:size(ax,1)
    ABW0(ay(k)-min(ay)+1,ax(k))= ai(k);
end
%--
ABW1 = logical(imdilate(logical(ABW0),strel('disk',20)));
ABW2 = bwpropfilt(      ABW1,'MajorAxisLength',[1000 Inf]);
ai00 = intersect(find(ABW0),find(ABW2));
ai03 = ABW0(ai00);

% --------
bx       = 500+round(   XYZetc(ai03,2)); % 500+round(   ryx2Ca(ai03,6))
by       = 000+round(20*XYZetc(ai03,5)); % 000+round(05*ryx2Ca(ai03,3)) --20*ryx2Ca(ai03,3)
% by       = 000+round(5*ryx2Ca(ai03,3));
bi       = 1:size(bx,1);
bx0      = bx;
by0      = by;
bi0      = bi;

BBW0    = uint64(zeros(max(by)-min(by),02500));

idbx    = find(bx<1         | bx>02500);
idby    = find(by<min(by)   | by>max(by));
bx(unique([idbx; idby])) = [];
by(unique([idbx; idby])) = [];
bi(unique([idbx; idby])) = [];
for k = 1:size(bx,1)
    BBW0(by(k)-min(by)+1,bx(k))= bi(k);
end
%--
BBW1 = logical(imdilate(logical(BBW0),strel('disk',20)));
BBW2 = bwpropfilt(      BBW1,'MajorAxisLength',[100 Inf]);
bi00 = intersect(find(BBW0),find(BBW2));
bi03 = BBW0(bi00);
XYZetcm = XYZetc(ai03,:);
XYZetcn = XYZetcm(bi03,:);

% ---------
% delete random points beyond the outer range
XYZetc(find(XYZetc(:,2)<-50 | XYZetc(:,2)>1550 | XYZetc(:,1)<0),:)=[]; 

% delete random points at the rack corner
XYZetc(find(XYZetc(:,2)>1250 & XYZetc(:,1)<2000),:)=[]; 
XYZetc(find(XYZetc(:,2)>1400 & XYZetc(:,1)<2300),:)=[];
XYZetc(find(XYZetc(:,2)>750 & XYZetc(:,1)<1250),:)=[]; 
XYZetc(find(XYZetc(:,2)>800 & XYZetc(:,1)<1400),:)=[];
XYZetc(find(XYZetc(:,2)>900 & XYZetc(:,1)<1570),:)=[];

% ---------
% clear out and interpolation
fx = 1;
fy = 5;
fz = 20*100;
% fz = 5*100;
fzp= 10;

AA0     = pointCloud(...
           [XYZetc(               :,1)*1,... 
            XYZetc(               :,2)*1,...
            XYZetc(               :,5)*1*fzp]);
        
AAA     = pointCloud(...
           [XYZetc(               :,1)*fx,... 
            XYZetc(               :,2)*fy,...
            XYZetc(               :,5)*fz]);       

[labels,numClusters] = pcsegdist(AAA,minDistance);

%--
DDD = [NaN NaN NaN NaN NaN];
for j = 1:numClusters
    DDtmp = sortrows(AA0.Location((labels==j),:),3);
    DDtmp(:,3) = 2*round(1/2*DDtmp(:,3));
    DDtmp(:,4) = XYZetc((labels==j),3);% add
    DDuni = unique(AA0.Location((labels==j),3));
    for tt = 1: size(DDuni,1)
       if sum(DDtmp(:,3) == DDuni(tt))>1
        DDD = [DDD;[median(DDtmp(DDtmp(:,3) == DDuni(tt),:)) j]];
       else
           if sum(DDtmp(:,3) == DDuni(tt))>0
           end
       end      
    end
end
DDD = DDD(2:end,:);

%--
EEE = [NaN NaN NaN NaN NaN];
EEEuni = unique(DDD(:,5));

for e = 1:size(EEEuni,1)
    
    id = find(EEEuni(e)==DDD(:,5));
    if size(id,1)>1
    tt = DDD(id(1),3):0.5:DDD(id(end),3);
    
    Xt = interp1(   DDD(id(1):id(end),3),...
                    DDD(id(1):id(end),1),tt);
    Yt = interp1(   DDD(id(1):id(end),3),...
                    DDD(id(1):id(end),2),tt);
    Zt = interp1(   DDD(id(1):id(end),3),...
                    DDD(id(1):id(end),4),tt);
    EEE = [EEE;[Xt' Yt' Zt' tt' EEEuni(e)*tt'./tt']];
    end        
end

EEE         = EEE(2:end,:);
EEE(:,4)    = EEE(:,4)./fzp;
EEEuni = unique(EEE(:,5));

% ---------
% delete very short tracks
for k = 1:size(EEEuni,1)
    id = find(EEE(:,5) == EEEuni(k));
    if size(id,1) < lgtracks % threshold adjustment
        EEE(id,:)=[];
    end
end
EEEuni = unique(EEE(:,5));

% -- Coordinate transformation
XYZetc(:,1) = (XYZetc(:,1)-500)/1000;
XYZetc(:,2) = XYZetc(:,2)/1000;
XYZetc(:,3) = XYZetc(:,3)/1000;
EEE(:,1) = (EEE(:,1)-500)/1000;
EEE(:,2) = EEE(:,2)/1000;
EEE(:,3) = EEE(:,3)/1000;

% -------
figi00tmp = figure('Position', [680 87 1011 560]);
set(figi00tmp,'Name',[MAINFOLDERS{i},'_figi00tmp_',funcID,'.fig'])
hold on
plot3(XYZetc(:,1),XYZetc(:,2),XYZetc(:,3),'.','Color',[0.0 0.0 0.0],'MarkerSize',4);

for r = 1:size(EEEuni,1)
    id = find(EEE(:,5) == EEEuni(r));
    plot3(             EEE(id,1),...
                       EEE(id,2),...
                       EEE(id,3),...
          '.','MarkerSize',12)
end

plot3(HCOS(1:2,1),HCOS(1:2,2),HCOS(1:2,3),'k-','Color',[1 0 0]);
plot3(HCOS(3:6,1),HCOS(3:6,2),HCOS(3:6,3),'k-','Color',[1 0 0]);
plot3(HCOS(7:8,1),HCOS(7:8,2),HCOS(7:8,3),'k--','Color',[1 0 0]);
plot3(CCOS( : ,1),CCOS( : ,2),CCOS( : ,3),'ko','Color',[1 0 0]);

xlabel('{\itx} (mm)');
ylabel('{\ity} (mm)');
zlabel('{\itz} (mm)');
axis on
grid on
%--
figi00tmp.UserData.avifps     = avifps;
figi00tmp.UserData.XYZetc     = XYZetc;
figi00tmp.UserData.EEE        = EEE;
figi00tmp.UserData.EEEuni     = EEEuni;
h                             = datacursormode(figi00tmp);
h.UpdateFcn                   = @myupdatef55_02_201xtmb;
h.SnapToDataVertex            = 'on';
h.Enable = 'on';
%--
set(gca,'View',[0 0]);
xlim([-0.5 8]);ylim([-0.5 2]);zlim([-0.1 1]);
daspect([1 1 1]);

figi01tmp = figure('Position', [680 87 1011 560]);
set(figi01tmp,'Name',[MAINFOLDERS{i},'_figi01tmp_',funcID,'.fig'])
hold on
plot3(XYZetc(:,1),XYZetc(:,2),XYZetc(:,3),'.','Color',[0.0 0.0 0.0],'MarkerSize',4);

for r = 1:size(EEEuni,1)
    id = find(EEE(:,5) == EEEuni(r));
    plot3(             EEE(id,1),...
                       EEE(id,2),...
                       EEE(id,3),...
          '.','MarkerSize',12)
end

plot3(HCOS(1:2,1),HCOS(1:2,2),HCOS(1:2,3),'k-','Color',[1 0 0]);
plot3(HCOS(3:6,1),HCOS(3:6,2),HCOS(3:6,3),'k-','Color',[1 0 0]);
plot3(HCOS(7:8,1),HCOS(7:8,2),HCOS(7:8,3),'k--','Color',[1 0 0]);
plot3(CCOS( : ,1),CCOS( : ,2),CCOS( : ,3),'ko','Color',[1 0 0]);

xlabel('{\itx} (mm)');
ylabel('{\ity} (mm)');
zlabel('{\itz} (mm)');
axis on
grid on
% set(gca,'ZDir','Reverse');
%--
figi01tmp.UserData.avifps     = avifps;
figi01tmp.UserData.XYZetc     = XYZetc;
figi01tmp.UserData.EEE        = EEE;
figi01tmp.UserData.EEEuni     = EEEuni;
h                             = datacursormode(figi00tmp);
h.UpdateFcn                   = @myupdatef55_02_201xtmb;
h.SnapToDataVertex            = 'on';
h.Enable = 'on';
%--
set(gca,'View',[0 90]);
daspect([1 1 1]);
xlim([-0.5 8]);ylim([-0.5 2]);zlim([-0.1 1]);

% ----------
% save2figures
saveFld4 = [savdir, filesep,['3D_Tracks']];
if ~isfolder(saveFld4); mkdir(saveFld4); end
saveas(figi00tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi00tmp_',funcID,'.fig']);
saveas(figi00tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi00tmp_',funcID,'.png']);
saveas(figi01tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi01tmp_',funcID,'.fig']);
saveas(figi01tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi01tmp_',funcID,'.png']);
save([saveFld4,filesep,MAINFOLDERS{i},'.mat'],...
    'XYZetc','EEE','EEEuni');
end
