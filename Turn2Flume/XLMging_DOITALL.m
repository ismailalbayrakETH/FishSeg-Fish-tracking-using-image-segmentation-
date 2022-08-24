% 3D Fish Tracking Project

% Part 3 - Equalize and apply calibration to the generated tracks
% Fan Yang
% July 2022
% Matlab 2021b, VLC Player

%% Required functions/Files in Matlab Path: 
%Output from Mask R-CNN (in excel files)
%Output from CALIBRATION (CLB Folder)

% undistFisheySng.m
% registDoubleCam.m
% showItDoubleCam.m
% trigomDoubleCam.m

%% Steps for Usage: 
%0) funcID should be consistent over all used codes/functions!
%1) Specify correct directory for datdir - Output of Calibration and Tracking
%3) Specify directory for savedir - Ideally on local machine to avoid junk on NAS
%4) Add correct Calibration to KALIFOLDERS. Should be of the form  'CLB_2019_07_11_14_29_xxx#xx1';
%5) Part 10 - order Data - Add which calibration should be used for which experiments
%6) Part 48 pre work and Part 50 work - Specify which tracks you want to run - M = 3 means the 3rd track in the tracking
%Folder will be undistorted and finalized.
%7) Adapt savefld 4 to laufnummer where you want final results
%8) Copy the Output (521, 522, 523) to the directory X1 on NAS and the Output 
%#000XXX_2019_00_00_00_04 to the directory X2 on NAS. 

% The data corresponding to the plots in #000XXX_2019_00_00_00_04 is saved
% in the field UserData. It can be retracted using h = gcf; h.UserData

%% 
clear all
close all
clc

%% Step 0: extract output files

% datdir = 'D:\Tracking_Workflow\initial_results';
% outdir = 'S:\vaw_public\yanfan\FishTracking\Outputs';
% folders = dir([outdir,filesep,'*_2020_*_*_*_*_mog2_results']);
% for i = 1:length(folders)
%     ID = str2num(folders(i).name(1:3));
%     if ID>=304 && ID<=309
%         new_path = [outdir,filesep,folders(i).name];
%         files = dir([new_path,filesep,'tracks_Cam_*_mog2.xlsx']);
%         mkdir([datdir,filesep,folders(i).name]);
%         destinyfile = [datdir,filesep,folders(i).name];
%         for j = 1:5
%             sourcefile = [new_path,filesep,files(j).name];           
%             copyfile(sourcefile,destinyfile);
%         end
%     end
% end

%% Step 1: main definitions
funcID      = '58';
datdir = 'D:\FishSeg_Results\TrackingResults\Results_initial';
savdir = 'D:\FishSeg_Results\TrackingResults\Results_final';

% -- parameter definitions for 'showItDoubleCam'
vidOn   	= 0;
sig         = 1.0;
alp       	= 0.8;
dnnMin      = 500; % [px] for nearest neighbor criteria
avifps      = 25;  % [fps] bugfix MDE 20181212

% -- points in flume-COS
FCOS        = xlsread([datdir,filesep,'20181017_KAL00'],'Tabelle2','C3:E53');
HCOS        = xlsread([datdir,filesep,'20181017_KAL00'],'Tabelle2','H3:J10');
CCOS        = xlsread([datdir,filesep,'20181017_KAL00'],'Tabelle2','M3:O7');

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
    if meas(i)<301 || meas(i)>309 % Set range for analysis
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
    dirname(i).X = dir([datdir,filesep,MAINFOLDERS{i},filesep,'tracks_Cam_*_*.xlsx']); 
    IDmf = str2num(MAINFOLDERS{i}(1:3));
    if 300<=IDmf && IDmf<=370 % changed according to video names
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
        savFldr     = [savdir,dirname(i).X(j).folder(savID-3:end)];
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
        savFldr     = [savdir,dirname(i).X(j).folder(savID-3:end)];
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
    ryx2Ka = [];
    ryx2La = [];
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
ax      = 500+round(   ryx2Ca(:,5)); % 500+round(   ryx2Ca(:,5))
ay      = 000+round(20*ryx2Ca(:,3)); % turn time to frames--(25*ryx2Ca)
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
bx       = 500+round(   ryx2Ca(ai03,6)); % 500+round(   ryx2Ca(ai03,6))
by       = 000+round(20*ryx2Ca(ai03,3)); % 000+round(05*ryx2Ca(ai03,3)) --20*ryx2Ca(ai03,3)
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
ryx2Ka = ryx2Ca(ai03,:);
ryx2La = ryx2Ka(bi03,:);

% ---------
% delete random points beyond the outer range
ryx2Ca(find(ryx2Ca(:,6)<-50 | ryx2Ca(:,6)>1550 | ryx2Ca(:,5)<0),:)=[]; 

% delete random points at the rack corner
ryx2Ca(find(ryx2Ca(:,6)>1250 & ryx2Ca(:,5)<2000),:)=[]; 
ryx2Ca(find(ryx2Ca(:,6)>1400 & ryx2Ca(:,5)<2300),:)=[];
ryx2Ca(find(ryx2Ca(:,6)>750 & ryx2Ca(:,5)<1250),:)=[]; 
ryx2Ca(find(ryx2Ca(:,6)>800 & ryx2Ca(:,5)<1400),:)=[];
ryx2Ca(find(ryx2Ca(:,6)>900 & ryx2Ca(:,5)<1570),:)=[];

% ---------
% clear out and interpolation
fx = 1;
fy = 5;
fz = 20*100;
fzp= 10;

AA0     = pointCloud(...
           [ryx2Ca(               :,5)*1,... 
            ryx2Ca(               :,6)*1,...
            ryx2Ca(               :,3)*1*fzp],...
            'Color',uint8(ones(size(ryx2Ca,1), 3).*[255 255 255]));
        
AAA     = pointCloud(...
           [ryx2Ca(               :,5)*fx,... 
            ryx2Ca(               :,6)*fy,...
            ryx2Ca(               :,3)*fz]);       
%--
minDistance = 500; % default: 140, threshold adjustment

[labels,numClusters] = pcsegdist(AAA,minDistance);

%--
DDD = [NaN NaN NaN NaN];
for j = 1:numClusters
    DDtmp = sortrows(AA0.Location((labels==j),:),3);
    DDtmp(:,3) = 2*round(1/2*DDtmp(:,3));
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
EEE = [NaN NaN NaN NaN];
EEEuni = unique(DDD(:,4));

for e = 1:size(EEEuni,1)
    
    id = find(EEEuni(e)==DDD(:,4));
    if size(id,1)>1
    tt = DDD(id(1),3):0.5:DDD(id(end),3);
    
    Xt = interp1(   DDD(id(1):id(end),3),...
                    DDD(id(1):id(end),1),tt);
    Yt = interp1(   DDD(id(1):id(end),3),...
                    DDD(id(1):id(end),2),tt);
    EEE = [EEE;[Xt' Yt' tt' EEEuni(e)*tt'./tt']];
    end        
end

EEE         = EEE(2:end,:);
EEE(:,3)    = EEE(:,3)./fzp;
EEEuni = unique(EEE(:,4));

% ---------
% delete very short tracks
for k = 1:size(EEEuni,1)
    id = find(EEE(:,4) == EEEuni(k));
    if size(id,1) < 50 % threshold adjustment
        EEE(id,:)=[];
    end
end
EEEuni = unique(EEE(:,4));

% ----------
figi52tmp = figure('Position', [680 87 560 1011]);
set(figi52tmp,'Name',[MAINFOLDERS{i},'_figi52tmp_',funcID,'.fig'])
hold on
plot3(ryx2Ca(:,5), ryx2Ca(:,6), duration(0,0,ryx2Ca(:,3)),'.','Color',[0.0 0.0 0.0],'MarkerSize',4);

for r = 1:size(EEEuni,1)
    id = find(EEE(:,4) == EEEuni(r));
    plot3(             EEE(id,1),...
                       EEE(id,2),...
                    ...EEE(id,3),...
          duration(0,0,EEE(id,3)/1),...
          '.','MarkerSize',12)
end
    
plot3(HCOS(1:2,1),HCOS(1:2,2),duration(0,0,ones(size(HCOS(1:2,1)))*max(ryx2Ca(:,3))),'k-','Color',[1 0 0]);
plot3(HCOS(3:6,1),HCOS(3:6,2),duration(0,0,ones(size(HCOS(3:6,1)))*max(ryx2Ca(:,3))),'k-','Color',[1 0 0]);
plot3(HCOS(7:8,1),HCOS(7:8,2),duration(0,0,ones(size(HCOS(7:8,1)))*max(ryx2Ca(:,3))),'k--','Color',[1 0 0]);
plot3(CCOS( : ,1),CCOS( : ,2),duration(0,0,ones(size(CCOS( : ,1)))*max(ryx2Ca(:,3))),'ko','Color',[1 0 0]);
%--     
xlabel('{\itx} (mm)')
ylabel('{\ity} (mm)');
zlabel('real {\itt} (s)');  
axis on
grid on
set(gca,'ZDir','Reverse');
%--
figi52tmp.UserData.avifps     = avifps;
figi52tmp.UserData.ryx2Ca     = ryx2Ca;
figi52tmp.UserData.EEE        = EEE;
figi52tmp.UserData.EEEuni     = EEEuni;
h                             = datacursormode(figi52tmp);
h.UpdateFcn                   = @myupdatef55_02_201xtmb;
h.SnapToDataVertex            = 'on';
h.Enable = 'on'
%--
set(gca,'View',[-90 90])
xlim([-530 10030])
ylim([-530  2030])
zlim([duration(0,0,min(ryx2Ca(:,3))) duration(0,0,max(ryx2Ca(:,3)))]) 

% ----------
% save2figures
saveFld4 = [savdir, filesep,['#000XXX_2020_00_00_00_01']];
if ~isfolder(saveFld4); mkdir(saveFld4); end
saveas(figi52tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi52tmp_',funcID,'.fig']);
saveas(figi52tmp,[saveFld4,filesep,MAINFOLDERS{i},'_figi52tmp_',funcID,'.png']);
save([saveFld4,filesep,MAINFOLDERS{i},'.mat'],...
    'ryx2Ca','EEE','EEEuni');
end
