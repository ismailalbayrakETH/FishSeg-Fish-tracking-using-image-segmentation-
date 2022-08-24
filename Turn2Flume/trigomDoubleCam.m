function trigomDoubleCam(camFldr,camFld2,kalFldr,kalFld2,savFldr,camN0,camN1,funcID)
%% -- rename option to be able to read 'funcID-n' as well
fundID = num2str(100+str2num(funcID)-0);
fundID = fundID(2:3);  

%% -- load data
% -- calibration files
load([kalFldr,filesep,'camPrmsSTEREOetc','_',fundID,'_c',num2str(str2num(camN0)),'to',num2str(str2num(camN1))]);

aCam = load([kalFldr,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN0))]);
bCam = load([kalFld2,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN1))]);    

% -- tracking files
load([savFldr,filesep,'tabiidr_',funcID,'_c',camN0,'to',camN1]);

aunF = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN0)),'_doitall']);
bunF = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN1)),'_doitall']);

% -- tracks in cam_A
if isempty(aunF.tracks)
    disp('isempty(aunF.tracks) == 1!');
    aunF.tracks = -99*ones(1,7); % -99*ones(1,8)
end
aunF.tracksmn = [aunF.tracks,zeros(size(aunF.tracks,1),4)]; % 11 columns (7+4)
if ~strcmp(mat2str(aunF.tracks),mat2str(-99*ones(1,7))) % mat2str(-99*ones(1,8))
    aunF.tracksmn(:,8:9) = undistortFisheyePoints(aunF.tracks(:,4:5),aCam.params.Intrinsics,aCam.facW);
end % aTrk.iXYZSmn(:,6:7)

% -- tracks in cam_B
if isempty(bunF.tracks)
    disp('isempty(bunF.tracks) == 1!'); 
    bunF.tracks = -99*ones(1,7); % -99*ones(1,8)
end
bunF.tracksmn = [bunF.tracks,zeros(size(bunF.tracks,1),4)];
if ~strcmp(mat2str(bunF.tracks),mat2str(-99*ones(1,7))) % mat2str(-99*ones(1,8))
    bunF.tracksmn(:,8:9) = undistortFisheyePoints(bunF.tracks(:,4:5),bCam.params.Intrinsics,bCam.facW);
end % bTrk.iXYZSmn(:,6:7)

%% -- do the triangulation
XYZetc  = [triangulate(aunF.tracksmn(tabiidr(:,2),8:9),...
                       bunF.tracksmn(tabiidr(:,3),8:9), stereoParams),...
                       tabiidr(:,[1]),...
                       aunF.tracksmn(tabiidr(:,2),3),... % 5:time
                       tabiidr(:,[4:5]),...
                       str2double(camN0).*ones(size(tabiidr(:,1))),...
                       str2double(camN1).*ones(size(tabiidr(:,1))),...
                       tabiidr(:,[6:end])]; 

%% -- rigid transformation
load([kalFldr,filesep,'camPrmsRIGEDetc_',funcID,'_c',num2str(str2num(camN0)),'to',num2str(str2num(camN1))],'tformT2');

if size(XYZetc,1)>1
    [k,l,m]     = transformPointsForward(tformT2,XYZetc(:,1),...
                                                 XYZetc(:,2),...
                                                 XYZetc(:,3));
    XYZetc(:,1) = k;
    XYZetc(:,2) = l;
    XYZetc(:,3) = m;
    clear k l m
else
    disp('isempty(XYZetc) == 1!');
    XYZetc      = -9999*zeros(1,10); % -9999*zeros(1,13)
end

%% remaining 2D tracks transformation
ryx2Da = [                                      aunF.tracks(:,[1:3]),...
                       str2num(camN0)*ones(size(aunF.tracks,1),1),...
                         undistortFisheyePoints(aunF.tracks(:,4:5),aCam.params.Intrinsics,aCam.facW)];
ryx2Db = [                                      bunF.tracks(:,[1:3]),...
                       str2num(camN1)*ones(size(bunF.tracks,1),1),...
                         undistortFisheyePoints(bunF.tracks(:,4:5),bCam.params.Intrinsics,bCam.facW)];

% --
aTMP        = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN0)),'_doitall']);
warning('off')
bTMP        = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN1)),'_doitall']);
warning('off')
% --
aViF        = aTMP.vidFramM;
bViF    	= bTMP.vidFramM;
if isfield(aTMP,'maskBW'); aMbw = aTMP.maskBW; else; aMbw = 1; end
if isfield(bTMP,'maskBW'); bMbw = bTMP.maskBW; else; bMbw = 1; end
clear aTMP bTMP

% --
if aCam.params.Intrinsics.ImageSize(1) > aCam.params.Intrinsics.ImageSize(2)
    A       = undistortFisheyeImage(imrotate(aViF.*uint8(aMbw),90),aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
else
    A       = undistortFisheyeImage(         aViF.*uint8(aMbw)    ,aCam.params.Intrinsics,'ScaleFactor',aCam.facW);   
end

if bCam.params.Intrinsics.ImageSize(1) > bCam.params.Intrinsics.ImageSize(2)
    B       = undistortFisheyeImage(imrotate(bViF.*uint8(bMbw),90),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
else
    B       = undistortFisheyeImage(         bViF.*uint8(bMbw)    ,bCam.params.Intrinsics,'ScaleFactor',bCam.facW);   
end

load([kalFldr,filesep,'cNatoNb02all.mat'],'xy','FCOS');

figN0 = figure('Visible','on'); imshow(A);
hold on; 
plot(ryx2Da(:,5),ryx2Da(:,6),'.'); 
eval(['plot(xy.c',camN0,'(:,end-1),xy.c',camN0,'(:,end),''ro'')']);

figN1 = figure('Visible','on'); imshow(B);
hold on; 
plot(ryx2Db(:,5),ryx2Db(:,6),'.'); 
eval(['plot(xy.c',camN1,'(:,end-1),xy.c',camN1,'(:,end),''ro'')']);

% -- camN0
hasi = 1;
ptsN0mov    = eval(['xy.c',camN0]);
ptsN0mov    = ptsN0mov(:,end-1:end);
ptsN0fix    = FCOS(:,2:3);
ptsN0fix(isnan(ptsN0mov(:,1)),:) = [];
ptsN0mov(isnan(ptsN0mov(:,1)),:) = [];
tformN0     = fitgeotrans(ptsN0mov,ptsN0fix,'projective');
[xN0, yN0]  = transformPointsForward(tformN0, ryx2Da(:,5), ryx2Da(:,6));

% -- camN1
hasi = 1;
ptsN1mov    = eval(['xy.c',camN1]);
ptsN1mov    = ptsN1mov(:,end-1:end);
ptsN1fix    = FCOS(:,2:3);
ptsN1fix(isnan(ptsN1mov(:,1)),:) = [];
ptsN1mov(isnan(ptsN1mov(:,1)),:) = [];
tformN1     = fitgeotrans(ptsN1mov,ptsN1fix,'projective');
[xN1, yN1] = transformPointsForward(tformN1, ryx2Db(:,5), ryx2Db(:,6));

%-----
ryx2Da(:,5)     = xN0;
ryx2Da(:,6)     = yN0;

ryx2Db(:,5)     = xN1;
ryx2Db(:,6)     = yN1;

%---
ryx2Ea      =   ryx2Da; 
ryx2Eb      =   ryx2Db;
ryx2Ca      =   ryx2Da(:,:); % ryx2Da( aTrk.conn02,:)
ryx2Cb      =   ryx2Db(:,:); % ryx2Db( bTrk.conn02,:)
ryx2Da      =   ryx2Da(:,:); % ryx2Da(~aTrk.conn02,:)
ryx2Db      =   ryx2Db(:,:); % ryx2Db(~bTrk.conn02,:)

%% save data
if isempty(tabiidr); tabiidr = ones(1,9); end
%--
save([savFldr,filesep,'XYZetr_',funcID,'_c',camN0,'to',camN1],...
      'XYZetc','ryx2Ca','ryx2Cb','ryx2Da','ryx2Db','ryx2Ea','ryx2Eb');
   
saveas(figN0,[savFldr,filesep,'figN0x_',funcID,'_c',camN0],'png');
saveas(figN1,[savFldr,filesep,'figN0x_',funcID,'_c',camN1],'png');

end
