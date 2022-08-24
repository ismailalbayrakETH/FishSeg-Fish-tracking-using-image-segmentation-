function registDoubleCam(camFldr,camFld2,kalFldr,kalFld2,savFldr,camN0,camN1,funcID)
%% -- rename option to be able to read 'funcID-n' as well
fundID      = num2str(100+str2num(funcID)-0);
fundID      = fundID(2:3);    
%% -- load data
aCam        = load([kalFldr,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN0))]);
bCam        = load([kalFld2,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN1))]);
%--
aTMP        = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN0)),'_doitall']);
bTMP        = load([camFld2,filesep,'tracks_Cam_',num2str(str2num(camN1)),'_doitall']);
%--
aViF        = aTMP.vidFramM;
bViF    	= bTMP.vidFramM;
if isfield(aTMP,'maskBW'); aMbw = aTMP.maskBW; else; aMbw = 1; end
if isfield(bTMP,'maskBW'); bMbw = bTMP.maskBW; else; bMbw = 1; end
clear aTMP bTMP

%% -- prepare images
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
AA1         = locallapfilt(A,0.5,0.3);
BB1         = locallapfilt(B,0.5,0.3);
%--
load([kalFldr,filesep,'pts',camN0,'to',camN1,'ok']);

%% -- register images
MOVINGREG   = fitgeotrans(AB(~isnan(AB(:,1)),3:4),AB(~isnan(AB(:,1)),1:2),'projective');
BBt         = imwarp(B,MOVINGREG,'OutputView',imref2d(size(B)));

%% -- plot figures
figR00 = figure; 
imshowpair(BBt,A,'scaling','joint'); 
xlabel('{\itx} (px, no fisheye)');
ylabel('{\ity} (px, no fisheye)');
axis on
grid on

%% -- save data
save([savFldr,filesep,'regDbl_',funcID,'Xc',camN0,'to',camN1,'.mat'],...
       'MOVINGREG','A','B','AA1','BB1','BBt');

%% -- save2figures
saveas(figR00,[savFldr,filesep,'regDbl_',funcID,'Xc',camN0,'to',camN1],'png');
% close gcf
end
