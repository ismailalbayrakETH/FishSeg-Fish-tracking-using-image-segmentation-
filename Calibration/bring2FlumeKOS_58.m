function [ptscamA,ptscamB] = bring2FlumeKOS_58(datdir, savdir, camN0id, camN1id, funcID, cam0, cam1)
%% -- prepare images
aVid = VideoReader([datdir,filesep,savdir(end-27:end-8),filesep,'Cam_',num2str(cam0),'_final.avi']);
bVid = VideoReader([datdir,filesep,savdir(end-27:end-8),filesep,'Cam_',num2str(cam1),'_final.avi']);

%--
aVid.CurrentTime = camN0id/aVid.FrameRate;
bVid.CurrentTime = camN1id/bVid.FrameRate;

%--
aCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_',num2str(cam0)]);
bCam = load([savdir,filesep,'camPrmsSNGRATetc_',funcID,'_',num2str(cam1)]);

%--
if cam0==0 && strcmp(savdir(4:end),'0876_FitHydrH0\CLB_2019_22_05_08_05_xxx#xxx') %% PATH PROBLEM
  A  = undistortFisheyeImage(imrotate(rgb2gray(readFrame(aVid)),90),aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
else
  A  = undistortFisheyeImage(rgb2gray(         readFrame(aVid)),    aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
end
  B  = undistortFisheyeImage(rgb2gray(         readFrame(bVid)),    bCam.params.Intrinsics,'ScaleFactor',bCam.facW);

%% -- start the cp-selector
A(1:round(0.05*size(A,1)),:) = 200;
B(1:round(0.05*size(B,1)),:) = 200;
%--
[ptscamA,ptscamB]   = cpselect( (adapthisteq(A,'Distribution','rayleigh')),...
                                (adapthisteq(B,'Distribution','rayleigh')),...
                                'Wait',true); % cpselect: Control point selection tool
  ptscamA(ptscamA(:,2)< round(0.05*size(A,1)),:)   = NaN;
  ptscamB(ptscamB(:,2)< round(0.05*size(B,1)),:)   = NaN;   
end
