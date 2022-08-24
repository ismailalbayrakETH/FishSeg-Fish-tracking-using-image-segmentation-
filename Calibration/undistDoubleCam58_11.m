function undistDoubleCam55_11(camFldr,camN0,camN1,funcID,maxErp9x,NtkEv2c)
%--
load([camFldr,filesep,'camPrmsDOUBLEetc_',funcID,'txt_c',camN0,'to',camN1]);
id_XXdSS            = isnan(txtCAB(:,1).*txtCAB(:,2));
txtCxx(id_XXdSS,:)  = [];
%--
imageFileNames      = cellstr(cat(1,txtCxx))';
%--
trXA                = load([camFldr,filesep,'camPrmsSNGRATetc_',funcID,'_',camN0,'.mat']);
trXB                = load([camFldr,filesep,'camPrmsSNGRATetc_',funcID,'_',camN1,'.mat']);
boardsize           = trXA.boardsize;
squareSize          = trXA.squareSize;

%% --
jj                  = 1;
tkEv2c              = 1+ceil((size(imageFileNames,2)-NtkEv2c)/NtkEv2c);

%--
for ii = 1:tkEv2c:size(imageFileNames,2)
    load([camFldr,filesep,'mf0toXX_',camN0,'_',funcID,filesep,imageFileNames{ii}]);
    %--
    if prod(brzf == boardsize)
        pts_fin = pnzf;
    else
        if prod(brz0 == boardsize)
            pts_fin = pnz0;
        else
           if prod(brz1 == boardsize)
               pts_fin  = pnz1;
           else
           end
        end
    end
    if pts_fin(1,1) < median(pts_fin(:,1))
           pts_fin = flipud(pts_fin);
    end
    %--
    dst         = undistortFisheyeImage( frame,  trXA.params.Intrinsics,'ScaleFactor',trXA.facH);
    pts_fi0     = undistortFisheyePoints(pts_fin,trXA.params.Intrinsics,              trXA.facH);
	%--
    pts_fi0     = pts_fi0';
    ds0         = dst(:,:,:);                   %MDE trick; 'dst' is also a MATLAB-function
    %--
    load([camFldr,filesep,'mf0toXX_',camN1,'_',funcID,filesep,imageFileNames{ii}]);
    %--
    if prod(brzf == boardsize)
        pts_fin = pnzf;
    else
        if prod(brz0 == boardsize)
            pts_fin = pnz0;
        else
           if prod(brz1 == boardsize)
               pts_fin  = pnz1;
           else
           end
        end
    end
    if pts_fin(1,1) < median(pts_fin(:,1))
           pts_fin = flipud(pts_fin);
    end
    %--
    dst         = undistortFisheyeImage( frame,  trXB.params.Intrinsics,'ScaleFactor',trXA.facH);
    pts_fi1     = undistortFisheyePoints(pts_fin,trXB.params.Intrinsics,              trXA.facH);
    %--
    pts_fi1     = pts_fi1';
    %--
    imagePoints(:,1,jj,1) = pts_fi0(1,:);
    imagePoints(:,2,jj,1) = pts_fi0(2,:);
    imagePoints(:,1,jj,2) = pts_fi1(1,:);
    imagePoints(:,2,jj,2) = pts_fi1(2,:);
    jj = jj + 1;
end

% Generate world coordinates of the checkerboard keypoints
worldPoints = generateCheckerboardPoints(boardsize, squareSize);

% Calibrate the camera
tic
disp('estimateCameraParameters')
[stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', true, 'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'mm', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', []);
toc

% check the calibration and perform it new optionally
xrpA = stereoParams.CameraParameters1.ReprojectionErrors;
xrpB = stereoParams.CameraParameters2.ReprojectionErrors;
%--
mrpA        = squeeze(mean((  [    xrpA(:,1,:).^2+    xrpA(:,2,:).^2]).^0.5));
nrpA        = squeeze(max(max([abs(xrpA(:,1,:)),  abs(xrpA(:,2,:))  ])     ));
mrpB        = squeeze(mean((  [    xrpB(:,1,:).^2+    xrpB(:,2,:).^2]).^0.5));
nrpB        = squeeze(max(max([abs(xrpB(:,1,:)),  abs(xrpB(:,2,:))  ])     ));
if strcmp(camFldr,'D:\0876_FitHydrH0\CLB_2019_07_05_08_46_xxx#xxx') && camN0 == '2' && camN1 == '3'
idAB        = unique([find(mrpA>quantile(mrpA,(maxErp9x-0.1)));...
                      find(mrpB>quantile(mrpB,(maxErp9x-0.1)))]);
else
idAB        = unique([find(mrpA>quantile(mrpA,(maxErp9x)));...
                      find(mrpB>quantile(mrpB,(maxErp9x)))]);  
end            

%--
if ~isempty(idAB)
    imagePoints(:,:,idAB,:) = [];
    disp('estimateCameraParameters II')
    [stereoParams, pairsUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', true, 'EstimateTangentialDistortion', true, ...
    	'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'mm', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', []);
    disp(['------------']);
    disp(['max pixel error correction applied'])
    disp(['------------']);
end
%--
save([camFldr,filesep,'camPrmsSTEREOetc_',funcID,'_c',camN0,'to',camN1],...
      'stereoParams', 'pairsUsed', 'estimationErrors','imageFileNames',...
      'maxErp9x','NtkEv2c');

%% --
h1 = figure; showReprojectionErrors(stereoParams);                                  % View reprojection errors
saveas(h1,[camFldr,filesep,'camPrmsSTEREOetc_',funcID,'err','_c',camN0,'to',camN1],'jpg');
%--
h2 = figure; showExtrinsics(stereoParams, 'CameraCentric');                         % Visualize pattern locations
saveas(h2,[camFldr,filesep,'camPrmsSTEREOetc_',funcID,'ext','_c',camN0,'to',camN1],'jpg');
%--
if size(ds0) == size(dst)
    [J1, J2] = rectifyStereoImages(         ds0,      dst, stereoParams,'OutputView', 'full');
else
    [J1, J2] = rectifyStereoImages(imrotate(ds0,-90), dst, stereoParams,'OutputView', 'full');
end
h3 = figure; imshowpair(imfuse(J1,J2), J2,'montage');
saveas(h3,[camFldr,filesep,'camPrmsSTEREOetc_',funcID,'mnt','_c',camN0,'to',camN1],'jpg');
%--
if size(ds0) == size(dst)
    [J3, J4] = rectifyStereoImages(         ds0,      dst, stereoParams,'OutputView', 'valid');
else
    [J3, J4] = rectifyStereoImages(imrotate(ds0,-90), dst, stereoParams,'OutputView', 'valid');
end
h4 = figure; imshowpair(imfuse(J3,J4), J4,'montage');
saveas(h4,[camFldr,filesep,'camPrmsSTEREOetc_',funcID,'mnu','_c',camN0,'to',camN1],'jpg');

end
