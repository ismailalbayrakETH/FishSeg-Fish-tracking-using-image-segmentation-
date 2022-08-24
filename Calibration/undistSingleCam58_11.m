function undistSingleCam58_11(camFldr,camN,funcID,squareSize,boardsize,dSS_crit,maxErp01,maxErp11,facW,pk)
%% 00.50 parameter and header
facH            = facW;                                         % height factor  = width factor
VisOnOff        = 'off';                                        % visibility for figure
idCalibC        = [];                                           % initialisation
%--
camFldrX        = [camFldr,filesep,'tr0toXX_',camN,'_',funcID];
of0Fldr         = [camFldr,filesep,'of0toXX_',camN,'_',funcID];
if exist(of0Fldr); 
    rmdir(of0Fldr,'s'); 
end
mkdir(of0Fldr); % make new folders of of0toXX_*_58
%--
matFldr         = [camFldr,filesep,'mf0toXX_',camN,'_',funcID];
matnam          = dir([matFldr,filesep,'*.mat']); 
N               = size(matnam,1); 
pts_s           = cell(1,N); 
%% 10.00 handle chessboard points

%%-- prepare calibration patterns

%%-- fill calibration patterns
for i = 1:N 
    load([matFldr,filesep,matnam(i).name]);
    dSS = sqrt(sum(median(diff(pnzf).^2)));
    disp([num2str(N-i),'...',matnam(i).name(1:6),'...',num2str(dSS,'%5.1f')]);
    %--
    if prod(brzf == boardsize)
        diagChessPx = max(pnzf)-min(pnzf);
        pts_tmp = pnzf;
    else
        if prod(brz0 == boardsize)
            diagChessPx = max(pnz0)-min(pnz0);
            pts_tmp = pnz0;
            dSS     = sqrt(sum(median(diff(pnz0).^2)));
        else
           if prod(brz1 == boardsize)
               diagChessPx = max(pnz1)-min(pnz1);
               pts_tmp  = pnz1;
               dSS      = sqrt(sum(median(diff(pnz1).^2)));
           else
           end
        end
    end
    %--  
    if exist('diagChessPx')
        %--
        if pts_tmp(1,1) < median(pts_tmp(:,1))
           pts_tmp = flipud(pts_tmp);
        end
        matnam(i).dSS       = dSS;
        matnam(i).dSS_ok    = double(uint8(ceil(dSS-dSS_crit)))/double(uint8(ceil(dSS-dSS_crit)));
        %--
        matnam(i).ptMED_i   = matnam(i).dSS_ok*median(pts_tmp);
        matnam(i).ptXYR_i   = matnam(i).dSS_ok*sqrt(sum((0.5*[size(frame,2),size(frame,1)] - median(pts_tmp)).^2));
        matnam(i).ptSGN_i   = matnam(i).dSS_ok*sign(     0.5*[size(frame,2),size(frame,1)] - median(pts_tmp)     );
        matnam(i).ptXmn_i   = matnam(i).dSS_ok*pts_tmp(find(pts_tmp(:,1)==nanmin(pts_tmp(:,1)),1,'first'),:);
        matnam(i).ptXmx_i   = matnam(i).dSS_ok*pts_tmp(find(pts_tmp(:,1)==nanmax(pts_tmp(:,1)),1,'first'),:);
        matnam(i).ptYmn_i   = matnam(i).dSS_ok*pts_tmp(find(pts_tmp(:,2)==nanmin(pts_tmp(:,2)),1,'first'),:);
        matnam(i).ptYmx_i   = matnam(i).dSS_ok*pts_tmp(find(pts_tmp(:,2)==nanmax(pts_tmp(:,2)),1,'first'),:);

        for j = 1: ((boardsize(1)-1)*(boardsize(2)-1))
            pts_s{i}{j} = pts_tmp(j,:);
        end
        matnam(i).pts_i = i; 
    end
            
    clear brz0 brzf brz1 pnz0 pnzf pnz1 diagChessPx %pts_tmp
end

%% 30.00 find IDs of non-valid and valid chessboard points
ima_proc    = NaN;
ima_dlet    = NaN;

%% 40.00 find IDs of chessboards in center - and at extremes of axis (i.e, {min,max}(x,y)) as well as quadrants (i.e, near corners)
%%-- find idCalibC - do it the easy way
idCalibC                    = cat(1,matnam.pts_i).*cat(1,matnam.dSS_ok);
idCalibC(isnan(idCalibC))   = [];
tkpk                        = 1+ceil((size(idCalibC,1)-pk)/pk);
idCalibC                    = idCalibC(1):tkpk:idCalibC(end);
idCalibC                    = idCalibC';

%%-- sort findings
idCalibC    = unique(sortrows(idCalibC));
if idCalibC(1)==0; idCalibC(1)=[]; end         %delete the falsified pseudo chessboards (diagnale < dCP_crit)  

%% 50.00 do the initial calibration
%%-- preparation
imgSiz = [size(frame,1),size(frame,2)];

%%-- do the calibration
tic
disp(['start fisheye calibration @', datestr(now,'HH:MM:SS')]);

%%-- re-format the already detected checkerboard points' data
for ii = 1:size(idCalibC,1)
    a                   = pts_s(idCalibC(ii));
    aa                  = a{:};
    imagePoints(:,:,ii) = cat(1,aa{:,:});
    clear a aa
end

%%-- generate world coordinates for the corners of the checkerboard squares.
worldPoints = generateCheckerboardPoints(boardsize,squareSize);

%%-- estimate the fisheye camera calibration parameters based on the image and world points.
params      = estimateFisheyeParameters(imagePoints,worldPoints,[size(frame,1) size(frame,2)]);
toc
disp(['final fisheye calibration @', datestr(now,'HH:MM:SS')]);

%% 55.00 do the final calibration
% check the calibration and perform it new optionally
xrpA        = params.ReprojectionErrors;
mrpA        = squeeze(mean((  [    xrpA(:,1,:).^2+    xrpA(:,2,:).^2]).^0.5));
nrpA        = squeeze(max(max([abs(xrpA(:,1,:)),  abs(xrpA(:,2,:))  ])     ));
idA         = unique([find(mrpA>maxErp01); find(nrpA>maxErp11)]);
imagePointo = imagePoints;
%--
if ~isempty(idA)
    imagePointo(:,:,idA,:) = [];
    params      = estimateFisheyeParameters(imagePointo,worldPoints,[size(frame,1) size(frame,2)]);
    disp(['------------']);
    disp(['max pixel error correction applied'])
    disp(['------------']);
end

for i = 1:size(idCalibC,1)
    load([matFldr,filesep,matnam(idCalibC(i)).name]);
    disp([num2str(size(idCalibC,1)-i),'...',matnam(idCalibC(i)).name(1:6),'...',num2str(matnam(idCalibC(i)).dSS,'%5.1f')]);
    dst 	= undistortFisheyeImage(frame,params.Intrinsics,'ScaleFactor',facW);
    warning('off')
    f1      = figure('Name',num2str(idCalibC(end)),'Visible',VisOnOff);
    imshow(dst);
    hold on
    pts_fin = undistortFisheyePoints(imagePoints(:,:,i),params.Intrinsics,facW);
    %--
    if ~isempty(find(i==idA,1))
    plot(pts_fin(:,1),pts_fin(:,2),'r.-');
    else
    plot(pts_fin(:,1),pts_fin(:,2),'go-');
    end
    plot(pts_fin(1,1),pts_fin(1,2),'rx','MarkerSize',9);
    hold off
    %--
    iptsetpref('ImshowBorder','tight')
    print(gcf,  [of0Fldr,filesep,matnam(idCalibC(i)).name(1:6),'figj.jpg'],'-djpeg','-r0');
    save(       [of0Fldr,filesep,matnam(idCalibC(i)).name(1:6),'datb.mat'],...
                 'dst', 'pts_fin',...
                 'camN', 'funcID', 'squareSize', 'boardsize');
   closereq
end

%% 90.00 save further data
    save([camFldr,filesep,'camPrmsSNGRATetc_',funcID,'_',camN,'.mat'],          ...
                'camN', 'funcID', 'squareSize', 'boardsize',                    ...
                'ima_dlet', 'ima_proc', 'imagePoints', 'imagePointo', 'idA',    ...
                'pk', 'idCalibC', 'dSS_crit', 'facW', 'facH',                   ...
                'params', 'matnam',...
                'pts_s');
            
    %%-- visualize calibration accuracy and camera extrinsics.
	sRE = figure; showReprojectionErrors(params);
    saveas(sRE,[camFldr,filesep,'camPrmsSINGLEetc_',funcID,'err_c',camN],'jpg');
    %--
	sEx = figure;         showExtrinsics(params);
    saveas(sEx,[camFldr,filesep,'camPrmsSINGLEetc_',funcID,'ext_c',camN],'jpg');
    %--
	sXY = figure;         
    imshow(frame)
    hold on
    for i = 1: size(pts_s,2)
        pts_xy = cat(1,pts_s{i}{:});
        plot(pts_xy(:,1),pts_xy(:,2),'r.')
    end
    for i = 1: size(idCalibC,1)
        plot(imagePoints(:,1,i),imagePoints(:,2,i),'bo-')
        plot(matnam(idCalibC(i)).ptMED_i(1),matnam(idCalibC(i)).ptMED_i(2),'bo','MarkerSize',12) 
    end
    idCalibCok = idCalibC; idCalibCok(idA) = [];
    for i = 1: size(idCalibCok,1)
        plot(imagePointo(:,1,i),imagePointo(:,2,i),'y-')
        plot(matnam(idCalibCok(i)).ptMED_i(1),matnam(idCalibCok(i)).ptMED_i(2),'yo','MarkerSize',14)
        plot(matnam(idCalibCok(i)).ptMED_i(1),matnam(idCalibCok(i)).ptMED_i(2),'ko','MarkerSize',16) 
    end
    saveas(sXY,[camFldr,filesep,'camPrmsSINGLEetc_',funcID,'sXY_c',camN],'jpg');
    %--
	sXX = figure;         
    imshow(dst)
    hold on
    for i = 1: size(pts_s,2)
        pts_xy = cat(1,pts_s{i}{:});
        pts_xy = undistortFisheyePoints(pts_xy,params.Intrinsics,facW);
        plot(pts_xy(:,1),pts_xy(:,2),'r.')
    end
    for i = 1: size(idCalibC,1)
        pts_ip = undistortFisheyePoints(imagePoints(:,:,i),params.Intrinsics,facW);
        plot(pts_ip(:,1),pts_ip(:,2),'bo-')
        plot(mean(pts_ip(:,1)),mean(pts_ip(:,2)),'bo','MarkerSize',12); 
    end
    idCalibCok = idCalibC; idCalibCok(idA) = [];
    for i = 1: size(idCalibCok,1)
        pts_io = undistortFisheyePoints(imagePointo(:,:,i),params.Intrinsics,facW);
        plot(pts_io(:,1),pts_io(:,2),'yo-')
        pts_mn = undistortFisheyePoints(matnam(idCalibCok(i)).ptMED_i(:)',params.Intrinsics,facW);
        plot(pts_mn(  1),pts_mn(  2),'yo','MarkerSize',14) 
        plot(pts_mn(  1),pts_mn(  2),'ko','MarkerSize',16) 
     end
    saveas(sXX,[camFldr,filesep,'camPrmsSINGLEetc_',funcID,'sXX_c',camN],'jpg');
