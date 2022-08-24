function orderDupliFiles55_11(camFldr, camFld2, camN0, camN1, funcID)
%--
 damFldA 	= [camFldr,filesep,'mf0toXX_',          camN0,'_',funcID];
 damFldB   	= [camFld2,filesep,'mf0toXX_',          camN1,'_',funcID];
 eamFldA   	= [camFldr,filesep,'d',                 camN0,'to',camN1,'_',funcID];
 eamFldB    = [camFld2,filesep,'d',                 camN1,'to',camN0,'_',funcID];
%--
trXA   = load([camFldr,filesep,'camPrmsSNGRATetc_',funcID,'_',camN0,'.mat']);
trXB   = load([camFld2,filesep,'camPrmsSNGRATetc_',funcID,'_',camN1,'.mat']);
%--
if exist(eamFldA); rmdir(eamFldA,'s');  end
if exist(eamFldB); rmdir(eamFldB,'s');  end
mkdir(eamFldB);
% --
txtC01      = cat(1,trXA.matnam.name);
txtC02      = cat(1,trXB.matnam.name);
for i = 1:size(txtC01,1); numC01(i)=str2num(txtC01(i,1: 6)); end
for i = 1:size(txtC02,1); numC02(i)=str2num(txtC02(i,1: 6)); end
numC01      = numC01';
numC02      = numC02';
%--
idC01  = zeros(size(numC01));
for i = 1:size(numC01,1)
    if find(numC01(i)==numC02)
        idC01(i) = find(numC01(i)==numC02);
    end
end
%--
allC01 = [numC01, idC01, idC01./idC01];
txtCxx = txtC01(~isnan(allC01(:,3)),:);
txtCAB = ones(size(txtCxx,1),2);

%% --
for i = 1:size(txtCxx,1) 
%--  
        IDa    = find(strcmp(cat(1,          {trXA.matnam.name}),[txtCxx(i,1:6),'data.mat'])); 
        IDb    = find(strcmp(cat(1,          {trXB.matnam.name}),[txtCxx(i,1:6),'data.mat'])); 
%--
        load([damFldA,filesep,txtCxx(i,1:6),'data.mat'],'frame');
        dst    = undistortFisheyeImage( frame,                    trXA.params.Intrinsics,'ScaleFactor',trXA.facH);
        pst    = undistortFisheyePoints(cat(1,trXA.pts_s{IDa}{:}),trXA.params.Intrinsics,              trXA.facH);
        xyr    = [median(pst),10*abs(median(median(diff(pst))))];
        if isnan(trXA.matnam(IDa).dSS_ok)
           dst = insertMarker(dst,pst,'o','color','red','size',4); 
           dst = insertShape(dst,'circle',xyr,'color','cyan','LineWidth',5);
           txtCAB(i,1) = NaN;
        else
           dst = insertMarker(dst,pst,'o','color','green','size',4);
        end

        load([damFldB,filesep,txtCxx(i,1:6),'data.mat'],'frame');
        dstB   = undistortFisheyeImage( frame,                    trXB.params.Intrinsics,'ScaleFactor',trXB.facH);
        pst    = undistortFisheyePoints(cat(1,trXB.pts_s{IDb}{:}),trXB.params.Intrinsics,              trXB.facH);
        xyr    = [median(pst),10*abs(median(median(diff(pst))))]; 
        if isnan(trXB.matnam(IDb).dSS_ok)
           dstB = insertMarker(dstB,pst,'o','color','red','size',4); 
           dstB = insertShape(dstB,'circle',xyr,'color','cyan','LineWidth',5);
           imwrite(imfuse(imresize(dst,0.5),imresize(dstB,0.5),'montage'),[eamFldB,filesep,txtCxx(i,1:end-5),'_.jpg']);
           txtCAB(i,2) = NaN;
        else
           dstB = insertMarker(dstB,pst,'o','color','green','size',4); 
           imwrite(imfuse(imresize(dst,0.5),imresize(dstB,0.5),'montage'),[eamFldB,filesep,txtCxx(i,1:end-5),'c.jpg']);
        end
%--
end

%% --
save([camFldr,filesep,'camPrmsDOUBLEetc_',funcID,'txt_c',camN0,'to',camN1,'.mat'],'txtCxx','txtCAB');
