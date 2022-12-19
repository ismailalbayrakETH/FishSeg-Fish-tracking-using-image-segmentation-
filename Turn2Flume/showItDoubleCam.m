function showItDoubleCam(camFldr,camFld2,kalFldr,kalFld2,savFldr,camN0,camN1,funcID,vidOn,sig,alp,dnnMin)
%% -- rename option to be able to read 'funcID-n' as well
fundID      = num2str(100+str2num(funcID)-0);
fundID      = fundID(2:3);                               

%% -- initial parameter definition
t           = 0;
tabiidr 	= [];

%% -- load data
aCam        = load([kalFldr,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN0))]);
bCam        = load([kalFld2,filesep,'camPrmsSNGRATetc','_',fundID,'_',num2str(str2num(camN1))]);

aTMP        = load([camFldr,filesep,'tracks_Cam_',num2str(str2num(camN0)),'_doitall']);
warning('off')
bTMP        = load([camFld2,filesep,'tracks_Cam_',num2str(str2num(camN1)),'_doitall']);
warning('off')
%--
aViF        = aTMP.vidFramM;
bViF    	= bTMP.vidFramM;
if isfield(aTMP,'maskBW'); aMbw = aTMP.maskBW; else; aMbw = 1; end
if isfield(bTMP,'maskBW'); bMbw = bTMP.maskBW; else; bMbw = 1; end
clear aTMP bTMP

load([savFldr,filesep,'regDbl_',funcID,'Xc',camN0,'to',camN1]);

aunF = load([camFldr,filesep,'tracks_cam_',num2str(str2num(camN0)),'_doitall']);
bunF = load([camFld2,filesep,'tracks_cam_',num2str(str2num(camN1)),'_doitall']);
[bunT.tracks(:,6),bunT.tracks(:,7)] = ...
    transformPointsForward(MOVINGREG, bunF.tracks(:,6), bunF.tracks(:,7));

aXYF.fg = aunF.tracks(:,6:7); % aunF.fg(aunF.conn02,:);
bXYT.fg = bunT.tracks(:,6:7); % bunT.fg(bunF.conn02,:);

%% -- do only computation
if ~vidOn
 figvid01            = figure;
 currAxes            = axes;
 %--
 tic
 %--
 for dummycount = 1:max([max(aunF.tracks(:,2)),max(bunF.tracks(:,2))]) % arTf.itsxyMmo00(:,2)
    t = t+1;
    %--
    if (round(t/100)-(t/100))==0 
    fprintf('%s ... ',num2str(uint16(max([max(aunF.tracks(:,2)),max(bunF.tracks(:,2))]))-t));
    if (round(t/1000)-(t/1000))==0; fprintf('\n'); end 
    end
    %--
    ia = find(aunF.tracks(:,2)==t); % aTrk.iXYZS(:,4)
    ib = find(bunF.tracks(:,2)==t); % bTrk.iXYZS(:,4) 
    ja = find(aunF.tracks(:,2)==t); % arTf.itsxyMmo00(:,2)
    jb = find(bunF.tracks(:,2)==t); % brTf.itsxyMmo00(:,2)
    %-- 
%     if vidOn
%         vidFramA = undistortFisheyeImage(rgb2gray(readFrame(aVid)).*uint8(aMbw),aCam.params.Intrinsics,'ScaleFactor',aCam.facW);
%         vidFramB = undistortFisheyeImage(rgb2gray(readFrame(bVid)).*uint8(bMbw),bCam.params.Intrinsics,'ScaleFactor',bCam.facW);
%         vidTramB = imwarp(vidFramB,         MOVINGREG,               'OutputView',imref2d(size(vidFramB)));
%         %--
%         imshow(locallapfilt( imfuse(vidFramA,vidTramB,'blend'      ),sig,alp), 'Parent', currAxes);
%         hold on
%         %--------    
%         plot([aXYF.a1(ia,1) aXYF.a2(ia,1)]',[aXYF.a1(ia,2) aXYF.a2(ia,2)]','c-');
%         plot([bXYT.a1(ib,1) bXYT.a2(ib,1)]',[bXYT.a1(ib,2) bXYT.a2(ib,2)]','b-');
%         plot(aunF.fg(ja,1),aunF.fg(ja,2),'g.');
%         plot(bunT.fg(jb,1),bunT.fg(jb,2),'r.');
%     end
    
    %--------
    X = double([aXYF.fg(ia,1),aXYF.fg(ia,2)]);
    q = double([bXYT.fg(ib,1),bXYT.fg(ib,2)]);
    %--
    if size(X,1)>=3
         dt = delaunayTriangulation(X);
         [xi,dnn] = nearestNeighbor(dt, q);
          if size(q,1)==2       
                Xadd    = X(~ismember(1: size(X,1), xi),:);
                dn1     = (sum(((Xadd-q(1,:)).^2)')).^0.5;
                dn2     = (sum(((Xadd-q(2,:)).^2)')).^0.5;
                xi      = [xi; find(~ismember(1: size(X,1), xi))'];
                dnn     = [dnn; min([dn1;dn2])'];
                [iq,~]  = ind2sub(size([dn1;dn2]),find(ismember([dn1;dn2], min([dn1;dn2]))));
                q       = [q;   q(iq,:)];
          end 
         xi(dnn>dnnMin)      = [];
          q(dnn>dnnMin,:)    = [];
         xnn = X(xi,:);
    else
        if size(q,1)>=3
         qtmp = X;
         X = q; q = qtmp;
         %--
         dt = delaunayTriangulation(X);
         [xi,dnn] = nearestNeighbor(dt, q);
          if size(q,1)==2       
                Xadd    = X(~ismember(1: size(X,1), xi),:);
                dn1     = (sum(((Xadd-q(1,:)).^2)')).^0.5;
                dn2     = (sum(((Xadd-q(2,:)).^2)')).^0.5;
                xi      = [xi; find(~ismember(1: size(X,1), xi))'];
                dnn     = [dnn; min([dn1;dn2])'];
                [iq,~]  = ind2sub(size([dn1;dn2]),find(ismember([dn1;dn2], min([dn1;dn2]))));
                q       = [q;   q(iq,:)];
          end 
         xi(dnn>dnnMin)      = [];
          q(dnn>dnnMin,:)    = [];
         xnn = X(xi,:);
         %--
         qtmp = xnn;
         xnn = q; q = qtmp;
        else
            if size(X,1)>=1 && size(q,1)>=2
            dn1     = (sum(((X-q(1,:)).^2)')).^0.5;
            xi(1)   = find(dn1 == min(dn1));
            dnn     = dn1(xi(1));
                if size(q,1)==2
                dn2     = (sum(((X-q(2,:)).^2)')).^0.5;
                xi(2)   = find(dn2 == min(dn2));
                dnn     = [dnn, dn2(xi(2))];
                end
            xi(dnn>dnnMin)      = [];
             q(dnn>dnnMin,:)    = [];
            xnn = X(xi,:);    
            else
                if size(X,1)>=1 && size(q,1)==1
                qtmp = X;
                X = q; q = qtmp;
                %--    
                dn1     = (sum(((X-q(1,:)).^2)')).^0.5;
                xi(1)   = find(dn1 == min(dn1));
                dnn     = dn1(xi(1));
                    if size(q,1)==2
                    dn2     = (sum(((X-q(2,:)).^2)')).^0.5;
                    xi(2)   = find(dn2 == min(dn2));
                    dnn     = [dnn, dn2(xi(2))];
                    end
                xi(dnn>dnnMin)      = [];
                q(dnn>dnnMin,:)    = [];
                xnn = X(xi,:);
                %--
                qtmp = xnn;
                xnn = q; q = qtmp;
                end  
            end
        end
    end 
    %--------
    if exist('xnn','var') && ~isempty(xnn) 
        if vidOn  
        plot([xnn(:,1) q(:,1)]',[xnn(:,2) q(:,2)]', '-y');
        plot( xnn(:,1),          xnn(:,2),          'ko','MarkerSize',3);
        plot(          q(:,1),            q(:,2),   'yo','MarkerSize',3);
        end
    %-------- 
        X0  = double([aXYF.fg(ia,1),aXYF.fg(ia,2)]);
        xi0 = [];
        for ci = 1: size(X0,1)
            xitmp = ismember(xnn, X0(ci,:));
            xitmp = find(xitmp(:,1).*xitmp(:,2));
            xitmp = [ci*ones(size(xitmp,1),1), xitmp];
            xi0   = [xi0; xitmp];
        end
        %-------- 
        q0 = double([bXYT.fg(ib,1),bXYT.fg(ib,2)]);
        qi0 = [];
        for di = 1: size(q0,1)
            qitmp = ismember(q,   q0(di,:));
            qitmp = find(qitmp(:,1).*qitmp(:,2));
            qitmp = [di*ones(size(qitmp,1),1), qitmp];
            qi0   = [qi0; qitmp];
        end
    end
    %-------- 
    if exist('xi0','var') && ~isempty(xi0) && ~isempty(qi0)
    xi0 = sortrows(xi0,2);
    qi0 = sortrows(qi0,2);
%     tabiidr = [tabiidr;...
%                t*ones(size(    ia(xi0(:,1),1))),...
%                                ia(xi0(:,1)),...
%                                ib(qi0(:,1)),...
%                     aTrk.iXYZS(ia(xi0(:,1))),...
%                     bTrk.iXYZS(ib(qi0(:,1))),...
%                     aXYF.d(         ia(xi0(:,1))),...
%                     bXYF.d(         ib(qi0(:,1))),...
%                     aXYF.r(         ia(xi0(:,1))),...
%                     bXYF.r(         ib(qi0(:,1)))];
    tabiidr = [tabiidr;...
               t*ones(size(    ia(xi0(:,1),1))),...
                               ia(xi0(:,1)),...
                               ib(qi0(:,1)),...
                    aunF.tracks(ia(xi0(:,1))),...
                    bunF.tracks(ib(qi0(:,1)))];    
              
    end 
    %--
    clear X q dnn xi qtmp iq xi0 qi0
    %--
 end    %end of 'dummycount = 1:...' 
 %--
 toc
 %--
end

%% -- save data
if isempty(tabiidr); tabiidr = ones(1,9); end
%--
save([savFldr,filesep,'tabiidr_',funcID,'_c',camN0,'to',camN1],...
       'tabiidr','sig','alp','dnnMin');  

end

