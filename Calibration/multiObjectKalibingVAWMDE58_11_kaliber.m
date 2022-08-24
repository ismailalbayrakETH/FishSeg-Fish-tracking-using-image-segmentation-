function multiObjectKalibingVAWMDE55_11_kaliber(avipath,savpath,funMFID,maskBW,hini,vini,hok,vok,vidFramM,pXthr0,pXthr1,mXthr0,strIC0,tkdID,medDpnzf,medDpnz0,medDpnz1,startI,finalI)
%-- main function
obj             = setupSystemObjects();
nextId          = 1;                % initial ID of the next track
dummyId         = 0;                % initial ID of the next track 2 save

T               = uint8(pXthr0*adaptthresh(vidFramM));

while ~isDone(obj.reader) && dummyId<=finalI
    frame       = readFrame();
    displayTrackingResults();
end

%% Create System Objects
    function obj = setupSystemObjects()
        obj.reader          = vision.VideoFileReader( avipath , ...
                                                 'ImageColorSpace', 'RGB', ...
                                                 'VideoOutputDataType', 'uint8');
        %--
        set(0,'showHiddenHandles','on') ;                                    
        %--
        obj.videoPlayer     = vision.VideoPlayer('Position', [20, 150,1280, 960]);
        fig_handle          = gcf;  
        ftw                 = fig_handle.findobj('TooltipString', 'Maintain fit to window'); 
        ftw.ClickedCallback();  
        %--
        obj.maskPlayer      = vision.VideoPlayer('Position', [1400,50,512,384]);
        fig_handle          = gcf;  
        ftw                 = fig_handle.findobj('TooltipString', 'Maintain fit to window'); 
        ftw.ClickedCallback();  
        %--   
    end

%% Read a Video Frame
    function frame = readFrame()
        frame               = obj.reader.step();  % 
        frame               = uint8(maskBW).*frame;
    end

%% Compute and Display Tracking Results
    function displayTrackingResults()
        %-----00 header
        mask                = uint8(abs(single(frame(:,:,1))-single(vidFramM(:,:,1))));
        mask(mask<pXthr1)   = NaN;
        mask                = mask - T;
        mask                = imbinarize(mask);
        %--
        mXthri              = sum(sum(mask))/(size(mask,1)*size(mask,2));
        if mXthr0 < mXthri
            mask = imclose(mask,strel('disk',strIC0));
            mask = imfill(mask,'holes');
        end
        %--
        dummyId             = dummyId + 1;
        nummy               = num2str(dummyId+1000000);
        nummy               = nummy(2:end);
        
        %-----20 computations
        if((dummyId)/tkdID)-floor((dummyId)/tkdID)==0   && ...
            mXthr0 < mXthri                             && ...
            dummyId > startI
            %--
            disp(['frame: ',num2str(nummy)]);% add ':' to stop continous output
            tmp = uint8(mask).*imadjust(uint8(rgb2gray(frame)),[],[],100);
            %--
            mask1           = imclose(tmp, strel('disk',10));
            mask1           = imfill(mask1, 'holes');
            cc              = bwconncomp(mask1); 
            stats           = regionprops(cc, 'Area');
            idx             = sortrows([[stats.Area];1:size([stats.Area],2)]',-1);
            if size(idx,1)>1
            idx             = idx(1:2,:);    
            end
            
            if isempty(idx)
                mask3   = uint8(mask1);
            else
                mask2   = ismember(labelmatrix(cc),idx(:,2));
                mask3	= imdilate(mask2,strel('disk',round(0.1*sum([stats(idx(:,2)).Area]).^0.5)));
            end
                mask3	= uint8(mask3);
                
            %--
            mask = uint8(mask)+mask3;
            %-- 
            [pnzf,brzf]     = detectCheckerboardPoints(uint8(mask3).*rgb2gray(frame));
            szPf            = size(pnzf,1); 
            if szPf==0
                pnzf = [NaN NaN]; 
            end
            
            %--
            if      brzf(1)>=vini && brzf(2)>=hini
                %--
                if     ((brzf(1)==vok && brzf(2)==hok) && abs(median(diff(pnzf(:,2))))> medDpnzf)
                        %--
                        save([savpath,filesep,funMFID,filesep,nummy,'data.mat'],...% save .mat files into folder "mf0toXX_*_58"
                              'frame','mask','pnzf','brzf');
                        %--
                else 
                    framF       = rgb2gray(frame);
                    framF       = uint8(mask3).*framF;
                    framF       = locallapfilt(framF, 0.1, 0.5, 0.1);
                    framG       = imerode(tmp,strel('disk',3));
                    %--
                    [pnz0,brz0]  = detectCheckerboardPoints(uint8(mask3).*framF); % [imagePoints,broadSize]
                    [pnz1,brz1]  = detectCheckerboardPoints(uint8(mask3).*framG);
                    szP0              = size(pnz0,1); if szP0==0; pnz0 = [NaN NaN]; end
                    szP1              = size(pnz1,1); if szP1==0; pnz1 = [NaN NaN]; end 

                    if ((brz0(1)==vok && brz0(2)==hok) && (abs(median(diff(pnz0(:,2))))>medDpnz0))  || ...
                       ((brz1(1)==vok && brz1(2)==hok) && (abs(median(diff(pnz1(:,2))))>medDpnz1))
                        %--
                        save([savpath,filesep,funMFID,filesep,nummy,'data.mat'],... 
                              'frame','mask','pnzf','pnz0','pnz1','brzf','brz0','brz1');
                        %--
                    end
                end
                %--
            end         % concerning 'if brzf(1)>=vini && brzf(2)>=hini'
            %--              
        end             % concerning 'dummyId > mndId && ((dummyId)/tkdID)-floor((dummyId)/tkdID)==0 && mXthr0 < mXthri'
        
        %-----40 prepare 'frame' for video
        frame   = insertText(frame,[size(mask,2)-300 size(mask,1)-100],...
                             num2str(mXthri,'%.5f'),'FontSize',54,'TextColor','white');
        %--
        if exist('pnzf')
            if ~isempty(pnzf) && prod(prod(~isnan(pnzf)))
                frame   = insertMarker(frame,pnzf,'o','color','blue', 'size',16);
                frame   = insertMarker(frame,pnzf,'x','color','blue', 'size',16);
            end
        end
        if exist('pnz0')
            if ~isempty(pnz0) && prod(prod(~isnan(pnz0)))
                frame   = insertMarker(frame,pnz0,'o','color','black','size',12);
                frame   = insertMarker(frame,pnz0,'+','color','black','size',12);
            end
        end
        if exist('pnz1')
            if ~isempty(pnz1) && prod(prod(~isnan(pnz1)))
                frame   = insertMarker(frame,pnz1,'o','color','red',  'size',10);
                frame   = insertMarker(frame,pnz1,'o','color','red',  'size', 5);
           	end
        end
        %--
        obj.videoPlayer.step(frame);  % VideoPlayer                                              
        obj.maskPlayer.step(label2rgb(uint8(mask),colormap([1 1 1; 1 0 0]),'k'));    
        
        %-----60 save 'frame' as *.png                
                    if isfile(        [savpath,filesep,funMFID,filesep,nummy,'data.mat'])
                        %--
                        imwrite(frame,[savpath,filesep,funMFID,filesep,nummy,'figi.png']);% save .png files into folder "mf0toXX_*_58"
                        %--
                    end
        
        %-----
    end                 % concerning 'function displayTrackingResults()'

%--
end                     % concerning 'main function'