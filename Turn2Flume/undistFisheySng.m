function undistFisheySng(camFldr,kalFldr,savFldr,camN0,funcID)
fundID = num2str(100+str2num(funcID)-0);
fundID = fundID(2:3);     

%% -- load data
load([kalFldr,filesep,'camPrmsSNGRATetc_',fundID,'_',num2str(camN0)]);
% load([camFldr(1:30),'TempData',filesep,'tr0toXX_',num2str(camN0),'_',funcID,'_doitAll']); % for vidFramM & maskBW
load([camFldr(1:21),'TempData',filesep,'tr0toXX_',num2str(camN0),'_',funcID,'_doitAll']);

%% -- transvert excels to mat files
excel_name = [camFldr,filesep,'tracks_Cam_',num2str(camN0),'_mog2.xlsx'];
sheet = sheetnames(excel_name);
opts = detectImportOptions(excel_name);
opts.Sheet = sheet(1);
opts.SelectedVariableNames = [2:4];
opts.DataRange = 'A2';
temp = readmatrix(excel_name,opts);
time = temp(:,1)/20; % calculated time from frames (fps=20)
% time = temp(:,1)/5; % calculated time from frames (fps=5)
temp = [temp(:,1),time,temp(:,2:3)];
pat =  digitsPattern;
fishIDs = ones(size(temp,1),1)*str2num(extract(sheet(1),pat));
tracks = [fishIDs,temp];

for k = 2:length(sheet)
    opts.Sheet = sheet(k);
    opts.SelectedVariableNames = [2:4];
    opts.DataRange = 'A2';
    temp = readmatrix(excel_name,opts);
    time = temp(:,1)/20; 
%     time = temp(:,1)/5;
    temp = [temp(:,1),time,temp(:,2:3)];
    pat =  digitsPattern;
    fishIDs = ones(size(temp,1),1)*str2num(extract(sheet(k),pat));
    track = [fishIDs,temp];
    tracks = [tracks;track];
end

% -- undistort fisheye
tracks(:,6:7) = undistortFisheyePoints(tracks(:,4:5),params.Intrinsics,facW);

% -- save mat file
save([camFldr,filesep,'tracks_Cam_',num2str(camN0),'_doitall'],...
    'vidFramM','maskBW','tracks');

%% save further data
figure(1);
plot(tracks(:,6),tracks(:,7),'o');
axis equal;
xlabel('X (px)');
ylabel('Y (px)');
title(['Cam ',num2str(camN0)]);
set(gca,'LineWidth',1,'FontSize',12,'FontName','Times New Roman','Box','on');
set(gcf,'position',[200 200 1000 550]);
% print(figure(1),'-djpeg','-r300',[camFldr,filesep,'tracks_',camFldr(31:33),'_Cam_',num2str(camN0)]);
% print(figure(1),'-djpeg','-r300',[camFldr,filesep,'tracks','_Cam_',num2str(camN0),'_',camFldr(59:61)]);
print(figure(1),'-djpeg','-r300',[camFldr,filesep,'tracks_',camFldr(38:40),'_Cam_',num2str(camN0)]);

end
