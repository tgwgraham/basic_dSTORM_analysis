function cellpicker(snapfolder,maskfolder,range,outfile,varargin)
% cellpicker(snapfolder,range,outfile,[optional arguments])
%
% version 6
%
% %%%%%% INPUTS %%%%%% 
% 
%  REQUIRED PARAMETERS
% 
% snapfolder - base file name for snapshots that were used for
% segmentation
%
% maskfolder - base file name containing segmented masks
% 
% range - min and max file numbers to include
%  if this is not specified, then the file number range is detected
%  automatically
% 
% outfile - output (.mat) file to store the categorization of good vs.
% bad
% 
% OPTIONAL PARAMETERS:
% 
% roifile - file name of text file containing ROI coordinates.
% If this is specified, then it is assumed that only the ROI was imaged in
% single-molecule tracking. If this is not specified, then it is assumed
% that the full FOV was imaged in single-molecule tracking.
% 
% snapfolder2 - base file name for snapshots in a second
% channel
%
% snapfolder3 - base file name for snapshots in a third
% channel
% 
% gridsize - size of grid for display (default = [2,2])
% 
% ncat - number of categories that classifications can adopt (up to 9)
% default = 2
% 
% scale1 - 2-element vector giving minimum and maximum pixel values for
% display of first channel. If not provided, autoscales to min/max of image 
% pixels
% 
% scale2 - 2-element vector giving minimum and maximum pixel values for
% display of second channel. If not provided, autoscales to min/max of 
% image pixels
%  
% scale3 - 2-element vector giving minimum and maximum pixel values for
% display of third channel. If not provided, autoscales to min/max of 
% image pixels
%
% placeholderim - image to use as a placeholder if this FOV number is
% not present
%
% %%%%%% OUTPUTS %%%%%% 
% 
% the output is a .mat file (outfile) containing the following variables:
% 
% 'classification' - vector that indicates whether or not each FOV was
% selected. 1 if selected, 0 if not selected.
% 
% 'range' - file number range [min,max]
% 
% 'masks' - cell array containing a 2D array of segmented labels for each
% FOV with at least one selected cell
% 
% 'roimasks' - cell array containing a 2D array of segmented labels for
% each smaller ROI. If "roifiles" is not defined, then this is the same as
% "masks"
% 
% 'categories' - cell array in which each element is a vector giving the
% category in which each cell in that FOV was classified by the user.
% 
% 'use_whole_roi' - If this is set to true, then it will use the entire ROI
% as the mask. Default: false
%
% Thomas Graham, Tjian-Darzacq Group, UC Berkeley
% 20230526

p = inputParser;
addOptional(p,'gridsize',[2,2]);
addOptional(p,'ncat',2);
addOptional(p,'snapfolder2',[]);
addOptional(p,'snapfolder3',[]);
addOptional(p,'roifile',[]);
addOptional(p,'scale1',[]);
addOptional(p,'scale2',[]);
addOptional(p,'scale3',[]);
addOptional(p,'placeholderim',magic(256)/65536);
addOptional(p,'use_whole_roi',false);
parse(p,varargin{:});

gridsize = p.Results.gridsize;
ncat = p.Results.ncat;
snapfolder2 = p.Results.snapfolder2;
snapfolder3 = p.Results.snapfolder3;
roifile = p.Results.roifile;
scale1 = p.Results.scale1;
scale2 = p.Results.scale2;
scale3 = p.Results.scale3;
placeholderim = p.Results.placeholderim;
use_whole_roi = p.Results.use_whole_roi;

rois = dlmread(roifile);

% Define colorblind-friendly RGB values
color1 = [0, 86, 149] ./ 255;   % Dark blue
color2 = [213, 94, 0] ./ 255;   % Dark orange
color3 = [240, 228, 66] ./ 255; % Yellow
color4 = [0, 158, 115] ./ 255;  % Dark green
color5 = [204, 121, 167] ./ 255; % Mauve
color6 = [86, 180, 233] ./ 255;  % Sky blue
color7 = [230, 159, 0] ./ 255;  % Dark yellow/orange
color8 = [0, 114, 178] ./ 255;   % Light blue
color9 = [213, 94, 0] ./ 255;   % Dark orange (same as color2)
% Combine RGB values into a MATLAB array
colors = [color1; color2; color3; color4; color5; color6; color7; color8; color9];

if isempty(range)
    fnums = [];
    fileNames = dir(snapfolder);
    fileNames = {fileNames.name};
    for i = 1:numel(fileNames)
        match = regexp(fileNames{i}, '\d+', 'match');
        if ~isempty(match)
            fnums(end+1) = str2num(match{1});
        end
    end
    range = [min(fnums),max(fnums)];
end

if exist(outfile,'file')
    temp = load(outfile);
    classification = temp.classification;
    masks = temp.masks;
    if ~isempty(roifile) % output a "roimasks" variable only if roifile is defined
        roimasks = temp.roimasks;
    end
    categories = temp.categories;
else
    % in this version the classification array and masks cell array will
    % start at 1, even if the range of movies does not
    classification = zeros(1,range(2));
    masks = cell(1,range(2));
    if ~isempty(roifile) % only output a "roimasks" variable if roifile is defined
        roimasks = cell(1,range(2));
    end
    categories = cell(1,range(2));
end

f = figure;

nrows = gridsize(1);
ncols = gridsize(2);

nmontages = ceil((1+range(2)-range(1))/(nrows*ncols));
n = range(1);
disp(range(1))
m = 0;
firstdraw = 1;
channelnum = 2;
quitnow = 0;
while ~quitnow

    if firstdraw
        handles = [];
        rects = [];
        celloutlines = {};
        clf;
        for j=n:min(n+nrows*ncols-1, range(2))

            currf = sprintf('%s%s%d.tif',snapfolder,filesep,j);
            
            if ~isempty(snapfolder2)
                currf2 = sprintf('%s%s%d.tif',snapfolder2,filesep,j);
            end
            
            if ~isempty(snapfolder3)
                currf3 = sprintf('%s%s%d.tif',snapfolder3,filesep,j);
            end

            if ~exist(currf,'file')
                handles(end+1) = subplot(nrows,ncols,j-n+1);
                imshow(placeholderim); axis equal; axis off;
                title(sprintf('FOV %d: Rejected',j))
                continue;
            end

            im = double(imread(currf));
            if isempty(scale1)
                minpx = min(im(:));
                maxpx = max(im(:));
            else
                minpx = scale1(1);
                maxpx = scale1(2);
            end
            im = (im - minpx)/(maxpx-minpx);
            segf = sprintf('%s%s%d.csv',maskfolder,filesep,j);
            
            if ~isempty(snapfolder2)
                im2 = double(imread(currf2));
                if isempty(scale2)
                    minpx2 = min(im2(:));
                    maxpx2 = max(im2(:));
                else
                    minpx2 = scale2(1);
                    maxpx2 = scale2(2);
                end
                im2 = (im2 - minpx2)/(maxpx2-minpx2);
            end
            
            if ~isempty(snapfolder3)
                im3 = double(imread(currf3));
                if isempty(scale3)
                    minpx3 = min(im3(:));
                    maxpx3 = max(im3(:));
                else
                    minpx3 = scale3(1);
                    maxpx3 = scale3(2);
                end
                im3 = (im3 - minpx3)/(maxpx3-minpx3);
            end

            if isempty(masks{j})
                if use_whole_roi
                    % if it is set to use the whole imaged ROI, then create
                    % a mask for this
                    curroi = rois(j,:);
                    seg = zeros(size(im));
                    seg(curroi(2):curroi(4),curroi(1):curroi(3)) = 1;
                else
                    seg = csvread(segf);
                end
                masks{j} = seg;
            else
                seg = masks{j};
            end


            handles(end+1) = subplot(nrows,ncols,j-n+1);
            
            if (isempty(snapfolder2) && isempty(snapfolder3)) % if only one channel is defined
                imshow(im); hold on;
            elseif (~isempty(snapfolder2) && isempty(snapfolder3)) % if channels 1 and 2 are defined, but not channel 3
                if channelnum == 2
                    imshow(cat(3,im,im2,im2)); hold on;
                    titlecolor = 'k';
                elseif channelnum == 0
                    imshow(cat(3,im,im2*0,im2*0)); hold on;
                    titlecolor = 'r';
                else
                    imshow(cat(3,im*0,im2,im2)); hold on;
                    titlecolor = [0,0.75,0.75];
                end
            elseif (isempty(snapfolder2) && ~isempty(snapfolder3)) % if channels 1 and 3 are defined, but not channel 2
                if channelnum == 2
                    imshow(cat(3,im,im2,im3)); hold on;
                    titlecolor = 'k';
                elseif channelnum == 0
                    imshow(cat(3,im,im2*0,im3*0)); hold on;
                    titlecolor = 'r';
                else
                    imshow(cat(3,im*0,im2,im3)); hold on;
                    titlecolor = [0,0.75,0.75];
                end
            else % if all three channels are defined
                if channelnum == 2
                    imshow(cat(3,im,im2,im3)); hold on;
                    titlecolor = 'k';
                elseif channelnum == 0
                    imshow(cat(3,im,im2*0,im3*0)); hold on;
                    titlecolor = 'r';
                else
                    imshow(cat(3,im*0,im2,im3)); hold on;
                    titlecolor = [0,0.75,0.75];
                end
                
            end
                
            set(get(gca,'Toolbar'),'Visible','off')
            
            celloutlines{end+1} = [];
                       
            if isempty(categories{j})
                categories{j} = zeros(1,max(seg(:))); % initialize all category classifications to 0
            end
            
            for regind = 1:numel(categories{j})
                currcat = categories{j}(regind);
                currmask = seg==regind;
                bounds = bwboundaries(currmask);
                celloutlines{end}(regind) = plot(bounds{1}(:,2),...
                    bounds{1}(:,1),'-',...
                    'LineWidth',2,'Color',colors(currcat+1,:));            
            end

            if any(categories{j} > 0)
                classification(j) = 1;
            else
                classification(j) = 0;
            end

            rects(end+1) = rectangle('Position',[1,1,fliplr(size(im))],'LineWidth',2);
            if classification(j) == 1
                set(rects(end),'EdgeColor',[1,1,0]);
            else
                set(rects(end),'EdgeColor',[0,0,0]);
            end
            
            % draw ROI rectangle and output smaller mask for ROI if roifile is specified
            if ~isempty(roifile)
                curroi = rois(j,:);
                rectangle('Position',[curroi(1),curroi(2),curroi(3)-curroi(1),curroi(4)-curroi(2)],'LineWidth',1,'EdgeColor','w');
                if isempty(roimasks{j})
                    roimasks{j} = seg(curroi(2):curroi(4),curroi(1):curroi(3));
                end
            end
            
            

            axis equal; axis off;
            
            title(sprintf('FOV %d (%d, %d)',j,minpx,maxpx),'Color',titlecolor);
            set(gcf,'name',sprintf('Cells %d to %d of %d.',n,min(n+nrows*ncols-1, range(2)),range(2)))
        end
        firstdraw = 0;
    end
    
    gottenkey = 0;
    while ~gottenkey
        [x,y,key] = ginput(1); % get keyboard or mouse input from user
        
        if key == 113 % q - quit
            m = nmontages;
            if ~isempty(roifile)
                save(outfile,'classification','range','masks','roimasks','categories'); 
            else
                save(outfile,'classification','range','masks','categories'); 
            end
            close(f)
            quitnow = 1;
        
        elseif key == 28 % reverse arrow - move back one page
            if m > 0
                n = n - nrows*ncols;
                m = m-1;
                if ~isempty(roifile)
                    save(outfile,'classification','range','masks','roimasks','categories'); 
                else
                    save(outfile,'classification','range','masks','categories'); 
                end
                firstdraw = 1;
            end
            
        elseif key == 29 % forward arrow - move forward one page
            if m + 1 > nmontages
                msgbox('Reached the end. Press "q" to quit.')
            else
                n = n + nrows*ncols;
                m = m+1;
            end
            if ~isempty(roifile)
                save(outfile,'classification','range','masks','roimasks','categories'); 
            else
                save(outfile,'classification','range','masks','categories'); 
            end
            firstdraw = 1;
            
        elseif key == 115 % s - save
            if ~isempty(roifile)
                save(outfile,'classification','range','masks','roimasks','categories'); 
            else
                save(outfile,'classification','range','masks','categories'); 
            end
            
        elseif key == 1 % left mouse click - change category of cell under cursor
            clickedim = find(handles==gca);
            clickedindex = n + clickedim - 1;
            
            % Determine in which ROI the click lies, and change the sign of
            % that label
            if ~isempty(masks{clickedindex})
                if round(x) < 1 || round(y) < 1 || round(x) > size(masks{clickedindex},1) || round(y) > size(masks{clickedindex},2)
                    currlabel = 0;
                else
                    currlabel = masks{clickedindex}(round(y),round(x));
                end
                if currlabel % ignore if the user clicked in the background region (zero)
                    categories{clickedindex}(currlabel) = mod(categories{clickedindex}(currlabel) + 1,ncat);
                    %masks{clickedindex}(masks{clickedindex}==currlabel) = -currlabel;
                    set(celloutlines{clickedim}(currlabel),'Color',colors(categories{clickedindex}(currlabel)+1,:))

                    if any(categories{clickedindex} > 0)
                        classification(clickedindex) = 1;
                        set(rects(clickedim),'EdgeColor',[1,1,0])
                    else
                        classification(clickedindex) = 0;
                        set(rects(clickedim),'EdgeColor',[0,0,0])
                    end
                end
            end
            
         elseif key == 3 % right mouse click - deselect cell
            clickedim = find(handles==gca);
            clickedindex = n + clickedim - 1;
            
            % Determine in which ROI the click lies, and change the sign of
            % that label
            if ~isempty(masks{clickedindex})
                currlabel = masks{clickedindex}(round(y),round(x));
                if currlabel % ignore if the user clicked in the background region (zero)
                    categories{clickedindex}(currlabel) = 0;
                    set(celloutlines{clickedim}(currlabel),'Color',colors(1,:))

                    if any(categories{clickedindex} > 0)
                        classification(clickedindex) = 1;
                        set(rects(clickedim),'EdgeColor',[1,1,0])
                    else
                        classification(clickedindex) = 0;
                        set(rects(clickedim),'EdgeColor',[0,0,0])
                    end
                end
            end
            
        elseif key == 114 % r - toggle first channel (or toggle from second to first)
            if channelnum == 2
                channelnum = 1;
            elseif channelnum == 1
                channelnum = 2;
            elseif channelnum == 0
                channelnum = 1;
            end
            firstdraw = 1;

        elseif key == 116 % t - toggle second channel (or toggle from first to second)
            if channelnum == 2
                channelnum = 0;
            elseif channelnum == 1
                channelnum = 0;
            elseif channelnum == 0
                channelnum = 2;
            end
            firstdraw = 1;
        elseif key==110 % 'n' - ask user which file number to go to 
            userinput = inputdlg('Enter movie number:', 'Movie number', [1 50]);
            userinput = str2double(userinput{1});
            if ~isnan(userinput)
                m = floor(userinput/(nrows*ncols))+1;
                n = (m-1)*nrows*ncols + 1;
                save(outfile,'classification','range','masks','categories'); 
                firstdraw = 1;
            else
                disp('Please enter an integer.')
            end

        end
        gottenkey = 1;
    end
end
    
end

