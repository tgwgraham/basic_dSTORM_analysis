% Example of how to call the cellpicker function
% This section of the script runs cellpicker on the first set of files,
% and the next set runs it on the second set of files.
snapfolder = '../snaps2';      % where the snapshots are stored
roifile = '../rois.txt';       % text file providing coordinates of SMT ROIs 
maskfolder = '../masks/';      % where cell masks are stored
                            % Note: This is not used if use_whole_roi is
                            % set to true.
snapfolder2 = '../snaps2';     % second channel of snapshots (optional)
snapfolder3 = '../snaps2';     % third channel of snapshots (optional)
range = [];             % file number range to examine
outfile = 'out.mat';        % where to store selection output
gridsize=[2,3];             % size of image grid for display
ncat = 2;                   % number of categories
scale1 = [0,1e4];           % scale for display of first channel
scale2 = [0,1e4];          % scale for display of second channel
scale3 = [0,1e4];          % scale for display of thirdq channel
use_whole_roi = true;      % set this option to true if you want to take 
                            % all of the trajectories from each imaged ROI
                            % rather than only selecting the ones that
                            % overlap the cell mask.
tightplots = true;          % whether to use tight layout of plotsq


                            
cellpicker(snapfolder,maskfolder,range,outfile,...
    'gridsize',gridsize,...
    'ncat',ncat,...
    'snapfolder2',snapfolder2,...
    'snapfolder3',snapfolder2,...
    'roifile',roifile,...
    'scale1',scale1,...
    'scale2',scale2,...
    'scale3',scale3,...
    'use_whole_roi',use_whole_roi,...
    'tightplots',true);

% right arrow - move to next set of FOVs
% left arrow - move to next set of FOVs
% right mouse click - cycle through selection categories
% left mouse click - deselect cell
% s - save (saving also happens automatically after arrow presses)
% q - quit and save 
% n - jump to FOV number
% r - toggle first channel (or toggle from second to first)
% t - toggle second channel (or toggle from first to second)

% Before running this script, you need to segment nuclei using an
% appropriate segmentation script (e.g., segment_measure_time.py)

