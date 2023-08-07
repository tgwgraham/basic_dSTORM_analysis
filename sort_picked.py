# This is a small wrapper script that sorts trajectories from cell picked using the MATLAB cellPicker

import autosmt_utils as au

maskmat = 'out.mat' # mask file (generated using MATLAB cellpicker function)
csvfolder = '../tracking' # folder that contains the CSV files from quot tracking 
measfolder = '../roi_measurements'
outfolder = 'cell_by_cell_csvs_all' # folder for sorted CSV output (doesn't have to exist yet)

au.classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder,maskvarname='roimasks')
