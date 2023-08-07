# This script can be used to do segmentation and measurement of snapshot images 
# after runtime
# This version extracts the timestamp from the corresponding nd2 single-molecule tracking
# movie and includes that as a column in the csv file.

import glob
from os.path import exists
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from csbdeep.utils import normalize
from skimage import io
from stardist.models import StarDist2D
import re
from skimage.measure import regionprops_table



######################### ENTER PARAMETERS HERE #######################################
snapfolder = "snaps2"          # snapshots to use for segmentation
extra_snap_folders = []         # list of additional snapshot folders to measure intensity 
                                # in additional channels
ocnames = []                    # list of whatever names you want to call the channels
                                # corresponding to the additional snapshots
nd2folder = "spt"               # Folder containing nd2 movies.
#######################################################################################

def main(snapfolder=snapfolder,extra_snap_folders=extra_snap_folders,
                ocnames=ocnames,nd2folder=nd2folder):

    columns = ['y','x','I0','bg','y_err','x_err','I0_err','bg_err','H_det','error_flag', 
          'snr','rmse','n_iter','y_detect','x_detect','frame','loc_idx','trajectory',  
          'subproblem_n_traj','subproblem_n_locs']

    os.makedirs(f'masks',exist_ok=True)
    os.makedirs(f'roi_measurements',exist_ok=True)

    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # get a list of all snapshot files that have not yet been segmented and measured
    flist = glob.glob(f"{snapfolder}/*.tif")
    # include files in the file list for which snapshots in all channels are available
    for folder in extra_snap_folders:
        flist = [f for f in flist if exists(f"{folder}/{os.path.basename(f)}")]

    for f in flist:
        # output of image segmentation
        segfname = f'masks/{os.path.basename(f)[:-4]}.csv'
        # output CSV containing ROI properties
        propfname = f'roi_measurements/{os.path.basename(f)[:-4]}.csv'
        
        I = io.imread(f, plugin='pil')
        labels, details = model.predict_instances(normalize(I), prob_thresh=0.5) 
        np.savetxt(segfname,labels,delimiter=',',fmt='%d')
        
        # get region properties of all the ROIs in the image
        rp = regionprops_table(labels, I,
                      properties=('centroid',
                                 'orientation',
                                 'axis_major_length',
                                 'axis_minor_length',
                                  'area',
                                  'area_filled',
                                  'eccentricity',
                                  'intensity_mean',
                                 )
                          )
        df = pd.DataFrame(rp,index = range(1,len(rp['area'])+1))
        df['background'] = I[labels==0].mean()

        # measure intensity mean and background in each snapshot channel, and append this as another column of the dataframe
        # Note that "background" for the other channels may be meaningless if there is signal in the other channels outside
        # of the nuclear masks defined by the first channel
        for i,folder in enumerate(extra_snap_folders):
            # get intensity mean for each addition channel
            # read in image for this channel                
            I = io.imread(f"{folder}/{os.path.basename(f)}", plugin='pil')
            # measure only mean intensity
            rp = regionprops_table(labels, I,properties=('intensity_mean',))
            newcolumns = pd.DataFrame(rp,index = range(1,len(rp['intensity_mean'])+1))
            # get "background" intensity outside of all ROIs. Note that this may not be very meaningful
            # unless the segmentation in the first channel is valid for all channels
            newcolumns[f'background_{ocnames[i]}'] = I[labels==0].mean()
            # the intensity in each channel will be called by whatever name it was given in the macro settings file
            newcolumns.rename(columns={'intensity_mean':f'intensity_mean_{ocnames[i]}'},inplace=True)
            df = pd.concat([df,newcolumns],axis=1)
            
        # get timestamp for nd2 file associated with this snapshot file
        nd2fname = f'{nd2folder}/{os.path.basename(f)[:-4]}.nd2'
        df['time'] = get_ND2_time(nd2fname)
        
        # Write out properties to a csv file.
        df.to_csv(propfname)
    
def get_ND2_time(fname):
        
    """
    Get timestamp in Julian Date Number from a Nikon nd2 file.
    
    args
        fname  :   string - path to the nd2 file
        
    returns
        Julian Date Number as a floating point number 
    
    """
    
    bytesfromend = int(1e6) # where to start reading in the metadata relative the end of the file
    
    # Julian Date Number pattern to match in the metadata at the end of the ND2 file
    pattern = r"<ModifiedAtJDN runtype=\"double\" value=\"(.+?)\"/>" 

    with open(fname, "rb") as file:
        # Read the bytes at the end of the file
        file.seek(os.path.getsize(fname)-bytesfromend)
        bytes = file.read()
        decoded = bytes.decode(errors='ignore')
        numbers = re.findall(pattern, decoded)
        if not numbers:
            return float('nan')
        else:
            return float(numbers[0])


if __name__ == "__main__":
    main()
