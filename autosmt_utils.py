"""
autosmt_utils.py -- utilities for analysis of automated SMT experiments

"""

import os
import re
# Read config files in TOML format
import toml 
import pandas as pd
import numpy as np
from glob import glob
import scipy.io
import traceback


def read_config(path):
    """
    Read a config file.

    args
        path    :   str

    returns
        dict

    """
    assert os.path.isfile(path), \
        "read_config: path %s does not exist" % path
    return toml.load(path)
    
    
def save_config(path, config):
    """
    Save data in a dict to a TOML file.

    args
        path    :   str
        config  :   dict

    """
    with open(path, 'w') as o:
        toml.dump(config, o)
        
      
    
def sort_PAPA_DR(config,outf='sortedTrajectories'):
    """
sort_PAPA_DR(config,outf='sortedTrajectories')

For PAPA-SMT analysis.

Write out new csv files corresponding to PAPA, DR, and other segments from 
trajectory csv file f.

By default, the output is stored in folder sortedTrajectories. 
Within the output directory, the function creates a series of nested subfolders:
- for each experimental condition(keys of "conditions" sub-dictionary)
- for each experiment within that condition
- for PAPA, DR, and other trajectories within that experiment

Within the PAPA, DR, and other subfolders, it will write out CSV files containing 
PAPA, DR, and other trajectories, respectively. These CSV files are given the same
names as the CSV files in the original folder

args
    config  :   dict
    outf    :   string giving path for writing output files
    """
    
    # make output directory
    os.system(f'mkdir {outf}')
    
    # get frame indices for PAPA and DR frames
    fn = get_gv_framenum(config)
    PAPAframes = fn['gpost']
    DRframes = fn['vpost']

    # column names for writing output CSV files in the correct order
    columns = ['y','x','I0','bg','y_err','x_err','I0_err','bg_err','H_det','error_flag', 
          'snr','rmse','n_iter','y_detect','x_detect','frame','loc_idx','trajectory',  
          'subproblem_n_traj','subproblem_n_locs']
        
    # loop over conditions
    for c in config['conditions'].keys():
        os.makedirs(f"{outf}/{c}",exist_ok=True)
        
        # loop over experiments
        for e in config['conditions'][c]['experiments'].keys():
                
            os.makedirs(f"{outf}/{c}/{e}",exist_ok=True)
            os.makedirs(f"{outf}/{c}/{e}/PAPA",exist_ok=True)
            os.makedirs(f"{outf}/{c}/{e}/DR",exist_ok=True)
            os.makedirs(f"{outf}/{c}/{e}/other",exist_ok=True)
            
            csv_folder = config['conditions'][c]['experiments'][e]['csv_folder']
            csvfiles = glob(f"{csv_folder}/*.csv")
            
            for f in csvfiles:
                data = pd.read_csv(f)   
                PAPAtraj = data[data['frame'].isin(PAPAframes)]
                DRtraj = data[data['frame'].isin(DRframes)]
                othertraj = data[~data['frame'].isin(PAPAframes) & ~data['frame'].isin(DRframes)]
                
                basename = os.path.basename(f)
                
                PAPAtraj.to_csv(f'{outf}/{c}/{e}/PAPA/{basename}', 
                    index=False, columns=columns)
                DRtraj.to_csv(f'{outf}/{c}/{e}/DR/{basename}', 
                    index=False, columns=columns)
                othertraj.to_csv(f'{outf}/{c}/{e}/other/{basename}', 
                    index=False, columns=columns)
                
                


def get_gv_framenum(config):
    """get_gv_framenum(config)
    
    returns arrays containing the frame indices of frames before and after green and violet excitation
    (PAPA and DR, respectively)"""
    
    def add_fw_and_flatten(x,ncycles,fwrange):
        rv = x.reshape(ncycles,1) + fwrange
        rv = rv.flatten()
        return rv
    
    IS = config['illumination_sequence']
    ncycles = IS['ncycles']
    r = IS['r']
    v = IS['v']
    g = IS['g']
    fw = IS['framewindow']
    gfirst = IS['gfirst']
    
    t = 4*r + v + g
    
    fwrange = np.arange(0,fw)
    
    if gfirst: # if the green pulse occurs first in the cycle
        gpre = np.arange(0,ncycles)*t # indices of frames in frame window before green
        gpost = np.arange(0,ncycles)*t + r + g # indices of frames in frame window after green
        vpre = np.arange(0,ncycles)*t + 2*r + g # indices of frames in frame window before violet
        vpost = np.arange(0,ncycles)*t + 3*r + v + g # indices of frames in frame window after violet
    else: # if the violet pulse occurs first in the cycle
        vpre = np.arange(0,ncycles)*t # indices of frames in frame window before violet
        vpost = np.arange(0,ncycles)*t + r + v # indices of frames in frame window after violet
        gpre = np.arange(0,ncycles)*t + 2*r + v # indices of frames in frame window before green
        gpost = np.arange(0,ncycles)*t + 3*r + v + g # indices of frames in frame window after green
    
    gpre = add_fw_and_flatten(gpre,ncycles,fwrange)
    gpost = add_fw_and_flatten(gpost,ncycles,fwrange)
    vpre = add_fw_and_flatten(vpre,ncycles,fwrange)
    vpost = add_fw_and_flatten(vpost,ncycles,fwrange)
    
    return({'vpre':vpre,'vpost':vpost,'gpre':gpre,'gpost':gpost})
    
    
    
def makeStateArrayDataset(config,sortedFolder='sortedTrajectories',
            nworkers=1,isPAPA=False):
    """
    makeStateArrayDataset(config,sortedFolder='sortedTrajectories',
            nworkers=1,isPAPA=False)
    
    Make StateArrayDataset for the entire dataset of sorted PAPA, DR, and other trajectories.
    
    Requires saspt (https://github.com/alecheckert/saspt)
    
    args
        config  :   dict - configuration settings
        sortedFolder    :   directory where sorted PAPA/DR/other trajectories are stored
                            (only relevant to PAPA experiments)
        nworkers    :   number of workers to use for the computation; do not set this to
                        a value higher than the number of CPUs on your computer.
        isPAPA  : Whether or not this is a PAPA experiment (default: False)
    """
    # This has replaced my old function makeSA_sorted in papa_utils.py
    
    from saspt import StateArrayDataset, RBME
    
    if 'sample_size' in config['saspt'].keys():
        sample_size = config['saspt']['sample_size']
    else:
        sample_size = 1000000
    
    settings = dict(
        likelihood_type = RBME,
        pixel_size_um = config['track']['pixel_size_um'],
        frame_interval = config['track']['frame_interval'],
        focal_depth = config['saspt']['focal_depth'],
        sample_size = sample_size,
        path_col = 'filepath',
        condition_col = 'condition',
        progress_bar = True,
        num_workers=nworkers,
    )
        
    conditions = []
    filepaths = []
            
    for c in config['conditions']:
        
        # files are conveniently named whatever they were called originally
        # I'm going to need to loop over experiments, loop over PAPA, DR, and other
        # and append files and condition name to filepaths and conditions lists
                    
        for e in config['conditions'][c]['experiments']:
            
            # get list of files in the current experiment
            csv_folder = config['conditions'][c]['experiments'][e]['csv_folder']
            flist = glob(csv_folder + "/*.csv")
            
            if isPAPA:
                PDO = ['PAPA','DR','other']

                # extract just the name of the csv file without the rest of the path
                flist = [os.path.basename(f) for f in flist]

                for j in range(3):
                    # append condition name, underscore, PAPA/DR/other to conditions list an appropriate number of times
                    conditions.extend([f'{c}_{PDO[j]}'] * len(flist)) 
                    # append file names with correct path to filepaths list
                    filepaths.extend([f'{sortedFolder}/{c}/{e}/{PDO[j]}/{f}' for f in flist])

                    # The above list comprehension looks rather complicated, but it just gives
                    # the path to the file: sortedFolder (by default 'sortedTrajectories'),
                    # condition name, experiment name, PAPA/DR/other, and finally the file name.
            else: 
                # If this is not a PAPA experiment, then append the csv file list with folder names
                filepaths.extend(flist)
                conditions.extend([c]*len(flist))

    # the StateArrayDataset object for PAPA and non-PAPA experiments is essentially the same,
    # except in PAPA experiments, each condition is split into three sub-conditions: PAPA, DR, and other,
    # which are called [condition_name]_PAPA, [condition_name]_DR, and [condition_name]_other

    # Make a DataFrame with the condition/filepath pairs, and use this to make a StateArrayDataset.
    paths = pd.DataFrame({'condition':conditions, 'filepath':filepaths})
    SAD = StateArrayDataset.from_kwargs(paths, **settings)
    return [SAD,paths]


def mergeCellMeasurements(config,outf='merged_measurements.csv'):
    pass
    #TODO: Write a function to import all cell measurement csv files as Pandas DataFrames, merge these
    # dataframes, and write the output to a new csv file.


def makeSA_sorted_sameN(config,sortedFolder='sortedTrajectories', 
    subsampledFolder='subsampledTrajectories',nworkers=1,randseed=None,
                        ignoreOther=False,isPAPA=False):
    """
    makeSA_sorted_sameN(config,sortedFolder='sortedTrajectories',nworkers=1,randseed=None,ignoreOther=False):
    
    Make StateArray for the entire dataset of sorted PAPA, DR, and other trajectories.
    Identify the sorted category with the fewest trajectories, and randomly subsample the same number of trajectories from each of the other two categories.
    
    Requires saspt (https://github.com/alecheckert/saspt)
    
    args
        config  :   dict - configuration settings
        sortedFolder    :   directory where sorted PAPA/DR/other trajectories are stored
        subsampledFolder    :   directory where subsampled trajectories will be stored
        nworkers    :   number of workers to use for the computation; do not set this to
                        a value higher than the number of CPUs on your computer.
        randseed    :   (optional) number to serve as a random seed to initialize the          
                        random number generator (for reproducing the same "random" output when
                        the code is re-run)
        ignoreOther : Do not include "other" trajectories in the subsampling/analysis [default: False]
    """
    
    from saspt import StateArrayDataset, RBME
    from random import sample, seed
    
    # set random seed if one is provided
    if randseed is not None:
        seed(randseed)
    
    if ignoreOther:
        npdo = 2
        PDO = ['PAPA','DR']
    else:
        npdo = 3
        PDO = ['PAPA','DR','other']
    
    if 'sample_size' in config['saspt'].keys():
        sample_size = config['saspt']['sample_size']
    else:
        sample_size = 1000000    
    
    settings = dict(
        likelihood_type = RBME,
        pixel_size_um = config['track']['pixel_size_um'],
        frame_interval = config['track']['frame_interval'],
        focal_depth = config['saspt']['focal_depth'],
        sample_size = sample_size,
        path_col = 'filepath',
        condition_col = 'condition',
        progress_bar = True,
        num_workers=nworkers,
    )
    
    columns = ['y','x','I0','bg','y_err','x_err','I0_err','bg_err','H_det','error_flag', 
          'snr','rmse','n_iter','y_detect','x_detect','frame','loc_idx','trajectory',  
          'subproblem_n_traj','subproblem_n_locs']
    IS = config['illumination_sequence']
    ncycles = IS['ncycles']
    r = IS['r']
    v = IS['v']
    g = IS['g']
    framesPerMovie = ncycles*(4*r+g+v) # total number of frames per movie
   
    conditions = []
    filepaths = []
    
    # concatenate all trajectories from each category, incrementing the frame and trajectory 
    # indices appropriately to avoid overlaps
    if not os.path.isdir(subsampledFolder):
        os.mkdir(subsampledFolder)    
    
    # loop over all experimental conditions
    for c in config['conditions']:
        # initialize a list of PAPA, DR (and other) trajectories
        traj = [None] * npdo
        # initialize a list to store counts of trajectories for PAPA, DR (and other)
        maxtrajnum = [0] * npdo    
        
            # loop over all experiments within this condition
        for e in config['conditions'][c]['experiments']:
                # get list of files in the current experiment
            csv_folder = config['conditions'][c]['experiments'][e]['csv_folder']
            flist = glob(csv_folder + "/*.csv")
            # extract just the name of the csv file without the rest of the path
            flist = [os.path.basename(f) for f in flist]
            
            # loop over csv files in this list
            for f in flist:
                # loop over PAPA, DR (and other) trajectories
                for j in range(npdo):
                    # read in this CSV from among the sorted CSVs in sortedFolder
                    currdf = pd.read_csv(f'{sortedFolder}/{c}/{e}/{PDO[j]}/{f}')
                    # if there is actually data in this CSV
                    if not currdf.empty:
                        if traj[j] is None:
                            # if this is the first file, put it in traj as-is
                            traj[j] = currdf
                            maxtrajnum[j] = currdf['trajectory'].max()
                        else:
                            # if this is not the first file, increment the 
                            # trajectory indices, and then concatenate to the 
                            # DataFrame of trajectories
                            currdf['trajectory'] = currdf['trajectory'] + maxtrajnum[j] + 1
                            maxtrajnum[j] = currdf['trajectory'].max()
                            traj[j] = pd.concat([traj[j],currdf],ignore_index=True)

        # include only trajectories with length greater than 1
        ntraj = [0] * npdo
        for j in range(npdo):
            goodtraj = traj[j]['trajectory'].value_counts()>1
            goodtraj = goodtraj[goodtraj].index # indices of trajectories longer than 1
            sel = traj[j]['trajectory'].isin(goodtraj)
            traj[j] = traj[j][sel] # retain only these trajectories
            ntraj[j] = len(traj[j]['trajectory'].unique())
            
        if npdo==2:
            print(f'Number of trajectories in condition {c}')
            print(f'PAPA trajectories: {ntraj[0]}')
            print(f'DR trajectories: {ntraj[1]}')
            print
        else:
            print(f'Number of trajectories in condition {c}')
            print(f'PAPA trajectories: {ntraj[0]}')
            print(f'DR trajectories: {ntraj[1]}')
            print(f'Other trajectories: {ntraj[2]}')
        
        
        # export all trajectories from the condition with the fewest trajectories
        # subsample the same number from the two other conditions
        mintraj = min(ntraj)
        if npdo==2:
            print(f'Subsampling {mintraj} trajectories each for PAPA and DR.')
        else:
            print(f'Subsampling {mintraj} trajectories each for PAPA, DR, and other.')
        print()
        whichmin = ntraj.index(mintraj)
        for j in range(npdo):
            outfname = '%s/%s_%s.csv' % (subsampledFolder, c, PDO[j])
            if j==whichmin: # if this is the one with the fewest trajectories, no need to subsample
                traj[j].to_csv(outfname, index=False, columns=columns, header=True)
            else:
                trajind = list(traj[j]['trajectory'].unique())
                trajind = sample(trajind,mintraj)
                sel = traj[j]['trajectory'].isin(trajind)
                traj[j] = traj[j][sel]
                traj[j].to_csv(outfname, index=False, columns=columns, header=True)
            conditions.append('%s_%s' % (c,PDO[j]))
            filepaths.append(outfname)
    
    paths = pd.DataFrame({'condition':conditions, 'filepath':filepaths})
    
    SAD = StateArrayDataset.from_kwargs(paths, **settings)

    return [SAD,paths]
  

def analyze_PAPA_DR_stateArray(config,sortedFolder='sortedTrajectories',nworkers=1):
    """
    [SAD,posterior_occs,condition_names] = 
        analyze_PAPA_DR_stateArray(config,sortedFolder='sortedTrajectories',nworkers=1):
    
    Make StateArray for the entire dataset of sorted PAPA, DR, and other trajectories,
    and then infer posterior probability distribution by condition.
    
    Requires saspt (https://github.com/alecheckert/saspt)
    
    args
        config  :   dict - configuration settings
        sortedFolder    :   directory where sorted PAPA/DR/other trajectories are stored
        subsampledFolder    :   directory where subsampled trajectories will be stored
        nworkers    :   number of workers to use for the computation; do not set this to
                        a value higher than the number of CPUs on your computer.
        randseed    :   (optional) number to serve as a random seed to initialize the          
                        random number generator (for reproducing the same "random" output when
                        the code is re-run)
    
    returns
        [SAD,posterior_occs,condition_names]
        state array dataset, posterior occupations for each condition, names of each condition
    
    """
    
    import pickle
        
    [SAD,paths]=makeStateArrayDataset(config,sortedFolder=sortedFolder,nworkers=nworkers,isPAPA=True)
    print('Inferring posterior probability by condition.')
    posterior_occs, condition_names = SAD.infer_posterior_by_condition('condition')
    D = SAD.likelihood.diff_coefs
    plot_PAPA_DR(config,D,posterior_occs,condition_names,'figures')
    with open('state_array_pickle','wb') as fh:
        pickle.dump([SAD,posterior_occs,condition_names],fh)
    # to do: Write this all out to csv files as well
    return [SAD,posterior_occs,condition_names]


def analyze_PAPA_DR_stateArray_sameN(config,sortedFolder='sortedTrajectories',
                subsampledFolder='subsampledTrajectories',
                nworkers=1,randseed=None,ignoreOther = False):
    """
    [SAD,posterior_occs,condition_names] = 
        analyze_PAPA_DR_stateArray_sameN2(config,sortedFolder='sortedTrajectories',nworkers=1
        ignoreOther=False):
    
    Make StateArray for the entire dataset of sorted PAPA and DR trajectories,
    and then infer posterior probability distribution by condition.
    Within each condition, subsample the same number of PAPA and DR trajectories.
        
    Requires saspt (https://github.com/alecheckert/saspt)
    
    args
        config  :   dict - configuration settings
        sortedFolder    :   directory where sorted PAPA/DR trajectories are stored
        nworkers    :   number of workers to use for the computation; do not set this to
                        a value higher than the number of CPUs on your computer.
        ignoreOther : Do not include "other" trajectories in the subsampling/analysis [default: False]
    
    returns
        [SAD,posterior_occs,condition_names]
        state array dataset, posterior occupations for each condition, names of each condition
    
    """
    
    import pickle
    
    [SAD,paths]=makeSA_sorted_sameN(config,sortedFolder=sortedFolder,
           nworkers=nworkers,randseed=randseed,ignoreOther=ignoreOther)
    print('Inferring posterior probability by condition.')
    posterior_occs, condition_names = SAD.infer_posterior_by_condition('condition')
    D = SAD.likelihood.diff_coefs
    plot_PAPA_DR(config,D,posterior_occs,condition_names,'figures_sameN')
    with open('state_array_sameN_pickle','wb') as fh:
        pickle.dump([SAD,posterior_occs,condition_names],fh)
    # to do: Write this all out to csv files as well
    return [SAD,posterior_occs,condition_names]

def plot_PAPA_DR(config,D,posterior_occs,condition_names,figfname='figures'):
    """
    plot_PAPA_DR(config,D,posterior_occs,condition_names,figfname)
    
    args:
        config  :   dict of configuration settings
        D   :   array of diffusion coefficients
        posterior_occs  :   list of lists of posterior occupations
        condition_names :   list of associated condition names
        figfname    :   name of output folder for storing figures
    """
    
    # get posterior occupations for each condition
    # This either uses the pre-calculated values or runs the calculation if it is 
    # not yet calculated.
    
    if not os.path.isdir(figfname):
        os.mkdir(figfname)
    if not os.path.isdir(figfname + '/PAPA_vs_DR_Dspectra'):
        os.mkdir(figfname + '/PAPA_vs_DR_Dspectra')
    if not os.path.isdir(figfname + '/PAPA_vs_DR_Dspectra/csvs'):
        os.mkdir(figfname + '/PAPA_vs_DR_Dspectra/csvs')
        
    #posterior_occs, condition_names = SAD.infer_posterior_by_condition('condition')
    
    from saspt import normalize_2d
    from matplotlib import pyplot as plt
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)

    posterior_occs = normalize_2d(posterior_occs, axis=1)
    podict = {}
    for j in range(posterior_occs.shape[0]):
        currc = condition_names[j]
        podict[currc] = posterior_occs[j,:]
        pd.DataFrame({'D':D,'P':podict[currc]}).to_csv(
            figfname + '/PAPA_vs_DR_Dspectra/csvs/%s.csv' % currc)
    for c in config['conditions']:
        plt.figure(c)
        plt.title(config['conditions'][c]['title'])
        plt.plot(D,podict[c+'_PAPA'],'g-')
        plt.plot(D,podict[c+'_DR'],'-',color='#A000A0')
        plt.xscale('log')
        plt.xlabel('Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)')
        plt.ylabel('Mean posterior occupation')
        plt.savefig(figfname + '/PAPA_vs_DR_Dspectra/%s.png' % c,format='png',bbox_inches='tight')
        

# TODO: Add option to overlay plots for each individual experiment in the condition
def plot_Nlocs_wholemovie(config,locsbyframe,figfname='figures'):
    """
    plot_Nlocs_wholemovie(config)
    
    args:
        config  :   dict of configuration settings
        locsbyframe     :   dataframe containing localizations per frame for each file
                        (output of getLocsByFrameDF)
        figfname:   folder name for storing figures
        
    For each condition specified in config, makes a plot of average number of localizations 
    as a function of frame number.
    """
    from matplotlib import pyplot as plt
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)

    IS = config['illumination_sequence']
    ncycles = IS['ncycles']
    r = IS['r']
    v = IS['v']
    g = IS['g']
    fw = IS['framewindow']
    gfirst = IS['gfirst']
    cyclelen = 4*r + v + g
    nframes = ncycles*cyclelen
        
    for c in locsbyframe['condition'].unique():
    
        currlocs = locsbyframe[locsbyframe['condition']==c]
        
        meanlocs = currlocs['locsbyframe'].mean()
        
        os.makedirs(f'{figfname}/locsperframe/', exist_ok=True)
        np.savetxt(f'{figfname}/locsperframe/{c}.csv', meanlocs, delimiter=',')
                
        plt.figure(c)
        plt.title(config['conditions'][c]['title'])
        plt.plot(range(nframes),meanlocs)
        plt.xlabel('Frame number')
        plt.ylabel('Mean localization number')
        plt.savefig(figfname + '/locsperframe/%s.png' % c,format='png',bbox_inches='tight')
        
        
        

def plot_Nlocs_bycycle_colors(config,locsbyframe,figfname='figures',byexperiment=True):
    """
    plot_Nlocs_wholemovie(config,locsbyframe,figfname='figures')
    
    Plot number of localizations for each cycle of PAPA/DR excitation
    
    args:
        config          : configuration file giving parameters for PAPA experiment
        locsbyframe     :   dataframe containing localizations per frame for each file
                        (output of getLocsByFrameDF)
        figfname        :   folder name for storing figures
        byexperiment    : whether to also plot number of localizations by cycle for each
                            experiment within a condition
        
    For each condition specified in config, makes a plot of average number of localizations 
    as a function of frame number.
    
    This version color-codes green and violet the frames that are used to collect PAPA and DR trajectories.
    """
    from matplotlib import pyplot as plt
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)

    IS = config['illumination_sequence']
    ncycles = IS['ncycles']
    r = IS['r']
    v = IS['v']
    g = IS['g']
    fw = IS['framewindow']
    gfirst = IS['gfirst']
    cyclelen = 4*r + v + g
    nframes = ncycles*cyclelen
    
    # frame number ranges for plotting
    framenums1 = np.arange(0,r)
    framenums2 = np.arange(r+v+fw,3*r+v)
    framenums3 = np.arange(3*r+v+g+fw,cyclelen)
    
    if gfirst: # if a green pulse is first, then swap the green and violet indices
        framenums_green = np.arange(r+v,r+v+fw)
        framenums_violet = np.arange(3*r+v+g,3*r+v+g+fw)        
    else:
        framenums_violet = np.arange(r+v,r+v+fw)
        framenums_green = np.arange(3*r+v+g,3*r+v+g+fw)
    
    # formatstring = '%%s%%0%dd.csv' % config['file_format']['ndigits']
    
    # do modular division to give index of each frame within an illumination cycle
    cycle_index = np.arange(0,nframes) % cyclelen
    
    for c in locsbyframe['condition'].unique():
    
        currlocs = locsbyframe[locsbyframe['condition']==c]
        
        meanlocs = currlocs['locsbyframe'].mean()
        meanlocs_cycle = np.bincount(cycle_index,weights=meanlocs)/ncycles
        
        os.makedirs(f'{figfname}/locsperframe_cycle/', exist_ok=True)
        np.savetxt(f'{figfname}/locsperframe_cycle/{c}.csv', meanlocs_cycle, delimiter=',')
                
        plt.figure(c)
        plt.title(config['conditions'][c]['title'])

        plt.plot(framenums1,meanlocs[framenums1],'k-')
        plt.plot(framenums2,meanlocs[framenums2],'k-')
        plt.plot(framenums3,meanlocs[framenums3],'k-')
        plt.plot(framenums_violet,meanlocs[framenums_violet],'-',color='blueviolet')
        plt.plot(framenums_green,meanlocs[framenums_green],'-',color='green')
        
        if byexperiment:
            for e in currlocs['experiment'].unique():
                currlocs_exp = currlocs[currlocs['experiment']==e]
                meanlocs = currlocs_exp['locsbyframe'].mean()
                meanlocs_cycle = np.bincount(cycle_index,weights=meanlocs)/ncycles
                
                np.savetxt(f'{figfname}/locsperframe_cycle/{c}_{e}.csv', 
                            meanlocs_cycle, delimiter=',')
                
                plt.plot(framenums1,meanlocs[framenums1],'k-',linewidth=0.5)
                plt.plot(framenums2,meanlocs[framenums2],'k-',linewidth=0.5)
                plt.plot(framenums3,meanlocs[framenums3],'k-',linewidth=0.5)
                plt.plot(framenums_violet,meanlocs[framenums_violet],
                            '-',color='blueviolet',linewidth=0.5)
                plt.plot(framenums_green,meanlocs[framenums_green],
                            '-',color='green',linewidth=0.5)

        
        plt.xlabel('Frame number')
        plt.ylabel('Mean localization number')
        plt.savefig(f'{figfname}/locsperframe_cycle/{c}.png',format='png',bbox_inches='tight')
        

# TODO: update with new configuration file format with multiple experiments within each condition
def get_ND2_times(config):
        
    """
    List ND2 movie file names with timestamps (Julian Date Number or JDN)
    
    args
        config  :   dict
        
    returns
        dictionary containing file names and timestamps for each movie in each condition
    
    """
    
    rv = {}
    
    bytesfromend = int(1e6) # where to start reading in the metadata relative the end of the file
    
    # format string for movie file names
    formatstring = '%%s%%0%dd.%s' % (config['file_format']['ndigits'],config['file_format']['extension'])
    
    # Julian Date Number pattern to match in the metadata at the end of the ND2 file
    pattern = r"<ModifiedAtJDN runtype=\"double\" value=\"(.+?)\"/>" 
    
    for c in config['conditions']:
        rv[c] = []
        currc = config['conditions'][c]
        if 'basefname' not in currc.keys():
            raise Exception('You must include a basefname for every condition')
        else:
            fnum = get_condition_fnum(currc)
            for f in fnum:
                currfname = formatstring % (currc['basefname'],f)
                with open(currfname, "rb") as file:
                    # Read the bytes at the end of the file
                    file.seek(os.path.getsize(currfname)-bytesfromend)
                    bytes = file.read()
                    decoded = bytes.decode(errors='ignore')
                    numbers = re.findall(pattern, decoded)
                    #print(f"{currfname} {numbers}")
                    if not numbers:
                        rv[c].append([currfname,np.nan])   
                    else:
                        rv[c].append([currfname,float(numbers[0])])   
    return rv
    
def getalldisp(fname):
    """
getalldisp(fname)

returns a list of all single-molecule displacements from a csv file containing
single-molecule trajectories.

This function assumes that the x and y coordinates are in the first two columns
of the csv file.
    """
    t = np.genfromtxt(fname, skip_header=1, delimiter=',')
    traj = np.unique(t[:, 17])
    sz = 0
    for j in range(traj.size):
        tcount = np.sum(t[:, 17] == traj[j])
        if tcount > 1:
            sz = sz + tcount - 1
    rv = np.zeros(sz)
    counter = 0
    for j in range(traj.size):
        selector = (t[:, 17] == traj[j])
        if np.sum(selector) > 1:
            currtraj = t[selector, :2]
            rv[counter:(counter + currtraj.shape[0] - 1)] = \
                np.sqrt(np.power(currtraj[1:, 0] - currtraj[:-1, 0], 2) 
                        + np.power(currtraj[1:, 1] - currtraj[:-1, 1], 2))
            counter = counter + currtraj.shape[0] - 1
    return rv
    

# TODO: Loop over experiments within each condition
# Use new file name format
def getLocsByFrameDF(config,verbose=True,isPAPA=True,nframes=1000):
    """
getLocsByFrame(config,verbose=True,isPAPA=True,nframes=1000)

count localizations by frame index for each file in each condition

args
    config  :   dict
    verbose :   boolean; whether or not to print out detailed output [default: True]
    isPAPA  :   boolean; whether or not this is a PAPA experiment [default: True]
    nframes :   how many frames there are per movie if this isn't a PAPA experiment

"""
    if isPAPA:
        ncycles = config['illumination_sequence']['ncycles']
        nr = config['illumination_sequence']['r']
        ng = config['illumination_sequence']['g']
        nv = config['illumination_sequence']['v']
        gfirst = config['illumination_sequence']['gfirst']
        nframes = ncycles*(4*nr + ng + nv)

    #loc_counts = dict.fromkeys(config['conditions'].keys())
    
    conditions = []
    experiments = []
    fnames = []
    locsbyframe = []
    
    for c in config['conditions']:
        for e in config['conditions'][c]['experiments']:
        
            csv_folder = config['conditions'][c]['experiments'][e]['csv_folder']
            flist = glob(csv_folder + "/*.csv")
            if verbose:
                print(c,end=" ")
            allframenums = pd.Series()            
            countsbyfile = np.zeros([len(flist),nframes])
            
            for fname in flist:
                try:
                    currdata = pd.read_csv(fname)
                    currframes=currdata['frame']
                    vcs = currframes.value_counts(bins=range(-1,nframes)).sort_index().values
                    conditions.append(c)
                    experiments.append(e)
                    fnames.append(fname)
                    locsbyframe.append(vcs)
                except Exception as theproblem:
                    print(f'Encountered a problem with file {fname}.')
                    print(theproblem)
            if verbose:
                print()
    rv = pd.DataFrame({'condition':conditions,'experiment':experiments,
                        'fname':fnames,'locsbyframe':locsbyframe})
    return rv


def make_condition_template(path,tracked_subfolder="tracking"):
    """
make_condition_template(path,tracked_subfolder="tracking")
    
this function is useful for automatically generating the "conditions" portion of your settings file
it identifies folders in the specified path that contain tracked csv files and determines the minimum
and maximum file number

inputs:
  path - path to search for folders containing tracked data
  tracked_subfolder - subfolder name containing tracking data (default: "tracking")

output: template for "conditions" section of the settings file (string)
    """
    
    rv = ""
    
    folders = []

    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.name)

    pattern = re.compile(r'\d+')


    for f in folders:
        # Get a list of all the filenames in the directory
        if os.path.exists(f"{path}/{f}/tracking"):
            filenames = os.listdir(f"{path}/{f}/tracking/")
            filenames = [f for f in filenames if f[-4:]=='.csv']
            # Extract the file numbers from the filenames
            file_numbers = [int(pattern.search(filename).group()) 
                            for filename in filenames if pattern.search(filename)]

            # Find the minimum and maximum file numbers
            min_file_number = min(file_numbers)
            max_file_number = max(file_numbers)

            rv = rv + f"""
                [conditions.{f}]
                basefname = '{path}/{f}/tracking/'     # base file name of all movies in this condition
                first = {min_file_number}
                last = {max_file_number}
                title = '{f}' # title to use for plotting 
                """
    return rv
            

#TODO: Modify classify_and_write_csv to write out cell categories

def classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder,
                                    maskvarname = 'roimasks',
                                    verbose=True):
    """
classify_and_write_csv(maskmat,csvfolder,measfolder,outfolder,
                                maskvarname = 'roimasks',
                                verbose=True)
                                
classifies all of the trajectories in a csv file based on the ROI mask in 
the .mat file maskmat

inputs:
maskmat - .mat file that contains the segmented masks
csvfolder - folder that contains the tracked csvs
measfolder - folder that contains cell measurements and metadata for each FOV
outfolder - output folder for writing out classified csvs
maskvarname - name of variable within the .mat file that contains the 
relevant masks [default 'roimasks']
   (in earlier versions of cellpicker, this variable was called 'masks')

output:
returns a dataframe with cell measurements for each FOV
    """    
    # column number in which trajectory index is stored in tracking csv files
    TRAJECTORY_COLUMN = 17 
    
    os.makedirs(outfolder,exist_ok = True)
    masks = scipy.io.loadmat(maskmat)
    nrange = masks['range'][0]     # range of file numbers (upper and lower bounds inclusive)
    isfirst = True
    
    # variable specifying whether or not each FOV had selected cells or not
    FOV_is_selected = masks['classification'][0]
    
    for j in range(nrange[0],nrange[1]+1):

        # cell masks in current FOV
        currmask = masks[maskvarname][0][j-1] 
        # vector of categories in which each cell was classified by the user
        cell_categories = masks['categories'][0][j-1][0]

        # skip this FOV if the mask is empty or if the FOV is marked unselected
        if currmask.size==0 or FOV_is_selected[j-1]==0:
            continue

        # Display status in the terminal, overwriting previous output
        if verbose:
            print(f'Processing FOV {j}.', end="\r")
        
        try:
            fname = f"{csvfolder}/{j}.csv"
            
            # Loop over the file a first time to get maximum trajectory number in the file
            maxtraj = 0
            with open(fname) as fh:
                fh.readline()
                line = fh.readline()
                while line:
                    maxtraj = max(int(line.split(',')[TRAJECTORY_COLUMN]),maxtraj)
                    line = fh.readline()
                    
            # initialize an array of -1's with that many elements to contain the cell number 
            # for each trajectory (or NaN if the trajectory passes over multiple cells)
            trajcell = -np.ones(maxtraj+1) # array starting at 0 and ending at maxtraj, inclusive
            
            # loop over the csv file a second time, and determine in which cell mask each trajectory falls
            with open(fname) as fh:
                fh.readline()
                line = fh.readline()
                allcelln = set()
                while line:
                    linespl = line.split(',')
                    
                    # current trajectory number
                    trajn = int(linespl[TRAJECTORY_COLUMN])
                    
                    # current x and y coordinates
                    x = round(float(linespl[1]))
                    y = round(float(linespl[0]))
                    
                    # get cell number
                    # celln = 0 corresponds to background regions. Numbering of cells starts at 1
                    celln = currmask[y,x] 
                    
                    # add this cell index to the list of all cell indices
                    allcelln.add(celln)
                    
                    # if it has not yet been classified, classify it to the cell it is in
                    if trajcell[trajn] == -1: 
                        trajcell[trajn] = celln
                    # if it has previously been classified to another cell, set it to nan
                    elif trajcell[trajn] != celln:
                        trajcell[trajn] = np.nan
                    line = fh.readline()
            
            # loop over the file one last time and write out each line to a file for that cell
            with open(fname) as fh:
                header = fh.readline()

                # open output file handles and initialize each with a header row
                fh2 = {}
                for n in allcelln:
                    # category in which this cell was classified by the user
                    currcat = cell_categories[n-1]
                    # only generate an output file for this cell if it is selected 
                    # (i.e., if it is assigned a category not equal to zero)
                    # exclude trajectories in the background region (n = 0)
                    if currcat>0 and n>0:
                        os.makedirs(f"{outfolder}/{currcat}",exist_ok=True)
                        fh2[n] = open(f"{outfolder}/{currcat}/{j}_{n}.csv",'w')
                        fh2[n].write(header)

                line = fh.readline()
                while line:
                    linespl = line.split(',')
                    
                    # trajectory number of current localization
                    trajn = int(linespl[TRAJECTORY_COLUMN])
                    
                    # cell number of current trajectory
                    celln = trajcell[trajn]
                    
                    # only write out the current localization if it is part of a 
                    # trajectory within a cell that is selected
                    if not np.isnan(celln) and celln != 0:
                        if cell_categories[int(celln)-1] != 0:
                            celln = int(celln)
                            fh2[celln].write(line)
                    
                    line = fh.readline()
        
                # close all file handles
                for f in fh2.keys():
                    fh2[f].close()
            
            # read in measurements for all selected cells in this FOV, and append
            # to an output dataframe
            selected_cells = np.nonzero(cell_categories)[0]
            df = pd.read_csv(f"{measfolder}/{j}.csv")
            df.rename(columns={df.columns[0]: 'cellnum'}, inplace=True)
            df = df.iloc[selected_cells].copy()
            df['fovnum'] = j
            df['category'] = cell_categories[selected_cells]
            if isfirst:
                rv = df
                isfirst = False
            else:
                rv = rv.append(df,ignore_index=True)
            
        except Exception as e:
            print(f"Error with FOV {j}:", str(e))
            traceback.print_exc()
            print(f"Error occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}: {str(e)}")
    rv.to_csv(f"{outfolder}/measurements.csv")
    return rv

def locsPerFrame(foldername,nframes,ncategories=1):
    """
locsPerFrame(foldername,nframes,ncategories=1)

Tabulates localizations per frame for each CSV file in a folder, and returns 
this as a dataframe.

Inputs:
foldername - name of the folder containing sorted molecule tracking csv files. The function expects that these will be stored in subfolders--one for each user-classified category. In most cases, there will be only a single category (i.e., a single subfolder named "1")
nframes - number of frames per movie
ncategories - number of different categories in which cells have been classified (default: 1)

Output:
Pandas DataFrame containing two columns:
- file_name: name of each file
- locsperframe: numpy arrays with number of localizations per frame for each frame index
"""

    isfirst = True
    for catnum in range(1,ncategories+1):
        fnames = glob(f"{foldername}/{catnum}/*.csv")
        for fname in fnames:
            try:
                currdata = pd.read_csv(fname)
                currframes=currdata['frame']
                vcs = currframes.value_counts(bins=range(-1,nframes)).sort_index().values
                df = pd.DataFrame({'file_name':fname,'locsperframe':[vcs]})
                if isfirst:
                    rv = df
                    isfirst=False
                else:
                    rv = rv.append(df,ignore_index=True)
            except:
                print(f'Encountered a problem with file number {f}.')
    return rv      

def measurements_and_GV(foldername,nframes,config,measfile,outfbase=''):
    """
measurements_and_GV(foldername,nframes,config,measfile,outf='GV_by_cell.csv')

Inputs:
foldername - name of the folder containing molecule tracking csv files
nframes - number of frames per movie
config - configuration dictionary containing PAPA cycle parameters
measfile - CSV file containing pooled measurements for selected cells (output of classify_and_write_csv)
outf - base file name for output files

Outputs:
- Returns a DataFrame containing measurements of each selected cell together with localizations per frame (1D Numpy array), total green and violet localization counts, and green-to-violet ratio.
- Saves this DataFrame (without the localizations per frame column) to a CSV file called outfbase + "gv_by_cell.csv"
- Saves the complete localizations per frame data to another CSV file called outfbase + "locsperframe.csv". The first two columns are FOV number and cell number.
"""
    locs = locsPerFrame(foldername,nframes)
    
    # extract the FOV and cell numbers from the file names
    extracted_numbers = locs['file_name'].str.extract(r'.*/(\d+)_(\d+)\.csv$')
    locs['fovnum'] = extracted_numbers[0].astype('int')
    locs['cellnum'] = extracted_numbers[1].astype('int')
    
    # get green and violet frame indices
    frameindices = get_gv_framenum(config)
    gpre = frameindices['gpre']
    gpost = frameindices['gpost']
    vpre = frameindices['vpre']
    vpost = frameindices['vpost']
    
    # calculate green and violet localizations and gv ratio, and add these as new columns
    # green and violet localization counts are counts in framewindow after the photostimulation
    # pulse minus background counts before the photostimulation pulse
    locs['gcounts'] = locs['locsperframe'].apply(lambda x: x[gpost].sum()-x[gpre].sum())
    locs['vcounts'] = locs['locsperframe'].apply(lambda x: x[vpost].sum()-x[vpre].sum())
    locs['GV_ratio'] = locs['gcounts']/locs['vcounts']

    # import cell measurements as pandas dataframe
    measdf = pd.read_csv(measfile,index_col=0)

    # merge the two dataframes using FOV and cell number columns
    combined_df = pd.merge(measdf,locs,on=['cellnum','fovnum'])
    
    # write out one CSV file containing cell data and GV ratio
    df_for_output = combined_df.drop('locsperframe',axis=1)
    df_for_output.to_csv(outfbase + "gv_by_cell.csv")
    
    # write out another CSV file containing FOV number and cell number in the first two
    # columns and localizations per frame in the remaining columns
    fov_and_cell_numbers = combined_df[['fovnum', 'cellnum']].to_numpy().astype('int')
    locsperframe_as_array = np.vstack(combined_df['locsperframe'])
    combined_for_output = np.concatenate((fov_and_cell_numbers, locsperframe_as_array), axis=1)
    np.savetxt(outfbase + "locsperframe.csv",combined_for_output,delimiter=',')
    
    return combined_df





















