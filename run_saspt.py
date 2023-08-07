import os, numpy as np, pandas as pd
from glob import glob
from saspt import StateArray, StateArrayDataset, RBME, load_detections
import re
from matplotlib import pyplot as plt

# read in all files from the dataset, and perform a StateArray analysis on the combined data.

# This assumes that you have classified cells into just one category. The code will need to be modified for additional categories.
input_files = glob('cell_by_cell_csvs_all/1/*csv')
detections = load_detections(*input_files)
settings = dict(
            likelihood_type = RBME,
                pixel_size_um = 0.16,
                    frame_interval = 0.00748,
                        focal_depth = 0.7,
                            start_frame = 0,
                                progress_bar = True,
                                    sample_size = 1e6,
                                    )
SA = StateArray.from_detections(detections, **settings)
print(SA)
print("Trajectory statistics:")
for k, v in SA.trajectories.processed_track_statistics.items():
        print(f"{k : <20} : {v}")


# make some output plots, and write the overall posterior occupations to a CSV file

SA.occupations_dataframe.to_csv("posterior_occupations.csv", index=False)
SA.plot_occupations("posterior_occupations.png")
SA.plot_assignment_probabilities("assignment_probabilities.png")
SA.plot_temporal_assignment_probabilities("assignment_probabilities_by_frame.png")


po = pd.read_csv('posterior_occupations.csv')

grouped = po.groupby('diff_coef',as_index=False).sum()
D = grouped['diff_coef'].to_numpy()
mpo = grouped['mean_posterior_occupation']
plt.semilogx(D,mpo)
plt.xlabel('Diffusion coefficient (µm$^2$/s)')
plt.ylabel('Fraction of molecules');
plt.savefig('mpo.png',dpi=300,bbox_inches='tight')
plt.savefig('mpo.pdf',bbox_inches='tight')

# write out StateArray as a pickled file

import pickle
with open('SA_pickle','wb') as fh:
        pickle.dump(SA,fh)


