
"""
Example usage of encoding and readout functions
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import matplotlib.pyplot as plt

from utils import driftmodel, compute_pvalues
from encoding import encoding
from readout import readout

####################################################################################################
# Settings

iC = 2 # movement group (1:ASD, 2:TD, 0:both)
iG = 2 # observer group (1:ASD, 2:TD)

kinfeat = range(15) # kinematic features to use
                    # among the following:
FeatNames = ['WV',  # 0 Wrist Velocity
             'GA',  # 1 Grip Aperture
             'WH',  # 2 Wrist Height
             'IX',  # 3 Index X-coord
             'IY',  # 4 Index Y-coord
             'IZ',  # 5 Index Z-coord
             'TX',  # 6 Thumb X-coord
             'TY',  # 7 Thumb Y-coord
             'TZ',  # 8 Thumb Z-coord
            'DPX',  # 9 Dorsum Plane X-coord
            'DPY',  # 10 Dorsum Plane Y-coord
            'DPZ',  # 11 Dorsum Plane Z-coord
            'FPX',  # 12 Finger Plane X-coord
            'FPY',  # 13 Finger Plane Y-coord
            'FPZ']  # 14 Finger Plane Z-coord

CondNames = ['BOTH','ASD','TD']
nsub = 35

####################################################################################################
# Data

# The data file is available as Supplemental Material of the paper
# "Intersecting intention encoding and readout in autism" (**doi**)
execdata = pd.read_excel('ASD_enc_read_DATA.xlsx', sheet_name='Execution')
obsdata = pd.read_excel('ASD_enc_read_DATA.xlsx', sheet_name='Observation')

####################################################################################################
# Encoding model

# Train encoding model once, visualize coefficients and loss function:
enc_outs0 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                    plots=True, verbose=2)

# Train encoding model on 1000 resamplings of the data and visualize estimated coefficients:
enc_outs1 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                    nresample=1000, plots=True, verbose=1)
encbetas = np.mean(enc_outs1['betas'], axis=0)
enc_idx = np.argsort(np.abs(encbetas))[::-1]
encnames = [FeatNames[i] for i in enc_idx]

# Evaluate encoding model with 5-fold cross validation repeated 50 times:
enc_outs2 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                    cv=True, verbose=2)

# Perform permutation test (200 perms) on cross-validated accuracy and plot null distribution:
enc_outs3 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                    cv=True, permtest=True, verbose=1)
permacc = enc_outs3['permacc']
plt.figure()
plt.hist(permacc[1:], color='lightgray')
plt.axvline(permacc[0], color='k', linestyle='--')
plt.xticks([np.mean(permacc[1:]), permacc[0]],
        ['%.2f\nnull mean' %np.mean(permacc[1:]),'%.2f\nmodel accuracy' %permacc[0]])
plt.title('%s encoding accuracy: null distribution' %CondNames[iC])

#######################################################################################################
# Readout model

# Train readout models of all observers on 200 resamplings of the data:
read_outs1 = readout(driftmodel, execdata, obsdata, iC=iC, iG=iG, kinfeat=kinfeat,
                    nresample=200, verbose=1)
# Visualize estimated readout coefficients (mean over observers of normalised readout vectors):
readbetas_all = read_outs1['betas'].mean(axis=1)
readbetas_all = readbetas_all/np.linalg.norm(readbetas_all,ord=1,axis=1,keepdims=True)
readbetas = np.abs(readbetas_all).mean(axis=0)
rbsorted = readbetas[enc_idx]
f = plt.figure()
plt.bar(np.arange(len(rbsorted)), rbsorted, color='lightgray')
plt.xticks(np.arange(len(encnames)), encnames, rotation='vertical')
plt.xlabel('kinematic features')
plt.ylabel('mean fraction of coefficients')
plt.title('%s watch %s' %(CondNames[iG],CondNames[iC]))


# Compute null distribution for readout weights:
read_outs2 = readout(driftmodel, execdata, obsdata, iC=iC, iG=iG, kinfeat=kinfeat,
                    permtest=True, verbose=1)
permbetas = read_outs2['permbetas']
# Compare null distribution with resampled weights estimated above:
permbetas[:,0,:] = readbetas # comment out this line to use single-fit weights as estimate
pvalues = []
for subj in range(nsub):
    pvalues.append(compute_pvalues(permbetas[subj,:,:]))
selected_features = np.sum(np.array(pvalues)<.05, axis=0)
sfsorted = selected_features[enc_idx]
plt.figure()
plt.bar(np.arange(len(sfsorted)), sfsorted, color='lightgray')
plt.xticks(np.arange(len(encnames)), encnames, rotation='vertical')
plt.xlabel('kinematic features')
plt.ylabel('number of subjects')
plt.title('%s watch %s: selected features' %(CondNames[iG],CondNames[iC]))

