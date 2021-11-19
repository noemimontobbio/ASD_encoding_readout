
import torch
from torch.utils.data import Dataset

import numpy as np
import scipy
from scipy import io
import os
import copy


class EncReadDATA(Dataset):
    """ASD_encoding_readout dataset."""

    def __init__(self, execdata, iC, mode, datadir='.', kinfeat=range(15),
                 obsdata=None, iG=2, tasksubj=0, augment=0, transform=None, transform_on=True):
        """
        Args:
            execdata : DataFrame of kinematic data frome execution experiment
            iC : Condition index (1:ASD, 2:TD, 0:both)
            mode : 'enc' for intention encoding, 'read' for intention readout
            datadir : the directory containing the data
            kinfeat : indices of kinematic features to extract
            obsdata : DataFrame of observation data (only needed when mode=='read')
            iG : Group index (1:ASD, 2:TD)
            tasksubj : index of observer within selected group (from 0 to 34)
            augment : number of data replications for augmentation
            transform : optional transform to be applied on a sample.
            transform_on : allows to disable the transform 
                           also when transform is not None.
        """
        
        # Settings
        self.execdata = execdata
        self.iC = iC
        self.mode = mode
        self.datadir = datadir
        self.kinfeat = kinfeat
        self.obsdata = obsdata
        self.iG = iG
        self.tasksubj = tasksubj
        self.augment = augment
        self.transform = transform
        self.transform_on = transform_on
        # Names
        self.condnames = ['ASD','TD']
        self.intnames = ['Placing', 'Pouring']
        

        # Load kinematic data
        data = copy.deepcopy(execdata)
        
        # Extract indices for the selected mvt condition from the videos ordered "by encoding"
        # if mode=='read' or returnDuration:
        videonames = data['VIDEO_NAME'].to_numpy()
        if iC==1:
            condind = [ii for ii in range(len(videonames)) if 'C' not in videonames[ii]]
        elif iC==2:
            condind = [ii for ii in range(len(videonames)) if 'C' in videonames[ii]]
        else:
            condind = np.arange(len(videonames))
        cvideonames = list(videonames[condind])

        data = data.iloc[condind,:]

        # Extract kinematic variables and intention labels from dataframe
        kindata = data.iloc[:,3:].to_numpy()
        kindata = kindata.reshape((kindata.shape[0],-1,10))
        kindata = kindata[:, kinfeat, :]
        intentions = data['VIDEO_INTENTION'].to_numpy()
        subjtype = data['VIDEO_GROUP'].to_numpy()


        if mode=='read':
            taskdata = copy.deepcopy(obsdata)
            # Extract wanted observer group and movement group
            taskdata = taskdata.loc[taskdata.loc[:,'SUBJECT_GROUP']==iG,:] # observer group
            if iC!=0:
                taskdata = taskdata.loc[taskdata.loc[:,'VIDEO_GROUP']==iC,:] # movement group
            # Extract subject number and select wanted subject
            subjnumbers = taskdata['SUBJECT_ID'].unique() # subject IDs for the group
            self.subjn = subjnumbers[tasksubj] # keep track of subject ID
            subjind = taskdata.loc[:,'SUBJECT_ID']==self.subjn
            taskdata = taskdata.loc[subjind,:]
            # Extract subject answers
            stim = taskdata['VIDEO_INTENTION']
            answ = taskdata['SUBJECT_RESPONSE']

            # Match with kinematic data
            taskvideos = taskdata['VIDEO_NAME'] # ordering of videos for the current task subject
            videos, task_ind, kin_ind = np.intersect1d(taskvideos,
                            cvideonames, return_indices=True) # map subject video order to original order
            answ = np.array(answ)[task_ind] # match with original order
            kindata = kindata[kin_ind,:,:] # possibly remove invalid trials for current subject from kindata
            self.stim = np.array(stim)[task_ind] # match with original order
            self.kin_ind = kin_ind # keep track of invalid trials


        # Final data
        self.kindata = kindata
        if mode=='enc':
            self.target = intentions - 1
        elif mode=='read':
            self.target = answ - 1


        if augment:
            self.kindata = np.tile(self.kindata, (augment,1,1))
            self.target = np.tile(self.target, augment)


    def __len__(self):
        return len(self.target)


    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = torch.tensor(self.kindata[idx,:,:])
        
        target = self.target[idx].squeeze()
        

        if self.transform and self.transform_on:
            sample = self.transform(sample)

        outputs = sample, target

        return outputs
