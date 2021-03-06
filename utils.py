"""
Models and transforms.

"""


import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate


#####################################################################################

class driftmodel(nn.Module):
    """
    Logistic model with drift
    """
    def __init__(self, input_size=15, pdrop=0):
        """
        input_size : number of features
        pdrop : dropout probability
        """
        super(driftmodel, self).__init__()
        # Settings
        self.input_size = input_size
        self.pdrop = pdrop
        # Layers
        self.fc = nn.Linear(input_size,1)
        self.rfc = nn.Linear(1,1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=pdrop)
        self.relu = nn.ReLU()

    def forward(self, t, x, y):
        """
        t : time step
        x : input at time t
        y : output at time t-1
        """
        act = self.rfc(y-.5) + self.fc(self.drop(x))
        y = self.sigmoid(act)
        return y


#####################################################################################

class TimeWarping(object):
    """
    Warping transform over the time variable
    """
    def __init__(self, deltarange, randomdelta=True):
        """
        deltarange: exponent for time warping via t_new = T*(t_old/T)**delta
        randomdelta: if True, randomly choose exponent in [1/deltarange, deltarange]
        """
        super(TimeWarping, self).__init__()
        self.deltarange = deltarange
        self.randomdelta = randomdelta

    def __call__(self, x):
        if self.deltarange!=1:
            if self.deltarange < 1:
                self.deltarange = 1/self.deltarange
            delta = (self.deltarange - 1/self.deltarange)*np.random.random_sample() \
                    + 1/self.deltarange if self.randomdelta else self.deltarange
            timebins = x.shape[-1] # tensor is (features x timebins)
            newtimes = (timebins-1)*(np.arange(timebins)/(timebins-1))**delta
            f = interpolate.interp1d(np.arange(timebins), x)
            x = torch.tensor(f(newtimes))

        return x

#####################################################################################

def compute_pvalues(seq, dmean=None, optsided='two', direction=None):
    """
    seq = sequence of outputs (n_perms+1 x n_out)
          seq[0,:] = original values
          seq[1:,:] = values from permuted input
    dmean = distribution mean
    optsided = 'one' for one-sided, 'two' for two-sided
    direction = 1 or -1, for one-sided case (inferred from data if not specified)
    """

    seq = np.array(seq)
    if len(seq.shape)==1:
        seq = np.expand_dims(seq,1)
    if dmean==None:
        dmean = np.mean(seq[1:,:], axis=0)
    seq -= dmean
    nPerms = seq.shape[0] - 1
    nOut = seq.shape[1]
    if optsided=='one':
        pvalues = np.zeros(nOut)
        for iOut in range(nOut):
            one_sided_direction = np.sign(seq[0,iOut]) if direction==None else direction
            if one_sided_direction > 0:
                pvalues[iOut] = np.sum(seq[1:,iOut] >= seq[0,iOut])/nPerms
            else:
                pvalues[iOut] = np.sum(seq[1:,iOut] <= seq[0,iOut])/nPerms
    elif optsided=='two':
        pvalues = np.sum(np.abs(seq[1:,:]) >= np.abs(seq[0,:]), 0)/nPerms
    else:
        print('Invalid optsided value (must be either \'one\' or \'two\')')

    return pvalues
