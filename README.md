# ASD kinematic encoding and readout of intention

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

This repository contains Python code to train and evaluate the 
*kinematic encoding* and *kinematic readout* models 
introduced in [[1]](#1).
It is built to work with the .xslx data file available as 
Supplemental Material of the paper.

## Example usage 

### Setup

Import libraries and functions.
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import driftmodel, compute_pvalues
from encoding import encoding
from readout import readout
```

Select observed action (`iC`), observer group (`iG`) and desired subset of 
kinematic features (`kinfeat`).
```python
iC = 2 # movement group (1:ASD, 2:TD, 0:both)
iG = 2 # observer group (1:ASD, 2:TD)

kinfeat = range(15) # kinematic features to use
                    # among the following:
FeatNames = ['WV',  #  0 Wrist Velocity
             'GA',  #  1 Grip Aperture
             'WH',  #  2 Wrist Height
             'IX',  #  3 Index X-coord
             'IY',  #  4 Index Y-coord
             'IZ',  #  5 Index Z-coord
             'TX',  #  6 Thumb X-coord
             'TY',  #  7 Thumb Y-coord
             'TZ',  #  8 Thumb Z-coord
            'DPX',  #  9 Dorsum Plane X-coord
            'DPY',  # 10 Dorsum Plane Y-coord
            'DPZ',  # 11 Dorsum Plane Z-coord
            'FPX',  # 12 Finger Plane X-coord
            'FPY',  # 13 Finger Plane Y-coord
            'FPZ']  # 14 Finger Plane Z-coord

CondNames = ['BOTH','ASD','TD']
nsub = 35
```
Load data (the data file is available as Supplemental Material of [[1]](#1) ).
```python
execdata = pd.read_excel('ASD_enc_read_DATA.xlsx', sheet_name='Execution')
obsdata = pd.read_excel('ASD_enc_read_DATA.xlsx', sheet_name='Observation')
```

<br />

### Kinematic encoding examples

- Train kinematic encoding model once, on kinematic data (`execdata`) 
of selected group (`iC`). 
Visualize coefficients and loss function plot (`plots=True`).

    ```python
    enc_outs0 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                    plots=True, verbose=2)
    ```
  
<br />
    
- Evaluate kinematic encoding model using 5-fold cross validation 
(`cv=True`; `kcv` defaults to 5) 
repeated 50 times (`ncv`, defaults to 50). 
Display detailed info (`verbose=2`).
    ```python
    enc_outs1 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                        cv=True, verbose=2)
    ```
  Example output:
    ```
    ENCODING MODEL ON TD DATASET
  
  
    --------- FOLD #1 ---------
  
    Fold #1, Epoch 100
    train accuracy: 0.912
  
    Finished training for fold #1: best train accuracy = 0.963 at epoch 71
  
    Evaluation on out-of-sample elements: 0.900
    Mean CV perf until now = 0.900
  
  
    --------- FOLD #2 ---------
  
    Fold #2, Epoch 100
    train accuracy: 0.931
  
    Fold #2, Epoch 200
    train accuracy: 0.912
  
    Finished training for fold #2: best train accuracy = 0.969 at epoch 128
  
    Evaluation on out-of-sample elements: 1.000
    Mean CV perf until now = 0.950
    
    
    
    .........
    
    
    
    Encoding model on TD DATASET: : mean accuracy over CV folds = 96.1%
  
    NOTE : trained with data augmentation
    ```
  
<br />

- Perform permutation test on cross-validated accuracy 
(`permtest=True`, `cv=True`). 
The number of permutations (`n_perms`) defaults to 200.
    ```python
    enc_outs2 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                        cv=True, permtest=True, verbose=1)
    ```
  Extract cross-validated accuracy on permuted data, compute *p* value 
  and plot null-hypothesis distribution:
    ```python
    permacc = enc_outs2['permacc']
    print('%s encoding accuracy: p = %.3f' 
    %(CondNames[iC], compute_pvalues(permacc)))
    plt.figure()
    plt.hist(permacc[1:], color='lightgray')
    plt.axvline(permacc[0], color='k', linestyle='--')
    plt.xticks([np.mean(permacc[1:]), permacc[0]],
            ['%.2f\nnull mean' %np.mean(permacc[1:]),'%.2f\nmodel accuracy' %permacc[0]])
    plt.title('%s encoding accuracy: null distribution' %CondNames[iC])
    ```

<br />

- Train kinematic encoding model on 1000 resamplings of the data (`nresample=1000`). 
  Visualize bootstrap estimates (mean±SEM) of the coefficients (`plots=True`).
    ```python
    enc_outs3 = encoding(driftmodel, execdata, iC=iC, kinfeat=kinfeat,
                        nresample=1000, plots=True, verbose=1)
    ```
  <a id="encidx">Extract coefficient estimates and sort the kinematic 
  features by relevance:</a>
    ```python
    encbetas = np.mean(enc_outs3['betas'], axis=0)
    enc_idx = np.argsort(np.abs(encbetas))[::-1]
    encnames = [FeatNames[i] for i in enc_idx]
    ```

<br />

### Kinematic readout examples

- Train kinematic readout models of all observers of the selected group (`iG`) 
 watching selected actions (`iC`), on 200 resamplings of the data 
 (kinematics, `execdata`, and observer choices, `obsdata`):
    ```python
    read_outs1 = readout(driftmodel, execdata, obsdata, iC=iC, iG=iG, kinfeat=kinfeat,
                        nresample=200, verbose=1)
    ```
  Visualize estimated readout model coefficients 
  (mean over observers of normalized kinematic readout vectors), ordered by 
  [relevance of kinematic features in encoding](#encidx) (`enc_idx`):
    ```python
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
    ```

<br />

- Perform permutation test (`permtest=True`) on readout model coefficients 
(note: `cv` defaults to `False`, so the weights are estimated on *all* data at 
each permutation). 
The number of permutations (`n_perms`) defaults to 200.
    ```python
    read_outs2 = readout(driftmodel, execdata, obsdata, iC=iC, iG=iG, kinfeat=kinfeat,
                        permtest=True, verbose=1)
    ```
  Extract permuted weights and compare null distribution with resampled 
  weights estimated above. Visualize significant coefficients.
    ```python
    permbetas = read_outs2['permbetas']
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
    ```
  
<br />
    
- Evaluate kinematic readout models of all observers with 5-fold cross validation 
(`cv=True`; `kcv` defaults to 5) 
repeated 50 times (`ncv`, defaults to 50). 
Display detailed info (`verbose=2`).
    ```python
    read_outs3 = readout(driftmodel, execdata, obsdata, iC=iC, iG=iG, kinfeat=kinfeat,
                        cv=True, verbose=2)
    ```
  Example output:
    ```
    READOUT MODEL ON TD KINEMATICS, TD GROUP
  
  
    Loading TDwatchTD data...
    Done!
  
  
    TD kinematics, TD subject 1/35 (S001C)
  
  
    --------- FOLD #1 ---------
  
    Fold #1, Epoch 100
    train accuracy: 0.781
  
    Fold #1, Epoch 200
    train accuracy: 0.806
  
    Fold #1, Epoch 300
    train accuracy: 0.794
  
    Finished training for fold #1: best train accuracy = 0.844 at epoch 215
  
    Evaluation on out-of-sample elements: 0.700
    Mean perf until now = 0.700
    
    
    
    .........
    
    
    
    Readout model on TD kinematics, TD group 
  
    TD task subject S001C : mean accuracy over CV folds = 72.0%
    .........
    TD task subject S035C : mean accuracy over CV folds = 59.1%
  
    NOTE : trained with data augmentation
    ```

<br />

All code in this section can be found in `example_main.py`.

<br />

## References
<a id="1">[1]</a> 
Montobbio, N., Cavallo, A., Albergo, D., Ansuini, C., 
Battaglia, F., Podda, J., Nobili, L., Panzeri, S., 
Becchio, C. (2022).
_Intersecting kinematic encoding and readout of intention in autism_.
Proceedings of the National Academy of Sciences, 119(5). 
doi: [10.1073/pnas.2114648119](https://www.pnas.org/content/119/5/e2114648119)

