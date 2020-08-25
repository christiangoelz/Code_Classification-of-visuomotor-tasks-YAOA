#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:02:17 2020

@author: christiangolz
"""

import glob
import pickle 
import seaborn as sns
import numpy as np


part = [] 
confmat = [] 
wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP/'
files = glob.glob(wd + '/na*Classification_All.pkl')
l = len(wd) + 1
ll = l + 4
for file in files: 
    with open(file, 'rb') as handle:
        participant = file[l:ll]
        part.append(file[l:ll])
        data = pickle.load(handle)
        confmat.append(np.mean(data['All'].conf_mat, axis = 0))
        
m = np.mean(confmat,axis = 0)

sns.heatmap(m, annot=True, vmin = 0, vmax =1, yticklabels=False, xticklabels=False, cmap ='Greys')


files = glob.glob(wd + '/nj*Classification_All.pkl')
l = len(wd) + 1
ll = l + 4
for file in files: 
    with open(file, 'rb') as handle:
        participant = file[l:ll]
        part.append(file[l:ll])
        data = pickle.load(handle)
        confmat.append(np.mean(data['All'].conf_mat, axis = 0))
        
m = np.mean(confmat,axis = 0)

sns.heatmap(m, annot=True, vmin = 0, vmax =1, yticklabels=False, xticklabels=False, cmap ='Oranges')