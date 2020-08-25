#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:33:01 2020

@author: christiangolz
"""


import pickle 
import glob 
import pandas as pd

wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Preprocessing/preprocessed_python'
files = glob.glob(wd + '/n*.pkl')
bad_epochs = [] 
bad_ics = []
parts = []  
l = len(wd)+1
ll = len(wd)+5
for file in files: 
   parts.append(file[l:ll])
   with open(file, 'rb') as handle:
        data = pickle.load(handle)
        bad_epochs.append(data.bads[0])
        bad_ics.append(data.ica.exclude)        
    
bad_epochs = pd.DataFrame(data = bad_epochs, index = parts)
bad_epochs.to_excel('/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Preprocessing/preprocessed_python/bad_epochs.xlsx')

StR = bad_epochs[bad_epochs <=40].count(axis=1)
SinR = bad_epochs[(bad_epochs >40) & (bad_epochs <=80)].count(axis=1)
StL = bad_epochs[(bad_epochs >80) & (bad_epochs <=120)].count(axis=1)
SinL = bad_epochs[(bad_epochs >120) & (bad_epochs <=160)].count(axis=1)

mean_bad_epochs = pd.concat([StR, SinR, StL, SinL], axis = 1)
mean_bad_epochs['Mean'] = mean_bad_epochs.mean(axis = 1)
mean_bad_epochs['SD'] = mean_bad_epochs.std(axis = 1)
mean_bad_epochs.columns = ['StR', 'SinR', 'StL', 'SinL','Mean','SD']
mean_bad_epochs.to_excel('/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Preprocessing/preprocessed_python/bad_mean_epochs.xlsx')

bad_ics = pd.DataFrame(data = bad_ics, index = parts)
bad_ics.to_excel('/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Preprocessing/preprocessed_python/bad_ics.xlsx')