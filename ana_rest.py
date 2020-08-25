#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chrisitan Goelz <goelz@sportmed.upb.de>
# Description: Script for Classification DMD Aging Force Control Project - Additional Analysis of Rest data
# Dependencies: see .yml 
'''
Main Script to generate results for paper: XYZ 
Step 1: Preprocessing 
Step 2: Dynamic Mode Decomposition 
'''

import glob
import pickle 
import numpy as np
import random
from DMD import DMD
from preprocessing import PREPRO

random.seed(777) 
np.random.seed(777)
dmd_all = [] 
data_all = []
parti = []

# Define where to find all data and store results 
output_dir = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/'
wd = "/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Exp3/" # SET
EEGfiles = wd +  "EEG_Ruhe/Pre"
misc =list(range(32,47)); del misc[2] #number of miscellaneous channels [32,33,35,36,37,38,39,40,41,42,43,44,45,46,47]
Forcefiles = wd + "Verhalten_Motorik_Pre_Post/Pre/"
eeg_files = (glob.glob(EEGfiles +'/nj*pre.edf') + glob.glob(EEGfiles + '/Nj*pre.edf') +
             glob.glob(EEGfiles +'/na*pre.edf') + glob.glob(EEGfiles + '/NA*pre.edf'))
for file in eeg_files:
    participant = file[-17:-13].lower()
    data = PREPRO(participant,
                  file = file,
                  trial_length = 30,
                  t_epoch = [10,20],
                  eog_channels = [34], 
                  misc_channels = misc, 
                  stim_channel = [42],
                  montage = 'biosemi32',
                  event_detection = False,
                  marker_detection = True, 
                  event_dict = {'Rest:EC':1, 'Rest:EO':2}, 
                  sr_new = 200, 
                  ref_new = ['EXG5','EXG6'], 
                  filter_freqs = [4,30], 
                  ICA = True,
                  Autoreject = False).run()

    X = data.epochs.get_data() * (1e6) #scale to microvolt
    channels = data.epochs.info['ch_names'][:32]
    labels = data.epochs.events[:, -1]
    
    with open(output_dir + participant + '_cleanRest.pkl', 'wb') as handle:
            pickle.dump(data,handle, protocol=pickle.HIGHEST_PROTOCOL) 
    
    ######################### Dynamic Mode Decomposition############################ 
    dmd = DMD(X, labels, 
          channels = channels,
          dt = 1/200,
          win_size = 100,
          overlap = 50,
          stacking_factor = 4,
          truncation = True).DMD_win()
    
    with open(output_dir + participant + 'Rest_dmd.pkl', 'wb') as handle:
            pickle.dump(dmd,handle, protocol=pickle.HIGHEST_PROTOCOL)
