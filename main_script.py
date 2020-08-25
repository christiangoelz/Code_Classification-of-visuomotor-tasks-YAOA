#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chrisitan Goelz <goelz@sportmed.upb.de>
# Description: Main Script for Classification DMD Aging Force Control Project
# Dependencies: see .yml 
'''
Main Script to generate results for paper: XYZ 
Step 1: Preprocessing 
Step 2: Dynamic Mode Decomposition 
Step 3: Classification 
Step 4 Save as dictionary 
'''

import glob
import pickle 
import numpy as np
import pandas as pd 
import random
import gc 

from DMD import DMD
from dmd_fbcsp import DMD_FBCSP 
from preprocessing import PREPRO



random.seed(777) 
np.random.seed(777)
dmd_all = [] 
data_all = []
parti = []

# Define where to find all data and store results 
output_dir = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP/'
wd = "/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Exp3/" # SET
EEGfiles = wd +  "EEG_Motorik_Pre_Post/Pre/"
Forcefiles = wd + "Verhalten_Motorik_Pre_Post/Pre/"
eeg_files = (glob.glob(EEGfiles +'/nj*pre.edf') + glob.glob(EEGfiles + '/Nj*pre.edf') +
             glob.glob(EEGfiles +'/na*pre.edf') + glob.glob(EEGfiles + '/NA*pre.edf'))


misc =list(range(32,47)); del misc[2] #number of miscellaneous channels [32,33,35,36,37,38,39,40,41,42,43,44,45,46,47]
exclude = pd.read_excel('Exclude.xlsx', index_col = 0)

for file in eeg_files:
    try:
        participant = file[-18:-14].lower()    
        bads = np.asarray(exclude.loc[participant,:].dropna()) #excluded based on behavioral data 
        
        ############################# preprocessing ####################################
        data = PREPRO(participant,
                      file = file,
                      trial_length = 5,
                      t_epoch = [1,4],
                      eog_channels = [34], 
                      misc_channels = misc, 
                      stim_channel = [42],
                      montage = 'biosemi32',
                      event_detection = True, 
                      ext_file_folder = Forcefiles, 
                      event_dict = {'Steady_right':1, 'Sinus_right':2, 
                                    'Steady_left':3, 'Sinus_left':4}, 
                      sr_new = 200, 
                      ref_new = ['EXG5','EXG6'], 
                      filter_freqs = [4,30], 
                      ICA = True,
                      Autoreject = True,
                      bads = bads).run()
        
        # get relevant information from prepro
        X = data.epochs.get_data() * (1e6) #scale to microvolt
        channels = data.epochs.info['ch_names'][:32]
        labels = data.epochs.events[:, -1]
        
        with open(output_dir + participant + '.pkl', 'wb') as handle:
            pickle.dump(data,handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        ######################### Dynamic Mode Decomposition############################ 
        dmd = DMD(X, labels, 
                  channels = channels,
                  dt = 1/200,
                  win_size = 100,
                  overlap = 50,
                  stacking_factor = 4,
                  truncation = True).DMD_win()

        with open(output_dir + participant + 'dmd.pkl', 'wb') as handle:
            pickle.dump(dmd,handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        ######################## classification #######################################
        
        #'Steady_right':1, 'Sinus_right':2, 'Steady_left':3, 'Sinus_left:4'
        All = DMD_FBCSP(dmd).classify()
        L_R =  DMD_FBCSP(dmd,merge_labels = [[1,2],[3,4]]).classify()
        SinSt = DMD_FBCSP(dmd,merge_labels = [[1,3],[2,4]]).classify()
        #2class
        # Left / Rigt 
        SinLR = DMD_FBCSP(dmd, select_labels = [2,4]).classify() 
        StLR = DMD_FBCSP(dmd, select_labels = [1,3]).classify()
        # Sinus / Steady 
        LStSin = DMD_FBCSP(dmd, select_labels = [3,4]).classify()
        RStSin = DMD_FBCSP(dmd, select_labels = [1,2]).classify()
        results = {'All': All,
                  'L_R':L_R,
                  'SinSt':SinSt,
                  'SinLR':SinLR,
                  'StLR':StLR,
                  'LStSin':LStSin,
                  'RStSin':RStSin}
       
        ######################## save reslults #######################################
        with open(output_dir + participant + '_Classification.pkl', 'wb') as handle:
            pickle.dump(results,handle, protocol=pickle.HIGHEST_PROTOCOL)
        del results, All, L_R, SinSt, SinLR, StLR, LStSin, RStSin
        gc.collect() # free memory from garbage             
    except:
        parti.append(participant)
        gc.collect() # free memory from garbage 
        pass

        
 