#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:19:29 2020

@author: christiangolz
"""

import glob
import pickle
import numpy as np
from scipy import signal
import pandas as pd
from scipy import stats
import seaborn as sns 

results_path = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/Force' # where to put te reults 
force_path = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Exp3/Verhalten_Motorik_Pre_Post/Pre/' #location of force files 
files = glob.glob(force_path + 'na*') + glob.glob(force_path + 'nj*')
n_trials = 160 
fs = 120  #sampling rate 
length = 120 * 3 #3sec
b, a = signal.butter(4, 30/(fs/2), 'low')
diff_signal ={}
columns_results = ['mAccuracy_StR', 'mAccuracy_SinR', 'mAccuracy_StL', 'mAccuracy_SinL',  
                   'mVariability_StR', 'mVariability_SinR', 'mVariability_StL', 'mVariability_SinL',
                   'mTWR_StR', 'mTWR_SinR', 'mTWR_StL', 'mTWR_SinL'] 
part_all = []
mean_all_trial = [] 
Accuracy = [] 
Variability = [] 
exclude = []
for parti in files :
    diff = []
    wr_vec = []
    label = []
    ex = [] 
    part_all.append(parti[-4:])
    
    for trial in range(1,n_trials+1):
        # Import and filtering 
        this_part_files = glob.glob(parti + '/*r' + str(trial) + 'axis0.dat')
        data = pd.read_csv(this_part_files[0], skiprows = 2)
        raw = data['Pinch 1'].values[120:-120]
        target = data['MVC 1'].values[120:-120] 
        filt = signal.filtfilt(b, a, raw)
        fivep_range = target*0.05
        
        d = abs(target - filt) # absolut deviation from target 
        wr_vec.append((d-fivep_range) < 0)
        diff.append(abs(target - filt))  
        
        if trial <= 40: 
            label.append(['Steady_r'])
        elif trial > 40 and trial <= 80:
            label.append(['Sinus_r'])
        elif trial > 80 and trial <= 120 :
            label.append(['Steady_l'])
        else: 
            label.append(['Sinus_l'])
      
    diff = np.asarray(diff)
    TWR = [len(np.asarray(np.where(vec == True)).T)/120 for vec in wr_vec]
    label = np.ravel(label)
    df = pd.DataFrame(data = diff)
    df['label'] = label
    df['Accuracy'] = np.mean(diff, axis = 1)
    df['Variability'] = np.std(diff, axis = 1)
    df['TWR'] = TWR
    diff_signal[parti[-4:]] = df
    
    # #zscore outlier detection 
    zAll = df.groupby(['label'], sort = False).Accuracy.apply(stats.zscore)
    zAll = np.concatenate(zAll.to_list())
    
    #outlier detection and documentation
    ex = np.where(np.abs(zAll > 3))                                             
    df.drop(ex[0])
    exclude.append(np.array(ex[0]))
    
    # stats and add to results 
    df = df[['Accuracy', 'Variability', 'TWR', 'label']]
    this_result = df.groupby(['label'], sort = False).mean()
    mean_all_trial.append(this_result)
    
########## save results ######################################################  
exclude = pd.DataFrame(data = exclude, index = part_all)
mean_all = pd.concat(mean_all_trial, keys = part_all, names = ['Part', 'label']) 
exclude.to_excel('Exclude.xlsx')
# mean_all.to_excel(results_path + '/Force_results.xlsx')  
# with open(results_path + '/Diff_signal.pkl', 'wb') as handle:
#     pickle.dump(diff_signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
################## plotting of the results ###################################
mean_all.reset_index(inplace = True)
mean_all['Group'] = 52*['LMA'] + 52*['Young']
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})
with sns.axes_style("ticks"):
    g = sns.catplot(y = 'Accuracy', x = 'label', hue = 'Group', data = mean_all, kind = 'violin' ,palette = ['orange','dimgray'])
    (g.set_axis_labels('', "Mean deviation from target")
      .set_xticklabels(["Steady right", "Sinus right", "Steady left", "Sinus left"])
      .set_titles("{col_name} {col_var}"))  
# '