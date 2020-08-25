#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chrisitan Goelz <goelz@sportmed.upb.de>
# Description: Script for results visualization of C3C4 and PzFz indices 
# Dependencies: see .yml 

import mne 
import pickle 
import pandas as pd
import glob
import numpy as np
import seaborn as sns

output_dir = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/'
wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/DMD_ana/BHS/python/Task'
m = mne.channels.make_standard_montage('biosemi32') 
info = mne.create_info(
        ch_names=m.ch_names, sfreq=200., ch_types='eeg')
info.set_montage(m)

yStr_theta= [] 
yStr_alpha= [] 
yStr_beta1= [] 
yStr_beta2= [] 

ySinL_theta= [] 
ySinL_alpha= [] 
ySinL_beta1= [] 
ySinL_beta2= [] 

ySinR_theta= [] 
ySinR_alpha= [] 
ySinR_beta1= [] 
ySinR_beta2= [] 

yStL_theta= [] 
yStL_alpha= [] 
yStL_beta1= [] 
yStL_beta2= [] 

y_parts = [] 
l = len(wd)+1
ll = len(wd)+5
files = glob.glob(wd + '/nj*.pkl')

for file in files: 
   y_parts.append(file[l:ll])
   with open(file, 'rb') as handle:
        dmd = pickle.load(handle)
        yStr_theta.append(dmd.get_PSI(fband = [4,8], labels = [1]).abs().mean())
        yStr_alpha.append(dmd.get_PSI(fband = [8,12], labels = [1]).abs().mean())
        yStr_beta1.append(dmd.get_PSI(fband = [12,16], labels = [1]).abs().mean())
        yStr_beta2.append(dmd.get_PSI(fband = [16,30], labels = [1]).abs().mean())
        
        ySinL_theta.append(dmd.get_PSI(fband = [4,8], labels = [4]).abs().mean())
        ySinL_alpha.append(dmd.get_PSI(fband = [8,12], labels = [4]).abs().mean())
        ySinL_beta1.append(dmd.get_PSI(fband = [12,16], labels = [4]).abs().mean())
        ySinL_beta2.append(dmd.get_PSI(fband = [16,30], labels = [4]).abs().mean())   
        
        ySinR_theta.append(dmd.get_PSI(fband = [4,8], labels = [2]).abs().mean())
        ySinR_alpha.append(dmd.get_PSI(fband = [8,12], labels = [2]).abs().mean())
        ySinR_beta1.append(dmd.get_PSI(fband = [12,16], labels = [2]).abs().mean())
        ySinR_beta2.append(dmd.get_PSI(fband = [16,30], labels = [2]).abs().mean())
        
        yStL_theta.append(dmd.get_PSI(fband = [4,8], labels = [3]).abs().mean())
        yStL_alpha.append(dmd.get_PSI(fband = [8,12], labels = [3]).abs().mean())
        yStL_beta1.append(dmd.get_PSI(fband = [12,16], labels = [3]).abs().mean())
        yStL_beta2.append(dmd.get_PSI(fband = [16,30], labels = [3]).abs().mean())  
          
oStr_theta= [] 
oStr_alpha= [] 
oStr_beta1= [] 
oStr_beta2= [] 

oSinL_theta= [] 
oSinL_alpha= [] 
oSinL_beta1= [] 
oSinL_beta2= [] 

oSinR_theta= [] 
oSinR_alpha= [] 
oSinR_beta1= [] 
oSinR_beta2= [] 

oStL_theta= [] 
oStL_alpha= [] 
oStL_beta1= [] 
oStL_beta2= []       
o_parts = []     
files = glob.glob(wd + '/na*.pkl')       
for file in files: 
    o_parts.append(file[l:ll])
    with open(file, 'rb') as handle:
        dmd = pickle.load(handle)
        oStr_theta.append(dmd.get_PSI(fband = [4,8], labels = [1]).abs().mean())
        oStr_alpha.append(dmd.get_PSI(fband = [8,12], labels = [1]).abs().mean())
        oStr_beta1.append(dmd.get_PSI(fband = [12,16], labels = [1]).abs().mean())
        oStr_beta2.append(dmd.get_PSI(fband = [16,30], labels = [1]).abs().mean())
        
        oSinL_theta.append(dmd.get_PSI(fband = [4,8], labels = [4]).abs().mean())
        oSinL_alpha.append(dmd.get_PSI(fband = [8,12], labels = [4]).abs().mean())
        oSinL_beta1.append(dmd.get_PSI(fband = [12,16], labels = [4]).abs().mean())
        oSinL_beta2.append(dmd.get_PSI(fband = [16,30], labels = [4]).abs().mean())   
        
        oSinR_theta.append(dmd.get_PSI(fband = [4,8], labels = [2]).abs().mean())
        oSinR_alpha.append(dmd.get_PSI(fband = [8,12], labels = [2]).abs().mean())
        oSinR_beta1.append(dmd.get_PSI(fband = [12,16], labels = [2]).abs().mean())
        oSinR_beta2.append(dmd.get_PSI(fband = [16,30], labels = [2]).abs().mean())
        
        oStL_theta.append(dmd.get_PSI(fband = [4,8], labels = [3]).abs().mean())
        oStL_alpha.append(dmd.get_PSI(fband = [8,12], labels = [3]).abs().mean())
        oStL_beta1.append(dmd.get_PSI(fband = [12,16], labels = [3]).abs().mean())
        oStL_beta2.append(dmd.get_PSI(fband = [16,30], labels = [3]).abs().mean())  
        
        
yStr_theta= (pd.concat(yStr_theta, axis = 1)).iloc[:-4].T
yStr_alpha= (pd.concat(yStr_alpha, axis = 1)).iloc[:-4].T
yStr_beta1= (pd.concat(yStr_beta1, axis = 1)).iloc[:-4].T
yStr_beta2= (pd.concat(yStr_beta2, axis = 1)).iloc[:-4].T

yStL_theta= (pd.concat(yStL_theta, axis = 1)).iloc[:-4].T
yStL_alpha= (pd.concat(yStL_alpha, axis = 1)).iloc[:-4].T
yStL_beta1= (pd.concat(yStL_beta1, axis = 1)).iloc[:-4].T
yStL_beta2= (pd.concat(yStL_beta2, axis = 1)).iloc[:-4].T

ySinL_theta= (pd.concat(ySinL_theta, axis = 1)).iloc[:-4].T
ySinL_alpha= (pd.concat(ySinL_alpha, axis = 1)).iloc[:-4].T
ySinL_beta1= (pd.concat(ySinL_beta1, axis = 1)).iloc[:-4].T
ySinL_beta2= (pd.concat(ySinL_beta2, axis = 1)).iloc[:-4].T

ySinR_theta= (pd.concat(ySinR_theta, axis = 1)).iloc[:-4].T
ySinR_alpha= (pd.concat(ySinR_alpha, axis = 1)).iloc[:-4].T
ySinR_beta1= (pd.concat(ySinR_beta1, axis = 1)).iloc[:-4].T
ySinR_beta2= (pd.concat(ySinR_beta2, axis = 1)).iloc[:-4].T

oStr_theta= (pd.concat(oStr_theta, axis = 1)).iloc[:-4].T
oStr_alpha= (pd.concat(oStr_alpha, axis = 1)).iloc[:-4].T
oStr_beta1= (pd.concat(oStr_beta1, axis = 1)).iloc[:-4].T
oStr_beta2= (pd.concat(oStr_beta2, axis = 1)).iloc[:-4].T

oStL_theta= (pd.concat(oStL_theta, axis = 1)).iloc[:-4].T
oStL_alpha= (pd.concat(oStL_alpha, axis = 1)).iloc[:-4].T
oStL_beta1= (pd.concat(oStL_beta1, axis = 1)).iloc[:-4].T
oStL_beta2= (pd.concat(oStL_beta2, axis = 1)).iloc[:-4].T

oSinL_theta= (pd.concat(oSinL_theta, axis = 1)).iloc[:-4].T
oSinL_alpha= (pd.concat(oSinL_alpha, axis = 1)).iloc[:-4].T
oSinL_beta1= (pd.concat(oSinL_beta1, axis = 1)).iloc[:-4].T
oSinL_beta2= (pd.concat(oSinL_beta2, axis = 1)).iloc[:-4].T

oSinR_theta= (pd.concat(oSinR_theta, axis = 1)).iloc[:-4].T
oSinR_alpha= (pd.concat(oSinR_alpha, axis = 1)).iloc[:-4].T
oSinR_beta1= (pd.concat(oSinR_beta1, axis = 1)).iloc[:-4].T
oSinR_beta2= (pd.concat(oSinR_beta2, axis = 1)).iloc[:-4].T

Str_theta= pd.concat([yStr_theta, oStr_theta])
Str_alpha= pd.concat([yStr_alpha, oStr_alpha])
Str_beta1= pd.concat([yStr_beta1, oStr_beta1])
Str_beta2= pd.concat([yStr_beta2, oStr_beta2]) 

SinR_theta= pd.concat([ySinR_theta, oSinR_theta])
SinR_alpha= pd.concat([ySinR_alpha, oSinR_alpha])
SinR_beta1= pd.concat([ySinR_beta1, oSinR_beta1])
SinR_beta2= pd.concat([ySinR_beta2, oSinR_beta2]) 

StL_theta= pd.concat([yStL_theta, oStL_theta])
StL_alpha= pd.concat([yStL_alpha, oStL_alpha])
StL_beta1= pd.concat([yStL_beta1, oStL_beta1])
StL_beta2= pd.concat([yStL_beta2, oStL_beta2]) 

SinL_theta= pd.concat([ySinL_theta, oSinL_theta])
SinL_alpha= pd.concat([ySinL_alpha, oSinL_alpha])
SinL_beta1= pd.concat([ySinL_beta1, oSinL_beta1])
SinL_beta2= pd.concat([ySinL_beta2, oSinL_beta2]) 

cols = ['StR_t','StR_a', 'StR_b1', 'StR_b2', 
        'StL_t','StL_a', 'StL_b1', 'StL_b2',
        'SinR_t','SinR_a', 'SinR_b1', 'SinR_b2', 
        'SinL_t','SinL_a', 'SinL_b1', 'SinL_b2',]

StR_C3C4_theta =  Str_theta['PSI_C3']-  Str_theta['PSI_C4']
StR_C3C4_alpha=  Str_alpha['PSI_C3']-  Str_alpha['PSI_C4']
StR_C3C4_beta1 =  Str_beta1['PSI_C3']-  Str_beta1['PSI_C4']
StR_C3C4_beta2 =  Str_beta2['PSI_C3']-  Str_beta2['PSI_C4']

StL_C3C4_theta =  StL_theta['PSI_C3']-  StL_theta['PSI_C4']
StL_C3C4_alpha=  StL_alpha['PSI_C3']-  StL_alpha['PSI_C4']
StL_C3C4_beta1 =  StL_beta1['PSI_C3']-  StL_beta1['PSI_C4']
StL_C3C4_beta2 =  StL_beta2['PSI_C3']-  StL_beta2['PSI_C4']

SinR_C3C4_theta =  SinR_theta['PSI_C3']-  SinR_theta['PSI_C4']
SinR_C3C4_alpha=  SinR_alpha['PSI_C3']-  SinR_alpha['PSI_C4']
SinR_C3C4_beta1 =  SinR_beta1['PSI_C3']-  SinR_beta1['PSI_C4']
SinR_C3C4_beta2 =  SinR_beta2['PSI_C3']-  SinR_beta2['PSI_C4']

SinL_C3C4_theta =  SinL_theta['PSI_C3']-  SinL_theta['PSI_C4']
SinL_C3C4_alpha=  SinL_alpha['PSI_C3']-  SinL_alpha['PSI_C4']
SinL_C3C4_beta1 =  SinL_beta1['PSI_C3']-  SinL_beta1['PSI_C4']
SinL_C3C4_beta2 =  SinL_beta2['PSI_C3']-  SinL_beta2['PSI_C4']

StR_PzFz_theta =  Str_theta['PSI_Pz']-  Str_theta['PSI_Fz']
StR_PzFz_alpha=  Str_alpha['PSI_Pz']-  Str_alpha['PSI_Fz']
StR_PzFz_beta1 =  Str_beta1['PSI_Pz']-  Str_beta1['PSI_Fz']
StR_PzFz_beta2 =  Str_beta2['PSI_Pz']-  Str_beta2['PSI_Fz']

StL_PzFz_theta =  StL_theta['PSI_Pz']-  StL_theta['PSI_Fz']
StL_PzFz_alpha=  StL_alpha['PSI_Pz']-  StL_alpha['PSI_Fz']
StL_PzFz_beta1 =  StL_beta1['PSI_Pz']-  StL_beta1['PSI_Fz']
StL_PzFz_beta2 =  StL_beta2['PSI_Pz']-  StL_beta2['PSI_Fz']

SinR_PzFz_theta =  SinR_theta['PSI_Pz']-  SinR_theta['PSI_Fz']
SinR_PzFz_alpha=  SinR_alpha['PSI_Pz']-  SinR_alpha['PSI_Fz']
SinR_PzFz_beta1 =  SinR_beta1['PSI_Pz']-  SinR_beta1['PSI_Fz']
SinR_PzFz_beta2 =  SinR_beta2['PSI_Pz']-  SinR_beta2['PSI_Fz']

SinL_PzFz_theta =  SinL_theta['PSI_Pz']-  SinL_theta['PSI_Fz']
SinL_PzFz_alpha=  SinL_alpha['PSI_Pz']-  SinL_alpha['PSI_Fz']
SinL_PzFz_beta1 =  SinL_beta1['PSI_Pz']-  SinL_beta1['PSI_Fz']
SinL_PzFz_beta2 =  SinL_beta2['PSI_Pz']-  SinL_beta2['PSI_Fz']


c3c4 = pd.concat([StR_C3C4_theta, StR_C3C4_alpha ,StR_C3C4_beta1 ,StR_C3C4_beta2,
                 StL_C3C4_theta, StL_C3C4_alpha ,StL_C3C4_beta1 ,StL_C3C4_beta2,
                 SinR_C3C4_theta, SinR_C3C4_alpha ,SinR_C3C4_beta1 ,SinR_C3C4_beta2,
                 SinL_C3C4_theta, SinL_C3C4_alpha ,SinL_C3C4_beta1 ,SinL_C3C4_beta2], axis = 1)
c3c4.columns = cols
c3c4.index = np.r_[y_parts,o_parts]
pzfz = pd.concat([StR_PzFz_theta, StR_PzFz_alpha ,StR_PzFz_beta1 ,StR_PzFz_beta2,
                 StL_PzFz_theta, StL_PzFz_alpha ,StL_PzFz_beta1 ,StL_PzFz_beta2,
                 SinR_PzFz_theta, SinR_PzFz_alpha ,SinR_PzFz_beta1 ,SinR_PzFz_beta2,
                 SinL_PzFz_theta, SinL_PzFz_alpha ,SinL_PzFz_beta1 ,SinL_PzFz_beta2], axis = 1)
pzfz.columns = cols
pzfz.index = np.r_[y_parts,o_parts]

c3c4.to_excel(output_dir + 'C3C4.xlsx')
pzfz.to_excel(output_dir + '/pzfz.xlsx')             

# plotting 
sns.set_context(font_scale=2)
c3c4 = pd.concat([StR_C3C4_theta, StR_C3C4_alpha ,StR_C3C4_beta1 ,StR_C3C4_beta2,
                 StL_C3C4_theta, StL_C3C4_alpha ,StL_C3C4_beta1 ,StL_C3C4_beta2,
                 SinR_C3C4_theta, SinR_C3C4_alpha ,SinR_C3C4_beta1 ,SinR_C3C4_beta2,
                 SinL_C3C4_theta, SinL_C3C4_alpha ,SinL_C3C4_beta1 ,SinL_C3C4_beta2], axis = 0)

group = np.ravel(16*[13*['Young'], 13*['OA']])
task = np.ravel([104*['Steady Right'],104*['Steady Left'], 104*['Sine Right'], 104*['Sine Left'] ])
freq = np.ravel(4*[26*['theta'], 26*['alpha'], 26*['beta1'], 26*['beta2']])
c3c4 = c3c4.reset_index()
cols = ['index', 'C3C4', 'group', 'task', 'fband']
c3c4['group'] = group
c3c4['task'] = task
c3c4['Frequency'] = freq#
c3c4.columns = cols

              
g = sns.catplot(x="fband", y="C3C4", hue="group", col = 'task',
data=c3c4,kind = 'box', palette = ['orange','dimgray'],width = 0.2)
g.set_titles("{col_name} {col_var}")
g.set(ylim=(-.1, .1))


pzfz = pd.concat([StR_PzFz_theta, StR_PzFz_alpha ,StR_PzFz_beta1 ,StR_PzFz_beta2,
                 StL_PzFz_theta, StL_PzFz_alpha ,StL_PzFz_beta1 ,StL_PzFz_beta2,
                 SinR_PzFz_theta, SinR_PzFz_alpha ,SinR_PzFz_beta1 ,SinR_PzFz_beta2,
                 SinL_PzFz_theta, SinL_PzFz_alpha ,SinL_PzFz_beta1 ,SinL_PzFz_beta2], axis = 0)
pzfz = pzfz.reset_index()
cols = ['index', 'PzFz', 'group', 'task', 'fband']
pzfz['group'] = group
pzfz['task'] = task
pzfz['Frequency'] = freq#
pzfz.columns = cols

g = sns.catplot(x="fband", y="PzFz", hue="group", col = 'task',
data=pzfz,kind = 'box', palette = ['orange','dimgray'],width = 0.2)
g.set_titles("{col_name} {col_var}")
g.set(ylim=(-.1, .1))