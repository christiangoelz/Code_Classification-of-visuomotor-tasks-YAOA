#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:39:25 2020

@author: christiangolz
"""


import glob
import pickle 
import numpy as np
import pandas as pd

lr_acc = [] 
Sin_lr_acc = [] 
St_lr_acc = [] 
lr_f1 = [] 
Sin_lr_f1 = [] 
St_lr_f1 = [] 
lr_auc = [] 
Sin_lr_auc = [] 
St_lr_auc = [] 

SinSt_acc = [] 
l_SinSt_acc = [] 
r_SinSt_acc = [] 
SinSt_f1 = [] 
l_SinSt_f1 = [] 
r_SinSt_f1 = []
SinSt_auc = [] 
l_SinSt_auc = [] 
r_SinSt_auc = []

All_acc = []
All_auc = [] 
All_f1 = []

part = [] 
group = [] 

output_dir = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/'
wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP/'
files = glob.glob(wd + '/n*Classification.pkl')
files_all = glob.glob(wd + '/n*Classification_All.pkl') 
l = len(wd) + 1
ll = l + 4
for i, file in enumerate(files): 
    with open(file, 'rb') as handle:
        participant = file[l:ll]
        part.append(file[l:ll])
        data = pickle.load(handle)
    with open(files_all[i], 'rb') as handle2:
        data_a = pickle.load(handle2)
        
        All_acc.append([data_a['All'].metrics_[i]['accuracy'] for i in range(10)])
        lr_acc.append([data['L_R'].metrics_[i]['accuracy'] for i in range(10)])
        SinSt_acc.append([data['SinSt'].metrics_[i]['accuracy'] for i in range(10)])
        Sin_lr_acc.append([data['SinLR'].metrics_[i]['accuracy'] for i in range(10)])
        St_lr_acc.append([data['StLR'].metrics_[i]['accuracy'] for i in range(10)])
        l_SinSt_acc.append([data['LStSin'].metrics_[i]['accuracy'] for i in range(10)])
        r_SinSt_acc.append([data['RStSin'].metrics_[i]['accuracy'] for i in range(10)])
        
        All_auc.append(data_a['All'].auc_score)
        lr_auc.append(data['L_R'].auc_score)
        SinSt_auc.append(data['SinSt'].auc_score)
        Sin_lr_auc.append(data['SinLR'].auc_score)
        St_lr_auc.append(data['StLR'].auc_score)
        l_SinSt_auc.append(data['LStSin'].auc_score)
        r_SinSt_auc.append(data['RStSin'].auc_score)
        
        All_f1.append([data_a['All'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        lr_f1.append([data['L_R'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        SinSt_f1.append([data['SinSt'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        Sin_lr_f1.append([data['SinLR'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        St_lr_f1.append([data['StLR'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        l_SinSt_f1.append([data['LStSin'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
        r_SinSt_f1.append([data['RStSin'].metrics_[i]['macro avg']['f1-score'] for i in range(10)])
            
        if 'nj' in participant:
            group.append(1) 
        else: 
            group.append(0)
    
All_acc = pd.DataFrame(All_acc, index = part)       
lr_acc = pd.DataFrame(lr_acc, index = part)
SinSt_acc = pd.DataFrame(SinSt_acc,index = part)
Sin_lr_acc  = pd.DataFrame(Sin_lr_acc,index = part )
St_lr_acc = pd.DataFrame(St_lr_acc,index = part)
l_SinSt_acc= pd.DataFrame(l_SinSt_acc, index = part)
r_SinSt_acc = pd.DataFrame(r_SinSt_acc, index = part)

All_auc = pd.DataFrame(All_auc, index = part)       
lr_auc = pd.DataFrame(lr_auc, index = part)
SinSt_auc = pd.DataFrame(SinSt_auc,index = part)
Sin_lr_auc  = pd.DataFrame(Sin_lr_auc,index = part )
St_lr_auc = pd.DataFrame(St_lr_auc,index = part)
l_SinSt_auc= pd.DataFrame(l_SinSt_auc, index = part)
r_SinSt_auc = pd.DataFrame(r_SinSt_auc, index = part)

All_f1 = pd.DataFrame(All_f1, index = part)       
lr_f1 = pd.DataFrame(lr_f1, index = part)
SinSt_f1 = pd.DataFrame(SinSt_f1,index = part)
Sin_lr_f1  = pd.DataFrame(Sin_lr_f1,index = part )
St_lr_f1 = pd.DataFrame(St_lr_f1,index = part)
l_SinSt_f1= pd.DataFrame(l_SinSt_f1, index = part)
r_SinSt_f1 = pd.DataFrame(r_SinSt_f1, index = part)

columns = ['All_acc','All_auc','All_f1',
            'SinSt_acc','SinSt_auc','SinSt_f1', 
            'Sin_lr_acc','Sin_lr_auc','Sin_lr_f1',
            'St_lr_acc','St_lr_auc','St_lr_f1',
            'l_SinSt_acc','l_SinSt_auc', 'l_SinSt_f1',
            'lr_acc','lr_auc','lr_f1',
            'r_SinSt_acc','r_SinSt_auc','r_SinSt_f1']  
mean_d = np.c_[All_acc.mean(axis = 1), All_auc.mean(axis = 1),All_f1.mean(axis = 1),
            SinSt_acc.mean(axis = 1),SinSt_auc.mean(axis = 1),SinSt_f1.mean(axis = 1), 
            Sin_lr_acc.mean(axis = 1),Sin_lr_auc.mean(axis = 1),Sin_lr_f1.mean(axis = 1),
            St_lr_acc.mean(axis = 1),St_lr_auc.mean(axis = 1),St_lr_f1.mean(axis = 1),
            l_SinSt_acc.mean(axis = 1),l_SinSt_auc.mean(axis = 1), l_SinSt_f1.mean(axis = 1),
            lr_acc.mean(axis = 1),lr_auc.mean(axis = 1),lr_f1.mean(axis = 1),
            r_SinSt_acc.mean(axis = 1),r_SinSt_auc.mean(axis = 1),r_SinSt_f1.mean(axis = 1)]
std_d = np.c_[All_acc.std(axis = 1),All_auc.std(axis = 1),All_f1.std(axis = 1),
            SinSt_acc.std(axis = 1),SinSt_auc.std(axis = 1),SinSt_f1.std(axis = 1), 
            Sin_lr_acc.std(axis = 1),Sin_lr_auc.std(axis = 1),Sin_lr_f1.std(axis = 1),
            St_lr_acc.std(axis = 1),St_lr_auc.std(axis = 1),St_lr_f1.std(axis = 1),
            l_SinSt_acc.std(axis = 1),l_SinSt_auc.std(axis = 1), l_SinSt_f1.std(axis = 1),
            lr_acc.std(axis = 1),lr_auc.std(axis = 1),lr_f1.std(axis = 1),
            r_SinSt_acc.std(axis = 1),r_SinSt_auc.std(axis = 1),r_SinSt_f1.std(axis = 1)]

df_mean = pd.DataFrame(data = mean_d, columns = columns, index = part)
df_mean['group'] = group
df_std = pd.DataFrame(data = std_d, columns = columns, index = part)
df_std['group'] = group

df_mean.to_excel(output_dir + 'results_mean2.xlsx')
df_std.to_excel(output_dir+ '/results_std2.xlsx')


