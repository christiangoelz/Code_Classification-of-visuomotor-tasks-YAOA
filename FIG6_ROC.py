#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:49:55 2020

@author: christiangolz
"""


import glob
import pickle 
import matplotlib.pyplot as plt
import numpy as np

wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP'
files = glob.glob(wd + '/n*Classification.pkl')
l = len(wd) + 1
ll = l + 4


ylr_tprs = [] 
ySin_lr_tprs= [] 
ySt_lr_tprs= [] 
ySinSt_tprs= [] 
yl_SinSt_tprs= [] 
yr_SinSt_tprs= []



olr_tprs = [] 
oSin_lr_tprs= [] 
oSt_lr_tprs= [] 
oSinSt_tprs= [] 
ol_SinSt_tprs= [] 
or_SinSt_tprs= []

yAll_tprs= []
mean_fpr = np.linspace(0, 1, 100)
o = [] 
y = [] 
part = [] 

for file in files: 
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
        participant = file[l:ll]
        part.append(participant)
    if 'nj' in participant:
        ylr_tprs.append([np.interp(mean_fpr, data['L_R'].fpr[i], data['L_R'].tpr[i]) for i in range(10)])
        ySin_lr_tprs.append([np.interp(mean_fpr, data['SinLR'].fpr[i], data['SinLR'].tpr[i]) for i in range(10)])
        ySt_lr_tprs.append([np.interp(mean_fpr, data['StLR'].fpr[i], data['StLR'].tpr[i]) for i in range(10)])
        ySinSt_tprs.append([np.interp(mean_fpr, data['SinSt'].fpr[i], data['SinSt'].tpr[i]) for i in range(10)])
        yl_SinSt_tprs.append([np.interp(mean_fpr, data['LStSin'].fpr[i], data['LStSin'].tpr[i]) for i in range(10)]) 
        yr_SinSt_tprs.append([np.interp(mean_fpr, data['RStSin'].fpr[i], data['RStSin'].tpr[i]) for i in range(10)])
        y.append(0)
        
    else: 
        olr_tprs.append([np.interp(mean_fpr, data['L_R'].fpr[i], data['L_R'].tpr[i]) for i in range(10)])
        oSin_lr_tprs.append([np.interp(mean_fpr, data['SinLR'].fpr[i], data['SinLR'].tpr[i]) for i in range(10)])
        oSt_lr_tprs.append([np.interp(mean_fpr, data['StLR'].fpr[i], data['StLR'].tpr[i]) for i in range(10)])
        oSinSt_tprs.append([np.interp(mean_fpr, data['SinSt'].fpr[i], data['SinSt'].tpr[i]) for i in range(10)])
        ol_SinSt_tprs.append([np.interp(mean_fpr, data['LStSin'].fpr[i], data['LStSin'].tpr[i]) for i in range(10)]) 
        or_SinSt_tprs.append([np.interp(mean_fpr, data['RStSin'].fpr[i], data['RStSin'].tpr[i]) for i in range(10)])
        o.append(1)
       
ylr = np.mean(np.mean(ylr_tprs, axis = 1), axis = 0 )
ySin_lr = np.mean(np.mean(ySin_lr_tprs,axis = 1), axis = 0 )
ySt_lr = np.mean(np.mean(ySt_lr_tprs,axis = 1) , axis = 0 )   
ySinSt = np.mean(np.mean(ySinSt_tprs,axis = 1) , axis = 0 )  
yl_SinSt = np.mean(np.mean(yl_SinSt_tprs,axis = 1), axis = 0 )  
yr_SinSt = np.mean(np.mean(yr_SinSt_tprs,axis = 1), axis = 0 )

ylr[0]= 0  
ySin_lr[0]= 0  
ySt_lr[0]= 0  
ySinSt[0]= 0   
yl_SinSt[0]= 0  
yr_SinSt[0]= 0  

ylr_std = np.std(np.mean(ylr_tprs, axis = 1), axis = 0 )
ySin_lr_std = np.std(np.mean(ySin_lr_tprs,axis = 1), axis = 0 )
ySt_lr_std = np.std(np.mean(ySt_lr_tprs,axis = 1) , axis = 0 )   
ySinSt_std = np.std(np.mean(ySinSt_tprs,axis = 1) , axis = 0 )  
yl_SinSt_std = np.std(np.mean(yl_SinSt_tprs,axis = 1), axis = 0 )  
yr_SinSt_std = np.std(np.mean(yr_SinSt_tprs,axis = 1), axis = 0 )

ylr_min = np.minimum(ylr + ylr_std, 1); ylr_max = np.maximum(ylr - ylr_std, 0)
ySin_lr_min = np.minimum(ySin_lr + ySin_lr_std, 1); ySin_lr_max = np.maximum(ySin_lr - ySin_lr_std, 0)
ySt_lr_min = np.minimum(ySt_lr + ySt_lr_std, 1); ySt_lr_max = np.maximum(ySt_lr - ySt_lr_std, 0)
ySinSt_min = np.minimum(ySinSt + ySinSt_std, 1); ySinSt_max = np.maximum(ySinSt - ySinSt_std, 0)
yl_SinSt_min = np.minimum(yl_SinSt + yl_SinSt_std, 1); yl_SinSt_max = np.maximum(yl_SinSt - yl_SinSt_std, 0)
yr_SinSt_min = np.minimum(yr_SinSt + yr_SinSt_std, 1); yr_SinSt_max = np.maximum(yr_SinSt - yr_SinSt_std, 0)



olr = np.mean(np.mean(olr_tprs, axis = 1), axis = 0 )
oSin_lr = np.mean(np.mean(oSin_lr_tprs,axis = 1), axis = 0 )
oSt_lr = np.mean(np.mean(oSt_lr_tprs,axis = 1) , axis = 0 )   
oSinSt = np.mean(np.mean(oSinSt_tprs,axis = 1) , axis = 0 )  
ol_SinSt = np.mean(np.mean(ol_SinSt_tprs,axis = 1), axis = 0 )  
or_SinSt = np.mean(np.mean(or_SinSt_tprs,axis = 1), axis = 0 )

olr[0]= 0  
oSin_lr[0]= 0  
oSt_lr[0]= 0  
oSinSt[0]= 0   
ol_SinSt[0]= 0  
or_SinSt[0]= 0  

olr_std = np.std(np.mean(olr_tprs, axis = 1), axis = 0 )
oSin_lr_std = np.std(np.mean(oSin_lr_tprs,axis = 1), axis = 0 )
oSt_lr_std = np.std(np.mean(oSt_lr_tprs,axis = 1) , axis = 0 )   
oSinSt_std = np.std(np.mean(oSinSt_tprs,axis = 1) , axis = 0 )  
ol_SinSt_std = np.std(np.mean(ol_SinSt_tprs,axis = 1), axis = 0 )  
or_SinSt_std = np.std(np.mean(or_SinSt_tprs,axis = 1), axis = 0 )

olr_min = np.minimum(olr + olr_std, 1); olr_max = np.maximum(olr - olr_std, 0)
oSin_lr_min = np.minimum(oSin_lr + oSin_lr_std, 1); oSin_lr_max = np.maximum(oSin_lr - oSin_lr_std, 0)
oSt_lr_min = np.minimum(oSt_lr + oSt_lr_std, 1); oSt_lr_max = np.maximum(oSt_lr - oSt_lr_std, 0)
oSinSt_min = np.minimum(oSinSt + oSinSt_std, 1); oSinSt_max = np.maximum(oSinSt - oSinSt_std, 0)
ol_SinSt_min = np.minimum(ol_SinSt + ol_SinSt_std, 1); ol_SinSt_max = np.maximum(ol_SinSt - ol_SinSt_std, 0)
or_SinSt_min = np.minimum(or_SinSt + or_SinSt_std, 1); or_SinSt_max = np.maximum(or_SinSt - or_SinSt_std, 0)



#LR
fig, ax = plt.subplots()
ax.plot(mean_fpr, ylr, color='orange',label='Young')
ax.plot(mean_fpr, olr, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, ylr_min, ylr_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, olr_min, olr_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc="lower right", fontsize = 16 )
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

#Sin_LR
fig, ax = plt.subplots()
ax.plot(mean_fpr, ySin_lr, color='orange',label='Young')
ax.plot(mean_fpr, oSin_lr, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, ySin_lr_min, ySin_lr_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, oSin_lr_min, oSin_lr_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

#Steady_LR
fig, ax = plt.subplots()
ax.plot(mean_fpr, ySt_lr, color='orange',label='Young')
ax.plot(mean_fpr, oSt_lr, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, ySt_lr_min, ySt_lr_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, oSt_lr_min, oSt_lr_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

#SinSt
fig, ax = plt.subplots()
ax.plot(mean_fpr, ySinSt, color='orange',label='Young')
ax.plot(mean_fpr, oSinSt, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, ySinSt_min, ySinSt_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, oSinSt_min, oSinSt_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc="lower right", fontsize = 16 )
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

#r_SinSt
fig, ax = plt.subplots()
ax.plot(mean_fpr, yl_SinSt, color='orange',label='Young')
ax.plot(mean_fpr, ol_SinSt, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, yl_SinSt_min, yl_SinSt_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, ol_SinSt_min, ol_SinSt_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

#r_SinSt
fig, ax = plt.subplots()
ax.plot(mean_fpr, yr_SinSt, color='orange',label='Young')
ax.plot(mean_fpr, or_SinSt, color='dimgray',label='LMA')
ax.fill_between(mean_fpr, yr_SinSt_min, yr_SinSt_max, color='orange', alpha=.2)
ax.fill_between(mean_fpr, or_SinSt_min, or_SinSt_max, color='dimgray', alpha=.2)
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('False Positive Rate', fontsize = 16)
ax.set_ylabel('True Positive Rate', fontsize = 16)
plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)