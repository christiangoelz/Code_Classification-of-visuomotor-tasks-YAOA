#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Chrisitan Goelz <goelz@sportmed.upb.de>
# Description: Main Script for Classification DMD Aging Force Control Project
# Dependencies: see .yml 

import mne 
import pickle 
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
wd_task = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/DMD_ana/BHS/python/Task'
wd_rest = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/DMD_ana/BHS/python/Rest/EOEC/'
ystl_theta = [] 
ystl_alpha = [] 
ystl_beta1 = [] 
ystl_beta2 = [] 
ystr_theta = [] 
ystr_alpha = [] 
ystr_beta1 = [] 
ystr_beta2 = []
ysinl_theta = [] 
ysinl_alpha = [] 
ysinl_beta1 = [] 
ysinl_beta2 = [] 
ysinr_theta = [] 
ysinr_alpha = [] 
ysinr_beta1 = [] 
ysinr_beta2 = []
yrest_theta = [] 
yrest_alpha = [] 
yrest_beta1 = [] 
yrest_beta2 = []
y_parts = []
l = len(wd_task)+1
ll = len(wd_task)+5
files = glob.glob(wd_task + '/nj*.pkl')
m = mne.channels.make_standard_montage('biosemi32') 
info = mne.create_info(
        ch_names=m.ch_names, sfreq=200., ch_types='eeg')
info.set_montage(m)

for file in files: 
   y_part = (file[l:ll])
   y_parts.append(file[l:ll])
   rest_file = glob.glob(wd_rest + '/' + y_part + '*.pkl')
   with open(file, 'rb') as handle:
       dmd= pickle.load(handle)
   with open(rest_file[0], 'rb') as handle:
       dmd_rest = pickle.load(handle)
   
   statStR = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [1])
   statStL = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [3])
   statSinR = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [2])
   statSinL = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [4])
   statrest = dmd_rest.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [1]) #1 Eyes closed / 2 Eyes Open
    
   ystl_theta.append(statStL['4-8'].loc['mean'])
   ystl_alpha.append(statStL['8-12'].loc['mean'])
   ystl_beta1.append(statStL['12-16'].loc['mean'])
   ystl_beta2.append(statStL['16-30'].loc['mean'])
   
   ystr_theta.append(statStR['4-8'].loc['mean'])
   ystr_alpha.append(statStR['8-12'].loc['mean'])
   ystr_beta1.append(statStR['12-16'].loc['mean'])
   ystr_beta2.append(statStR['16-30'].loc['mean'])
   
   ysinl_theta.append(statSinL['4-8'].loc['mean'])
   ysinl_alpha.append(statSinL['8-12'].loc['mean'])
   ysinl_beta1.append(statSinL['12-16'].loc['mean'])
   ysinl_beta2.append(statSinL['16-30'].loc['mean'])
   
   ysinr_theta.append(statSinR['4-8'].loc['mean'])
   ysinr_alpha.append(statSinR['8-12'].loc['mean'])
   ysinr_beta1.append(statSinR['12-16'].loc['mean'])
   ysinr_beta2.append(statSinR['16-30'].loc['mean'])
   
   yrest_theta.append(statrest['4-8'].loc['mean'])
   yrest_alpha.append(statrest['8-12'].loc['mean'])
   yrest_beta1.append(statrest['12-16'].loc['mean'])
   yrest_beta2.append(statrest['16-30'].loc['mean'])

ystl_theta = pd.DataFrame(ystl_theta).T
ystl_alpha = pd.DataFrame(ystl_alpha).T
ystl_beta1 = pd.DataFrame(ystl_beta1).T
ystl_beta2 = pd.DataFrame(ystl_beta2).T
ystr_theta = pd.DataFrame(ystr_theta).T
ystr_alpha = pd.DataFrame(ystr_alpha).T
ystr_beta1 = pd.DataFrame(ystr_beta1).T
ystr_beta2 = pd.DataFrame(ystr_beta2).T
ysinl_theta = pd.DataFrame(ysinl_theta).T
ysinl_alpha = pd.DataFrame(ysinl_alpha).T
ysinl_beta1 = pd.DataFrame(ysinl_beta1).T
ysinl_beta2 = pd.DataFrame(ysinl_beta2).T
ysinr_theta = pd.DataFrame(ysinr_theta).T
ysinr_alpha = pd.DataFrame(ysinr_alpha).T
ysinr_beta1 = pd.DataFrame(ysinr_beta1).T
ysinr_beta2 = pd.DataFrame(ysinr_beta2).T   
yrest_theta = pd.DataFrame(yrest_theta).T
yrest_alpha = pd.DataFrame(yrest_alpha).T
yrest_beta1 = pd.DataFrame(yrest_beta1).T
yrest_beta2 = pd.DataFrame(yrest_beta2).T   

ostl_theta = [] 
ostl_alpha = [] 
ostl_beta1 = [] 
ostl_beta2 = [] 
ostr_theta = [] 
ostr_alpha = [] 
ostr_beta1 = [] 
ostr_beta2 = []
osinl_theta = [] 
osinl_alpha = [] 
osinl_beta1 = [] 
osinl_beta2 = [] 
osinr_theta = [] 
osinr_alpha = [] 
osinr_beta1 = [] 
osinr_beta2 = []
orest_theta = [] 
orest_alpha = [] 
orest_beta1 = [] 
orest_beta2 = []
o_parts = []
l = len(wd_task)+1
ll = len(wd_task)+5
files = glob.glob(wd_task + '/na*.pkl')

for file in files: 
   o_part = (file[l:ll])
   o_parts.append(file[l:ll])
   rest_file = glob.glob(wd_rest + '/' + o_part + '*.pkl')
   with open(file, 'rb') as handle:
       dmd= pickle.load(handle)
   with open(rest_file[0], 'rb') as handle:
       dmd_rest = pickle.load(handle)
   
   statStR = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [1])
   statStL = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [3])
   statSinR = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [2])
   statSinL = dmd.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [4])
   statRest = dmd_rest.mode_stats(fbands = [[4,8],[8,12],[12,16],[16,30]], labels = [2])
    
   ostl_theta.append(statStL['4-8'].loc['mean'])
   ostl_alpha.append(statStL['8-12'].loc['mean'])
   ostl_beta1.append(statStL['12-16'].loc['mean'])
   ostl_beta2.append(statStL['16-30'].loc['mean'])
   
   ostr_theta.append(statStR['4-8'].loc['mean'])
   ostr_alpha.append(statStR['8-12'].loc['mean'])
   ostr_beta1.append(statStR['12-16'].loc['mean'])
   ostr_beta2.append(statStR['16-30'].loc['mean'])
   
   osinl_theta.append(statSinL['4-8'].loc['mean'])
   osinl_alpha.append(statSinL['8-12'].loc['mean'])
   osinl_beta1.append(statSinL['12-16'].loc['mean'])
   osinl_beta2.append(statSinL['16-30'].loc['mean'])
   
   osinr_theta.append(statSinR['4-8'].loc['mean'])
   osinr_alpha.append(statSinR['8-12'].loc['mean'])
   osinr_beta1.append(statSinR['12-16'].loc['mean'])
   osinr_beta2.append(statSinR['16-30'].loc['mean'])   
   
   orest_theta.append(statrest['4-8'].loc['mean'])
   orest_alpha.append(statrest['8-12'].loc['mean'])
   orest_beta1.append(statrest['12-16'].loc['mean'])
   orest_beta2.append(statrest['16-30'].loc['mean'])   

ostl_theta = pd.DataFrame(ostl_theta).T
ostl_alpha = pd.DataFrame(ostl_alpha).T
ostl_beta1 = pd.DataFrame(ostl_beta1).T
ostl_beta2 = pd.DataFrame(ostl_beta2).T
ostr_theta = pd.DataFrame(ostr_theta).T
ostr_alpha = pd.DataFrame(ostr_alpha).T
ostr_beta1 = pd.DataFrame(ostr_beta1).T
ostr_beta2 = pd.DataFrame(ostr_beta2).T
osinl_theta = pd.DataFrame(osinl_theta).T
osinl_alpha = pd.DataFrame(osinl_alpha).T
osinl_beta1 = pd.DataFrame(osinl_beta1).T
osinl_beta2 = pd.DataFrame(osinl_beta2).T
osinr_theta = pd.DataFrame(osinr_theta).T
osinr_alpha = pd.DataFrame(osinr_alpha).T
osinr_beta1 = pd.DataFrame(osinr_beta1).T
osinr_beta2 = pd.DataFrame(osinr_beta2).T
orest_theta = pd.DataFrame(orest_theta).T
orest_alpha = pd.DataFrame(orest_alpha).T
orest_beta1 = pd.DataFrame(orest_beta1).T
orest_beta2 = pd.DataFrame(orest_beta2).T
   
# calculate stat comparisons 
from permute.core import one_sample, two_sample
from statsmodels.stats.multitest import fdrcorrection
from mne import EvokedArray

p_th_stL = [] ; t_th_stL  = [] 
p_a_stL = [] ; t_a_stL  = [] 
p_b1_stL = [] ; t_b1_stL  = [] 
p_b2_stL = [] ; t_b2_stL  = [] 

p_th_stR = [] ; t_th_stR  = [] 
p_a_stR = [] ; t_a_stR = [] 
p_b1_stR = [] ; t_b1_stR  = [] 
p_b2_stR = [] ; t_b2_stR  = [] 

p_th_sinL = [] ; t_th_sinL  = [] 
p_a_sinL = [] ; t_a_sinL  = [] 
p_b1_sinL = [] ; t_b1_sinL  = [] 
p_b2_sinL = [] ; t_b2_sinL  = [] 

p_th_sinR = [] ; t_th_sinR  = [] 
p_a_sinR = [] ; t_a_sinR = [] 
p_b1_sinR = [] ; t_b1_sinR  = [] 
p_b2_sinR = [] ; t_b2_sinR  = [] 

p_th_rest = [] ; t_th_rest  = [] 
p_a_rest = [] ; t_a_rest = [] 
p_b1_rest = [] ; t_b1_rest  = [] 
p_b2_rest = [] ; t_b2_rest  = [] 

for i in range(32):
    #steady left
    (p_, t_) = two_sample(ostl_theta.values[i,:], ystl_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_th_stL.append(p_); t_th_stL.append(t_)   

    (p_, t_) = two_sample(ostl_alpha.values[i,:], ystl_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_a_stL.append(p_); t_a_stL.append(t_) 
    
    (p_, t_) = two_sample(ostl_beta1.values[i,:], ystl_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b1_stL.append(p_); t_b1_stL.append(t_) 
    
    (p_, t_) = two_sample(ostl_beta2.values[i,:], ystl_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b2_stL.append(p_); t_b2_stL.append(t_) 
    
    #steady right
    (p_, t_) = two_sample(ostr_theta.values[i,:], ystr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_th_stR.append(p_); t_th_stR.append(t_)   

    (p_, t_) = two_sample(ostr_alpha.values[i,:], ystr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_a_stR.append(p_); t_a_stR.append(t_) 
    
    (p_, t_) = two_sample(ostr_beta1.values[i,:], ystr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b1_stR.append(p_); t_b1_stR.append(t_) 
    
    (p_, t_) = two_sample(ostr_beta2.values[i,:], ystr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b2_stR.append(p_); t_b2_stR.append(t_) 
    
    #sinus left
    (p_, t_) = two_sample(osinl_theta.values[i,:], ysinl_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_th_sinL.append(p_); t_th_sinL.append(t_)   

    (p_, t_) = two_sample(osinl_alpha.values[i,:], ysinl_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_a_sinL.append(p_); t_a_sinL.append(t_) 
    
    (p_, t_) = two_sample(osinl_beta1.values[i,:], ysinl_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b1_sinL.append(p_); t_b1_sinL.append(t_) 
    
    (p_, t_) = two_sample(osinl_beta2.values[i,:], ysinl_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b2_sinL.append(p_); t_b2_sinL.append(t_) 
    
    #sinus right
    (p_, t_) = two_sample(osinr_theta.values[i,:], ysinr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_th_sinR.append(p_); t_th_sinR.append(t_)   

    (p_, t_) = two_sample(osinr_alpha.values[i,:], ysinr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_a_sinR.append(p_); t_a_sinR.append(t_) 
    
    (p_, t_) = two_sample(osinr_beta1.values[i,:], ysinr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b1_sinR.append(p_); t_b1_sinR.append(t_) 
    
    (p_, t_) = two_sample(osinr_beta2.values[i,:], ysinr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b2_sinR.append(p_); t_b2_sinR.append(t_) 
    
        #sinus right
    (p_, t_) = two_sample(orest_theta.values[i,:], yrest_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_th_rest.append(p_); t_th_rest.append(t_)   

    (p_, t_) = two_sample(orest_alpha.values[i,:], yrest_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_a_rest.append(p_); t_a_rest.append(t_) 
    
    (p_, t_) = two_sample(orest_beta1.values[i,:], yrest_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b1_rest.append(p_); t_b1_rest.append(t_) 
    
    (p_, t_) = two_sample(orest_beta2.values[i,:], yrest_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
    p_b2_rest.append(p_); t_b2_rest.append(t_) 
    
p_stR = np.c_[p_th_stR,p_a_stR,p_b1_stR, p_b2_stR]
p_stL = np.c_[p_th_stL,p_a_stL,p_b1_stL, p_b2_stL]
p_sinR = np.c_[p_th_sinR,p_a_sinR,p_b1_sinR, p_b2_sinR]
p_sinL = np.c_[p_th_sinL,p_a_sinL,p_b1_sinL, p_b2_sinL]
p_rest = np.c_[p_th_rest,p_a_rest,p_b1_rest, p_b2_rest]
t_stR = np.c_[t_th_stR,t_a_stR,t_b1_stR, t_b2_stR]
t_stL = np.c_[t_th_stL,t_a_stL,t_b1_stL, t_b2_stL]
t_sinR = np.c_[t_th_sinR,t_a_sinR,t_b1_sinR, t_b2_sinR]
t_sinL = np.c_[t_th_sinL,t_a_sinL,t_b1_sinL, t_b2_sinL]
t_rest = np.c_[t_th_rest,t_a_rest,t_b1_rest, t_b2_rest]
for p in range(4): 
    _, p_corr = fdrcorrection(p_stR[:,p])
    p_stR[:,p] = p_corr
    _, p_corr = fdrcorrection(p_stL[:,p])
    p_stL[:,p] = p_corr
    _, p_corr = fdrcorrection(p_sinR[:,p])
    p_sinR[:,p] = p_corr    
    _, p_corr = fdrcorrection(p_sinL[:,p])
    p_sinL[:,p] = p_corr
    _, p_corr = fdrcorrection(p_rest[:,p])
    p_rest[:,p] = p_corr    
t_stR = EvokedArray(t_stR, info, tmin=0)
t_stL = EvokedArray(t_stL, info, tmin=0) 
t_sinR = EvokedArray(t_sinR, info, tmin=0)
t_sinL = EvokedArray(t_sinL, info, tmin=0)    
t_rest = EvokedArray(t_rest, info, tmin=0)
mask_stR = p_stR <= 0.05
mask_stL = p_stL <=0.05
mask_sinR = p_sinR <= 0.05
mask_sinL = p_sinL <=0.05
mask_rest = p_rest <= 0.05
fig1 = t_stR.plot_topomap(ch_type='eeg', scalings=1,
                    time_format=' ', vmin=-4.5, vmax=4.5,
                    units='t_values', mask=mask_stR,
                    size=3,
                    time_unit='s', title = None, nrows = 4, res = 1000)


fig2 = t_sinR.plot_topomap(ch_type='eeg', scalings=1,
                    time_format=' ', vmin=-4.5, vmax=4.5,
                    units='t_values', mask=mask_sinR,
                    size=3,
                    time_unit='s', title = None, nrows = 4, res = 1000)


fig3 = t_sinL.plot_topomap(ch_type='eeg', scalings=1,
                    time_format=' ', vmin=-4.5, vmax=4.5,
                    units='t_values', mask=mask_sinL,
                    size=3,
                    time_unit='s', title = None, nrows = 4, res = 1000)


fig4 = t_stL.plot_topomap(ch_type='eeg', scalings=1,
                    time_format=' ', vmin=-4.5, vmax=4.5,
                    units='t_values', mask=mask_stL,
                    size=3,
                    time_unit='s', title = None, nrows = 4, res = 1000)


fig5 = t_rest.plot_topomap(ch_type='eeg', scalings=1,
                    time_format=' ', vmin=-9, vmax=9,
                    units='t_values', mask=mask_rest,
                    size=3,
                    time_unit='s', title = None, nrows = 4, res = 1000)

# #old
# p_th_stLR = [] ; t_th_stLR  = [] 
# p_a_stLR = [] ; t_a_stLR  = [] 
# p_b1_stLR = [] ; t_b1_stLR  = [] 
# p_b2_stLR = [] ; t_b2_stLR  = [] 

# p_th_sinstR = [] ; t_th_sinstR  = [] 
# p_a_sinstR = [] ; t_a_sinstR = [] 
# p_b1_sinstR = [] ; t_b1_sinstR  = [] 
# p_b2_sinstR = [] ; t_b2_sinstR  = [] 

# p_th_sinLR = [] ; t_th_sinLR  = [] 
# p_a_sinLR = [] ; t_a_sinLR  = [] 
# p_b1_sinLR = [] ; t_b1_sinLR  = [] 
# p_b2_sinLR = [] ; t_b2_sinLR  = [] 

# p_th_sinstL = [] ; t_th_sinstL  = [] 
# p_a_sinstL = [] ; t_a_sinstL = [] 
# p_b1_sinstL = [] ; t_b1_sinstL  = [] 
# p_b2_sinstL = [] ; t_b2_sinstL  = [] 

# p_th_rest = [] ; t_th_rest  = [] 
# p_a_rest = [] ; t_a_rest = [] 
# p_b1_rest = [] ; t_b1_rest  = [] 


# for i in range(32):
#     #steady left
#     (p_, t_) = one_sample(ostl_theta.values[i,:], ostr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_stLR.append(p_); t_th_stLR.append(t_)   

#     (p_, t_) = one_sample(ostl_alpha.values[i,:], ostr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_stLR.append(p_); t_a_stLR.append(t_) 
    
#     (p_, t_) = one_sample(ostl_beta1.values[i,:], ostr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_stLR.append(p_); t_b1_stLR.append(t_) 
    
#     (p_, t_) = one_sample(ostl_beta2.values[i,:], ostr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_stLR.append(p_); t_b2_stLR.append(t_) 
    
#     #steady right
#     (p_, t_) = one_sample(osinr_theta.values[i,:], ostr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinstR.append(p_); t_th_sinstR.append(t_)   

#     (p_, t_) = one_sample(osinr_alpha.values[i,:], ostr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinstR.append(p_); t_a_sinstR.append(t_) 
    
#     (p_, t_) = one_sample(osinr_beta1.values[i,:], ostr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinstR.append(p_); t_b1_sinstR.append(t_) 
    
#     (p_, t_) = one_sample(osinr_beta2.values[i,:], ostr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinstR.append(p_); t_b2_sinstR.append(t_) 
    
#     #sinus left
#     (p_, t_) = one_sample(osinl_theta.values[i,:], osinr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinLR.append(p_); t_th_sinLR.append(t_)   

#     (p_, t_) = one_sample(osinl_alpha.values[i,:], osinr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinLR.append(p_); t_a_sinLR.append(t_) 
    
#     (p_, t_) = one_sample(osinl_beta1.values[i,:], osinr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinLR.append(p_); t_b1_sinLR.append(t_) 
    
#     (p_, t_) = one_sample(osinl_beta2.values[i,:], osinr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinLR.append(p_); t_b2_sinLR.append(t_) 
    
#     #sinus right
#     (p_, t_) = one_sample(osinl_theta.values[i,:], ostl_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinstL.append(p_); t_th_sinstL.append(t_)   

#     (p_, t_) = one_sample(osinl_alpha.values[i,:], ostl_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinstL.append(p_); t_a_sinstL.append(t_) 
    
#     (p_, t_) = one_sample(osinl_beta1.values[i,:], ostl_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinstL.append(p_); t_b1_sinstL.append(t_) 
    
#     (p_, t_) = one_sample(osinl_beta2.values[i,:], ostl_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinstL.append(p_); t_b2_sinstL.append(t_) 
        
# p_sinstR = np.c_[p_th_sinstR,p_a_sinstR,p_b1_sinstR, p_b2_sinstR]
# p_stLR = np.c_[p_th_stLR,p_a_stLR,p_b1_stLR, p_b2_stLR]
# p_sinstL = np.c_[p_th_sinstL,p_a_sinstL,p_b1_sinstL, p_b2_sinstL]
# p_sinLR = np.c_[p_th_sinLR,p_a_sinLR,p_b1_sinLR, p_b2_sinLR]
# p_rest = np.c_[p_th_rest,p_a_rest,p_b1_rest, p_b2_rest]
# t_sinstR = np.c_[t_th_sinstR,t_a_sinstR,t_b1_sinstR, t_b2_sinstR]
# t_stLR = np.c_[t_th_stLR,t_a_stLR,t_b1_stLR, t_b2_stLR]
# t_sinstL = np.c_[t_th_sinstL,t_a_sinstL,t_b1_sinstL, t_b2_sinstL]
# t_sinLR = np.c_[t_th_sinLR,t_a_sinLR,t_b1_sinLR, t_b2_sinLR]

# for p in range(4): 
#     _, p_corr = fdrcorrection(p_sinstR[:,p])
#     p_sinstR[:,p] = p_corr
#     _, p_corr = fdrcorrection(p_stLR[:,p])
#     p_stLR[:,p] = p_corr
#     _, p_corr = fdrcorrection(p_sinstL[:,p])
#     p_sinstL[:,p] = p_corr    
#     _, p_corr = fdrcorrection(p_sinLR[:,p])
#     p_sinLR[:,p] = p_corr
 
# t_sinstR = EvokedArray(t_sinstR, info, tmin=0)
# t_stLR = EvokedArray(t_stLR, info, tmin=0) 
# t_sinstL = EvokedArray(t_sinstL, info, tmin=0)
# t_sinLR = EvokedArray(t_sinLR, info, tmin=0)    

# mask_sinstR = p_sinstR <= 0.05
# mask_stLR = p_stLR <=0.05
# mask_sinstL = p_sinstL <= 0.05
# mask_sinLR = p_sinLR <=0.05

# fig6 = t_sinstR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinstR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig7 = t_sinstL.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinstL,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig8 = t_sinLR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinLR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig9 = t_stLR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_stLR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)

# #young
# p_th_stLR = [] ; t_th_stLR  = [] 
# p_a_stLR = [] ; t_a_stLR  = [] 
# p_b1_stLR = [] ; t_b1_stLR  = [] 
# p_b2_stLR = [] ; t_b2_stLR  = [] 

# p_th_sinstR = [] ; t_th_sinstR  = [] 
# p_a_sinstR = [] ; t_a_sinstR = [] 
# p_b1_sinstR = [] ; t_b1_sinstR  = [] 
# p_b2_sinstR = [] ; t_b2_sinstR  = [] 

# p_th_sinLR = [] ; t_th_sinLR  = [] 
# p_a_sinLR = [] ; t_a_sinLR  = [] 
# p_b1_sinLR = [] ; t_b1_sinLR  = [] 
# p_b2_sinLR = [] ; t_b2_sinLR  = [] 

# p_th_sinstL = [] ; t_th_sinstL  = [] 
# p_a_sinstL = [] ; t_a_sinstL = [] 
# p_b1_sinstL = [] ; t_b1_sinstL  = [] 
# p_b2_sinstL = [] ; t_b2_sinstL  = [] 




# for i in range(32):
#     #steady left
#     (p_, t_) = one_sample(ystl_theta.values[i,:], ystr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_stLR.append(p_); t_th_stLR.append(t_)   

#     (p_, t_) = one_sample(ystl_alpha.values[i,:], ystr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_stLR.append(p_); t_a_stLR.append(t_) 
    
#     (p_, t_) = one_sample(ystl_beta1.values[i,:], ystr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_stLR.append(p_); t_b1_stLR.append(t_) 
    
#     (p_, t_) = one_sample(ystl_beta2.values[i,:], ystr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_stLR.append(p_); t_b2_stLR.append(t_) 
    
#     #steady right
#     (p_, t_) = one_sample(ysinr_theta.values[i,:], ystr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinstR.append(p_); t_th_sinstR.append(t_)   

#     (p_, t_) = one_sample(ysinr_alpha.values[i,:], ystr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinstR.append(p_); t_a_sinstR.append(t_) 
    
#     (p_, t_) = one_sample(ysinr_beta1.values[i,:], ystr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinstR.append(p_); t_b1_sinstR.append(t_) 
    
#     (p_, t_) = one_sample(ysinr_beta2.values[i,:], ystr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinstR.append(p_); t_b2_sinstR.append(t_) 
    
#     #sinus left
#     (p_, t_) = one_sample(ysinl_theta.values[i,:], ysinr_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinLR.append(p_); t_th_sinLR.append(t_)   

#     (p_, t_) = one_sample(ysinl_alpha.values[i,:], ysinr_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinLR.append(p_); t_a_sinLR.append(t_) 
    
#     (p_, t_) = one_sample(ysinl_beta1.values[i,:], ysinr_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinLR.append(p_); t_b1_sinLR.append(t_) 
    
#     (p_, t_) = one_sample(ysinl_beta2.values[i,:], ysinr_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinLR.append(p_); t_b2_sinLR.append(t_) 
    
#     #sinus right
#     (p_, t_) = one_sample(ysinl_theta.values[i,:], ystl_theta.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_th_sinstL.append(p_); t_th_sinstL.append(t_)   

#     (p_, t_) = one_sample(ysinl_alpha.values[i,:], ystl_alpha.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_a_sinstL.append(p_); t_a_sinstL.append(t_) 
    
#     (p_, t_) = one_sample(ysinl_beta1.values[i,:], ystl_beta1.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b1_sinstL.append(p_); t_b1_sinstL.append(t_) 
    
#     (p_, t_) = one_sample(ysinl_beta2.values[i,:], ystl_beta2.values[i,:], reps = 1000, stat='t',alternative="two-sided", seed=4)
#     p_b2_sinstL.append(p_); t_b2_sinstL.append(t_) 
        
# p_sinstR = np.c_[p_th_sinstR,p_a_sinstR,p_b1_sinstR, p_b2_sinstR]
# p_stLR = np.c_[p_th_stLR,p_a_stLR,p_b1_stLR, p_b2_stLR]
# p_sinstL = np.c_[p_th_sinstL,p_a_sinstL,p_b1_sinstL, p_b2_sinstL]
# p_sinLR = np.c_[p_th_sinLR,p_a_sinLR,p_b1_sinLR, p_b2_sinLR]
# p_rest = np.c_[p_th_rest,p_a_rest,p_b1_rest, p_b2_rest]
# t_sinstR = np.c_[t_th_sinstR,t_a_sinstR,t_b1_sinstR, t_b2_sinstR]
# t_stLR = np.c_[t_th_stLR,t_a_stLR,t_b1_stLR, t_b2_stLR]
# t_sinstL = np.c_[t_th_sinstL,t_a_sinstL,t_b1_sinstL, t_b2_sinstL]
# t_sinLR = np.c_[t_th_sinLR,t_a_sinLR,t_b1_sinLR, t_b2_sinLR]

# for p in range(4): 
#     _, p_corr = fdrcorrection(p_sinstR[:,p])
#     p_sinstR[:,p] = p_corr
#     _, p_corr = fdrcorrection(p_stLR[:,p])
#     p_stLR[:,p] = p_corr
#     _, p_corr = fdrcorrection(p_sinstL[:,p])
#     p_sinstL[:,p] = p_corr    
#     _, p_corr = fdrcorrection(p_sinLR[:,p])
#     p_sinLR[:,p] = p_corr
 
# t_sinstR = EvokedArray(t_sinstR, info, tmin=0)
# t_stLR = EvokedArray(t_stLR, info, tmin=0) 
# t_sinstL = EvokedArray(t_sinstL, info, tmin=0)
# t_sinLR = EvokedArray(t_sinLR, info, tmin=0)    

# mask_sinstR = p_sinstR <= 0.05
# mask_stLR = p_stLR <=0.05
# mask_sinstL = p_sinstL <= 0.05
# mask_sinLR = p_sinLR <=0.05

# fig10 = t_sinstR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinstR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig11 = t_sinstL.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinstL,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig12 = t_sinLR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_sinLR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)


# fig13 = t_stLR.plot_topomap(ch_type='eeg', scalings=1,
#                     time_format=' ', vmin=-4.5, vmax=4.5,
#                     units='t_values', mask=mask_stLR,
#                     size=3,
#                     time_unit='s', title = None, nrows = 4, res = 1000)