# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob 
import pandas as pd
import pickle 
from random  import seed, sample, shuffle
import numpy as np 

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mne.decoding import CSP

np.random.seed(777)
seed(777)

def csp_cassification(data, task_list = [], merge_labels = []):
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''   
    metrics_ = [] 
    conf_mat = [] 
    epochs = data.epochs[task_list]
    epochs_data = epochs.get_data()
    y = epochs.events[:, -1]
    
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    param_grid = {'LDA__solver': ['lsqr','eigen'], 
                  'LDA__shrinkage': ['auto'], 
                  'LDA__n_components' : [1]}
    
    steps = [('CSP', csp),
             ('LDA',  LinearDiscriminantAnalysis())]
    
    cv = StratifiedShuffleSplit(10, test_size= 0.2)
    if len(merge_labels) > 0:
        # calculate labels p
        for n,l in enumerate(merge_labels):
            y[(y == l[0]) | (y == l[1])] = n
        samp_size = 40 * len(np.unique(y))
        train_size = int((samp_size/len(np.unique(y)))*(0.8))
        test_size = int((samp_size/len(np.unique(y)))*(0.2))
    for train,test in cv.split(epochs_data, y):
        # in case of merging get the same number per condition in merged conditions 
        if len(merge_labels) > 0:
            train_picks = [] 
            test_picks = [] 
            for n in np.unique(y):
                train_entry = np.where(y[train]==n)
                train_pick = sample(list(train_entry[0]), 
                                    k = train_size)
                    
                test_entry = np.where(y[test]==n)
                test_pick = sample(list(test_entry[0]), 
                                       k = test_size)
                    
                train_picks.append(train_pick); test_picks.append(test_pick)
            train_picks = np.ravel(train_picks); test_picks = np.ravel(test_picks);
            train = train[train_picks]; test = test[test_picks]
            shuffle(train); shuffle(test)
        X_train = epochs_data[train]
        X_test = epochs_data[test]
        y_train = y[train]
        y_test = y[test] 
        pipeline = Pipeline(steps)
        GS = GridSearchCV(pipeline, param_grid = param_grid, cv = cv)
        try:
            GS.fit(X_train, y_train)
            y_pred = GS.predict(X_test)
            metrics_.append(metrics.classification_report(y_test, y_pred, output_dict=True))
            conf_mat.append(confusion_matrix(y_test, y_pred, normalize = 'true'))
        except: 
            continue 
    return(metrics_, conf_mat)
    
wd = '/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Preprocessing/preprocessed_python/'
files = glob.glob(wd + '*.pkl')
metrics_all = [] 
conf_mat_all = [] 
metrics_lr = [] 
conf_mat_lr = [] 
metrics_sinst = [] 
conf_mat_sinst = [] 
metrics_sinlr = [] 
conf_mat_sinlr = []
metrics_stlr = [] 
conf_mat_stlr = []
metrics_sinstl = [] 
conf_mat_sinstl = []
metrics_sinstr = [] 
conf_mat_sinstr  = []

for file in files :
    with open(file, 'rb') as handle:
        data = pickle.load(handle)
        met1, conf_mat1 = csp_cassification(data, task_list = ['Sinus_left','Sinus_right','Steady_left','Steady_right'])
        met2, conf_mat2 = csp_cassification(data, task_list = ['Sinus_left','Sinus_right','Steady_left','Steady_right'],
                                            merge_labels = [[1,2],[3,4]])
        met3, conf_mat3 = csp_cassification(data, task_list = ['Sinus_left','Sinus_right','Steady_left','Steady_right'],
                                            merge_labels = [[1,3],[2,4]])
        met4, conf_mat4 = csp_cassification(data, task_list = ['Sinus_left','Sinus_right'])
        met5, conf_mat5 = csp_cassification(data, task_list = ['Sinus_right','Steady_right'])
        met6, conf_mat6 = csp_cassification(data, task_list = ['Sinus_left','Steady_left']) 
        met7, conf_mat7 = csp_cassification(data, task_list = ['Sinus_right','Steady_right'])                              
        
        metrics_all.append(met1)
        conf_mat_all.append(conf_mat1)
        metrics_lr.append(met2)
        conf_mat_lr.append(conf_mat2)
        metrics_sinst.append(met3)
        conf_mat_sinst.append(conf_mat3)
        metrics_sinlr.append(met4)
        conf_mat_sinlr.append(conf_mat4)
        metrics_stlr.append(met5)
        conf_mat_stlr.append(conf_mat5)
        metrics_sinstl.append(met6)
        conf_mat_sinstl.append(conf_mat6)
        metrics_sinstr.append(met7)
        conf_mat_sinstr .append(conf_mat7)
        
        
metrics_ = {'All': conf_mat_all, 'lr': metrics_lr, 'SinSt': metrics_sinst,
            'Sin_lr': metrics_sinlr, 'St_lr': metrics_stlr,'SinSt_l': metrics_sinstl,
           'SinSt_r': metrics_sinstr}
conf_mat = {'All': metrics_all, 'lr': conf_mat_lr, 'SinSt': conf_mat_sinst,
            'Sin_lr': conf_mat_sinlr, 'St_lr': conf_mat_stlr,'SinSt_l': conf_mat_sinstl,
           'SinSt_r': conf_mat_sinstr}

with open('/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP/classicCSP_metrics.pkl', 'wb') as handle:
    pickle.dump(metrics_,handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/Volumes/SEAGATE_BAC/Hard_drive/Project2_DMDMotorControl/Results/EEG/Classification/Results_raw/EXCLUDE_Z/2CSP/classicCSP_conf_mat.pkl', 'wb') as handle:
    pickle.dump(conf_mat,handle, protocol=pickle.HIGHEST_PROTOCOL)