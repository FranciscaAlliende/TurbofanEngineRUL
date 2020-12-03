#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import ndimage
from pykalman import KalmanFilter
import random
from sklearn.metrics import mean_squared_error, r2_score

def import_(train, test, RUL):
    """function to import the data, the inputs are the names of the files in string format
    the output, 3 datasets, with column names and the all NaN value columns removed"""
    
    train = pd.read_csv("data/" + train, sep=" ", header = None)
    train.drop(train.columns[27], axis=1, inplace=True)
    train.drop(train.columns[26], axis=1, inplace=True)
    train.columns = ["unit_number", "n_cycles", "op_setting_1", "op_setting_2", "op_setting_3", 
                "sm_1", "sm_2", "sm_3","sm_4", "sm_5", "sm_6", "sm_7", "sm_8","sm_9", "sm_10", "sm_11", "sm_12", 
                "sm_13","sm_14", "sm_15", "sm_16", "sm_17", "sm_18","sm_19", "sm_20", "sm_21"]
    
    test = pd.read_csv("data/" + test, sep=" ", header = None)
    test.drop(test.columns[27], axis=1, inplace=True)
    test.drop(test.columns[26], axis=1, inplace=True)
    test.columns = ["unit_number", "n_cycles", "op_setting_1", "op_setting_2", "op_setting_3", 
                "sm_1", "sm_2", "sm_3","sm_4", "sm_5", "sm_6", "sm_7", "sm_8","sm_9", "sm_10", "sm_11", "sm_12", 
                "sm_13","sm_14", "sm_15", "sm_16", "sm_17", "sm_18","sm_19", "sm_20", "sm_21"]
    
    RUL = pd.read_csv("data/" + RUL, sep=" ", header = None)
    RUL.drop(RUL.columns[1], axis=1, inplace=True)
    RUL.columns = ["RUL"]
    
    return train, test, RUL

def drop_cols(S):
    col_drop = ["op_setting_1", "op_setting_2","op_setting_3",
                "sm_1", "sm_5", "sm_6", "sm_10", "sm_16", "sm_18", "sm_19"]
    return S.drop(columns = col_drop)

def apply_median_filter(S):
    # new names for columns to avoid confusion
    filter_cols = []
    for col in S.columns:
        if col[0] == "s":
            new_name = col+str("F")
        else:
            new_name = col
        filter_cols.append(new_name)
        
    # create empy dataframe and add the 2 firsts columns
    SF = pd.DataFrame(columns = filter_cols)
    SF["unit_number"] = S["unit_number"]
    SF["n_cycles"] = S["n_cycles"]
    
    # filter all sensor data, by unit and append it to new df
    sensors = S.columns[2::]
    units = S["unit_number"].unique()
    for sm in sensors:
        colF = []
        for unit in units:
            y = S[S["unit_number"]==unit][sm].values
            y_filter = ndimage.median_filter(y, size = 20)
            colF.append(y_filter)
        colF = np.concatenate(colF)
        SF[sm+str("F")] = colF
    return SF

def apply_kalman_filter(S):
    # new names for columns to avoid confusion
    filter_cols = []
    for col in S.columns:
        if col[0] == "s":
            new_name = col+str("F")
        else:
            new_name = col
        filter_cols.append(new_name)
        
    # create empy dataframe and add the 2 firsts columns
    SF = pd.DataFrame(columns = filter_cols)
    SF["unit_number"] = S["unit_number"]
    SF["n_cycles"] = S["n_cycles"]
    
    # filter all sensor data, by unit and append it to new df
    sensors = S.columns[2::]
    units = S["unit_number"].unique()
    for sm in sensors:
        colF = []
        for unit in units:
            y = S[S["unit_number"]==unit][sm].values
            kf = KalmanFilter(initial_state_mean=y[0], n_dim_obs=1)
            y_filter = kf.em(y, n_iter=10).smooth(y)[0].flatten()
            colF.append(y_filter)
        colF = np.concatenate(colF)
        SF[sm+str("F")] = colF
    return SF

def get_index(unit, cycle, S):
    """get the index of a row, based on the unit number and the cycle"""
    
    s1 = set(S.index[S['unit_number'] == unit].tolist())
    s2 = set(S.index[S['n_cycles'] == cycle].tolist())
    return list(s1 & s2)[0]

def linearRUL(train):
    """function that takes as an input the train set
    the output is the dataset with an extra column: RUL, caculated as the number of cycles before failure"""
    
    # create an empty column 
    trainLR = train.assign(linear_RUL = " ")
    cols = trainLR.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    trainLR = trainLR[cols]
    
    for unit in trainLR["unit_number"].unique():
        #last cycle
        maxcycle = max(trainLR[trainLR["unit_number"] == unit]["n_cycles"])

        # initial values
        idx = get_index(unit, maxcycle, trainLR)
        cycle = maxcycle
        value = 0

        # place a 0 on the last day (failure)
        trainLR.loc[idx, 'linear_RUL'] = int(value)

        while value < (maxcycle - 1):
            cycle -= 1
            value += 1
            idx = get_index(unit, cycle, trainLR)
            trainLR.loc[idx, 'linear_RUL'] = int(value)
        
    return trainLR

def clipRUL(train, t):
    # create dataset with linear RUL
    trainLR = linearRUL(train)
    units = train["unit_number"].unique()
    all_clip_rul = []
    for unit in units:
        clip_rul = trainLR[trainLR["unit_number"]==unit]["linear_RUL"].clip(upper = t).values
        all_clip_rul.append(clip_rul)
    all_clip_rul = np.concatenate(all_clip_rul)
    
    trainC = trainLR[trainLR.columns[1::]]
    trainC.insert(0, "clip_RUL", all_clip_rul)
    return trainC

def prepare_train(train, filter_, RUL_def, t):
    """ filter_ = median, kalman or None
        RUL_def = linear or clipped, if clipped, define t"""
    # drop columns
    train = drop_cols(train)
    
    # choose filter
    if filter_ == None:
        T = train
    elif filter_ == "median":
        T = apply_median_filter(train)
    elif filter_ == "kalman":
        T = apply_kalman_filter(train)
    else:
        print("provide a valid filter (None, median or kalman)")
    
    # choose RUL definition 
    if RUL_def == "linear":
        T = linearRUL(T)
    elif RUL_def == "clipped":
        T = clipRUL(T, t)
    else:
        print("provide a valid RUL definition (linear or clipped)")
    
    y_train = T.iloc[:,0].values
    cols = T.columns[3::]
    X_train = T[cols]
    
    return y_train, X_train

def prepare_test(test, filter_):
    # drop columns
    X_test = drop_cols(test)
    
    # choose filter
    if filter_ == None:
        T = X_test
    elif filter_ == "median":
        T = apply_median_filter(X_test)
    elif filter_ == "kalman":
        T = apply_kalman_filter(X_test)

    # Select just the last row for every unit to match results on the RUL column
    X_test = X_test.groupby("unit_number").last().reset_index()[X_test.columns[2::]]
    return X_test

def evaluate(y_true, y_pred):
    # evaluation metrics
    MSE = mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_true, y_pred)
    print("MSE", round(MSE,3))
    print("RMSE", round(RMSE,3))
    print("r2 score", round(R2,3))
    
    # plot prediction vs reality 1
    plt.plot(y_true, color = "g")
    plt.plot(y_pred, color = "black")
    plt.xlabel("units")
    plt.ylabel("RUL")
    plt.legend(["y_true", "y_pred"])
    plt.title("Prediction vs Reality sorted by unit number")
    plt.show()
    
    # plot prediction vs reality 2
    d = {"units": range(1,101), "y_true": y_true, "y_pred": y_pred}
    ev = pd.DataFrame(d)
    ev = ev.sort_values(by = "y_true", ascending = False)
    
    plt.plot(ev["y_true"].values, color = "g")
    plt.plot(ev["y_pred"].values, color = "black")
    plt.legend(["y_true", "y_pred"])
    plt.ylabel("RUL")
    plt.title("Prediction vs Reality sorted by real RUL")
    plt.show()
    return MSE, RMSE, R2

