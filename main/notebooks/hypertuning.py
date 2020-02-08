from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import h5py
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
import pywt
from plotly import tools
from plotly.offline import init_notebook_mode, iplot, plot
# import cufflinks as cf
import xgboost as xgb
import plotly.graph_objs as go
from datetime import datetime
from datetime import timedelta
import warnings
import time
from collections import Counter
import scipy
import pywt
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelBinarizer


X_train = h5py.File("original_data/X_train.h5", "r")
y_train_ = pd.read_csv("original_data/y_train.csv").as_matrix()[:, 1].squeeze()
df_train = pd.DataFrame(data=X_train["features"][:])

def label_columns(df):
    l_columns = ['num_pso', 
             'mean_amp_pso',
             'mean_dur_pso',
              'amp_cso',
              'dur_cso',
              'curr_sleep_stage',
              'time_since_sleep',
              'time_in_ds',
              'time_in_ls',
              'time_in_rs',
              'time_in_ws']
    for i in range(12, 1261+1):
        l_columns.append('eeg_signal_%s'%(i-12+1))
    df.columns = l_columns

    return df

df_train=label_columns(df_train)

df_train_final = pd.read_hdf('df_train_final.h5', 'df_train_final')

equations = {
  
     'base_model': {'predictors': list(df_train_final.columns)[0:13]},
     'entropy_fft': {'predictors': list(df_train_final.columns[0:81])},
     'dummy_variables_only': {'predictors': list(df_train_final.columns)[0:11]+list(df_train_final.columns)[81:83]},
     'dummy_variables': {'predictors': list(df_train_final.columns)[0:83]},
     'dummy_variables_target_encoding': {'predictors':list(df_train_final.columns)[0:81] + [df_train_final.columns[-1]]},
    
     'wawelets': {'predictors': list(df_train_final.columns[0:17])+list(df_train_final.columns[85:181])},
     #'wawelets_spindles':{'predictors': list(df_train_final.columns[0:17])+list(df_train_final.columns[85:182])},
     'wawelets_target_encoding': {'predictors': list(df_train_final.columns[0:17])+list(df_train_final.columns[85:181])
                                  + [df_train_final.columns[-1]]},
     'wawelets_dummies': {'predictors': list(df_train_final.columns[0:17])+list(df_train_final.columns[81:83])
                                  + list(df_train_final.columns[85:181])},
     'wawelets_SO_detected': {'predictors': list(df_train_final.columns[0:17])+list(df_train_final.columns[81:83])
                                  + list(df_train_final.columns[85:181]) + [df_train_final.columns[-1]]},

     'fft_wawelets' : {'predictors': list(df_train_final.columns[0:81])+list(df_train_final.columns[85:181])},
     #'fft_wawelets_spindles': {'predictors': list(df_train_final.columns[0:81])+list(df_train_final.columns[85:182])},

#fft_2_sec
     'entropy_fft_2_sec':{'predictors': list(df_train_final.columns)[0:17] +list(df_train_final.columns)[183:246]}, 
     'fft_2_sec_wavelets':{'predictors': list(df_train_final.columns)[0:17] + list(df_train_final.columns[85:181]) +
                           list(df_train_final.columns)[183:246]},
     'fft_fft_2_sec_wavelets':{'predictors': list(df_train_final.columns)[0:81] + list(df_train_final.columns[85:181]) +
                           list(df_train_final.columns)[183:246]},

#wavelets_2_sec
    'wawelets_2_sec':{'predictors': list(df_train_final.columns)[0:17] +list(df_train_final.columns)[246:306]}, 
    'wavelets_wavelets_2_sec': {'predictors': list(df_train_final.columns)[0:17]  + list(df_train_final.columns[85:181]) 
                                + list(df_train_final.columns)[246:306]},
    
    'fft_wawelets_2_sec':{'predictors': list(df_train_final.columns)[0:81] +list(df_train_final.columns)[246:306]}, 
    'fft_2_sec_wawelets_2_sec':{'predictors': list(df_train_final.columns)[0:17] +list(df_train_final.columns)[183:306]},
    'fft_fft_2_sec_wawelets_2_sec':{'predictors': list(df_train_final.columns)[0:81] + list(df_train_final.columns)[183:306]}, 
    
    'fft_fft_2_sec_wawelets_wawelets_2_sec':{'predictors': list(df_train_final.columns)[0:81] + list(df_train_final.columns[85:181]) 
                                             +list(df_train_final.columns)[183:306]},
}

cols_eq = list(equations.keys())

var_to_pred = 'SO'

X_train, X_val, y_train, y_val = train_test_split(df_train_final.loc[:, df_train_final.columns != var_to_pred], df_train_final[var_to_pred], test_size=0.10,random_state=0, stratify=df_train_final[var_to_pred])

y_train = pd.DataFrame(y_train)
y_val = pd.DataFrame(y_val)

model='wawelets'
predictors = equations[model]['predictors']

eval_metric =  'error'
scoring = 'accuracy'

import boostparam #external python file to hypertune xgboost model
from importlib import reload
reload(boostparam)

df = pd.concat([X_train,y_train], axis=1)

params = boostparam.tune_all_hyper_params(df, predictors, var_to_pred, eval_metric, scoring, seed = 6)
model = xgboost.XGBClassifier(**params)
model.fit(df[predictors], df[var_to_pred])
xgboost.plot_importance(model)
y_pred = model.predict(X_test)

print('accuracy_score:', accuracy_score(y_val, y_pred))
