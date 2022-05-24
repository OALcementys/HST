#!/usr/bin/env python
# coding: utf-8

# In[142]:

import json
import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
import time
import torch
from torch.optim.lr_scheduler import *
from sklearn.linear_model import LinearRegression as LR
from sklearn.impute import KNNImputer
from torch.autograd import Variable
from sklearn.metrics import *
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import time


"""
The main part of the HST method have been implemented following Luc Chouinard and Vincnet Roy paper :
'PERFORMANCE OF STATISTICAL MODELS FOR DAM MONITORING DATA '
"""


# checking nan values for data frame
def check_nan_values(dataframe):
    for column in dataframe.columns:
        check=dataframe[column].isnull().values.any()
        if check==True:
            print(column, 'missing data: ',dataframe[column].isnull().sum())
    else : print('dataframe on point')

"""
time prep is computed for standard scaled time and for seasonal term whether it's used or not
"""
def time_prep(dataframe,column='Time'):
    window_observation=dataframe[column][len(dataframe)-1]-dataframe[column][0]
    dataframe['scaled_time']=dataframe[column].apply(lambda T :((T-dataframe[column][0]).total_seconds())/window_observation.total_seconds())
    ## preaparing seasonal time for linear combination of sinusoidal functions
    dataframe['seasonal_time']=dataframe[column].apply(lambda T :(2*np.pi*(T-dataframe[column][0]).total_seconds())/365*24*60*60)
    return dataframe

"""
fixed order of hydrostatic load term for multi linear regression  to 4
"""
def hydrostatic_load(dataframe,wl_column,regression_order=4):
    hydrostaic_columns=[]
    max_observation=np.max(dataframe[wl_column])
    min_observation=np.min(dataframe[wl_column])
    scaled_observation=(max_observation-dataframe[wl_column])/(max_observation-min_observation)
    dataframe['scaled_observation']=scaled_observation
    for k in range(regression_order+1):
        order='HL_'+str(k)
        dataframe[order]=scaled_observation**k
        hydrostaic_columns.append(order)
    return dataframe,hydrostaic_columns
#seasonal term
"""
Possible later computation of a miniman time threshold for including seasonal componnent"""
def seasonal_term(dataframe):
    #scaled_time=dataframe['seasonal_time']
    season_columns=['sin','cos','sin2','sincos']
    dataframe[season_columns[0]]=np.sin(dataframe['seasonal_time'])
    dataframe[season_columns[1]]=np.cos(dataframe['seasonal_time'])
    dataframe[season_columns[2]]=np.sin(dataframe['seasonal_time'])**2
    dataframe[season_columns[3]]=np.sin(dataframe['seasonal_time'])*np.cos(dataframe['seasonal_time'])
    return dataframe,season_columns

#long term time influence


def Time_influence(dataframe):
    Time_columns=['T1','T2','T3']
    dataframe[Time_columns[0]]=dataframe['scaled_time']
    dataframe[Time_columns[1]]=np.exp(dataframe['scaled_time'])
    dataframe[Time_columns[2]]=np.exp((-1)*dataframe['scaled_time'])
    return dataframe , Time_columns

"""
Following "Hybrid GA/SIMPLS as alternative regression model in dam deformation analysis" paper
the temperature componenent is computed conditionnaly on its observation existance. if it doesn't
we compute a sum of sin and cos based on seasonal time
"""

def temperature_component(data,temp_column):
    if temp_column is not None :
        scale=np.max(data[temp_column])-np.min(data[temp_column])
        data['temperature']=data[temp_column]-np.mean(data[temp_column])
        data['temperature']/=scale
        return data,['temperature']
    else:
        Temp_columns=[]
        for k in range(1,6):
            data['sin_temp_'+str(k)]=np.sin(k*data['seasonal_time'])
            data['cos_temp_'+str(k)]=np.cos(k*data['seasonal_time'])
            Temp_columns.append('sin_temp_'+str(k),'cos_temp_'+str(k))
    return data ,Temp_columns


"""
We normalize Custom variable components that can be added to the model
"""
def variable_component(data,variable_column):
    scale=np.max(data[variable_column])-np.min(data[variable_column])
    if scale==0:
        data[variable_column]=data[variable_column]-np.mean(data[variable_column])
    else:
        data[variable_column]=data[variable_column]-np.mean(data[variable_column])
        data[variable_column]/=scale
    return data , [variable_column]



##Dataframe cleaning and prep for model
def prepare_prediction(dataframe,model):
    #assert len(model_columns)==len(model)
    df=dataframe.copy()
    df=time_prep(df)
    kept_columns=[]
    static_colums=['H','S','T','temp']
    if 'H' in model :
        df,columns=hydrostatic_load(df,'water_level',regression_order=4)
        kept_columns+=columns
    if 'S' in model:
        df,columns=seasonal_term(df)
        kept_columns+=columns
    if 'T' in model :
        df,columns=Time_influence(df)
        kept_columns+=columns
    if 'temp' in model :
        df,columns=temperature_component(df,'temperature')
        kept_columns+=columns

    added_variables=list(set(model)-set(np.intersect1d(static_colums,model)))
    if len(added_variables)>0 :
        for i in range(len(added_variables)):
            df,columns=variable_component(df,added_variables[i])
            kept_columns+=columns
    return df[kept_columns]


## Linear regression model
def linear_regression(data,target):
    start=time.time()
    regr =LinearRegression()
    regr.fit(data,target)
    return regr



##Computing k fold cross validation
def regression_kfold(matrix,target,n_splits):
    cv_outer = KFold(n_splits, shuffle=True, random_state=0)
    # enumerate splits
    outer_results = list()
    X=matrix.T
    y=target
    best_score=np.NINF
    best_regr=0,0
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        #cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        #__,weights,biais=optimizer_variation(X_train.T,  y_train,  'Adam',  lr=0.1,  num_epoch=2500)
        regr=linear_regression(X_train,y_train)
        y_hat=regr.predict(X_test)
        score=r2_score(y_hat,y_test)
        """
        y_hat=weights.mm(torch.Tensor(X_test.T)) + biais
        y_hat=y_hat.squeeze()
        score=r2_score(y_test,y_hat.detach().numpy())"""

        if score>best_score :
            best_score=score
            best_regr=regr
    #print(f"final score : {score :.3f}")
    return best_regr



##best model
def standard_regression(data,target,use_cross_val):
    if use_cross_val:
        regr=regression_kfold(data.T,target,n_splits=3)
        y_hat=regr.predict(data)
        ##print( 'regr score for '+target_column+'is:' , regr.score(dataframe,monitored_[target_column]))
        #dataframe.insert(0,'corrected observation',monitored_[target_column] -np.array(regr.predict(dataframe)))
        #dataframe.insert(len(dataframe.columns),target_column,monitored_[target_column])
    else:
        regr=linear_regression(data,target)
        y_hat=regr.predict(data)
    return y_hat



# Main optimization loop

##Using a NN layer for optimizer switching
def optimizer_variation(matrix,  target,  optimizer,  lr=0.01,  num_epoch=2500):
    A = torch.randn((1, matrix.shape[0]), requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    if optimizer=='SGD':
        opt = torch.optim.SGD([A, b], lr=lr)
    elif optimizer=='Adam' :
        opt=torch.optim.Adam([A, b], lr=lr)
    elif optimizer=='AdamW':
        opt=torch.optim.AdamW([A, b], lr=lr)
    elif optimizer=='Adamax':
        opt=torch.optim.AdamW([A, b], lr=lr)
    elif optimizer=='ASGD':
        opt=torch.optim.ASGD([A, b], lr=lr)
    elif optimizer=='Adagrad':
        opt=torch.optim.Adagrad([A, b], lr=lr)
    ##setting up a learning rate scheduler
    criterion=torch.nn.MSELoss()
    scheduler = ExponentialLR(opt, gamma=0.1)
    ##Multi Linear type model
    def NN(x_input):
        return A.mm(x_input) + b
    best_score=np.NINF
    best_loss=np.inf
    best_pred=0
    ##Trainning loop
    for t in range(0,num_epoch):
        opt.zero_grad()
        y_predicted = NN(torch.Tensor(matrix))
        current_loss = criterion(y_predicted, torch.Tensor(target))
        current_loss.backward()
        opt.step()
        pred=y_predicted.detach().numpy().squeeze()
        score=r2_score(target,pred)
        ##keeping best model
        if score>best_score :
            best_score=score
            best_pred=target-pred
            best_loss=current_loss
            weights=A
            biais=b
    #print(f"t = {t}, loss: {best_loss:.2F}, score : {score :.3f}")
    prediction=NN(torch.Tensor(matrix))
    return prediction,weights,biais


# create dataset
# configure the cross-validation procedure
def k_fold_cross(matrix,target,optimizer):
    cv_outer = KFold(n_splits=6, shuffle=True, random_state=0)
    # enumerate splits
    outer_results = list()
    X=matrix.T
    y=target
    best_score=np.NINF
    best_weights,best_biais=0,0
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        #cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        __,weights,biais=optimizer_variation(X_train.T,  y_train,  optimizer,  lr=0.1,  num_epoch=1000)
        y_hat=weights.mm(torch.Tensor(X_test.T)) + biais
        y_hat=y_hat.squeeze()
        score=r2_score(y_test,y_hat.detach().numpy())

        if score>best_score :
            best_score=score
            best_weights=weights
            best_biais=biais
    #print(f"final score : {score :.3f}")
    final_pred=weights.mm(torch.Tensor(matrix)) + biais
    return final_pred


# In[220]:

###############################################

"""
Bayesian Approch implementation
Inspired by https://zjost.github.io/bayesian-linear-regression/
We consider the error follwing a normal distribution and compute  bayesian theory for regression
"""
def calculate_w(l, phi, t, N_rows, M_cols):
    LI = np.eye(M_cols) * l
    innerPrdt = (LI + np.matmul(np.transpose(phi),phi))
    w = np.matmul(inv(innerPrdt),np.matmul(np.transpose(phi),t))
    return w
def bayesian_model_selection(train, trainR, test, testR):
    start_time = time.time()
    alpha = 2.34
    beta = 3.22
    prev_alpha = 0
    prev_beta = 0
    N_rows_train = np.shape(train)[0]
    M_features_train = np.shape(train)[1]
    N_rows_test = np.shape(test)[0]
    i = 0
    while abs(prev_alpha-alpha) > 0.0001 and abs(prev_beta-beta) > 0.0001:
        prev_alpha = alpha
        prev_beta = beta
        eigen_phiT_phi = LA.eigvals(np.matmul(np.transpose(train),train))
        Sn_inv = alpha * np.eye(M_features_train) + beta * np.matmul(np.transpose(train),train)
        Mn = beta * np.matmul(inv(Sn_inv),np.matmul(np.transpose(train),trainR))
        lamda = beta * eigen_phiT_phi
        alpha_lamda = LA.eigvals(Sn_inv)
        gamma = sum(lamda/alpha_lamda)
        alpha = gamma / np.matmul(np.transpose(Mn),Mn)
        beta = 1/(sum((trainR - np.matmul(train,Mn)) ** 2) / (N_rows_train-gamma))
        i+=1

    w = calculate_w(alpha/beta,train,trainR,N_rows_train,M_features_train)
    mse = (sum((np.matmul(test,w) - testR) ** 2) / N_rows_test)
    #print("computed bayesian model in ", time.time()-start_time)
    return mse,w

##Kfold on bayesian model
def bayesian_selection(matrix,target,n_splits=2):
    X=matrix.T
    y=target

    if n_splits==1 :
        X_train, y_train, X_test, y_test=train_test_split(X,y,test_size = 0.25,shuffle=False)
        return bayesian_model_selection(X_train, y_train, X_test, y_test)
    else:
        cv_outer = KFold(n_splits=n_splits, shuffle=True, random_state=0)
            # enumerate splits

        best_score=np.NINF
        best_weights,best_biais=0,0
        best_score=np.NINF
        best_weight=0
        for train_ix, test_ix in cv_outer.split(X):
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            mse,weight=bayesian_model_selection(X_train, y_train, X_test, y_test)
            if mse>best_score :
                best_score=mse
                best_weight=weight
        best_weight=best_weight.reshape(best_weight.shape[0])
    return np.dot(X,best_weight)



#######################

"""
summarizing function
"""
def deploy(model,data,method='use_NN_optimizer',optimizer=None,target_column='target_variable'):

    methods_list=['std_regression',
              'std_regression_cross',
              'use_NN_optimizer',
              'use_NN_optimizer_cross',
             'use_bayesian_approch']
    optimizers_list=['SGD','Adam', 'AdamW','Adamax','ASGD','Adagrad']
    #assert method in methods_list
    matrix=np.array(prepare_prediction(data,model)).T
    target=np.array(data[target_column])
    dataframe=data.copy()
    if method==methods_list[0]:

        corrected_obs=standard_regression(matrix.T,target,use_cross_val=False)
        dataframe.insert(0,'corrected observation',data[target_column] -np.array(corrected_obs))
    elif  method==methods_list[1]:
        corrected_obs=standard_regression(np.transpose(matrix),target,use_cross_val=True)
        dataframe.insert(0,'corrected observation',data[target_column] -np.array(corrected_obs))
    elif  method==methods_list[2]:
        if optimizer is not None :
            assert optimizer in optimizers_list
            corrected_obs,__,__=optimizer_variation(matrix,  target,  optimizer,  lr=0.1,  num_epoch=2500)
            dataframe.insert(0,'corrected observation',data[target_column] -corrected_obs.squeeze().detach().numpy())
        else:
            corrected_obs,__,__=optimizer_variation(matrix,  target,  optimizer='SGD',  lr=0.1,  num_epoch=2500)
            dataframe.insert(0,'corrected observation',data[target_column] -corrected_obs.squeeze().detach().numpy())
    elif method==methods_list[3]:
        if optimizer is not None :
            assert optimizer in optimizers_list
            corrected_obs=k_fold_cross(matrix,target,optimizer)
            dataframe.insert(0,'corrected observation',data[target_column] -corrected_obs.squeeze().detach().numpy())
        else:
            corrected_obs=k_fold_cross(matrix,target,optimizer='SGD')
            dataframe.insert(0,'corrected observation',data[target_column] -corrected_obs.squeeze().detach().numpy())
    elif method==methods_list[4]:
        corrected_obs=bayesian_selection(matrix,target,n_splits=2)
        dataframe.insert(0,'corrected observation',data[target_column] -np.array(corrected_obs))
    #print(dataframe.shape)
    return dataframe




###time handling relative to merging data
def normalize_date_time(dataframes):
    for data in dataframes:
        data['timestamp'] = pd.to_datetime(data['timestamp'].astype('datetime64[ns]'))
    return dataframes

"""
interpolation techniques
---we used tolerance sampling with a window relative to data lengh in a way that two diffrent observation in a same window will have the same timestamp
--- second approch will be interpolation relative to time stamp (linear)
"""
def time_interpolation(d1,d2,option):
    if option==None:option='tolerance_sampling'
    if option=='upsampling':
        df=pd.concat([d1, d2])
        dataframes=pd.DataFrame(df['timestamp'])
        dataframes.set_index(df['timestamp'])
        df=df.set_index('timestamp')
        df = df.sort_values(by="timestamp")
        for column in df.columns:
            d1=df[column]
            dataframes[column]=pd.DataFrame(d1.interpolate()).values
        #merged['timestamp']=merged.index
        dataframes=dataframes.reset_index(drop=True)
        return dataframes

    elif  option=='tolerance_sampling':
        lens={len(d1):d1,len(d2):d2}
        maxdf=lens[max(len(d1),len(d2))]
        mindf=lens[min(len(d1),len(d2))]
        kept_columns=maxdf.columns.append(mindf.columns)
        maxdf.index = maxdf['timestamp']
        mindf.index = mindf['timestamp']
        maxdf=maxdf.set_index('timestamp')
        mindf=mindf.set_index('timestamp')
        kept_columns=maxdf.columns.append(mindf.columns)
        maxdf = maxdf.sort_values(by="timestamp")
        mindf = mindf.sort_values(by="timestamp")
        tol = (maxdf.index[-1]-maxdf.index[0])/len(maxdf)
        merged=pd.merge_asof(left=maxdf,right=mindf,right_index=True,
        left_index=True,direction='nearest',tolerance=tol)
        merged=merged.dropna()
        merged['timestamp']=merged.index
        merged=merged.reset_index(drop=True)
        return merged
    elif option=='downsampling':
        print('not coded yet')
        quit()


def datetime_interploation(dataframes,option):
    df=dataframes[0]
    for data in dataframes[1:]:
        df=time_interpolation(df,data,option)
    return df


def final_prep(dataframes,option='tolerance sampling'):
    target_column='target_variable'
    dataframes=normalize_date_time(dataframes)
    df=datetime_interploation(dataframes,option)
    df=df.dropna()
    df = df.rename(index=str,columns= {'timestamp': "Time"})
    df['Time'] = pd.to_datetime(df['Time'])
    #model_columns=list(df.columns)
    #model_columns.remove(target_column)
    #return df,model_columns
    return df


def prepare_corrected_obs(res,id_res_var):
    df_res  = res[['Time','corrected observation']]
    df_res=df_res.rename(index=str,columns= {'Time': "timestamp"})
    df_res=df_res.rename(index=str,columns= {'corrected observation': "value"})
    df_res['variable_id'] = id_res_var
    return df_res
##function to retrieve variables
def get_targetvr_df(queryManager,dataframes,id_tv_var):
    target_column= 'target_variable'
    target_variable_name=queryManager.get_variable_name_by_id(id_tv_var)
    df_def = queryManager.get_df_by_variable_id(id_tv_var, target_column)
    dataframes.append(df_def)
    return dataframes
def get_temp_df(queryManager,dataframes,id_temperature_var):
    df_temp= queryManager.get_df_by_variable_id(id_temperature_var, 'temperature')
    assert queryManager.get_variable_metric(id_temperature_var)=='°C'
    dataframes.append(df_temp)
    return dataframes
def get_waterlevel_df(queryManager,dataframes,id_water_level_var):
    df_def = queryManager.get_df_by_variable_id(id_water_level_var, 'water_level')
    assert queryManager.get_variable_metric(id_water_level_var)=='mA'
    dataframes.append(df_def)
    return dataframes
def get_var_df(queryManager,dataframes,id_tv_var,model):
    variable_name=queryManager.get_variable_name_by_id(id_tv_var)
    df_def = queryManager.get_df_by_variable_id(id_tv_var, variable_name)
    model.append(variable_name)
    dataframes.append(df_def)
    return dataframes,model

##data is the json file : we extract variables upon it
def get_model_comps(queryManager,data):
    for model in data:
        static_colums=['H','S','T','temp','target']
        model_comp=[]
        dataframes=[]
        variables,advanced=data['model']
        dataframes=get_targetvr_df(queryManager,dataframes,variables['target'])
        if variables['T']:model_comp.append('T')
        if variables['S']:model_comp.append('S')
        if 'H' in variables :
            dataframes=get_waterlevel_df(queryManager,dataframes,variables['H'])
            model_comp.append('H')
        if 'temp' in variables :
                dataframes=get_temp_df(queryManager,dataframes,variables['temp'])
                model_comp.append('temp')
        crops=list(variables.keys())
        added_variables=list(set(crops)-set(np.intersect1d(static_colums,crops)))
        if len(added_variables)>0 :
            for i in range(len(added_variables)):
                dataframes,model_comp=get_var_df(queryManager,dataframes,variables[added_variables[i]],model_comp)
        return dataframes,model_comp,advanced

#final function
def prepare_res(queryManager,data):
    start=time.time()
    print('computation started : ' ,  time.strftime("%c"))
    dataframes,model_comp,advanced=get_model_comps(queryManager,data)
    __,advanced=data['model']
    option,method,optimizer=advanced['interpolation'],advanced['Approch'],advanced['Optimizer']
    df=final_prep(dataframes,option=option)
    res=deploy(model_comp,df,method=method,optimizer=optimizer)
    print('computation took {:.3f} seconds ' .format(time.time()-start))
    id_res_var=advanced['res_id']
    df_res =prepare_corrected_obs(res,id_res_var)
    return df_res,id_res_var


"""
json file form example
data={'model': [{'target': 1238,
   'H': 1246,
   'T': True,
   'S': True,
   'temp': 1245,
   'V': 1238},
  {'Approch': 'use_NN_optimizer',
   'Optimizer': 'SGD',
   'loss': 'MSE',
   'metric': 'R2',
   'interpolation': 'upsampling',
   'res_id': 112}]}



event={'body':data}
   """



# In[223]:





# In[ ]:
