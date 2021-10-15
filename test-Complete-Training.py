# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:22:33 2021

@author: Sheshank_Joshi
"""

#%% Importing Libraries - Data Ready
import pandas as pd
import numpy as np

df=pd.read_csv("dataset.csv")


#%%Importing Engines
import model_training_engine as mt
tr=mt.models_training_engine(df)

#%% Preparing for Training
tr.set_desc_column("Description")
tr.set_dep_target("Accident Level")
tr.set_target_column("Potential Accident Level")

#%% Complete Training Test.
tr.train()
#%% Language Model - Initialization
#tr._lm_initialize()
#%% Neural Network Initialization
#tr._NN_initialize()
#%% Supervised Model - Initialization
#tr._sup_initialize()
#%%
#tr._sup_initialize()
import supervised_core_manager as sme
sup=sme.sup_manager()
#%%
mod=sup.Models_list["adaboost"]


#%%
#prediction_check=[[ 1.12010558,  0.30468807, 0.91733383,  0.49783231,  0.66002169, -0.6229224,-0.95731861,  0.35572871,  0.22690534, -0.14822723,  0.0910733,   0.77695864, 0.81458351,  0.02822851,  0.16211852, -0.22560308,  0.0978956,  -0.038654, -0.04464605, -0.06070019,  0.04865046, -0.53625326, -0.50941806]]
#%% Model 
import model_management_engine as me
engine=me.model_management_engine()

#test=pd.DataFrame(df.loc[100]).T

#test=engine.enc.transform(test).toarray()
#%% Loading check data
check={"Countries":"Country_02","Local":"Local_05","Industry Sector":"Metals","Accident Level":"II","Genre":"Female","Employee or Third Party":"Third Party","Critical Risk":"Cut"}

#check2=pd.DataFrame(check,index=[0])
#%% Prediction -- Final
k=engine.predict_sup(check)
#%%
#k=pd.Series(k)
#%%






#%%
c=pd.Series(k).value_counts()
print(c)
i=0
try:
    while 1:
        #print(i)
        #print(c[i])
        #print(c[i+1])
        if c[i]==c[i+1]:
            i+=1
            continue
        else:
            print(i+1)
            break
except:
    print("The Length of the series is :",len(c))
final_prediction=c.index[i]
print("final prediction :",final_prediction)
#%%
from pathlib import Path
p=Path("./model_saves/")
direct=[x for x in p.iterdir() if not x.is_dir()]
#direct=[x for x in direct if ]
print(direct)
#%%