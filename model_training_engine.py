# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:13:28 2021
This is the generalized NLP interface that handles both training and the language models 
left and right to the current situation.

NOTE to self : Write a Function that will choose the best performing model from the available models, if required.
@author: Sheshank_Joshi
"""
#%%
import numpy as np
import pandas as pd
import tensorflow.keras as k
import NLP_core_manager
import language_models as lm
import supervised_core_manager
import json
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pickle as pkl
from sklearn.preprocessing import OneHotEncoder
#%%
class models_training_engine():
    """The Interfacing class that acts a bridge between Language model, Neural Network model and supervised model for training purposes. Automatic training
    doesn't happen here. We have to manually step by step procedures as listed below
    1. set description column name
    2. set dependent column name
    3. set target column name
    4. Call the Train function (this will take care of the rest in the background)"""
    #_max_len=45 # Tunable parameter to the maximum length of the description for the words
    _lm=None # Where we store the Language models Manager
    _NN=None # Where we store the Neural Network Manager
    _sup=None # Where we store the supervised Model Manager
    def __init__(self,dataframe):
        """It is user's responsibility to pass pandas dataframe object to the specifications as given in the dataset."""
        self.df=None
        self._check_dataframe(dataframe=dataframe)
        self._target_col=None # The Original Target Col name i.e. The Potential Accident Level in our case
        self._desc_col=None # The name of the Description column, that contains the description of the incident.
        self._desc=None # Actual description column that has text data i..e non tokenized, original corpus with individual texts
        self._target=None # The pandas series that contains the potential accident level.
        self._data=None
        self.dep_col=None
        self.options_data=None
        self._target2=None
        self.enc=None
    
    #Signed -- Debugging -- Working correctly. Rank 0
    def _check_dataframe(self,dataframe):
        """Just trying to catch an exception in case the supplied data isn't a pandas dataframe."""
        try:
            assert type(dataframe) == type(pd.DataFrame())
            self.df=dataframe
            try:
                self.df.drop("Data",axis=1,inplace=True)
            except:
                pass
            try:
                self.df.drop("Unnamed: 0",axis=1,inplace=True)
            except:
                pass
            #print(self.df.columns)
            #self._data=dataframe
        except:
            raise TypeError("The Object passed is not a Pandas Dataframe object. Please check it and re-initialize")   
    #
    # Signed - Debugging -- Working Correctly. Rank 1
    def set_desc_column(self,name):
        """Setting the column name for Description"""
        if self._desc_col:
            print("The Description column is already set.")   
        else:
            #print(name)
            try:
                name in self.df.columns
                self._desc_col=name
                self._desc=pd.DataFrame(self.df.pop(name))
                #self._desc["orig_length"]=self._desc[name].apply(len)
                # This can be completely avoided.
            except AssertionError:
                print("There is no target_column in the dataframe. Please change dataframe or please change the target name")
            else:
                pass
                #print("Please also set target column for Potential Accident Level")       
    #
    # Signed - Debugged -- Working Correctly. Rank 2
    def set_dep_target(self,name):
        """Set the dependent element target here. It needs appropriate labeling."""
        if not self.dep_col:
            self.dep_col=name
            self._desc[self.dep_col]=self.df[self.dep_col]
        else:
            print("Dependent column is already set")
    #
    # Setting Target column for Supervised Model
    # Signed - Debuggin -- working correctly. Rank 3
    def set_target_column(self,name):
        """Sets the Target Column for final prediction"""
        #convert=self.options_data
        if not self._desc_col:
            raise ValueError("First Set the Description Column by calling appropriate method")
        elif self._desc_col==name:
            raise NameError("Please select other column for Supervised model Target column. Given column {}".format(self._target))
        else:
            try:
                assert name in self.df.columns
                self._target_col=name
                self._data,self._target=self.shape_resampling(self.df.drop(name,axis=1),self.df[name])
                self._target2=self.df.pop(self._target_col)
                #convert=dict([(value,key) for key,value in enumerate(convert[self._target_col])])
                #self.target2=self._target.replace(to_replace=convert)
            except:
                raise ValueError("Sorry, the value you have in input is not in the columns")
    #
    # Signed - Debugging Finished -- Working Correctly. Rank 5
    def _lm_initialize(self):
        """This will initialize the Language Model, and set things up for other trainings to happen. This is the crucial
        step to make any changes for any further analysis."""
        try:
            if not self._desc.empty:
                # order is not going to be specified here.
                self._lm=lm.NLP_LM(corpus=self._desc[self._desc_col],train=True)
                print("Language Model Trained Successfully")
            else:
                raise ValueError("The Description column is not set yet. Check about it.")
        except:
            #self._desc=None
            print("There is something wrong with the Description given")
            raise AttributeError("Description Column not appropriate")
    #
    # Signed - Debugging Done -- Working Correclty. Rank 6
    def _NN_initialize(self):
        """This will initialize the Neural Network Model and set things up ready, including saving the models."""
        try:
            # assert self._lm
            # assert self.dep_col # Checking if the dependent column is set or not.
            # Here vocab needs to be checked if it is appropriate. We will build all our models initially
            data_corpus=self._lm._corpus_clean2
            data_corpus=data_corpus.apply(self.word2idx)
            self._desc[self._desc_col]=data_corpus
            self._NN=NLP_core_manager.NN_NLP(dat=self._desc[[self._desc_col,self.dep_col]],targ=self.dep_col,vocab=self._lm.vocab,train=True,auto=True)
            print("Neural Networks Trained")
        except:
            raise ValueError("You need to first specify what is the text, dependent column.")
    #
    #Signed - Debugging Done -- Working Perfectly. Rank 7
    def _sup_initialize(self):
        """This will initialize the Supervised Learning Model and will set things up, including saving the models for later."""
        try:
            self._create_data_dictionary()
            #print(self._data.columns,self._target.name)
            x,y=self._data,self._target
            #print(y)
            #print(self.options_data["Accident Level"])
            #convert=dict([(value,key) for key,value in enumerate(self.options_data[self.dep_col])])
            #y=y.replace(to_replace=convert)
            #print(x.shape,y.shape)
            #print(x.columns)
            self._sup=supervised_core_manager.sup_manager(X=x,y=y,train=True,auto=True)
            print("Supervised Model Trained.")
        except:
            raise ValueError("There is something wrong with the supervised model. Check if files are available")   
    #
    # Signed - Debugging Done -- Working Correctly. Rank 4
    def _create_data_dictionary(self):
        """Will create a dictionary for values and encoding appropriately in the order in which they will be fed to supervised
        learning model. The Date column is completely ignored here. Further thoughts about including it as a time series is to be
        seen much later."""
        #Remove Get
        df=self._data
        encoder=OneHotEncoder(handle_unknown='ignore')
        # Should be called in by the supervised model trainer. It should take care of the whole thing.
        cols={}
        #print(self.df.columns)
        try:
            df=df.drop(["Data"],axis=1)  
        except:
            pass
        try:
            df=df.drop(["Unnamed: 0"],axis=1)
        except:
            pass
        #print(df.columns)
        encoder.fit(df)
        x=encoder.transform(df).toarray()
        #print(x)
        for index in range(len(df.columns)):
            col=df.columns[index]
            cols.update({col:list(encoder.categories_[index])})
        #temp=pd.get_dummies(self._target)
        #self._target=temp
        #y=encoder.transform(_target)
        #cols.update({self._target_col:list(temp.columns)})
        #
        self.options_data=cols
        file=open("./model_saves/options_data.json","w")
        #json_obj=json.dumps(col)
        json.dump(self.options_data,file)
        file.close()
        #
        f=open("./model_saves/encoder.pkl","wb")
        #json.dump(encoder)
        pkl.dump(encoder,f)
        f.close()
        #
        x=pd.DataFrame(x,columns=[item for cat in encoder.categories_ for item in cat])
        self._data=x
        #print(self._data)
    #
    # singed -- Debugged -- working correctly.
    def word2idx(self,text_in):
        """Used to convert the words into appropriate indexed words."""
        word_dict=self._lm.vocab_dict
        return [word_dict[tok] for tok in text_in]
    #
    # This is not at all needed.. Keep it aside.
    def train(self):
        """This function is going to train all the NN Models followed by all the Supervised models and then place them in particular
        order in their machine in appropriate order for a standby call."""
        self._lm_initialize()
        self._NN_initialize() # We will make extensive use of multiprocessing methodologies.
        self._sup_initialize()
        #self.optimize() # Decision on whether one single model should be used is not yet decided.
    #
    #
    def optimize(self):
        """This function is going to call all the models invovled and optimize them appropriately within the specified parameters given in
        their respective Model managers"""
        # Code calling for optimization in both supervised learning model and NN Training model for chosen models.
        return
    #
    #
    # Signed - Debugging done -- Working perfectly.
    def shape_resampling(self,X_train,y_train):
        """This will try to eliminate any Class imbalances observed within the data."""
        ros = RandomOverSampler()
        rus = RandomUnderSampler()
        X_train,y_train=ros.fit_resample(X_train,y_train)
        X_train,y_train=rus.fit_resample(X_train,y_train)
        return X_train,y_train
    #

        
#%%