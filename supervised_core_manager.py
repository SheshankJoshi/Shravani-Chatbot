# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:20:46 2021

@author: Sheshank_Joshi
"""
#%%
#import imblearn

from sklearn.decomposition import PCA
import pandas as pd
import supervised_models as sm
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pickle as pkl
from pathlib import Path
#%%
class sup_manager(sm.sup_models):
    """This manages all the supervised learning models placed in a separate file of supervised learning
    models."""
    saved_file_name="sup_learn_models.pkl"
    def __init__(self,X=None,y=None,train=False,auto=False):
        super(sup_manager,self).__init__()
        self.no_of_components=None
        self.minimum_threshold=0.005 # This is the variance ratio to be used while doing feature engineering with PCA
        self.percentage=None
        self.scores={}
        self.X_train=None
        self.y_train=None
        self.pca=None
        self.models=None
        
        if train:
            if type(X)==type(pd.DataFrame()):
                self.X_train=X
                self.y_train=y
            else:
                raise ValueError("The Data type provided is not the data type that is required.")
            if auto:
                self.PCA_transform_features()
                self.fit()
                self.save()
        else:
            self.Models_list={}
            sup_manager.load(self)
            print("Supervised Models are Loaded Successfully")
        #self.X_train,self.y_train=self.shape_resampling(self.X_train,self.y_train)
    #
    # Debugging done -- Working Perfectly
    def save(self):
        """This can be used to save the models. But, it will not save the data, it will only save the models."""
        #self.X_train=None
        #self.y_train=None
        self.models=list(self.Models_list.keys())
        try:
            f=open("./model_saves/"+self.saved_file_name,"wb")
            pkl.dump(self,f)
            f.close()
            for each in self.Models_list:
                model=self.Models_list[each]
                file=open("./model_saves/"+each+".sav","wb")
                pkl.dump(model,file)
                file.close()
            #print("Supervised Models are saved")
        except:
            raise FileNotFoundError("The Given file can't be saved because the structure doesn't exist.")
    #
    #Signed -- Debugging done -- working perfectly.
    @classmethod
    def load(cls,self):
        """This will load the model from the saved files and makes it ready for prediciton, though the original data will be gone."""
         #print("--------------Before loading----------------")
        f=open("./model_saves/"+cls.saved_file_name,"rb")
        obj=pkl.load(f)
        f.close()
        #models=
        #print(dir(obj))
        #print(obj._fit_done)
        #print("Total Vocabulary Size loaded :",obj.model["l_1"].vocab._len)
        #print("The length of the Vocabulary is :",len(obj.vocab))
        #print(obj.avg_scores)
        self.__dict__=obj.__dict__.copy()
        self.Models_list={}
        p=Path("./model_saves/")
        direct=[x.name for x in p.iterdir() if not x.is_dir()]
        #print(direct)
        for each in self.models:
            file_name=each+".sav"
            if file_name in direct:
                file=open("./model_saves/"+file_name,"rb")
                mod=pkl.load(file)
                self.Models_list.update({each:mod})
                file.close()
            else:
                print("The model missing for loading is :",each)
        # Write script to load each of the fitted model into the dictionary object.
        # copying the original data into memory     
    #
    # Signed -- Debugging done -- Working perfectly
    def pca_tuning(self,X_train):
        parameters_to_choose=range(len(X_train.columns))
        decision_maker={}
        for i in parameters_to_choose:
            pca=PCA(n_components=i)
            pca.fit(X_train)
            #print("Checkpoint")
            pca_variance=pca.explained_variance_ratio_.sum()
            decision_maker.update({i:pca_variance})
            #print(decision_maker)
        decision_maker=pd.Series(decision_maker).diff()
        out=decision_maker[decision_maker>self.minimum_threshold].index.max()
        return out
    #
    # Signed -- Debugging Done -- working Perfectly.
    def PCA_transform_features(self):
        # We can tune this later for whatever value we want for components that can be neglected for their influence
        n=0 #Initialize the number of features to be 0.
        #print(no_of_components)
        if self.no_of_components==None:
            n=self.pca_tuning(self.X_train)
            #print("Here in no components")
            #print(n)
        else:
            n=self.no_of_components
        #print("No of components selected is :",n)
        pca=PCA(n_components=n,random_state=self.random_state)
        pca.fit(self.X_train)
        self.pca=pca
        #print(pca.explained_variance_ratio_)
        self.X_train=pd.DataFrame(pca.transform(self.X_train))
    #
    # Debugging done -- working Perfectly.
    def fit(self):
        """This fits into all the models the given training and testing data."""
        for each in self.Models_list:
            try:
                model=self.Models_list[each]
                #print(self.X_train,self.y_train)
                model.fit(self.X_train,self.y_train)
                self.scores.update({each:model.score(self.X_train,self.y_train)})
            except:
                print("Problem with fitting the model :",each)
    #
    # Debugging done -- working perfectly.
    def predict(self,outside_data):
        """All the models' predictions are returned. These can be used further for analysis."""
        predictions=[]
        data=self.pca.transform(outside_data)
        #data=np.expand_dims(outside_data,axis=0)
        #print(data)
        #print(data.shape)
        for each_model in self.Models_list:
            model=self.Models_list[each_model]
            try:
                pred=model.predict(data)[0]
                #print(pred)
                predictions.append(pred)
            except:
                print("Something wrong with prediction for the model :",each_model)
        return predictions           
    #
    #
    def Decisiontree_features(self,percentage=None):
        if percentage==None:
            n_features_chosen=self.X_train.shape[1]*10/100
        else:
            n_features_chosen=self.X_train.shape[1]*percentage/100
        k=n_features_chosen
        regressor = DecisionTreeRegressor(random_state=self.random_state, max_depth=5)
        regressor.fit(self.X_train,self.y_train)
        feature_importances = regressor.feature_importances_
        feature_names = self.X_train.columns
        top_k_idx = (feature_importances.argsort()[-k:][::-1])
        #print(feature_names[top_k_idx], feature_importances)
        return self.X_train[feature_names[top_k_idx]]
    #

    





#%%
