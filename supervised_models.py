# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:21:10 2021

@author: Sheshank_Joshi
"""
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import json
#%%
class sup_models():
    random_state=4
    Models_list={}
    #
    LR_model = LogisticRegression()
    Models_list.update({"LogisticRegression":LR_model})
    #
    NB_model = GaussianNB()
    Models_list.update({"NaiveBayes":NB_model})
    KNN = KNeighborsClassifier()
    Models_list.update({"KNN":KNN})
    #
    support_vector = SVC()
    Models_list.update({"support_vector":support_vector})
    #
    decision_tree=DecisionTreeClassifier()
    Models_list.update({"decision_tree":decision_tree})
    #
    bagging=BaggingClassifier(random_state=random_state)
    Models_list.update({"bagging":bagging})
    #
    adaboost=AdaBoostClassifier(n_estimators=100,random_state=random_state)
    Models_list.update({"adaboost":adaboost})
    #
    gradientboost = GradientBoostingClassifier(random_state=random_state)
    Models_list.update({"gradientboost":gradientboost})
    #
    random_forest = RandomForestClassifier(random_state=random_state)
    Models_list.update({"random_forest":random_forest})
    #
    Mlpc = MLPClassifier()
    Models_list.update({"MLPC":Mlpc})
    #
    Nusvc=NuSVC()
    Models_list.update({"NuSVC":Nusvc})
    #
    parameters=None
    def __init_(self):
        try:
            file=open("./param_saves/sup_model_refer_params.json","r")
            self.parameters=json.load(file)
            file.close()
        except:
            file.close()
            raise Warning("The parameters file is missing, so going with unkonwn default parameters. You don't be able to tune.")
    #
    def parameter_generator(self,model_name):
        """Will prepare the parameter search engine for the model and the best parameters."""
        gs = GridSearchCV(model_name,param_grid=self.parameters[model_name],cv=10)
        return gs
    #
    def tune(self,X_train,y_train):
        """This will tune the models appropriately for the object"""
        best_param_gridsearch={}
        for each in self.Models_list:
            gs=self.parameter_generator(each)
            gs.fit(X_train,y_train)
            best_param_gridsearch.update({each:gs.best_params_})
            model=self.Models_list[each]
            model.set_params(best_param_gridsearch)
        try:
            file=open("./param_saves/sup_model_best_params.json","w")
            json.dump(best_param_gridsearch,file)
            file.close()
        except:
            print("The appropriate folder hasn't been found. Hence saving the parameters is not implemented.")
    #
    
    
    
        


#%%


#%%


#%%

