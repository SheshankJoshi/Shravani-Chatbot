# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 15:50:27 2021
This module will define a class of model management engine that is going to bring three different model managers as parts into
a single engine that will interface with the chabot.

NOTE : There should be a better automatic threshold set for prediction based on the average score of equal likelihood among the given classes. So, here it is directly
coded to have _threshold_confidence_AL as 0.40. Since there are five accident levels, random probability will be 20 for each. So, double of any probability eliminates (very crudely)
any chance of likelihood of it being a random event or chance prediction. So, threshold is set to 40% or 0.40. It can be tuned that way, but way more analysis is to be done before taking 
that decision, especially conservative estimations. So, plan ahead for that.

@author: Sheshank_Joshi
"""
#%%
import language_models
import NLP_core_manager
import supervised_core_manager
import numpy as np
import pandas as pd
import json
import copy
from scipy import stats
import pickle as pkl
import model_training_engine as me
#%%
class model_management_engine():
    """This is the original management engine that brings in 3 different managers together. The methods are controls lie here for thresholds. Prelimnary integrity check also happens here
    for proper finding of the files. Since the corresponding managers are responsible for checking file integrities and saved models, it is delegated to the corresponding managers."""
    _threshold_confidence_AL=0.40 # This means any given prediction from the model is having atleast one prediction that has 45% accuracy in any one prediction i.e. just above average. 
    _prob_threshold_score=0.25
    parameter=1 # This paramter decides how strict our predictions or uncertainity we can allow in our NN model.
    #
    def __init__(self):
        self._lm=None
        self._NN=None
        self._sup=None
        self.vocab_dict=None # This is the master vocabulary used for word indexing both for testing and training.
        # More parameters that actually control the mechanism is given here.
        self.sample_scores=None
        self.AL_prediction=None
        self.tokens=None
        self.models_imp=["l_2_order_2",'m_2_order_2'] #,'m_3_order_3','l_3_order_3']
        # As is mentioned, only second order bigram data is considered
        self.init_lm()
        self.init_NN()
        self.init_sup()
        self.options=self.load_options_data()
        self.enc=None
        self.encode()
        print("Management Engine Initializing done")
    #
    def init_lm(self):
        try:
            self._lm=language_models.NLP_LM()
            print("Language model is initalized")
            #print(len(self._lm.vocab_dict))
        except:
            # Have to change this based upon the 
            raise FileNotFoundError("The language_models module doesn't exist. Please place it in the same folder as this file")
    #
    def init_NN(self):
        #Have to write this so that appropriate model is chosen here itself.
        try:
            self._NN=NLP_core_manager.NN_NLP()
            print("Neural Network Model is Initialized")
            
        except:
            raise FileNotFoundError("The NLP_core_manage module doesn't exist or is tampered with. Please place it in the same folder as this file")
    #
    def init_sup(self):
        try:
            #print("working on")
            self._sup=supervised_core_manager.sup_manager()
            print("Supervise Learning Model is Initialized")
        except:
            raise FileNotFoundError("The Supervised models are not found")
    #
    # Signed - Debuggig done -- Working Perfectly.
    def predict_AL(self,prob=False):
        """This will handle all the prections related to accident level.Ideally should use more than one model to do the actual prediction and choose the average and best ranking among the models.
        But here only one particular model is used along with analysis of all the given four prediction.
        Basically, the four probability outputs given out are taken and checked for the cumulative differences. Then a skew data is 
        framed across all the predictions. If that skew threshold is crossed by how many argmax and the maximum vote on that
        is taken as the final output.
        NOTE : That is not implemented in this as of now. Instead it is touched upon, just how many standard deviations a particular argmax is above the rest. If it
        is significant, then we consider that output. Else we take the conservative estimate.
        If there are two predictions, then the highest prediction is should be atleast one std away from the one next in its line.
        Returns : The most conservative estimate if the probability of the prediction is nearly similar to any two given cases."""
        if prob:
            return self.AL_prediction
        else:
            prediction=self.multimodal_predict()
            #print(prediction)
            votes=[]
            for each in prediction:
                votes.append(each.argmax())
            votes=pd.Series(votes)
            #print(votes)
            c=pd.Series(votes).value_counts()
            #print(c)
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
                #print("The Length of the series is :",len(c))
                pass
            final_prediction=c.index[i]
            #print("Final Prediction is :",final_prediction)
            return final_prediction
            #print(c)
            #print (prediction)
            #return prediction
    #
    # Signed - Debugging Done -- Working perfectly.
    def multimodal_predict(self):
        """This function gives prediction from multiple models."""
        #toks=self.word2idx(self.tokens)
        #print(toks)
        #X_test=self._NN.pad_sequences([toks])[0]
        #print(X_test)
        #models=None # These are to be delegated to the outside module
        #predictions=[]
        #for each in models:
        #    predictions.append(models.predict(X_test))
        predictions=self._NN.predict(self.tokens)
        #print(self.tokens)
        return predictions
    #
    def encode(self):
        """Encode the test data into the same mechanism and working as that of the original training data."""
        file=open("./model_saves/encoder.pkl","rb")
        enc=pkl.load(file)
        self.enc=enc
    #
    def predict_sup(self,entry):
        """takes in entire dataframe (along with accident level) and then tries to predict the potential accident level."""
        #Delegate to supervised model manager.
        # make a one hot encoder on the train data and then fix it up with test data.
        df=pd.DataFrame(entry,index=[0])
        #print(df)
        df=self.enc.transform(df).toarray()
        #print(df)
        predictions=self._sup.predict(df)
        # use the predictions given by various models here just like predict_AL funciton and then use them.
        k=pd.Series(predictions)
        c=pd.Series(k).value_counts()
        #print(c)
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
                    #print(i+1)
                    break
        except:
            #print("The Length of the series is :",len(c))
            pass
        final_prediction=c.index[i]
        #print("final prediction :",final_prediction)
        return final_prediction
    #
    # Signed -- debugging Done working perfectly.
    def predict_rough(self,corps):
        """This handles a rough accident level prediction to go with go ahead mechanism that will return a Go ahead or not as a boolean"""
        word_nums=self.word2idx(corps)
        word_nums=self._NN.pad_sequences([word_nums])
        #print(word_nums)
        best_model=self._NN.model_best
        #print(best_model)
        probs=self._NN._models_list[best_model].predict(word_nums) # This is a single model prediction, not a multi-model prediction. So, can't be finalized.
        # Checking if the highest probability of any given prediction is greater than threshold
        if probs.max()>self._prob_threshold_score:
            self.AL_prediction=probs.argmax()
            self.tokens=word_nums
            return True
        else:
            return False
    #
    # Signed - Debugging Done -- working perfectly.
    def generate_options(self,option):
        """Depending upon the option given there will be suggestions by the function that will be used to give user suggestions or thoughtful idea to the user.
        It will always give four options in random."""
        if option==1:
            choices=self._lm.recommend_desc()
        elif option==2:
            choices=self._lm.recommend_contexts()
        elif option==3:
            choices=self._lm.recommend_options()
        #choices=list(set(choices))
        indices=np.random.randint(0,len(choices),5)
        ret=[choices[opt] for opt in set(indices)]
        return ret
    #
    #
    def avg_score(self):
        """Will fetch a bunch of scores from downstream and returns the threshold best score. We can check here for more detailed one, even though it is not
        implemented here extensively."""
        # Here there should be code for analyzing the average scores and choosing the best language model
        # and which one is the best one to better decision making. But, here l_model trigram is used for testing purposes.
        scores=self._lm.avg_scores.describe().loc["25%"]
        scores=scores[self.models_imp].mean()
        # Within 15% of that tolerated mean value score is acceptable, though the minimum value is what is required. But here
        # 25% mean average value is taken as a threshold value.
        return scores
    #
    # Signed - Debugging Done -- Working Perfectly.
    def fetch_score(self,text_in):
        """Will pull a bunch of scores, return the score,and the corresponding corpus of the data"""
        options=self._lm.validity_check(text_in)
        if options:
            # Has options i.e. tokenized texts and a bunch of other scores to
            # Code for analyzing the fetched scores more deeply after seeing the working conditions and the logic for all these.
            # But for now, implementing only l_model trigram score. Similary, only 
            temp=options[0]
            self.tokens=options[1]
            temp=temp[self.models_imp].mean()
        else:
            # This case came up means there is not a single useful keyword in the description
            temp=0
            self.tokens=[]
        return temp,self.tokens
    #
    # Signed - Debugging Done -- Working Perfectly.
    def word2idx(self,text_in):
        word_dict=self._lm.vocab_dict
        return [word_dict[tok] for tok in text_in]
    #
    # Signed - Debugging Done -- Working Perfectly.
    def load_options_data(self):
        """This is going to load the options data that was created as part of training.Yet to be worked upon."""
        try:
            file=open("./model_config/options_data.json","r")
            data=json.load(file)
            file.close()
        except:
            raise FileNotFoundError("The Trained file is not to be found")
        else:
            #_=data.pop("Accident Level")
            return data
    #
        
        
        
