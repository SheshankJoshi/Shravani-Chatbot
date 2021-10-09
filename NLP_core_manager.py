# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 20:10:06 2021
This is the left hand module that contains the core interfacing for NN Models. The models are placed in a separate file. Only example models are placed in this. So, modularity
and scalability is achieved.

NOTE : A Class decorator function is required here that will pass all the methods to all the models stored in the manager at the same time and will work in exact class methods.

NOTE to Self : Try to implement the classifier models for proper guess and evaluating the self model based on how good our model is going to predict things, based on 
validation input given the user by selecting the country etc. 

NOTE to self : This needs some model management to be done. Saving the parameters and then serving them appropriately. Can use pickling the model and then resue them for prediction, along with 
all the other notes. It should be quite visible.

@author: Sheshank_Joshi
"""

#%%
import tensorflow as tf
from tensorflow import keras as k
import NN_models as NN
import pandas as pd
import numpy as np
#from tensorflow.keras import models
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import load_model
from pathlib import Path
import pickle as pkl
import json
import copy
#X=np.random.random((50,12))
#y=np.random.randint(0,2,(50,5))

#%%
class NN_NLP():
    """ The Class contains a bunch of models that is shown through its structures by calling the method  "show_models". The actual creation of models
    is given a choice at the time of initiation of the model object and then trained appropriately. All models are not pre-trained, though it can be done and called
    upon purpose and situation. We can choose to see the model architecture from "show_model_arch"
    
    Parameters chosen :
    optimizer_chosen : Intance of optimizers from Tensorflow Library or custom Optimizers
    losses_chosen : Losses of custom function or from Tensorflow Library
    metric_chosn : Metrics from a custom function or from Tensorflow Library
    
    NOTE : If appropriate model is not chosen, the Tensorflow functions and customizations will not be available. A single model
    needs to be chosen to increase the customization options and fine tuning the model.
    
    """
    # These are the default parameters to be used on all the models.
    max_len=50 # Maximum length of the sequences
    Embedding_dimensions=50 # Tunable parameter for better success.
    units=48
    #saved_file_name="NN_NLP.pkl" # This idea is not working, tested properly.
    epoch=55
    ####################################################
    # Initialize Function.
    ####################################################
    def __init__(self,dat=None,targ=None,vocab=None,train=False,model_chosen=None,auto=False):
        """Initializing is delegated to initiate method so multiple models can be initialized but only one running at any given time.
        Data : Should be index converted words i.e. numbers not words. Padding is not needed. Should have Target prediction along with Text Columns.
        vocab : Passing vocabulary size is enough, though passing entire vocabulary is also accepted.
        target : Should specify the name of the target column within the dataframe given
        model_chosen : specify the name of the model you want to initiate with. If none is given all the models will be initiated (waste of resources). 
        Can be later chosen, if not sure what are the models avaiable, just use the method NN_NLP.show_available_models() and then choose from the given strings.
        auto : If you want automatic fitting and compiling with default best-chosen parameters for the models or not. If not, be careful to save your models using obj.save()
        method."""        
        self.vocab_size=None # Stores the list vocabulary of the original data (imported from Langauge Model)
        self.data=None # Stores the Original data without the target column and exclusively the text corpus as a Series
        self.data2=None
        self.target=None # Stores the target data in a Pandas dataframe, with dummies stored in it in floating point.
        # self.model=None # This is the place one big chosen model resides
        self._models_list={} # This is the place where all the models in the NN_models are imported and stored as standby with best default parameters.
        # Once the models are initiated or chosen, access to the parameters for tuning the model become available like all the variables within the model
        # which are listed by dir.
        self.models_built=False # A Check whether the models are built or not
        self._model=None # If a specific model is initiated it is loaded here else it will remain None.
        self._initiated=False # Check the models have been initiated or not, especially when training purposes
        # These are the parameters chosen after extensive testing based on specifications made on the training set.
        self.optimizer_chosen="adam"#Adam(learning_rate=0.01) # Setting the default optimizer chosen
        self.loss_chosen=CategoricalCrossentropy()#"categorical_crossentropy"#CategoricalCrossentropy() # setting the default loss; can be reset anytime and models rebuilt.
        self.metric_chosen=CategoricalAccuracy()#"categorical_accuracy" #CategoricalAccuracy() # setting the default metric; can be reset anytime and models rebuilt.
        self._fit_done=False
        self.model_compiled=False
        self.model_best=None
        self._AL=None
        self.t=None
        self.call_back=EarlyStopping(monitor='categorical_accuracy',
                                min_delta=0.05,
                                patience=10, 
                                verbose=1, 
                                mode='auto', 
                                baseline=0.90, 
                                restore_best_weights=True)
        
        #
        if train:
            # This only prepares the data given for training. You call need to call train method to train chosen model or bunch of models.
            data=dat
            target=targ
            #print(data)
            #print(target)      
            try:
                data.empty
                #print(type(data))
                #print(data.empty)
                #assert data.empty==True
                #print(vocab)
                #assert data==None
                try:
                    vocab!=0
                except:
                    try:
                        len(vocab)==0
                    except:
                        raise ValueError("The Vocabulary length can't be zero")                
                #assert target.empty
                type(data)==type(pd.DataFrame())
                assert target in data.columns
            except:
                raise ValueError("The Arguments for Vocab and Data is not provided right. Please provide them")
            else:
                if (type(vocab)==int) | (type(vocab)==float):
                    self.vocab_size=vocab
                else:
                    self.vocab_size=len(vocab)
                self.data=data # Setting up the data.
                self.target=target # Setting up the Target
                self.setup(train) # setting up the data and target
                if model_chosen:
                    self.initiate(model_chosen) # Directly initiate the model that is chosen with no hassles.
                else:
                    self.initiate() # Initiate all the models possible with default paramters, best chosen. Can be retrained appropraitely through objects after accessing or selecting.                                    
                self.build_models()
                 # Will show model names and encourage to choose from
                # Now We have to introduce a method to train all the models or the chosen model with input data
                if auto:
                    #print("entered auto")
                    #self.setup(train)
                    #self.compile_model() # Automatically Fits the data
                    self.fit(batch_size=8,epochs=self.epoch,use_multiprocessing=True) # Automatically Fits the data
                    self.save_models()
                    self.model_best,_=self.choose_best_model()
                else:
                    self.show_models_names()
        else:
            try:
                #NN_NLP.load(self) # Loading the self object with data.
                self.load_models()
                print("Neural Network Models are loaded Successfully")
            except:
                print("There is something wrong here")
                raise FileNotFoundError("Model Files aren't found. Please check it.")
            # Write code to restore the data into the models, includeing the object state and also the model weights and load them appropriately.           
        #self._classifier=None   # This is not being dealt with currently.
        ## Write script to save the entire object as it is given here, and then after loading object, appropriate
        # model weights are to be loaded accordingly, when the entire object is pickled. 
            
    ####################################################
    #  EOF 
    ####################################################
    # Signed Debugging -- Working Correctly.
    def setup(self,train):
        """This sets up the models and the entire management core if training is chosen. Just an alias procedure."""
        if train:
            self.data=self.data.sample(frac=1).reset_index(drop=True)
            self.t=copy.copy(self.target) # Saving the Target Variable column name.
            self.target=self.data.pop(self.target) # This is the y for training data
            self.target=self.prepare_target() # Preparing the actual target data by generating the 
            self.data2=self.data[self.data.columns[0]]
            self.data2=self.pad_sequences(self.data2) # Padding the sequences with default paramters for training directly.
             # All the models are built with a default input shape, but none of the models are actually used, until specified.
   
    #
    #
    def choose_best_model(self):
        """This automatically sets the best model based on the situation and the characteristics. The Criteria is 
        median accuracy. As the accuracy can change, median value is a good indicator of how stable the model is. As the
        accuracy keeps increasing the median value keeps shifting, or it will say the same.
        This only works if there is no chosen model. """
        model_character={}
        information=[]
        #print("\n-------------------------------------------------\n")
        for each in self._models_list:
            try:
                model=self._models_list[each].history.history
                #print(model)
                #print(each)
                data_info=pd.DataFrame(model,columns=list(model.keys())).describe()
                model_character.update({each:data_info})
                #print("The Mean Loss is :",data_info["loss"].loc["min"])
                #print("The Maximum Accuracy is :",data_info["categorical_accuracy"].loc["max"])
                #print("The Median Accuracy is :",data_info["categorical_accuracy"].loc["50%"])
                #print("----------------------------------")
                median=data_info["categorical_accuracy"].loc["50%"]
                #print("The Median value is :",median)
                information.append(median)
            except:
                print("Choosing Best Model is not working out for the Model :",each)
        ind=information.index(max(information))
        model_chosen=list(model_character.keys())[ind]
        return model_chosen,model_character
    #
    #Signed - Debugging done -- Perfectly Working
    def save_models(self):
        """This is going to save the models exactly as they are. They can be retrained on additional data."""
        for each in self._models_list:
            model=self._models_list[each]
            try:
                model.save("./model_saves/"+model.name)
            except:
                print("Couldn't save Model :",each)
                
        try:
            file=open("./model_config/NLP_data.json","w")
            AL_data=dict([(key,value) for key,value in enumerate(self.target.columns)])
            data_to_store={self.t:AL_data,"best_model":self.model_best}
            json.dump(data_to_store,file)
            file.close()
        except:
            print("There seems to be some problem with saving data. Will not save them. But beware")
        
    # Signed - Debugging done -- Perfectly Working
    def load_models(self):
        """Models and their weights, both of them are saved.Just load and then can be retrained on additional data."""
        self._initiated=True
        # This extremely hardcoded. But should be done later on to search for alternatives.
        self.models_built=True
        p=Path("./model_saves/")
        direct=[x for x in p.iterdir() if x.is_dir()]
        if self._model:
            # Think of a method to load appropriate model.
            self._model.load_weights("./model_saves/" + self._model.name + ".h5")
        else: 
            for each in direct:
                model=load_model(str(each))
                model_name=each.name
                self._models_list.update({model_name:model})
                #self._models_list[model_name].load_weights("./model_saves/" + each)
        #self.compile_model()
        self._fit_done=True
        try:
            file=open("./model_config/NLP_data.json","r")
            j=json.load(file)
            file.close()
            self.model_best=j["best_model"]
        except:
            raise FileNotFoundError("Supplementary data file for NLP Core is missing. Plese check it.")
        
        #self.model_best="MultiAtt_cLSTM" # Hardcoded here but should be deciphered from outside file that stores the config.
        #self.model_best,_=self.choose_best_model()
        #print("NN models loaded successfully")
        
    # 

    ####################################################
    #  Function that conditions the target for training. 
    ####################################################
    # Signed Debugging -- Working Correctly.
    def prepare_target(self):
        """This is created when things are first trained.Saved and then loaded appropriately"""
        df=pd.get_dummies(self.target,dtype="int")
        return df
    #
    ####################################################
    #  EOF 
    ####################################################
    
    ####################################################
    # Function that initiates different models depending upon the choice and
    # can be dynamically initiated
    ####################################################
    # Signed Debugging -- working correctly.
    def initiate(self,model_name=None):
        """This will actually initiatize the chosen model and make it ready for training, fitting and all kinds of functions. At the same time once the model is chosen, it is finalized. If we want to change our model, we have to reinitialize our entire class and call appropriate methods.\n This method actually compiles the chosen arch and then """
        model=self.Model_1()
        self._models_list.update({model.name : model})
        model=self.Model_2()
        self._models_list.update({model.name : model})
        model=self.Model_3()
        self._models_list.update({model.name : model})
        model=self.Model_4()
        self._models_list.update({model.name : model})
        model=self.Model_5()
        self._models_list.update({model.name : model})
        if model_name:
            self._model=self._models_list[model_name]
        self._initiated=True
        #else:
        #    self._models_list.update({"Model_1" : self.Model_1()})
        #    self._models_list.update({"Model_2" : self.Model_2()})
        #    self._models_list.update({"Model_3" : self.Model_3()})
        #    self._models_list.update({"Model_4" : self.Model_4()})
        #    self._models_list.update({"Model_5" : self.Model_5()})
        #self._classifier=self.Model_country() # Modeling other columns isn't undertaken yet.

    #    
    ####################################################
    # EOF
    ####################################################
    
    ####################################################
    # Classifier Model for guessing the Industry - Currently on Hold
    # Theoretically, we can use the same method as the other models for target 
    # column to make the model learn automatically.
    ####################################################    
    #def Model_country(self):
    #    inp=k.layers.Input(shape=(12,),dtype="float") # You have to change this
    #    layer1=k.layers.Dense(12,activation=tf.nn.relu)(inp)
    #    layer2=k.layers.Dense(24,activation=tf.nn.relu)(layer1)
    #    dropout=k.layers.Dropout(0.2)(layer2)
    #    layer3=k.layers.Dense(3,activation=tf.nn.softmax)(dropout)
    #    l=k.Model(inputs=inp,outputs=layer3,name="classifier")
    #    return l
    ####################################################
    # End of Model
    ####################################################   

    ####################################################
    # Function that pads sequences as per the requirement given in the module
    ####################################################
    # Signed Debugging -- Working Correctly.
    def pad_sequences(self,seq):
        """Squences are padded according to the limits as specified by the tunable parameter here."""
        return pad_sequences(seq,maxlen=self.max_len)
    #
    ####################################################
    # EOF
    ####################################################

    ####################################################
    # Model_1 Initialization  ## Tune the Model here
    ####################################################
    # Signed Debugging -- Working correctly.
    def Model_1 (self): # Model name to change according to Architecture
        """Initiating the Model here itself, this is going to help set and tune the parameters
        and models perfectly according to the situation. The Most import point this is going to do
        is set the parameters for the model."""
        embedding_dim=self.Embedding_dimensions # Default parameter for the entire engine but is tuanble according to preference.
        length_of_sequence=self.max_len
        #logic to decide the no of units based on the chosen length of sequence. But the default parameter is chosen here.
        units=[self.units,int(self.units/2),len(self.target.columns)]
        model= NN.Simple_LSTM(vocab_size=self.vocab_size, 
                              embedding_dim=embedding_dim,
                              units_list=units, 
                              length_of_sequence=length_of_sequence)
        # Decide this later based on
        # Setting the parameters for the model here itself
        #model.act="relu" 
        #model.drop_out=0.2
        #model.lr=0.01
        #model.loss="categorical_crossentropy"
        #model.optimizer="adam"
        #model.metrics=["categorical_accuracy"]           
        return model
    #
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # Model_2 Initialization ## Tune the model here
    ####################################################
    # Signed Debugging -- Working Correclty.
    def Model_2 (self):
        """Initiating the Model here itself, this is going to help set and tune the parameters
        and models perfectly according to the situation. The Most import point this is going to do
        is set the parameters for the model."""
        embedding_dim=self.Embedding_dimensions # Default parameter for the entire engine but is tuanble according to preference.
        length_of_sequence=self.max_len
        #logic to decide the no of units based on the chosen length of sequence. But the default parameter is chosen here.
        units=[self.units,self.units,int(self.units/4),len(self.target.columns)]
        model= NN.Simple_BiLSTM(vocab_size=self.vocab_size, 
                              embedding_dim=embedding_dim,
                              units_list=units, 
                              length_of_sequence=length_of_sequence)
        # Decide this later based on
        # Setting the parameters for the model here itself
        #model.act="relu" 
        #model.drop_out=0.2
        #model.lr=0.01
        #model.loss="categorical_crossentropy"
        #model.optimizer="adam"
        #model.metrics=["categorical_accuracy"]
        return model    
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # Model_3 Initialization ## Change the model here
    ####################################################
    # Signed -- Debugging Working Correclty.
    def Model_3 (self): # Model name to change according to Architecture
        """Initiating the Model here itself, this is going to help set and tune the parameters
        and models perfectly according to the situation. The Most import point this is going to do
        is set the parameters for the model."""
        embedding_dim=self.Embedding_dimensions # Default parameter for the entire engine but is tuanble according to preference.
        length_of_sequence=self.max_len
        #logic to decide the no of units based on the chosen length of sequence. But the default parameter is chosen here.
        units=[self.units,self.units,int(self.units/4),len(self.target.columns)]
        model= NN.SelfAtt_LSTM(vocab_size=self.vocab_size, 
                              embedding_dim=embedding_dim,
                              units_list=units, 
                              length_of_sequence=length_of_sequence)
        # Decide this later based on
        # Setting the parameters for the model here itself
        #model.act="relu"
        #model.drop_out=0.2
        #model.lr=0.01
        #model.loss="categorical_crossentropy"
        #model.optimizer="adam"
        #model.metrics=["categorical_accuracy"]
        #model.attention_width_given=15
        #model.attention_activation_given="sigmoid"
        return model
    #
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # Model_4 Initialization ## Tune the model here
    ####################################################
    # Signed -- Debugging Correctly.
    def Model_4 (self): # Model name to change according to Architecture
        """Initiating the Model here itself, this is going to help set and tune the parameters
        and models perfectly according to the situation. The Most import point this is going to do
        is set the parameters for the model."""
        embedding_dim=self.Embedding_dimensions # Default parameter for the entire engine but is tuanble according to preference.
        length_of_sequence=self.max_len
        #logic to decide the no of units based on the chosen length of sequence. But the default parameter is chosen here.
        units=[self.units,self.units,int(self.units/4),len(self.target.columns)]
        model= NN.SelfAtt_cBiLSTM(vocab_size=self.vocab_size, 
                              embedding_dim=embedding_dim,
                              units_list=units, 
                              length_of_sequence=length_of_sequence)
        # Decide this later based on
        # Setting the parameters for the model here itself
        #model.act="relu" 
        #model.drop_out=0.2
        #model.lr=0.01
        #model.loss="categorical_crossentropy"
        #model.optimizer="adam"
        #model.metrics=["categorical_accuracy"]
        #model.attention_width_given=15
        #model.attention_activation_given="sigmoid"
        #model.cKernel_size=4 # The Kernel size for the convolution
        #model.cFilters=200 # No of convolution filters that are placed.
        #model.cPadding="same" # padding that is requied for Convolutional layer
        return model
    #
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # Model_5 Initialization ## Tune the model here
    ####################################################
    # Signed -- Debugged working correctly.
    def Model_5 (self): # Model name to change according to Architecture
        """Initiating the Model here itself, this is going to help set and tune the parameters
        and models perfectly according to the situation. The Most import point this is going to do
        is set the parameters for the model."""
        embedding_dim=self.Embedding_dimensions # Default parameter for the entire engine but is tuanble according to preference.
        length_of_sequence=self.max_len
        #logic to decide the no of units based on the chosen length of sequence. But the default parameter is chosen here.
        units=[self.units,int(self.units/4),len(self.target.columns)]
        model= NN.MultiAtt_cLSTM(vocab_size=self.vocab_size, 
                              embedding_dim=embedding_dim,
                              units_list=units, 
                              length_of_sequence=length_of_sequence)
        # Decide this later based on
        # Setting the parameters for the model here itself
        #model.act="relu"
        #model.drop_out=0.2
        #model.lr=0.01
        #model.loss="categorical_crossentropy"
        #model.optimizer="adam"
        #model.metrics=["categorical_accuracy"]
        #model.attention_heads=3
        #model.dimension_keys=3
        #model.cKernel_size=4 # The Kernel size for the convolution
        #model.cFilters=200 # No of convolution filters that are placed.
        #model.cPadding="same" # padding that is requied for Convolutional layer
        return model
    #
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # Function Initiating and building the models and their names
    ####################################################
    # Signed  Debugged -- Working Correctly.
    def build_models(self):
        """This function will build models and keep the models along with their architectures aside for selection and 
        training later on. Models are compiled only once the selection of the architecture is done. Building of models 
        happen automatically if its a train=true is chose.
        Actual input shapes are left to the fit time, based on user inputs, or depending upon the dataset provided."""
        if self._model:
            self._model.build(input_shape=(None,None,)) #         
        else:
            for each in self._models_list:
                model=self._models_list[each]
                model.build(input_shape=(None,None,)) # Can be instantiaed later on when things get going, to actually fix the size later on. 
        self.models_built=True    
    #    
    ####################################################
    # EOF
    ###################################################
    
    ####################################################
    # Function showing the list of available Models
    # If initiali
    ####################################################  
    # Signed Debugged -- Working Correctly.
    def show_models_names(self):
        """Shows the list of available Models"""
        if self.models_built:
            print("Please select from among below models with appropriate call function\n")
            x=["{}".format(each) for each in self._models_list]
            for each in x:
                print(each)
            print("If you want to see Network architecture, use the method \"show_model_arch()\" for showing the architecture summary")
            return
        else :
            raise NotImplementedError("The models are not yet built")
    #
    ####################################################
    # EOF
    ####################################################
    
    
    ####################################################
    # Function showing the different Model Architectures
    ####################################################
    # Signed Debugged -- Working Correctly.
    def show_model_arch(self,mod):
        """ Different model architectures as chosen are represented in this function.
        If model name is not in the list, it asks the user to check the spelling of the chosen model.
        If the model is in the list, the architecture summary of the model is returned. It can be fed to print function"""
        if self.models_built:
            try:
                assert mod in self._models_list.keys()
                return self._models_list[mod].summary()
            except:
                print(" Sorry ! Please check the spelling of your input or the name. If its right, please check if input is the key, not the model name")
                raise ValueError("Model you choose is not in the Architectures available")
        else:
            raise ValueError("The Model you choose haven't been built yet. Please build them first or follow the guidelines")
    #
    ####################################################
    # EOF
    ####################################################
    #
    ####################################################
    # List of functions for classifier model, handled separately, but not implemented here.
    ####################################################
    #
    #
    #def classifier_model_summary(self):
    #    return self._classifier.summary()
    #
    #def classifier_model_history(self):
    #    return self._classifer.history()
    #
    #def classifier_model_fit(self,*args,**kwargs):
    #    if self._initiated:
    #        self._classifier.fit(*args,**kwargs) # This needs modification
    #
    #def classifier_model_compile(self,*args,**kwargs):
    #    if self._initiated:
    #        self._classifier.compile(*args,**kwargs) # This needs modification
    #
    ####################################################
    # EOF
    ####################################################
    #
    # There should be a separate train function that will automatically
    # call all these funcitons with appropriate parameters.
    ####################################################
    # Important Training related functions.
    ####################################################
    #
    # Should write the script to fit the model with the data already here.
    # Signed Debugging -- Working Done perfectly.
    def fit(self,*args,**kwargs):
        """This is where the actual fitting is done. Batch size dimensions and """
        X_train=self.data2
        y_train=self.target
        callback=self.call_back
        if self._initiated:
            if self._model:
                self._model.fit(x=X_train,y=y_train,callbacks=callback,*args,**kwargs)
            else:
                for mod in self._models_list:
                    #print("New Model Started :",mod)
                    #x=copy.copy(X_train)
                    #y=copy.copy(y_train)
                    #print("X Shape : ",x.shape)
                    #print("y Shape : ",y.shape)
                    model=self._models_list[mod]
                    opt=copy.copy(self.optimizer_chosen)
                    los=copy.copy(self.loss_chosen)
                    met=copy.copy(self.metric_chosen)
                    if not self.model_compiled:
                        try:
                            model.compile(optimizer=opt,loss=los,metrics=[met])
                        except:
                            pass
                    try:
                        #model.fit(x=x,y=y,epochs=self.epoch,*args,**kwargs)
                        model.fit(x=X_train,y=y_train,*args,**kwargs)
                    except:
                        print(mod)
                    else:
                        print("Finished Building Model :", mod)
            self._fit_done=True     
        else:
            print("Please select an appropriate model architecture")
            raise ValueError("Model Not Selected")
        return
    #              
    #def build_sup(self,*args,**kwargs):
    #    if self._initiated:
    #        self._model_chosen.build(*args,**kwargs)
    #    else:
    #        print("Please select an appropriate model architecture")
    #        raise ValueError("Model Not Selected")
    #
    # Signed -- Debugging done - working perfectly.
    def predict(self,in_array,*args,**kwargs):
        """Parameters specifically chosen for predict paramters."""
        # Note very important point to reshape your model appropriately.
        #print("The arrived shape is :",in_array.shape)
        #x_in=np.expand_dims(in_array,axis=0)
        x_in=in_array
        if self._fit_done:
            try:
                if self._model:
                    l=self._model.predict(*args,**kwargs)
                else:
                    l=[]
                    for each in self._models_list:
                        model=self._models_list[each]
                        #print(model.summary())
                        try:
                            l.append(model.predict(x=x_in,*args,**kwargs))
                        except:
                            pass
                return l
            except:
                raise ValueError("The proper method is not given")
        else:
            raise NotImplementedError("The fitting hasn't been done. So, you can't predict.")
    # Signed -- Debugging done - working Perfectly. But Don't use it before. Use it in connection with fit function.
    def compile_model(self,*args,**kwargs):
        """This will compile the NN Model. It will over ride certain setting like optimizer and all, but usually shouldn't be a 
        problem while training."""
        opt=self.optimizer_chosen
        los=self.loss_chosen
        met=self.metric_chosen
        #met="categorical_crossentropy"
        try:
            assert self.models_built
        except:
            print("Please build the model properly before compiling it")
            raise ValueError("Model Not Selected")       
        else:           
            if self._model:
                self._model.compile(optimizer=opt,loss=los,metrics=[met],*args,**kwargs)            
            else:
                for each in self._models_list:
                    opt=copy.copy(self.optimizer_chosen)
                    los=copy.copy(self.loss_chosen)
                    met=copy.copy(self.metric_chosen)
                    model=self._models_list[each]
                    model.compile(optimizer=opt,loss=los,metrics=[met],*args,**kwargs)
                    print(model.summary())
        self.model_compiled=True
        return
    #
    def optimize(self):
        """This function will optimize the model in case a given model is selected. If not it will simply raise an error"""
        try: 
            assert self._model
            print("Model is being optimized here")
        except:
            raise NotImplementedError("It is not possible to optimize all the given models here. Please select one to proceed with optimization.")            
        else:
            pass
            # Implement the code to optimize the model here.
            
    #def get_config(self):
    #    if not self._model_chosen == None:        
    #        try:
    #            #j=super(NN_NLP,self).get_config()
    #            return self._model_chosen.get_config()
    #        except:
    #            raise ValueError("This is presently not supported")              
    #    else:
    #        raise ValueError("Please Choose a Model First")
    #
    
    ####################################################
    # End of Overload Methods
    ####################################################

#%%
