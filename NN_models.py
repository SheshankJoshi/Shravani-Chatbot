# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 00:28:17 2021
This is the core file where all the models are stored as they are.

@author: Sheshank_Joshi
"""
#%%
import tensorflow as tf
#from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, MultiHeadAttention, Conv1D, Dropout
from tensorflow.keras import regularizers as reg
from keras_self_attention import SeqSelfAttention
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import CategoricalCrossentropy
#from tensorflow.keras.metrics import CategoricalAccuracy
#from tensorflow.keras import models
#%%
class Simple_LSTM(tf.keras.Model):
    """Default values are set here based on testing stage. But, they can always be set different values by the
    management engine."""
    act="relu" # Activation layer that can be used for various layers
    _sm="softmax" # Softmax input for last layer. This can't be changed.
    drop_out=0.2 # Resettable dropout for the entire class
    lr=0.01 # Predefined learning rate can be tuned
    los="categorical_crossentropy" # Loss function can be returend later on
    opt="adam" # Optimizer function that can be retuned later on
    met=["categorical_accuracy"] # List of metrics that can be monitored from manager side.
    def __init__(self, vocab_size, embedding_dim, units_list, length_of_sequence):
        super(Simple_LSTM, self).__init__(name="Simple_LSTM")
        self.units = units_list
        #self.callback=callback
        self.embedding = Embedding(vocab_size, embedding_dim,trainable=True,input_length=length_of_sequence)
        self.layer1=LSTM(self.units[0],activation=self.act,recurrent_dropout=0.2,dropout=self.drop_out)
        self.layer2=Dense(self.units[1],activation=self.act,kernel_regularizer=reg.L1(l1=self.lr))
        self.layer3=Dense(self.units[2],activation=self._sm,kernel_regularizer=reg.L2(l2=self.lr))
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.layer1(x)
        x=self.layer2(x)
        outputs=self.layer3(x)
        return outputs
    
    
#%%
#k=Simple_BiLSTM(25,25,[12,15,10,5],4)
#%%
class Simple_BiLSTM(tf.keras.Model):
    """This model has total 3 layers. Total Number of units of each layer is specified in units_list, in appropriate order. If not enough parameters are not provided,
    it will throuigh out of index error. So, be careful."""
    act="relu" # Activation layer that can be used for various layers
    _sm="softmax" # Softmax input for last layer. This can't be changed.
    drop_out=0.2 # Resettable dropout for the entire class
    lr=0.01 # Predefined learning rate can be tuned
    loss="categorical_crossentropy" # Loss function can be returend later on
    optimizer="adam" # Optimizer function that can be retuned later on
    metrics=["categorical_accuracy"] # List of metrics that can be monitored from manager side.
    def __init__(self, vocab_size, embedding_dim, units_list, length_of_sequence):
        super(Simple_BiLSTM, self).__init__(name="Simple_BiLSTM")
        self.units = units_list
        #self.callback=callback
        self.embedding = Embedding(vocab_size, embedding_dim,trainable=True,input_length=length_of_sequence)
        self.drop1=SpatialDropout1D(self.drop_out)
        self.layer1=Bidirectional(LSTM(self.units[0],dropout=self.drop_out,return_sequences=True,activation=self.act,recurrent_dropout=0.2))
        self.layer2=LSTM(self.units[1],activation=self.act,dropout=self.drop_out,recurrent_dropout=0.2)
        self.layer3=Dense(self.units[2],activation=self.act,kernel_regularizer=reg.L1(l1=self.lr))
        self.layer4=Dense(self.units[3],activation=self._sm,kernel_regularizer=reg.L1L2(l1=self.lr,l2=self.lr))
    
    def call(self, inputs,training=False):
        x = self.embedding(inputs)
        #if training:
        x=self.drop1(x)
        x = self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        outputs=self.layer4(x)
        return outputs
#%%
#k=Simple_BiLSTM(25,25,[12,15,10,5],4)
#%%
class SelfAtt_LSTM(tf.keras.Model):
    act="relu" # Activation layer that can be used for various layers
    _sm="softmax" # Softmax input for last layer. This can't be changed.
    drop_out=0.2 # Resettable dropout for the entire class
    lr=0.01 # Predefined learning rate can be tuned
    los="categorical_crossentropy" # Loss function can be returend later on
    opt="adam" # Optimizer function that can be retuned later on
    met=["categorical_accuracy"] # List of metrics that can be monitored from manager side.
    attention_width_given=15
    attention_activation_given="sigmoid"
    def __init__(self, vocab_size, embedding_dim, units_list, length_of_sequence):
        super(SelfAtt_LSTM, self).__init__(name="SelfAtt_LSTM")
        self.units = units_list
        #self.callback=callback
        self.embedding = Embedding(vocab_size, embedding_dim,trainable=True,input_length=length_of_sequence)
        self.drop1=SpatialDropout1D(self.drop_out)
        self.layer1=Bidirectional(LSTM(self.units[0],activation=self.act,return_sequences=True,recurrent_dropout=0.2))
        self.att=SeqSelfAttention(attention_width=self.attention_width_given,attention_activation=self.attention_activation_given)
        self.layer2=LSTM(self.units[1],activation=self.act,dropout=self.drop_out,recurrent_dropout=0.2)
        self.layer3=Dense(self.units[2],activation=self.act,kernel_regularizer=reg.L1(l1=self.lr))
        self.layer4=Dense(self.units[3],activation=self._sm,kernel_regularizer=reg.L1L2(l1=self.lr,l2=self.lr))
    
    def call(self, inputs,training=False):
        x = self.embedding(inputs)
        if training:
            x=self.drop1(x,training=training)
        x=self.att(x)
        x = self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        outputs=self.layer4(x)
        return outputs
#%%
#k=SelfAtt_LSTM(25,25,[12,15,10,5],4)
#%%
class SelfAtt_cBiLSTM(tf.keras.Model):
    act="relu" # Activation layer that can be used for various layers
    _sm="softmax" # Softmax input for last layer. This can't be changed.
    drop_out=0.2 # Resettable dropout for the entire class
    lr=0.01 # Predefined learning rate can be tuned
    los="categorical_crossentropy" # Loss function can be returend later on
    opt="adam" # Optimizer function that can be retuned later on
    met=["categorical_accuracy"] # List of metrics that can be monitored from manager side.
    attention_width_given=15 # The sequence length to which attention is applied.
    attention_activation_given="sigmoid" # 
    cKernel_size=4 # The Kernel size for the convolution
    cFilters=200 # No of convolution filters that are placed.
    cPadding="same" # padding that is requied for Convolutional layer
    #
    def __init__(self, vocab_size, embedding_dim, units_list, length_of_sequence):
        super(SelfAtt_cBiLSTM, self).__init__(name="SelfAtt_cBiLSTM")
        self.units = units_list
        #self.callback=callback
        self.embedding = Embedding(vocab_size, embedding_dim,trainable=True,input_length=length_of_sequence)
        self.drop1=SpatialDropout1D(self.drop_out)
        self.conv=Conv1D(filters=self.cFilters,kernel_size=self.cKernel_size,padding='same')
        self.drop2=Dropout(self.drop_out)
        self.layer1=Bidirectional(LSTM(self.units[0],dropout=self.drop_out,activation=self.act,return_sequences=True,recurrent_dropout=0.2))
        self.att=SeqSelfAttention(attention_width=self.attention_width_given,attention_activation=self.attention_activation_given)
        self.layer2=LSTM(self.units[1],activation=self.act,dropout=self.drop_out,recurrent_dropout=0.4)
        self.layer3=Dense(self.units[2],activation=self.act,kernel_regularizer=reg.L1(l1=self.lr))
        self.layer4=Dense(self.units[3],activation=self._sm,kernel_regularizer=reg.L2(l2=self.lr))
    
    def call(self, inputs,training=False):
        x = self.embedding(inputs)
        if training:
            x=self.drop1(x,training=training)
        x=self.conv(x)
        if training:
            x=self.drop2(x,training=training)
        x=self.layer1(x)
        x=self.att(x)
        x=self.layer2(x)
        x=self.layer3(x)
        outputs=self.layer4(x)
        return outputs
#%%

#%%
class MultiAtt_cLSTM(tf.keras.Model):
    act="relu" # Activation layer that can be used for various layers
    _sm="softmax" # Softmax input for last layer. This can't be changed.
    drop_out=0.2 # Resettable dropout for the entire class
    lr=0.01 # Predefined learning rate can be tuned
    los="categorical_crossentropy" # Loss function can be returend later on
    opt="adam" # Optimizer function that can be retuned later on
    met=["categorical_accuracy"] # List of metrics that can be monitored from manager side.
    attention_heads=3 # The number of heads for Multihead-Attention
    dimension_keys=3 # The number of key dimensions for the Multiple heads.
    cKernel_size=4 # The Kernel size for the convolution
    cFilters=100 # No of convolution filters that are placed.
    cPadding="same" # padding that is requied for Convolutional layer
    #
    def __init__(self, vocab_size, embedding_dim, units_list, length_of_sequence):
        super(MultiAtt_cLSTM, self).__init__(name="MultiAtt_cLSTM")
        self.units = units_list
        self.embedding = Embedding(vocab_size, embedding_dim,trainable=True,input_length=length_of_sequence)
        #self.drop1 = SpatialDropout1D(self.drop_out)
        self.conv = Conv1D(filters=self.cFilters,kernel_size=self.cKernel_size,padding='same')
        self.mAtt=MultiHeadAttention(num_heads=self.attention_heads,key_dim=self.dimension_keys)
        #self.drop2 = Dropout(self.drop_out)
        #self.layer1=Bidirectional(LSTM(self.units[0],return_sequences=True,activation=self.relu,recurrent_dropout=0.2))
        #self.att=SeqSelfAttention(attention_width=self.attention_width_given,attention_activation=self.attention_activation_given)
        self.layer1=LSTM(self.units[0],activation=self.act,dropout=self.drop_out,recurrent_dropout=0.2)
        self.layer2=Dense(self.units[1],activation=self.act,kernel_regularizer=reg.L1(l1=self.lr))
        self.layer3=Dense(self.units[2],activation=self._sm,kernel_regularizer=reg.L2(l2=self.lr))
    #
    def call(self, inputs,training=False):
        x=self.embedding(inputs)
        x=self.conv(x)
        x=self.mAtt(x,x)
        x=self.layer1(x)
        x=self.layer2(x)
        outputs=self.layer3(x)
        return outputs
#%%
#k=MultiAtt_cLSTM(25,25,[12,15,10,5],4)
#%%
