# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:22:14 2021

@author: Sheshank_Joshi
"""
#%% Importing Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
import numpy as np
import dialogue
import warnings
warnings.filterwarnings("ignore")
import copy
#%% Loading the Engine
df=pd.read_csv("dataset.csv")
test=df.loc[12]
text=test["Description"]
chat=dialogue.chat_interface_handler()
#%% Testing data 2
test2=df.loc[128]
text2=test2["Description"]
"""The collaborator reports that he was working in the Ustulación and realized that the cyclone duct was obstructed and opened the door to try to unclog the same and the material detached and projected towards the employee causing small burn in the right heel."""
#%% Testing - parts
text1="The edward collaborator reports that he was working in the Ustulación and realized that"
text2="the cyclone duct was obstructed and opened the door to try to unclog the same and"
text3="the material detached and projected towards the employee causing small burn in the right heel."
#%% Creating a copy
chat2=copy.copy(chat)
chat2.debug()
#%% First Attempt - This is debugged
print("\n -------- OUTPUT--------\n",chat2.dialog_in(text),"\n-----------------------------\n")
chat2.debug()
#%% Second Attempt - This is debugged
print("\n -------- OUTPUT--------\n",chat2.dialog_in(text),"\n-----------------------------\n")
chat2.debug()
#%% Copying the data for reset
#chat3=copy.copy(chat2)
#chat3.debug()
#%% Third Attempt - This is debugged
#chat3.debug()
print("\n -------- OUTPUT--------\n",chat2.dialog_in(text),"\n-----------------------------\n")
chat2.debug()
#%%

#%% Fourth Attempt -
print("\n -------- OUTPUT--------\n",chat2.dialog_in("no"),"\n-----------------------------\n")
chat2.debug()
#%% Fifth Attempt -
print("\n -------- OUTPUT--------\n",chat2.dialog_in("2"),"\n-----------------------------\n")
chat2.debug()
#%%
################################################################
# 
#
#
#End of The First Phase
#
#
#
#
##############################################################
#%% Moving to Sup - Test
# Debug Status - First input options ---- Generating new options for Country
print("\n -------- OUTPUT--------\n",chat2.dialog_in("next_item"),"\n-----------------------------\n")
chat2.debug()

#%% First Sup - Country
# Debug Status - Country is selected here -- Generating new options for locality
print("\n -------- OUTPUT--------\n",chat2.dialog_in("0"),"\n-----------------------------\n")
chat2.debug()

#%% Second Sup - Locality
# Debug Status - Locality is selected here -- Generating new options for Industry
print("\n -------- OUTPUT--------\n",chat2.dialog_in("2"),"\n-----------------------------\n")
chat2.debug()

#%% Third Sup - Industry
# Debug Status - Industry is selected here -- Generating options for Gender
print("\n -------- OUTPUT--------\n",chat2.dialog_in("0"),"\n-----------------------------\n")
chat2.debug()

#%% Fourth Sup - Male
# Debug Status - Genre is selected here -- Generating options for Employee or Third Party
print("\n -------- OUTPUT--------\n",chat2.dialog_in("1"),"\n-----------------------------\n")
chat2.debug()

#%% Fifth Sup - Empoyee or Third Party
# Debug Status - Employee or not is selected here -- Generating options for Critical Risk
print("\n -------- OUTPUT--------\n",chat2.dialog_in("2"),"\n-----------------------------\n")
chat2.debug()

#%% Sixth Sup - Critical Risk Sector
# Debug Status - Critical Risk is selected here.
print("\n -------- OUTPUT--------\n",chat2.dialog_in("2"),"\n-----------------------------\n")
#chat2.debug()

#%% Resetting the chat
print("\n -------- OUTPUT--------\n",chat2.dialog_in("reset_chat"),"\n-----------------------------\n")
chat2.debug()


#%% Next Item - check
print("\n -------- OUTPUT--------\n",chat2.dialog_in("next_item"),"\n-----------------------------\n")
chat2.debug()

#%%
#%%

k=chat._engine.predict_AL()
#%%
votes=[]
for each in k:
    votes.append(each.argmax())
c=pd.Series(votes).value_counts()

#%%
k._lm.vocab_dict

#%%
print(chat3.choice_desc_generator(2))
#%%
_data=df.drop(["Potential Accident Level","Description"],axis=1)
_target=df["Potential Accident Level"]

#%%
score=chat._engine._lm.avg_scores
k=["l_2_order_2",'m_2_order_2']
score=score[k]

#%%

import seaborn as sns
#%%
sns.rugplot(score[k[1]])
#%%
corps=['collabor', 'work', 'realiz', 'cyclon', 'duct', 'obstruct', 'open', 'door', 'tri', 'unclog', 'materi', 'detach', 'project', 'employe', 'caus', 'small', 'burn', 'right', 'heel']
#%%

k=chat2._engine.predict_AL()
#%%

print(chat2._engine.tokens)


#%%



#%%

#%%
t1=["Can you please describe the incident a little more ?",
    "The information you have given is not sufficient enough for our analysis",
    "It seems that the description you have given isn't enough for me. Pardon me. Can you be more specific ?",
    "Would you mind describing the incident a little more ?",
    "Oops ! My systems were unable to assess the situation you described. Can you elaborate a little more ?",
    "That description is appreciated. But, my systems need a little more details. Can you elaborate ?",
    "I am very sorry. The description doesn't seem to be not match my system standards for proper assessment. Can you describe a little more ?",
    "Perhaps you missed the key details. Can you describe the incident a little more, with key details at the beginning ?",
    "Bad Luck !! My systems weren't able to assess the accident level. Can you please help me by detailing the incident a little more ?",
    "My Bad! I was unable to assess your situation. Would you please describe a little more ?"]
#%%
t2=["I am very sorry pal. The details were still not enough. Can you give more details, like below",
    "Alas ! Sorry friend. I am still unable. Will it be something like the options below",
    "That wasn't enough. Here are some example descriptions for you",
    "That wasn't enough still. Don't worry. I will help you out. Here, have a look at some examples below",
    "Sorry mate, that wasn't enough for me. You know it happens. Here are some examples to help you out",
    "Buddy ! That wasn't enough description given, even with additional data. Here, I give some examples below",
    "Tough luck Boss ! It wasn't enough for me. Can you try again. Here are some examples",
    "Oh Friend ! My systems are running hot. Couldn't figure out anything from that. I give you some details here",
    "Sir, I tried. It wasn't enough. Can you take que from some examples given below",
    "Oh Dear ! My engine doesn't seem to classify the threat. Here let me give a hint on how to use me with below examples"
    ]
#%%
t3=["Seems like the information was still not sufficient. Either choose from options below through the number or describe it more",
    "That still wasn't enough sir. Choose from below options by entering the number, or give a little more description",
    "Please choose mate, either from the contexts below or describe a little more",
    "I give you some common contexts I encountered. If you are in one of the similar ones, select them or give more description",
    "Hard luck once again friend. I will give you some examples from my experience. Choose something from one of the options or give similar description"
    ]
#%%
t4=["Tough luck mate. Just choose a valid option from below, if it comes closer to what happened. If it doesn't involve any given below, just reply \"no\"",
    "Does the incident involve any of the below. If not answer \"no\". Please make a hard choice from options",
    "I am running out of luck. Perhaps I need a proper training. Meanwhile, in a last attempt, make a hard choice if any of the following was involved. \"no\" if it doesnt",
    "From the below options, check if anything matches your description. Make a hard choice, or say \"no\"",
    "This is sad. In a final attempt, let me give you some common occurrences. Make a hard choice, if any of the below is involved, if not just say \"no\""
    ]
#%%
successful_pred="\n".join(["Thanks for using me. The above is the potential for your accident level based on the responses given by you. Please contact appropriate authorities.",
"If you want to try again, please reset using the button given and start again. I am going to take a leave by ending this conversation now.","Its a pleasure serving you."])
#%%
not_a_valid_option="The option you have selected is not in the options I have made you available. Please go through the options again, or just say \"no\", if it does not meet your criteria"
#%%
err_desc_output="\n".join(["Out of luck. Seems like your description does not match any at all. I will get ready for next session. To try again, please press reset.","If you have tried multiple times before, please check the document on how to use me.","Hey, do you know you can even train me to custom dataset. Check the website for links to that"])
#%%
welcome_msg="\n".join(["Hello user. I am Shrav.You can call me Vani too. I am here to help you prevent Industrial disaster","Please describe your indcident in detail please."])
#%%
desc_accepted="Thanks for the info. Your report is appreciated. Now, can you answer a few more questions ?"
#%%
err_sup_output="The choice you have made, doesn't seem to be in the listed options. Please choose the right option"
#%%
responses={str(0):t1,str(1):t2,str(2):t3,str(3):t4,"successful_pred":successful_pred,"not_a_valid_option":not_a_valid_option,"err_desc_output":err_desc_output,"welcome_msg":welcome_msg,"desc_accepted":desc_accepted,"err_sup_output":err_sup_output}
#%%
Countries="Can you specify, which country form the below options ?"
Local="A little more details on locality would help. Can you choose from below options ?"
ist="Industry is the most important input. Can you choose the industry where this happened from below options ?"
Genre="Was the Person injured Male or Female. Can you choose from below ?"
emp="Assuming you are the reporting person. Can you specify from below ?"
risk="I know it is tough. But, it is appreciated if you could answer what was the critical risk involved with accident, from your guess. Select from below frequently occuring risks."
#%%
responses.update({"Countries":Countries,"Local":Local,"Industry Sector":ist,"Genre":Genre,"Employee or Third Party":emp,"Critical Risk":risk})
#%%
import json
file=open("responses.json","w")
json.dump(responses,file)
file.close()
#%%
r=json.load(open("responses.json","r"))
#r.update({"welcome_msg":welcome_msg})
#%%
file=open("responses.json","w")
json.dump(r,file)
file.close()
#%%
k=r.keys()
for each in list(r.keys()):
    if each in df.columns:
        print(each)
#%%
trick="Hello World. This is edward tools bee hive"
#%%
import language_models
lm=language_models.NLP_LM()
#%%
o=lm.validity_check(trick)
#%%
k=lm.clean_test(["edward"])
#%%
from nltk.corpus import wordnet as eng_words
wor=eng_words.words()
#%%