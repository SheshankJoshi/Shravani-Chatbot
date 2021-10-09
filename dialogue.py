# -*- coding: utf-8 -*-
"""
This file contains the Dialogue Management Enginer, the first layer behind the Core Flask application
@Sheshank Joshi

NOTE : Spelling handling and the information is not yet handled. It is delegated to the language model. 
However, spelling mistakes in yes or no options is not entertained, though the case-sensitive issue 
can be eliminated.
NOTE : Delete method yet to be implemented that will log the data that is being generated here for 
future training purposes.
NOTE : Flask handle will destroy the existing object and create a new object.
"""
#%%
import random
import json
import model_management_engine as me
import copy
#%% Cell

class chat_interface_handler():
    # Don't forget to write functions for data preparations and language interfacing.
    _engine=None
    def __init__(self):
        """The state-machine has three states. 1. Dialogue is yet to be input 2. Dialogue input has been given
        but the score is not appropriate to take decision. 3. Dialogue state is satisfied and next input is to be taken"""
        self._engine=me.model_management_engine()
        self._current_desc_state=1 #(This is for debugging purposes only) #1 # Description state analyze and send the appropriate data and check on it.
        self.inps=[]
        self.outs=[]
        self.dialogues={} # This is the actual storage of the data where everything else is stored.
        self.current_state=1 #2 #(This is for debugging purposes only) #1 # Indicating the Nodal point of the dialogflow ( Crucial parameter for smooth flow of dialogue)
        self.NN_prediction=None # This prediction is a rough interfacing
        self.text_in=""
        self.score=0 # Indicative of whether the processing of description already done.
        self.target_name=None
        self.responses=None # Set of responses based on the dialogue flow state.
        self.response_state=0
        self.col=None # This is the columns of the original dataset as per requirements. A better mechanism for interfacing is needed.
        self._AL=None # Indicating which column we are considering given we already have text description
        self.return_statement="" # WE have to put default entry level introduction text here. Handled by initializer method.
        self.supervised_entry={} # This is where we collect the actual data for supervised learning model
        self.col_entry={} # This is the dictionary holder for current state in dialog values
        self.supervised_data=self._engine.options # Keys should be strings here.
        self.no_of_cols=len(self.supervised_data.keys())-1
        # Entry is of type parameter with an id and a dictionary associated with possible values.
        self.options_given=None
        self.load_response() # Loading the standard responses from saved file.
    #
    # Printing Welcome message as soon as the machine is initialized is not yet done. Do it.
    #    
    #
    def debug(self):
        print("\n------------DEBUG---------------")
        print("Current Description State :",self._current_desc_state)
        print("Current State of the Chatbot :",self.current_state)
        print("Current Description Score :",self.score)
        print("Current Response state for Description :",self.response_state)
        print("Current Col :",self.col)
        # print("Paramter at hand :",self._parameter)
        print("Current Supervised Entry :",self.supervised_entry)
        print("------------------------------------")
        print("Options given to the user :\n",self.options_given)
        print("-----------------------------------\n")
    #
    #
    def load_response(self):
        file=open("./model_config/responses.json","r")
        resp=json.load(file)
        self.responses=resp
        file.close()
        f=open("./model_config/NLP_data.json","r")
        AL=json.load(f)
        self._AL=AL
        _=self._AL.pop("best_model")
        self.target_name=list(self._AL.keys())[0]
        self._AL=self._AL[self.target_name]
        print("Succesfully Loaded Responses")
        #print(self._AL)
    #   
    #
    def dialog_in(self,text):
        """"This will store the temporary input and based on it will handle what is the current state in the dialog flow and to which
        handler data needs to be output to and return appropriate response from the response methods."""
        #print("Dialog in :",text)
        self.return_statement=""
        #state=self.current_state
        #print("The Number of columns in the Data :",self.no_of_cols)
        #print("Current State :",self.current_state)
        try:
            assert self.current_state<self.no_of_cols+2 # This exception means that dialogue management has finished and prediction is over. Ask user to reset the options
            if text=="reset_chat":
                raise ValueError
            if text=="next_item": # This is the request from client to take the conversation to next step. Its a magic code. Initialized after the page is loaded.
                if self._current_desc_state==1:
                    self.initializer(kind="desc")           
                    # Description initializer
                if self._current_desc_state==3:
                    self.initializer(kind="sup")
                    self.return_statement=self.responses["desc_accepted"]+"\n"+self.return_statement
            else:
                self.inps.append(text)
                if self._current_desc_state==3:
                    # This means the description event has been successfully accepted and now we are moving towards supervised model
                    self.supervised_data_handler()
                else :
                    # This means the descritpion event hasn't been finished yet.
                    self.dialogue_handler()
        except:
            txt="I am closing this session. Please Reset to use me again."+"\n"+" Or Let me reset it for you. Just type \"reset_chat\" in the chatbox"
            if text=="reset_chat":
                self._current_desc_state=1
                self.response_state=0
                self.supervised_entry={}
                self.score=0
                self.current_state=1
                self.options_given=None
                txt="Thanks for coming back. Start reporting a new incident again. Describe it"
            # Flask will re-initialize the state of resetting the object and clearing everything. Old object is logged into database.
            self.return_statement=txt
        else:
            #print("The Number of columns in the Data :",self.no_of_cols)
            #print("Current State :",self.current_state)
            if self.current_state==self.no_of_cols+2:
                predicted_AL=self._engine.predict_AL()
                #print("Predicted Accident Level",predicted_AL)
                #print("Supervised Model Data Collected :",self.supervised_entry)
                self.supervised_entry.update({self.target_name:self._AL[str(predicted_AL)]})
                # This is the crucial moment of the code.
                predicted_PAL=self._engine.predict_sup(self.supervised_entry)
                statement=self.responses["successful_pred"]
                # Your accident level is predicted to be ____ with confidence ____
                statement = predicted_PAL +"\n\n"+ statement
                self.return_statement=statement
        return self.return_statement
    #
    #
    #
    #            
    def supervised_data_handler(self):
        """Supervised Data handling should happen with the data points. This handles all the supervised Learning model data."""
        entry=self.inps[-1]
        #if self.col==None:
        #    self.col_entry=self.supervised_data[self.current_state]
        #    self.col=list(self.col_entry.keys())
        #print("Entry Received :",entry)
        try:
            self.choice_sup_analyzer(entry)
        except:
            self.return_statement = self.responses["err_sup_output"] #+ "\n\n" + self.outs[-1]
            # Here is the key error
            # 
            # "That's not a valid choice. Please on"
            # Should say its not a valid choice. please select proper data
        else:
            # Write code here to generate more options
            #self.current_state+=1
            if self.current_state==self.no_of_cols+2:
                #print("End has been reached")
                pass
            else:
                self.choice_sup_generator()
    # Generates next set of options for the next state.
    #
    #
    #
    def choice_sup_generator(self):
        """Generating supervised model options based on current dialog flow state."""
        self.col=list(self.col_entry.keys())[0]
        #print("Column selected",self.col)
        #print(self.col_entry[self.col])
        choices=self.col_entry.pop(self.col)
        #print(choices)
        if len(choices)==0:
            #This indicates the end has been reached.
            self.current_state+=1
        text="\n".join([str(key) + " . " + str(value) for key,value in enumerate(choices)])
        self.options_given=text
        statement=self.responses[self.col]
        #print(statement)
        self.return_statement=statement+"\n"+text
        self.outs.append(text)
        #self.dialogues["server"].append(text) #Remove this after debugging.
        #print("Response given :\n",self.return_statement)
    #
    #
    #
    #
    def choice_sup_analyzer(self,entry):
        """ This crucially analyzes if the option input is the valid option or not. If not sends appropriate responses or mitigates it further"""
        choices=self.supervised_data[self.col]
        state=self.current_state
        #print("-----------INPUT ANALYSIS------------------")
        for key,value in enumerate(choices):
            if str(key) == entry or str(value) == entry: # Checking if the project
                self.supervised_entry.update({self.col:value}) # One column data entry update is finished
                self.current_state+=1
                break
            #print("Key :",key,"Value :",value)
        #print("-------------------------------------------")
        try:
            assert not (state==self.current_state)
        except:
            raise ValueError
    #
    #
    #
    #
    def initializer(self,kind):
        """This will initialize the interface to face various options for the model under consideration."""
        if kind=="desc":
            try: 
                assert self._current_desc_state==1
                self.return_statement=self.responses["welcome_msg"]
            except:
                print("There is some fatal error somewhere")
                pass
        elif kind=="sup":
            try:
                assert self._current_desc_state==3
                assert self.current_state>1
                self.options_data=None
                try:
                    temp=self.supervised_data.pop(self.target_name)
                except:
                    pass
                # This is hard coded here. Take care of this.
                #self._AL = dict([(str(key),value) for key,value in enumerate(temp)])
                self.col_entry=copy.copy(self.supervised_data)
                self.choice_sup_generator()
            except:
                print("There is some fatal error somewhere")
                raise ValueError("There is a fatal Error Somewhere")
        else:
            print("There is a fatal error somewhere while initializing, check out.")
    #
    #
    #
    #
    def dialogue_handler(self):
        """diagloues are designed such that whenever a new input comes in, only the latest input is considered while the processing is happening in the background.
        This handles all the dialogues that are input"""
        #print("Entered Dialogue Handler")
        txt=self.inps[-1]
        if self._current_desc_state==1:
            self.dialogues.update({"user":[],"server":[]})
            try:
                self.dialogue_analyze(txt)
                # Write code here in case of a success
                self.return_statement=self.return_statement+self.responses["desc_accepted"]
            except ValueError:
                #print("Error Generated. Description was not sufficient")
                ret_statement=self.lm_error_msg_generator()
                #print(ret_statement)
                self.dialogues["server"].append(ret_statement)
                self._current_desc_state=2
                self.return_statement=ret_statement
            else:
                self.dialog_in("next_item")
                #make function elaborate in dialogue system while changing states.
        elif self._current_desc_state==2:
            try:
                assert self.response_state<4 # this is a tunable parameter. It can be modified on how many attempts user is allowed to make at describing event.
            except:
                # This means the attempts have been exhaust and the user is inputting some 
                # irrelevant data so the dialogues should restart and end. Implementation of that is pending.
                #print("Error ! It seems that you are trying to give a description of event that hasn't occurred or is not according to ")
                #print("this is a debug statement for everything else")
                self.return_statement=self.responses["err_desc_output"] # Ask to reset and try again.
            else:
                self.additional_desc_handler(txt)
    #
    #
    #
    def additional_desc_handler(self,txt):
       """After the first two attempts at getting description failed, this will handle the yes or not option for adding data from the user based on language model inputs"""
       text=txt # Taking in the latest input
       #print("The present response State is :",self.response_state)
       if self.response_state>2:
           self.choice_desc_analyzer(text)
           # A Yes or No Question is being asked here, based on most possible bigram correlation. If no is given as input            
       elif self.response_state<=2:
           try:
               self.dialogue_analyze(text) # Only to be called when choice handler fails
               # Way to generate options is not taken
           except ValueError:
               ret_statement=self.lm_error_msg_generator()
               # Script to generate options data.
               #self.dialogues["server"].append(ret_statement)
               #self.response_state+=1
               self.return_statement=ret_statement
           else:
               self.dialog_in("next_item")
               self._current_desc_state+=1
               self.current_state+=1
               #expression to handle if there is no exceptions i.e. details were enough.
       else:
           print("There is something wrong. This shouldn't be called.")
    #
    #
    #
    #
    def dialogue_analyze(self,txt):
        """The core function that will analyze the dialgoue and decide if there is enough score for given description."""
        #self.dialogues["user"].append(text)
        #print(txt)
        #print("Enter Dialogue Analyze \n-------------------------------\n")
        self.text_in=self.text_in+ " " + txt # Storing the description text for future analysis
        score,corp=self._engine.fetch_score(self.text_in) # Languge model score extraction along with obviously cleaned dataset comes out.
        std_score=self._engine.avg_score()
        #print("Score found is :",score)
        #print("Average Score is :",std_score)
        if score>=std_score:# if score is greater than average score
            #print("The corpus received is :",corp)
            #print("The Score for the input found satisfactory is :",score)
            self.NN_prediction=self._engine.predict_rough(corp)
            #print("Rough Prediction is :",self._engine.AL_prediction)
             # Fetching rough score from the engine for go-ahead.
            try :
                assert self.NN_prediction == True # rough prediction for threshold analysis approved or not
                #print("NN Prediction Test passed")
            except:
                raise ValueError
            else:
                self._current_desc_state=3 # Indicating successful description grasp and status change
                self.score=score # Storing the scoring paramters for confidence in future
                self.current_state+=1 # Changing to next node in dialog flow
                self.dialogues["user"].append(self.text_in) # Finalized user input is taken as an input successfully capture at one particular state
        else:
            self._current_desc_state=2 # Marking that more description is needed.
            raise ValueError
    #
    #
    #
    #            
    def choice_desc_analyzer(self,txt):
        """Checks if the user has input a choice in terms of 1 or 2 or 3, or, Yes or No or not known, if its not yes or no unknown is default taken
        and based on that it either returns """
        # Also need a choice creator via lm_error_msg_generator
        # A small text analysis code here to convert to lower code or to convert number to option
        text="" # lower the cases and everything involved
        try:
            assert self.options_given
        except:
            raise NotImplementedError("The Particular Dialogue state hasn't been reached. Please check program")
        if not txt.isalpha():
            try:
                assert str(txt) in self.options_given.keys()
                added_data=self.options_given[txt]
                self.dialogue_analyze(added_data)
            except:
                self.return_statement=self.responses["not_a_valid_option"]
                raise ValueError("Not a valid option")
        else:
            # This means one of the options given is not selected.
            try:
                text=txt.lower()
            except:
                print("Invalid option") # Error handler appropriately.
            else:
                if self.response_state==2:
                    if text!="no":
                    # handler for bigram in texts; a yes or no question on bigram check
                        self.dialogue_analyze(text)
                    else: # or "1" is also accepted as possible choice
                        #Jump to asking for trigram and call the error message generator
                        ret_statement=self.lm_error_msg_generator()
                        self.dialogues["server"].append(ret_statement)
                        self.return_statement=ret_statement
                        self.response_state==3
                elif self.response_state==3:
                    if text!="no":
                        # handler for trigram in texts; a yes or no question on trigram check
                        self.dialogue_analyze(text)
                    else:
                        # Case of absolute failure
                        self.response_state=4
                        self.dialogue_handler()
                else:
                    print("There is something wrong here.")
    #                
    #
    #
    #
    def lm_error_msg_generator(self):
        """Language model error handler that handles exclusively language model errors and generates dynamic new messages based on current state
        of the chat for present description"""
        #num=random.randint(1,10)
        #print("The present response State is :",self.response_state)
        try:
            assert self._current_desc_state==2
            if self.response_state==0:
                # This is the primary response indicating that user's description wasn't enough.
                #self.response_state+=1
                #print("This is reached means first warning that description wasn't enough")
                response_text=self.responses[str(self.response_state)][random.randint(0,9)]
            elif self.response_state==1:
                # simple response of asking for more information giving example descriptions randomly
                choices=self.choice_desc_generator(choice=1)
                response_text=self.responses[str(self.response_state)][random.randint(0,9)]
                response_text=response_text+"\n"+choices
                #print("This is reached means second warning and choices are given")
                #self.response_state+=1
            else:
                # This handles user suggestions on frequently occurring bigrams and incident mechanisms.
                response_text=self.responses[str(self.response_state)][random.randint(0,4)]
                if self.response_state==2:
                    # Handle the extra tests here with appropriate choice generator method
                    # self.bigram_used=lm.get_bigram(self.text_in) # determine the bigram used # Write code that delegates this to the engine
                    choices=self.choice_desc_generator(choice=2)
                    self.options_given=choices
                    response_text=response_text + "\n" + choices # Use some most frequent bigram words with collocation sentences
                    
                elif self.response_state==3:
                    # self.trigram_used=lm.get_trigram(self.text_in) # determine the trigram used # Write code that delegates this to the engine
                    choices=self.choice_desc_generator(choice=3)
                    self.options_given=choices
                    response_text=response_text + "\n" + choices # Use some most frequent trigram words with collocation sentences
                else:
                    print("Error debug statement. Something is wrong.")
            self.outs.append(response_text) # append it as out
            #print("\n---------------------\nFrom inside Error Message Generator :")
            #print(response_text)
            #print("-----------------------------------")
            self.response_state+=1
            #print("Changed Response State :",self.response_state)
            return response_text
        except:
            raise ValueError ("funciton is triggered from ")
    #
    #
    #
    #        
    def choice_desc_generator(self,choice):
        """Checks if the description given is according to the choices or not after first two options are exhaust"""
         # Check if the code has bigram and trigrams both of them same. If same, move to the next one.
        texts=[]
        if choice == 1:
            texts=self._engine.generate_options(option=1)
        else:
            if choice == 2:
                texts=self._engine.generate_options(option=2)#- Option generating for common contexts
                for i in range(len(texts)):
                    texts[i]=str(i)+ " . " + texts[i]
                 # choice generator for bigram data with third or fourth bigram
                #self.options_given=dict([(key+1,value) for key,value in enumerate(texts)])
            elif choice==3:
                texts=self._engine.generate_options(option=3)#- Option generating for concordance
                # Choice generator for trigram data
                for i in range(len(texts)):
                    texts[i]=str(i)+ " . " + texts[i]   
            self.options_given=dict([(key+1,value) for key,value in enumerate(texts)])
        #print("Given options are :",self.options_given)
        return "\n".join(texts)
    #
    #
    #
    #