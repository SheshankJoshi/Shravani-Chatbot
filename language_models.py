# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 17:44:18 2021
This is the right hand module that has the actual language model initited based on the corpus created.
Yet to finish : The train phase of the language models. It can be mitigated or can be run locally.

NOTE : Also mention the research paper links that is going to state and connect the langauge model
probability distributions given here to the actual models.

NOTE : Training Set is assumed to have no spelling mistakes and is proof read is what is assumed. Any Nouns and Named Entities in 
training set are not entertained if not in English language. If wish to use, should have used name tagged dataset and upset the 
generalization mechanism.

NOTE TO ME : Have to build a recommendation system.
NOTE : A Destructor method needs to be called to clear all the variables and free the memory for further usage.

NOTE : A User note is required where we need to issue guidelines to the user on how to use the chatbot effectively, with most 
intensive and effective usage with best description coming within first 'n' number of words (which is a tunable parameter)
@author: Sheshank_Joshi
"""
#%%
import nltk
# Uncomment below comments while deployment.
nltk.download("stopwords")
nltk.download("punkt")
#nltk.download("words")
nltk.download("wordnet")
import pandas as pd
from nltk.lm.models import KneserNeyInterpolated,MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary #NgramCounter,
from nltk.tokenize import word_tokenize
from nltk.text import TextCollection
from nltk.util import flatten,ngrams,pad_sequence,bigrams,trigrams,ngrams
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as eng_words
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import ContextIndex
import string
from dateutil.parser import parse
from string import punctuation as punct
from textblob import Word as suggest
import pickle as pkl
import copy
import numpy as np
#%%
class NLP_LM():
    _order=2 # This is the language model we choose i.e. bigram
    _l_model={} # KneserNeyInterpolated model for vairous ngrams
    _m_model={} # Storing Maximum Likelihood based model for various ngrams
    _vocab=None # Original vocabulary of the input while training
    stemmer = PorterStemmer() # Can be tuned according to situation
    lemmatizer=WordNetLemmatizer() # Since we are using Wordnet basis for language model, this lemmatizer is required.
    word_confidence_level=0.8 # A Tunable parameter, primarily used for spelling match and configuration, but can be tuned.
    max_order=4 # A Tunable parameter indicating the maximum order for the langauge models to be considered. It is tunable.
    tok=["<s>","</s>","<UNK>"] # Indicating start and end of sequences for user inputs
    saved_file_name="lm_model_saves.pkl"
    #
    def __init__(self,corpus=None,order=None,train=False):
        """If being trained on custom data, the data should be a list of texts without tokenization or anything at all. Just a list of texts.
        Do not provide corpus if it is just a running state and not training state. Even if provided, it will be ignored.
        By default if order is specified only that particular order model is initiated and the best model for that order is chosen i.e. an Interpolated language.
        Inputs : 
            Corpus : A List of texts, or a pandas series with each text considered as a document in itself. Is required for training.
            train : If the object is initialized as a train_phase or normal use. If normal, pretrained model is loaded.
            order : Pre-chosen langauge model to be chosen. If not given, it tries all the language models upto max_order=4
        """
        self.load=None
        self.model=None
        self._fit_done=False
        self._fixed_order=False
        self.corpus=None
        self.vocab=None
        self.vocab2=None
        #self.train=None
        self._tokens=None # Access by "._texts" method.
        self.org_vocab=None
        self._train=None            
        self._corpus_clean1=None
        self._corpus_clean2=None
        self.avg_scores=[]
        self.vocab_dict=None # Used for word2idx conversions. If any word gets added to corpus it will have to be added here too.
        self.reference={} # Used for reference for readibility and stuff.
        self.context=None
        # This whole thing will go under try.
        # First build empty models if order is given or not depending on it
        if order!=None: #If you want your model to be fixed
            self._order=order
            self.set_models(self._order)
            self._fixed_order=True
            self.choose_model(model_chosen="l_"+str(self._order))
            # We are only going to copy and create class object, in case if we want to operate and run multiple 
            # processings i.e. trainings at the same time on the same machine for better testing methodologies.
            #self.model.update({"l_" + str(self._order):self._l_model[self._order].copy()})
            #self.model.update({"m_" + str(self._order):self._m_model[self._order].copy()})
        else:
            for o in range(self.max_order):
                self.set_models(o)
            self.choose_model()
        # If we later want to set the model or change the langauge model, its made available through provided functions.
        # We have to manuall call the set models function if the order is not provided while initialization time.
        # Chief chosen model concept is abandoned, though it can be appropriately activately at relevant places.
        self.given_ng=None # Bigrams of user given text # Needs to be cleared when reset
        # Trigrams of user given text # Needs to be cleared when reset
        self.given_scores={} # Given text Scores # Needs to be cleared when reset
        self.keywords=[] # Keywords in the given user description having the highest score according to differen models is taken here. If no model is specified, all models are clubbed and used.
        self.received=None
        self.given=None
        # A Train method that assimilates all these member functions 
        if train:
            if type(corpus)=="NoneType":
                raise ValueError("Corpus is not given or is not in the format specified. Make sure you give it in the proper format as spcified.")
            self.corpus=corpus #Don't forget to give the option to save the corpus once the data processing is done.
            #print(self.corpus)
            self.start()
            self._build_model()
            print("Building models Finished")
            self._train=None    
            self.save()
        else:
            try :
                #obj=NLP_LM.load() # This is wrong, it should turn on model initiation mechanism
                # This needs to be further analyzed. Saved models needs to check for first.
                NLP_LM.load(self)
                #print("----------after loading-------------")
                #print(type(self))
                #print(dir(self))
                #print(self._order)
                #print(self._fit_done)
                #print(self.model["l_1"].vocab._len)
                print("Language Models loaded Successfully")
            except:              
                raise FileNotFoundError("The pickle file for saved models is not available. Either place it current working directly with name \"lm_model_saves.pkl\".\n Or give the corpus data to train on the ordering given or default ordering.")  
    #
    def save(self):
        f=open("./model_saves/"+self.saved_file_name,"wb")
        pkl.dump(self,f)
        f.close()
        print("Language Models are saved")
    #
    @classmethod
    def load(cls,self):
        #print("--------------Before loading----------------")
        f=open("./model_saves/"+cls.saved_file_name,"rb")
        obj=pkl.load(f)
        #print(dir(obj))
        #print(obj._fit_done)
        #print("Total Vocabulary Size loaded :",obj.model["l_1"].vocab._len)
        #print("The length of the Vocabulary is :",len(obj.vocab))
        #print(obj.avg_scores)
        self.__dict__=obj.__dict__.copy() # copying the original data into memory
        f.close()
    #
    @classmethod
    def clean(self):
        """This method is going to clean and delete the objects that are created as part of trianing."""
        
        return
    # Signed --- Finished debugging, its working perfect
    def start(self):
        text=self.tokenize()
        #print("Tokenization Done")
        self._tokens=TextCollection(pd.Series(text,dtype="object")) # Access by "._texts" method.
        self.org_vocab=self._tokens.vocab()         
        self._corpus_clean1=self._tokens._texts.apply(self.clean_train) # This is the primary source for recommendations and similarity
        # After deleting unnecessary words, will have to retain the words
        self._text=TextCollection(self._corpus_clean1)
        self.clean_vocab=self._tokens.vocab()
        #print("Clean Train Done")
        self._corpus_clean2=self._corpus_clean1.apply(self.process)
        #print("Process Training Done")
        self._train,self.vocab=padded_everygram_pipeline(self.max_order,self._corpus_clean2) #This is hard-coded for safety purposes. Can be changed.
        self.vocab=list(self.vocab)
        self._train=[list(lit) for lit in self._train]
        #self.vocab1=list(set(list(flatten)))
        self.vocab2=list(set(list(flatten(list(self._corpus_clean2)))))
        #self.vocab2=Vocabulary(self.vocab.counts,unk_cutoff=1)
        #print("The Length of the Vocabulary is :",len(self.vocab2))
        self.vocab_dict=dict([(word,key) for key,word in enumerate(self.vocab2)])
        self.create_references()
        self._tokens=TextCollection(self._corpus_clean2)
        #self._tokens=None
        #self._clean_corpus1=None
        #print("Start Up Over")
    # debugging finished.    
    #This idea is abandoned, though the method is left alone and should be ignored.
    def choose_model(self,model_chosen=None):
        """Appropriate model is chosen here from the list of available models. If not provided, all the models in the list will be chosen. However
        it is not recommended for large datasets"""
        if model_chosen:
            try:
                if model_chosen in self._l_model.keys():
                    self.model=self._l_model[model_chosen] # Only copies of the models will be taken
                elif model_chosen in self._l_model.keys():
                    self.model=self._m_model[model_chosen]
                # We are also going to completely reset the model and its flavour.
                self.vocab=self.model.vocab
                self.vocab2=Vocabulary(self.vocab.tokens)
            except:
                self.model=self._m_model[model_chosen]
        else:
            self.model={} # A Dictionary of models will be created.
            # Append all the models to the current models for test case. But, this is not recommended here.
            for each in self._l_model:
                self.model.update({"l_"+str(each):self._l_model[each]})
            for each in self._m_model:
                self.model.update({"m_"+str(each):self._m_model[each]})
            # Can add other models too, if possible.
    #      
    # (Keep this). This can be called as many times as required to set the models of various n-grams or further expansions for importing other models from outside
    # Signed -- Verified debugging, working correctly.
    def set_models(self,ordering):
        """" This sets models for whatever order we ask it to and sets it as a class variable. Becomes an object attribute only after fitting"""
        self._m_model.update({str(ordering):MLE(ordering)})
        self._l_model.update({str(ordering):KneserNeyInterpolated(ordering)})
        # can add other language models too, depending upon the situation.   
    # change this
    # Signed -- verified debugging, working correctly.
    def _fit_model(self,model_chosen,train_set,vocab):
        """" This fits the models. You can either specify the model name or just pass an empty string in case of fixed order while initializing. """
        train=train_set
        vocab2=vocab
        try:
            if self._fixed_order:
                assert self._fit_done==False
                self.model.fit(train,vocabulary_text=vocab2)
                self._fit_done=True
            else:
                leng=self.model[model_chosen].vocab.counts.keys()
                assert len(leng)==0
                #print(len(leng))
                #print("The vocab size for model",model_chosen,self.model[model_chosen].vocab._len)
                self.model[model_chosen].fit(train,vocabulary_text=vocab2)                
                #print(len(leng))  
        except:
            print("The language model that is not fitting :",model_chosen)
            raise AssertionError("Fitting already done")
        
    # Signed -- verified debugging - working properly. Certified
    def _build_model(self):
        """ Fits the models and builds them up for further analysis.
        NOTE : Currently there is no checking mechanism to see if training for any particular language model is already done. 
        So, if new builds or fits are happening, its going to retrain all the models on the given dataset again."""       
        # Build a pipeline here that will create a padded pipeline and then fit into each of those models
        if self._fixed_order:
            self._fit_model(model_chosen="",train_set=self._train,vocab=self.vocab2)
        else:
            for each in self.model:
                #print(len(self._train))
                #print(each)
                self._fit_model(model_chosen=each,train_set=self._train,vocab=self.vocab2)
        self._fit_done=True
        #print("fitting Models finished.")
        #Uncomment this after debugging.
        self.avg_scores=self.set_avg_scores()
    #
    # Signed -- Debugging done - Working perfectly.
    def _give_best_words(self):
        """This takes the words already present in the user input prir given, if found any significant words in it based on the score, it will give the words that scored highest. The words with the highest score
        are picked and the input is asked more keep that word in context as in other methods."""
        try:
            self.given_scores!=None
        except:
            raise ValueError("Perhaps validation hasn't been done on atleast one entry yet. Please first try with primary user input.")
        words=[]
        scores=[]
        q=self.given_scores
        for each in q.keys():
            model_param=q[each]
            #print("Model :",each)
            for item in model_param:
                ord_param=model_param[item]
                #print("Order :", item)
                word_score,word_index=ord_param.max(),ord_param.argmax()
                words.append(self.given[word_index])
                scores.append(word_score)
                #print(word_score," : ",word_index)
                #print("Word Suggested : ",my_model.given[word_index])
            #print("-----------------------------")
        self.keywords=self.keywords+list(set(words))
        return words,scores
    #
    #This is a doubtful but not completely functional design. It needs proper checking and support.  
    # Signed -- Debugged and Working.
    def recommend_contexts(self):
        """Recommends contexts for the words based on the highest score words that are already given in the description by the user to give him an idea of 
        what might be the description. It returns some context words that might help remember or extend the thoughts of the user to input more data."""
        w,s=self._give_best_words()
        #print(w,s)
        self.context=ContextIndex(self._text)
        #print(list(set(self.keywords)))
        kw=[self.reference[wor] for wor in set(self.keywords)]
        #print(kw)
        contexts=[list(self.context.common_contexts([word]).keys()) for word in kw] # Check contexts as a list.
        contexts=[" ".join(word) for each in contexts for word in each]
        return contexts   
    # Signed -- Debugged and working.
    def recommend_desc(self): # recommendation texts for users based on the current input
        """Some concordance of top words to given some options for the words and ideas based on the ideas. So, some concordance words are given
        to make the user select statements, or given some descriptions in similar concordance"""
        lines=8
        width=50
        concord=[]
        w,s=self._give_best_words()
        kw=[self.reference[wor] for wor in set(self.keywords)]
        for word in kw:
            recs=self._text.concordance_list(word,width=width,lines=lines)
            for each in recs:
                concord.append(each.line)
        #print(concord[0])
        #for each in concord:
        #    each.left_print
        # Calling function should pick random concordances from these
        #return [print(dir(concs)) for concs in concord ]
        return concord
    #
    #Signed -- Working Correctly
    def recommend_options(self):
        """Finding some common collocations in training set and passing them as option recommendations to improve boosting the description output
        with appending words."""
        colloc_list=self._text.collocation_list(num=8,window_size=5)
        return [" ".join(each) for each in colloc_list]
    # Signed Debugged -- working as expected.
    def validity_check(self,outside_input):
        """This is all we need for every text input that is given to the chatbot. It only handles descriptions over a certain length.
        It doesn't handle anything else.
        The outputs of overall score for different models is given here. It is for analysis purpose only. Care should be given that, if any,
        past inputs should be added with the present input to get a full picture.
        Outupt:
        Scores : A Dataframe of Scores by each model on the given chunk of text, given on an average
        text : The cleaned corpus text, that is done according the the present module, so that it can used further in NN processing.
        """
        #out=[] # The list containing output data for model management engine
        text=outside_input
        text=self.tokenize(text=text)
        try:
            text2=self.clean_test(text)
            #print("Text after clean test",text2)
            text3=self.clean_train(text2)
            #print("Text clean train",text3)
        except:
            print("Exception raised : Not enough text")
            return None
        else:
            text=text3
            text=self.process(text)
            self.received=text
            #print(test)
            text=[self.tok[0]]+text+[self.tok[1]] # Appending end of sentence and start of sentence tokens for better analysis
            self.ngram_test(text)
            #Here model is not yet decided along with proper model mechanism
            scores={} # This obtained score is the score for ngram for the given model. so, here model represents the rows not columns
            for each_model in self.model:
                model=self.model[each_model]
                n=2 #staring with bigrams for contexts, though unigram can be used for individual probability scores.
                obtained_scores={}
                for each_ng in self.given_ng:
                    #print("The lengh of the ngram is :",len(each_ng))
                    current_ord=len(each_ng[0])
                    #print(current_ord)
                    # Each is a bigram, trigram and quadgram
                    #obtained_score.append(self.ngram_score_calculator(each_ng,n,model))
                    s=self.ngram_score_calculator(each_ng,n,model)
                    obtained_scores.update({current_ord:np.array(s)})
                    scores.update({each_model+"_order_"+str(n):sum(s)})
                    n+=1
                if n==self.max_order:
                    n=2
                #print(type(obtained_scores[0]))
                self.given_scores.update({each_model:obtained_scores})
                #scores.update({each_model:obtained_score}) 
            #s=[self.ngram_score_calculator(test,i+2,model) for entry  for i in range(len(test))]
            #score_bigram_l_model=self.ngram_score_calculator(self.given_bi,2,self.l_model_2)
            #score_trigram_l_model=self.ngram_score_calculator(self.given_tri,3,self.l_model_3)
            #score_bigram_m_model=self.ngram_score_calculator(self.given_bi,2,self.m_model_2)
            #score_trigram_m_model=self.ngram_score_calculator(self.given_tri,3,self.m_model_3)
            #self.given_scores.append(score_bigram_l_model)
            #self.given_scores.append(score_trigram_l_model)
            #self.given_scores.append(score_bigram_m_model)
            #self.given_scores.append(score_trigram_m_model)
            text.remove(self.tok[0]) # Removing start of sequence token
            text.remove(self.tok[1]) # Removing end of sequence token
            # Add 2 more different scores
            self.given=text
            #out=out+[sum(score_bigram_l_model),sum(score_trigram_l_model),sum(score_bigram_m_model),sum(score_trigram_m_model),text]
            #print(test)
            #print("Validity of the input is checked")
            return [pd.Series(scores),self.given]#test#self.given#out
    # Signed -- Working Correctly.
    def ngram_test(self,text_in):
        """Takes in text tokens and generates a dataframe of all kinds of ngrams added to it and gives back the dataframe"""
        #temp=pd.Series(text_in)
        #temp=pd.DataFrame(text_in,columns=["tok"])
        self.given_ng=[list(ngrams(text_in,n=order)) for order in range(2,self.max_order+1)]
        
    # Signed and approved -- Debugging done.---  Perfectly working.
    def set_avg_scores(self):
        """This calculates the average scores for given training set that is used for comparison.
        This is the heart of the language model with score setting and all. So, its crucial"""
        temp=pd.DataFrame(self._corpus_clean2,columns=["tok"])
        #max_order=5
        models_peep=[]
        i=0
        for model_given in self.model:
            try:
                i+=1
                model=self.model[model_given]
                scores=[]
                #print("-----------------------")
                #print("model :",model_given)
                for order in range(2,self.max_order+1):
                    #temp["ngrm_"+str(order)]=temp["tok"].apply(ngrams,n=order).apply(list)
                    k=temp["tok"].apply(ngrams,n=order).apply(list)
                    #print("Order :",order)
                    k=k.apply(self.ngram_score_calculator,n=order,model=model).apply(sum)
                    #k = k.rename(columns={"tok":order})
                    k.name=model_given+"_order_"+str(order)
                    #print(k.name)
                    scores.append(k)
                models_peep.append(pd.DataFrame(scores).T)
                if i==self.max_order:
                    i=0
            except:
                print("Not working for the model :",model_given)
        ret=pd.concat(models_peep,axis=1) # This is the complete dataframe created for further analysis and is the important part of language models.
        print("Langauge Model Average Scores are Set")
        return ret
    #Debugged -- Working perfectly.
    def ngram_score_calculator(self,ngram_list, n, model):
        """This create ngram model's score on a given corpus's ngram list for a given entry. The choice is ours. It will return the scores for each and every
        word in the list of words."""
        score=[]
        for each in ngram_list:
            try:
                score.append(model.score(each[0],[each[i] for i in range(1,n)]))
            except:
                print("Not working out",each)
        #print(score)
        #print("-----------------\n")
        return score
        #entry_score=[]
        #s=[]
        #for each in ngram_list:
        #    print(each)
        #    s.append(model.score(each[0],[each[i] for i in range(1,NLP_LM.max_order)]))    
        #entry_score=[model.score() for each in ngram_list]
        #print(entry_score)
        #return sum(entry_score)
    #
    def clean_test(self,tokenized_text):
        """This will handle all the unknown words that are given by the user but not in the train corpus. If not present then it will return the
        will check for spelling mistakes and if any will correct it based on certain threshold confidence level. If everything is okay, then synonyms of the word
        that are present in trained corpus will be found out and replaced appropriately."""
        text=tokenized_text
        #spell=SpellChecker()
        to_remove=[]
        for index in range(len(text)):
            each_word=text[index]
            #print(each_word)
            # Have to carefully checkout if the corpus and vocab of the set is the same.
            #print(each_word in self.org_vocab)
            if each_word in self.org_vocab:
                continue
            else :
                #print("Word not in vocabulary :",each_word)
                #First check if its in english words or not.
                if each_word in eng_words.words():
                    # Addressing the presence of new word in the description given
                    synonyms=[]
                    for synset in eng_words.synsets(each_word):
                        # creating a synonym set here
                       for lemma in synset.lemmas():
                          synonyms.append(lemma.name())    # Creating list of Synonyms
                    #print(synonyms)
                    syns=copy.copy(synonyms)
                    try:
                        assert len(synonyms)!=0
                        #syns=synonyms
                        for word in synonyms:
                            #print(word)
                            # Finding if the word is in the original vocabulary
                            if word in self.org_vocab:
                                #print("Got the Word ",word)
                                text[index]=word
                                break
                            else:
                                syns.remove(word)
                                #print(syns)
                            #print(len(syns))
                    except:
                        #print("Exception Reached")
                        #simply delete the word from the description and consider it as a foriegn language word or something
                        to_remove.append(text[index])
                    else:
                        if len(syns)==0:
                            to_remove.append(text[index])
                else:
                    # It is assumed that all spelling correct words are already included in the wordnet.
                    words = suggest("thigns").spellcheck()
                    for word,confidence in words: 
                        if confidence>self.word_confidence_level and word in self.org_vocab:
                            #Addressing spelling mistake
                            text[index]=word
                            break
        #print("Testing Clean Done")
        #print ("After clean",text)
        #print ("To delete words",to_remove)
        for word in to_remove:
            text.remove(word)
        #print("AFter cleaning",text)
        return text
    # Signed -- working properly.
    def generate_text(self,*args,**kwargs):
        """You have to supply the model you want to use from among the available models, specify the number of words by n=number_of_words, specify the random seed by giving
        kwargs random_seed=value, specify the text_seed=["list","of","words"] kind of methodology.
        Returns a list of keywords"""
        out=[]
        for model_name in self.model:
            model=self.model[model_name]
            out.append(model.generate(*args,**kwargs))
        return out
    ######################################################################
    # Do NOT Touch this section. It is static.
    ######################################################################
    #
    #Keep this
    # Signed and approved --  working after debug (perfect)
    def tokenize(self,text=None):
        """This is a simple tokenizer for data analysis and generating tokens."""
        k=[]
        if type(text)==str:
            k=word_tokenize(text)     
        else:
            try:
                for each in self.corpus:
                    #print("-----------------------")
                    #print(each)
                    k.append(word_tokenize(each))
            except:
                raise TypeError("Given value is not a String type")
        #print(k)
        #print("Tokenization Done")
        return k
    # 
    # Signed and approved -- working after debug
    def process(self,text_token_list):
        """lemmatize and stem the words"""
        text=text_token_list
        text=[self.lemmatizer.lemmatize(word) for word in text]
        text=[self.stemmer.stem(word) for word in text]
        #print("Processing Done")
        return text
    #
    # Signed and approved -- working after debug
    def clean_train(self,texts_token_list):
        """"This will clean the data with elimination of punctuations and non-english words, and stop words"""
        text=texts_token_list
        text=[word for word in text if word not in punct]
        text=[word for word in text if not word.isdigit()]
        text=[word for word in text if word in eng_words.words()]
        stopwords = sw.words('english')
        text=[word.lower() for word in text if word not in stopwords]
        #print("Training Clean Done")
        return text
    #
    #
    def create_references(self):
        """This creates references for the dataset, so that information can be properly conveyed and
        sent to the user. This is used so that stemming can be reversed to make the data more readable."""
        org_data=self.clean_vocab
        stem_data=self.process(org_data)
        self.reference=dict(zip(stem_data,org_data))
    #
    ####################################################################
    #
    ########################################################################    
        
        
        
    
#%%

