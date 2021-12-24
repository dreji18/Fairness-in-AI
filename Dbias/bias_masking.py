# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 10:12:06 2021

@author: Deepak.Reji
"""

# loading all the packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import requests
import numpy as np
import re
import spacy
import transformers
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time

#%%
# classification model
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability

# ner model
#pip install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl
nlp = spacy.load("en_pipeline")

# masked language model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
unmasker = pipeline('fill-mask', model='bert-base-uncased')

#%%

def masking(text_input):
    # running the classification model to predict whether a sentence is biased or not
    classification_output = classifier(text_input)
    classi_out = classification_output[0]['label']
    probability = classification_output[0]['score']
    
    # running ner to extract the biased words
    doc = nlp(text_input)
    biased_words = []
    for ent in doc.ents:
         biased_words.append(ent.text)
    
    if classi_out != "Biased":
        print("The Sentence is Non biased hence no masking")
        ###--end--
    
    if classi_out == "Biased" and len(biased_words) == 0 :
        print("The Sentence is biased but the model failed to pick up portion of bias hence couldn't mask")
        ###--end--
    
    
    if classi_out == "Biased" and len(biased_words) != 0 :
            # collects bias phrases and creates multiple instances of sentences based on no. of biased words
            masked_sentences = []
            for instance in biased_words:
                masked_sentences.append(text_input.replace(instance, '[MASK]'))
            
            # run multiple instances of these masked sentences and retrieves the possible replacement words
            masked_words = []
            for masked_sentence in masked_sentences:

                masked_out = unmasker(masked_sentence)
                
                words_list = []
                for words in masked_out:
                    words_list.append(words['token_str'])
                masked_words.append(words_list) 
                masked_words_flatten = sum(masked_words, [])
                
            # a single sentence with masking based on the phrases
            bias_regex = re.compile('|'.join(map(re.escape, biased_words)))
            combined_mask_sent = bias_regex.sub('[MASK]', text_input)   
            
            return [{'masked': combined_mask_sent, 'words': biased_words}]







