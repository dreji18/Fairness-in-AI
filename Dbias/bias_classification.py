# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:39:10 2021

@author: Deepak.Reji
"""

# loading all the packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import transformers
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

#%%
# classification model
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer) # cuda = 0,1 based on gpu availability

#%%
def classify(text_input):    
    # running the classification model to predict whether a sentence is biased or not
    classification_output = classifier(text_input)
    # classi_out = classification_output[0]['label']
    # probability = classification_output[0]['score']
    
    return classification_output



