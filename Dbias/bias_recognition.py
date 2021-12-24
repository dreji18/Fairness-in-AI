# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:56:06 2021

@author: Deepak.Reji
"""

# loading packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import spacy

#%%
# ner model
#pip install https://huggingface.co/d4data/en_pipeline/resolve/main/en_pipeline-any-py3-none-any.whl
nlp = spacy.load("en_pipeline")

#%%
# running ner to extract the biased words
def recognizer(text_input):
    doc = nlp(text_input)
    biased_words = []
    for ent in doc.ents:
         biased_words.append({'entity': ent.text, 'start':ent.start_char, 'stop':ent.end_char, 'label':ent.label_})
         
    return biased_words
         