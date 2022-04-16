# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 19:41:55 2020

@author: nurul
"""



from __future__ import print_function, division
"exec(%matplotlib inline)"
from matplotlib import pyplot as plt
import json
import random
import numpy as np
import os
import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_crimes
import sys
from debiaswe.debias import debias
from pandas import DataFrame

# embeddings = os.listdir("embeddings")
embeddings = ["Law2Vec.200d.txt"]
for i in range(len(embeddings)):
    embeddings[i] = embeddings[i].replace(".txt", "")


for embedding in embeddings:
    # load google news word2vec
    E = WordEmbedding("embeddings/"+embedding+".txt")
    
    # load professions
    crimes = load_crimes()
    crime_words = [p[0] for p in crimes]
    
    # gender direction
    try:
        v_gender = E.diff('she', 'he')
    except:
        v_gender = E.diff('woman', 'man')
    
    # # profession analysis gender
    # sp = sorted([(E.v(w).dot(v_gender), w) for w in crime_words])
    sp=[]
    for w in crime_words:
        try:
            sp.append([E.v(w).dot(v_gender),w])
        except:
            pass
    sp = sorted(sp)    
    
    df = DataFrame (sp,columns=['words','projections'])
    df.set_index('words', inplace=True)
    df.to_csv(embedding+"_word_projections_biased.csv")
    
    # Lets load some gender related word lists to help us with debiasing
    with open('./data/definitional_pairs.json', "r") as f:
        defs = json.load(f)
    # print("definitional", defs)
    
    with open('./data/equalize_pairs.json', "r") as f:
        equalize_pairs = json.load(f)
    
    with open('./data/'+embedding+'_gender_specific_full.json', "r") as f:
        gender_specific_words = json.load(f)
    # print("gender specific", len(gender_specific_words), gender_specific_words[:10])
    
    debias(E, gender_specific_words, defs, equalize_pairs)
   
    # gender direction
    try:
        v_gender = E.diff('she', 'he')
    except:
        v_gender = E.diff('woman', 'man')
    
    # # profession analysis gender
    # sp = sorted([(E.v(w).dot(v_gender), w) for w in crime_words])
    sp=[]
    for w in crime_words:
        try:
            sp.append([E.v(w).dot(v_gender),w])
        except:
            pass
    sp = sorted(sp)    
    
    df = DataFrame (sp,columns=['words','projections'])
    df.set_index('words', inplace=True)
    df.to_csv(embedding+"_word_projections_debiased.csv")
    
    E.save_w2v(embedding+"debiased2.txt",binary=False)

