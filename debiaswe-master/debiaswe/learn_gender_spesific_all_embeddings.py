# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:43:10 2020

@author: nurul
"""

import os


embeddings_path = "../embeddings"
embeddings = os.listdir(embeddings_path)

for i in range(len(embeddings)):
    embeddings[i] = embeddings[i].replace(".txt", "")

for embedding in embeddings:
    os.system("python learn_gender_specific.py "+embeddings_path+"/"+embedding+".txt 50000 ../data/gender_specific_seed.json "+embedding+"_gender_specific_full.json")