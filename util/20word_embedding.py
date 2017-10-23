# -*- coding: utf-8 -*-
"""
Created on Wed May 27 19:50:17 2015

@author: happywwy1991
"""

import numpy as np
import cPickle

dic_file = open("/home/wenya/Word2Vec_python_code_data/data/w2v_yelp200_10.txt", "r")
dic = dic_file.readlines()

dictionary = {}

for line in dic:
    word_vector = line.split(",")[:-1]
    vector_list = []
    for element in word_vector[len(word_vector)-200:]:
        vector_list.append(float(element))
    word = ','.join(word_vector[:len(word_vector)-200])
        
    vector = np.asarray(vector_list)
    dictionary[word] = vector
    

final_input = cPickle.load(open("data_semEval/final_input_res15", "rb"))
vocab = final_input[0]
word_embedding = np.zeros((200, len(vocab)))

count = 0

for ind, word in enumerate(vocab):
    if word in dictionary.keys():
        vec = dictionary[word]
        row = 0
        for num in vec:
            word_embedding[row][ind] = float(num)
            row += 1
        count += 1
    else:
        print word,
        for i in range(200):
            word_embedding[i][ind] = 2 * np.random.rand() - 1
    
print len(vocab)
print count
#print word_embedding


cPickle.dump(word_embedding, open("data_semEval/word_embeddings200_res15", "wb"))
