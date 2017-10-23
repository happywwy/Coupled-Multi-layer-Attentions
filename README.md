# Coupled-Multi-layer-Attentions
Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms

********************README**************************************************************************

This is an instruction file for successfully running the CMLA model of the paper "Coupled Multi-Layer Attentions for Co-Extraction of Aspect and Opinion Terms" published in AAAI 2017:
http://www.aaai.org/Conferences/AAAI/2017/PreliminaryPapers/15-Wang-W-14441.pdf

****************************************************************************************************

This code makes use of theano for the implementation of GRU.

****************************************************************************************************

Please follow these steps to run the model:

1. Upload your sentence file and label file in util/data_semEval/

2. Under the folder 'util', run 
   $ python 10seqLabel.py 
   to generate structured input data.

3. Under the folder 'util', run 
   $ python 20word_embedding.py 
   to generate pre-trained word embeddings

4. Under the main folder, run 
   $ python train_GRU_dropout_attention.py 
   to train a model as well as the evaluation.

***************************************************************************************************
