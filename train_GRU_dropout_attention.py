# -*- coding: utf-8 -*-
import numpy as np
import cPickle, time, argparse
import util.seqItem

import random
import gru_tensor_pdropout_attention

#f1 score
def score_aspect(true_list, predict_list):
    
    correct = 0
    predicted = 0
    relevant = 0
    
    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == '1':
                if num < len(true_seq) - 1:
                    #if true_seq[num + 1] == '0' or true_seq[num + 1] == '1':
                    if true_seq[num + 1] != '2':
                        #if predict[num] == '1':
                        if predict[num] == '1' and predict[num + 1] != '2':
                        #if predict[num] == '1' and predict[num + 1] != '1':
                            correct += 1
                            #predicted += 1
                            relevant += 1
                        else:
                            relevant += 1
                    
                    else:
                        if predict[num] == '1':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '2':
                                    if predict[j] == '2' and j < len(predict) - 1:
                                    #if predict[j] == '1' and j < len(predict) - 1:
                                        continue
                                    elif predict[j] == '2' and j == len(predict) - 1:
                                    #elif predict[j] == '1' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1
                                        
                                    else:
                                        relevant += 1
                                        break
                                    
                                else:
                                    if predict[j] != '2':
                                    #if predict[j] != '1':
                                        correct += 1
                                        #predicted += 1
                                        relevant += 1
                                        break
    
                                
                        else:
                            relevant += 1
                            
                else:
                    if predict[num] == '1':
                        correct += 1
                        #predicted += 1
                        relevant += 1
                    else:
                        relevant += 1
                        
                            
        for num in range(len(predict)):
            if predict[num] == '1':
                predicted += 1
        
                        
        i += 1
                
    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


def score_opinion(true_list, predict_list):
    
    correct = 0
    predicted = 0
    relevant = 0
    
    i=0
    j=0
    pairs = []
    while i < len(true_list):
        true_seq = true_list[i]
        predict = predict_list[i]
        
        for num in range(len(true_seq)):
            if true_seq[num] == '1':
                if num < len(true_seq) - 1:
                    #if true_seq[num + 1] == '0' or true_seq[num + 1] == '3':
                    if true_seq[num + 1] != '2':
                        #if predict[num] == '3':
                        #if predict[num] == '1' and predict[num + 1] != '1':
                        if predict[num] == '1' and predict[num + 1] != '2':
                            correct += 1
                            #predicted += 1
                            relevant += 1
                        else:
                            relevant += 1
                    
                    else:
                        if predict[num] == '1':
                            for j in range(num + 1, len(true_seq)):
                                if true_seq[j] == '2':
                                    #if predict[j] == '1' and j < len(predict) - 1:
                                    if predict[j] == '2' and j < len(predict) - 1:
                                        continue
                                    #elif predict[j] == '1' and j == len(predict) - 1:
                                    elif predict[j] == '2' and j == len(predict) - 1:
                                        correct += 1
                                        relevant += 1
                                        
                                    else:
                                        relevant += 1
                                        break
                                    
                                else:
                                    #if predict[j] != '1':
                                    if predict[j] != '2':
                                        correct += 1
                                        #predicted += 1
                                        relevant += 1
                                        break
    
                                
                        else:
                            relevant += 1
                            
                else:
                    if predict[num] == '1':
                        correct += 1
                        #predicted += 1
                        relevant += 1
                    else:
                        relevant += 1
                        
                            
        for num in range(len(predict)):
            if predict[num] == '1':
                predicted += 1
            
                    
        i += 1
        
    precision = float(correct) / (predicted + 1e-6)
    recall = float(correct) / (relevant + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1
    


def convertToArray(train_lex,seq_size,winsize):
    trainingData = list(train_lex)
    #trainingLabel = list(label_lex)
    padding = (winsize-1)/2
    #index2word.append("<s>")
    returnArray = []
    #returnLabel = []
    #paddingIndex = index2word.index("<s>")
    for i in range(padding):
        trainingData.insert(0,seq_size-2)
        trainingData.append(seq_size-2)
    for i in range(padding,len(trainingData)-padding):
        returnArray.append(trainingData[i-padding:i+padding+1])
        #temp = [0]*len(index2label)
        #temp[trainingLabel[i-padding]] = 1
        #returnLabel.append(trainingLabel[i])
    return returnArray#,returnLabel
    
def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in xrange(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in xrange(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win, seq_size):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [seq_size - 2] + l + win/2 * [seq_size - 2]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out
    
    
def evaluate(s, out, epoch, aux, test_dict, We, vocab, lstm_attention, d, c, mixed = False):

    #store attention weight
    wt_file = open('att_weight_res', 'w')
    true_list = []
    true_a = []
    true_o = []

    pred_a = []
    pred_o = []
    
    for data in test_dict:
        nodes = data.get_nodes()
        
        for node in nodes:
            if node.word.lower() in vocab:
                node.vec = We[:, node.ind].reshape( (d, 1) )
            else:
                node.vec = 0.2 * np.random.uniform(-1.0,1.0,(d,1))

        #calculate lstm label
        h_input = np.zeros((len(data.get_nodes()) + 2, d))
        y_label = []
        ya_label = []
        yo_label = []
        index2word = []
        
        sent = []
        index = 0
        for ind, node in enumerate(data.nodes):
            if data.get(ind).is_word == 0:
                y_label.append(0)
                ya_label.append(0)
                yo_label.append(0)
                sent.append('None')
                index2word.append(len(data.get_nodes()) + 1)
                    
                    
            else:
                y_label.append(node.trueLabel)
                if node.trueLabel == 1 or node.trueLabel == 2:
                    ya_label.append(node.trueLabel)
                    yo_label.append(0)
                elif node.trueLabel == 3 or node.trueLabel == 4:
                    ya_label.append(0)
                    yo_label.append(node.trueLabel - 2)
                else:
                    ya_label.append(0)
                    yo_label.append(0)
                sent.append(node.word)
                index2word.append(index)

                for i in range(d):
                    h_input[index][i] = node.vec[i]
                index += 1
                           
               
        for i in range(d):
            h_input[len(data.get_nodes())][i] = aux['padding'][i]
            h_input[len(data.get_nodes()) + 1][i] = aux['punkt'][i]
                    
        #convert to lstm input format
        idxs = np.asarray(contextwin(index2word, s['win'], h_input.shape[0]))
        lstm_attention.dropout_layer(0.)
        ya_pred, yo_pred = lstm_attention.classify(idxs, h_input, None, 0.)
        #print ya_pred
        #print yo_pred
        pred_y = []
        '''
        for la, lo in zip(ya_pred, yo_pred):
            if la >= 0.5 and lo < 0.5:
                pred_y.append(1)
            elif la >= 0.5 and lo >= 0.5:
                pred_y.append(1 if la > lo else 2)
            elif la < 0.5 and lo >= 0.5:
                pred_y.append(2)
            elif la < 0.5 and lo < 0.5:
                pred_y.append(0)
        '''
        
        true_list.append([str(y) for y in y_label])
        true_a.append([str(y) for y in ya_label])
        true_o.append([str(y) for y in yo_label])
        #predict_list.append([str(y) for y in pred_y])
        pred_a.append([str(y) for y in ya_pred])
        pred_o.append([str(y) for y in yo_pred])
        
        #precision_aspect, recall_aspect, f1_aspect = score_aspect(true_list, predict_list)
        #precision_op, recall_op, f1_op = score_opinion(true_list, predict_list)
        precision_aspect, recall_aspect, f1_aspect = score_aspect(true_a, pred_a)
        precision_op, recall_op, f1_op = score_opinion(true_o, pred_o)
        
        '''
        #for printing attention weight
        wt_file.write(' '.join(w for w in sent))
        wt_file.write('\n')
        wt_aspect, wt_opinion = lstm_attention.att_wt(idxs, h_input, None, 0.)
        wt_file.write(' '.join(str(weight) for weight in wt_aspect))
        wt_file.write('\n')
        wt_file.write(' '.join(str(weight) for weight in wt_opinion))
        wt_file.write('\n')
        '''

    print "precision_aspect: \n", precision_aspect
    print "recall_aspect: \n", recall_aspect
    print "f1_aspect: \n", f1_aspect
    print "precision_opinion: \n", precision_op
    print "recall_opinion: \n", recall_op
    print "f1_opinion: \n", f1_op
    out.write(str(epoch))
    out.write('\n')
    out.write("aspect_precision: ")
    out.write(str(precision_aspect))
    out.write("aspect_recall: ")
    out.write(str(recall_aspect))
    out.write("aspect_f1: ")
    out.write(str(f1_aspect))
    out.write('\n')
    out.write("opinion_precision: ")
    out.write(str(precision_op))
    out.write("opinion_recall: ")
    out.write(str(recall_op))
    out.write("opinion_f1: ")
    out.write(str(f1_op))
    out.write('\n')

    wt_file.close()

# splits the training data into minibatches
def train(s, lstm_attention, aux, data, L, d, c, len_voc, train_size):

    
    error = 0.0
    size = 0

    nodes = data.get_nodes()
    
    for node in nodes:
        node.vec = L[:, node.ind].reshape( (d, 1) )

    size += len(nodes)

    #compute lstm input
    #add 'unk' and 'padding'
    h_input = np.zeros((len(data.get_nodes())+2, d))
    ya_label = []
    yo_label = []
    index2word = []
    sent = []
    word_index = 0
    
    for ind, node in enumerate(data.nodes):
        if data.get(ind).is_word == 0:
            #y_label.append(0)
            ya_label.append(0)
            yo_label.append(0)
            sent.append(None)
            index2word.append(len(data.get_nodes()) + 1)
            
        else:
            #y_label.append(node.trueLabel)
            if node.trueLabel == 1 or node.trueLabel == 2:
                ya_label.append(node.trueLabel)
                yo_label.append(0)
            elif node.trueLabel == 3 or node.trueLabel == 4:
                ya_label.append(0)
                yo_label.append(node.trueLabel - 2)
            else:
                ya_label.append(0)
                yo_label.append(0)
                
            sent.append(node.word)
            index2word.append(word_index)

            for i in range(d):
                h_input[word_index][i] = node.vec[i]
            
            word_index += 1
   
    for i in range(d):
        h_input[len(data.get_nodes())][i] = aux['padding'][i]
        h_input[len(data.get_nodes()) + 1][i] = aux['punkt'][i]
        
    
    #convert to lstm input format    
    context_words = np.asarray(contextwin(index2word, s['win'], h_input.shape[0]))
    #words = map(lambda x: np.asarray(x).astype('int32'), minibatch(context_words, s['bs']))
    #features = minibatch(feat_input, s['bs'])
    lstm_attention.dropout_layer(0.5)
    [error, grad_emb] = lstm_attention.train(context_words, h_input, None, ya_label, yo_label, s['lr'], 0.5)
    #grad_emb = lstm_attention.grad(context_words, h_input, None, ya_label, yo_label, 0.5)
    h_input -= s['lr'] * grad_emb
    aux['padding'] -= s['lr'] * grad_emb[h_input.shape[0] - 2, :]
    aux['punkt'] -= s['lr'] * grad_emb[h_input.shape[0] - 1, :]
    
    '''
    for word_batch, feat_batch, label_last_word in zip(words, features, y_label):
        #update inputs for every subsequence
        error += lstm_model.train(word_batch,h_input,feat_batch,label_last_word,s['lr'])
        grad_emb = lstm_model.grad(word_batch,h_input,feat_batch,label_last_word)
        
        h_input -= s['lr'] * grad_emb
        
        aux['padding'] -= s['lr'] * grad_emb[h_input.shape[0] - 2, :]
        aux['punkt'] -= s['lr'] * grad_emb[h_input.shape[0] - 1, :]
    '''
        
    for ind, node in enumerate(data.get_nodes()):
        L[:, node.ind] = h_input[ind].ravel()

    cost = error

    return cost, aux, L



# train qanta and save model
if __name__ == '__main__':
    

    # command line arguments
    parser = argparse.ArgumentParser(description='Attention Network for Fine-grained Opinion Mining.')
    parser.add_argument('-data', help='location of dataset', default='util/data_semEval/final_input_res15')
    parser.add_argument('-We', help='location of word embeddings', default='util/data_semEval/word_embeddings200_res15')
    parser.add_argument('-d', help='word embedding dimension', type=int, default=200)
    
    # no of classes
    parser.add_argument('-c', help='number of classes', type=int, default=3)
                    
    parser.add_argument('-b', '--batch_size', help='adagrad minibatch size', type=int,\
                        default=1)
    parser.add_argument('-ep', '--num_epochs', help='number of training epochs, can also determine \
                         dynamically via validate method', type=int, default=8)

    args = vars(parser.parse_args())
    outcome = open('outcomes_gru_tensor_pdropout_res15.txt', 'a')
    
    #build lstm model
    s = {'fold':5, # 5 folds 0,1,2,3,4
		'lr':0.07,
		'verbose':1,
		'decay':False, # decay on the learning rate if improvement stops
		'win':3, # number of words in the context window
		'bs':6, # number of backprop through time steps
		'nhidden':50, # number of hidden units
		'seed':345,
		'emb_dimension':200, # dimension of word embedding
		'nepochs':5}
  
    lstm_attention = gru_tensor_pdropout_attention.model(nh=s['nhidden'],
                            nc=3,
                            ne=100,
                            de=s['emb_dimension'],
                            cs=s['win'],
                            csv=1,
                            iteration=1,
                            featdim=0,
                            nt=20,
                            nt_=20)
   
    ## load data
    vocab, train_dict, test_dict = \
        cPickle.load(open(args['data'], 'rb'))


    We = cPickle.load(open(args['We'], 'rb'))

    print 'number of training sentences:', len(train_dict)
    print 'number of classes:', args['c']

    #add train_size
    train_size = len(train_dict)

    c = args['c']
    d = args['d']

    
    aux = {'padding':np.random.uniform(-0.2, 0.2, (d,)), 'punkt':np.random.uniform(-0.2, 0.2, (d,))}
    for tdata in [train_dict]:

        min_error = float('inf')

        for epoch in range(0, args['num_epochs']):

            lstring = ''

            # create mini-batches
            random.seed(7)
            random.shuffle(tdata)
            #batches = [tdata[x : x + args['batch_size']] for x in xrange(0, len(tdata), 
            #           args['batch_size'])]

            epoch_error = 0.0
            
            for inst_ind, inst in enumerate(tdata):
                now = time.time()

                
                err, aux, We = train(s, lstm_attention, aux, inst, We, \
                  args['d'], args['c'], len(vocab), train_size)

                lstring = 'epoch: ' + str(epoch) + ' batch_ind: ' + str(inst_ind) + \
                        ' error, ' + str(err) + ' time = '+ str(time.time()-now) + ' sec'
                print lstring

                epoch_error += err
                
                if inst_ind % 500 == 0 and inst_ind != 0:                    
                    evaluate(s, outcome, epoch, aux, test_dict, We, vocab, lstm_attention, d, c)

            # done with epoch
            print 'done with epoch ', epoch, ' epoch error = ', epoch_error, ' min error = ', min_error
            lstring = 'done with epoch ' + str(epoch) + ' epoch error = ' + str(epoch_error) \
                     + ' min error = ' + str(min_error) + '\n\n'

            # save parameters if the current model is better than previous best model
            if epoch_error < min_error:
                min_error = epoch_error
                print 'saving model...'


    outcome.close()
    

    
    



