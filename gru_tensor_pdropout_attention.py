import theano
import numpy
import os
import cPickle

from theano import tensor as T

# from collections import OrderedDict
from theano.compat.python2x import OrderedDict

dtype = theano.config.floatX
uniform = numpy.random.uniform
sigma = T.nnet.sigmoid
softmax = T.nnet.softmax

srng = T.shared_randomstreams.RandomStreams(1234)
 
class model(object):

    #def __init__(self, nh, nc, ne, de, cs, featdim, em=None, init=False):
    def __init__(self, nh, nc, ne, de, cs, csv, iteration, featdim, nt, nt_):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings
        cs :: word window context size 
        iter :: number of memory iterations
        '''
        # parameters of the model

        self.featdim = featdim

        # weights for LSTM
        n_in = de * cs
        n_hidden = n_i = n_c = n_o = n_f = nh
        n_y = nc
        #n_v = n_hidden * 2
        #n_v = 2 * nt
        n_v = nt + nt_
        n_inv = n_v * csv

        # forward weights
        self.Wxi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_i)).astype(dtype))
        self.Whi = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_i)).astype(dtype))
        #self.Wci = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_i)).astype(dtype))
        self.bi = theano.shared(numpy.zeros(n_i, dtype=theano.config.floatX))
        self.Wxf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_f)).astype(dtype))
        self.Whf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_f)).astype(dtype))
        #self.Wcf = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_f)).astype(dtype))
        self.bf = theano.shared(numpy.zeros(n_f, dtype=theano.config.floatX))
        self.Wxc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_c)).astype(dtype))
        self.Whc = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_c)).astype(dtype))
        self.bc = theano.shared(numpy.zeros(n_c, dtype=theano.config.floatX))
        #self.Wxo = theano.shared(0.2 * uniform(-1.0, 1.0, (n_in, n_o)).astype(dtype))
        #self.Who = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden, n_o)).astype(dtype))
        #self.Wco = theano.shared(0.2 * uniform(-1.0, 1.0, (n_c, n_o)).astype(dtype))
        #self.bo = theano.shared(numpy.zeros(n_o, dtype=theano.config.floatX))
        

        self.c0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        #self.h0 = T.tanh(self.c0)
        self.h0 = theano.shared(numpy.zeros(n_hidden, dtype=dtype))
        

        
        # classification weights
        # self.Wy0_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.Wy1_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.Wy2_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.by_a = theano.shared(numpy.zeros(n_y, dtype=dtype))
        # self.Wy0_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.Wy1_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.Wy2_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_hidden + featdim, n_y)).astype(dtype))
        # self.by_o = theano.shared(numpy.zeros(n_y, dtype=dtype))
        self.Wy_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_v + featdim, n_y)).astype(dtype))
        self.Wy_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_v + featdim, n_y)).astype(dtype))
        #self.Wy_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_y)).astype(dtype))
        #self.Wy_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_y)).astype(dtype))
        #self.Wy_a = theano.shared(0.2 * uniform(-1.0, 1.0, (n_v+1, n_y)).astype(dtype))
        #self.Wy_o = theano.shared(0.2 * uniform(-1.0, 1.0, (n_v+1, n_y)).astype(dtype))
        self.by_a = theano.shared(numpy.zeros(n_y, dtype=dtype))
        self.by_o = theano.shared(numpy.zeros(n_y, dtype=dtype))
        

        # attention weights
        #self.Wa = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        #self.Wo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        #self.Ra = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        #self.Ro = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Ua = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nt, n_hidden, n_hidden)).astype(theano.config.floatX))
        self.Uo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nt_, n_hidden, n_hidden)).astype(theano.config.floatX))
        self.Va = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nt_, n_hidden, n_hidden)).astype(theano.config.floatX))
        self.Vo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (nt, n_hidden, n_hidden)).astype(theano.config.floatX))
        #self.Vao = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        #self.Voa = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        

        
        self.va = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v)).astype(theano.config.floatX))
        self.vo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v)).astype(theano.config.floatX))
        
        self.Wha_1 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Who_1 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Wha_2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Who_2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Wha_3 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Who_3 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        self.Wxa_1 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        self.Wxo_1 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        self.Wxa_2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        self.Wxo_2 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        self.Wxa_3 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        self.Wxo_3 = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_inv, n_v)).astype(theano.config.floatX))
        

        # initial values
        self.m0_a = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_hidden)).astype(theano.config.floatX))
        self.m0_o = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_hidden)).astype(theano.config.floatX))
        self.r0_a = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v)).astype(theano.config.floatX))
        self.r0_o = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v)).astype(theano.config.floatX))
        #self.pad = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v)).astype(theano.config.floatX))

        # memory update weights (can also use GRU units)
        self.Ma = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_hidden, n_hidden)).astype(theano.config.floatX))
        self.Mo = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_hidden, n_hidden)).astype(theano.config.floatX))
        #self.Ca = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))
        #self.Co = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (n_v, n_v)).astype(theano.config.floatX))

        
        self.params = [self.Wxi,self.Whi,self.Wxf,self.Whf,self.Wxc,self.Whc,self.h0,self.bi,self.bf,self.bc,\
                        self.m0_a,self.m0_o,self.Ma,self.Mo,self.Wy_a,self.Wy_o,self.by_a,self.by_o, \
                        self.Ua,self.Uo,self.Va,self.Vo,self.va,self.vo,self.r0_a,self.r0_o,\
                        self.Wha_1,self.Wha_2,self.Wha_3,self.Who_1,self.Who_2,self.Who_3,self.Wxa_1,self.Wxa_2,self.Wxa_3,\
                        self.Wxo_1,self.Wxo_2,self.Wxo_3]
                        
        mask_params = [self.Wxi,self.Wxf,self.Wxc,self.Ua,self.Va,\
                        self.Uo,self.Vo,\
                        self.Wxa_1,self.Wxa_2,self.Wxa_3,self.Wxo_1,self.Wxo_2,self.Wxo_3]
        
        self.ms = [theano.shared(W.get_value() * numpy.asarray(0., dtype=dtype)) for W in mask_params]                

        #self.allcache = [theano.shared(W.get_value() * numpy.asarray(0., dtype=dtype)) for W in self.params]
       
        #idxs = T.ivector()
        emb = T.fmatrix('emb')
        idxs = T.imatrix()
        p = T.scalar('p')
        #ridxs = idxs[::-1]
        #x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
        x = emb[idxs].reshape((idxs.shape[0], de*cs))
        #xr = emb[ridxs].reshape((idxs.shape[0], de*cs))
        f = T.matrix('f')
        f.reshape( (idxs.shape[0], featdim))
        ya_true = T.ivector('ya_true') # label
        yo_true = T.ivector('yo_true')
        #y = T.ivector('y') 
        
        
        def tensor_product(a, b, t):
            s, _ = theano.scan(lambda r, p, q: T.dot(p, T.dot(r, T.transpose(q))), sequences=[t], non_sequences=[a,b], n_steps=t.shape[0])
            s = s.reshape((s.shape[0],))
            return s
        
        def mask(param, prob):
            m = srng.binomial(n=1, p=1-prob, size=param.shape)
            m = T.cast(m, theano.config.floatX)
            return m
            
        def multimask(params, prob):
            return [mask(param, prob) for param in params]
        
        def dropout(param, mask, prob):
            scale = 1. / (1. - prob)
            return param * mask * scale
            
        #masks_gru = multimask([self.Wxi,self.Whi,self.Wxf,self.Whf,self.Wxc,self.Whc], p)
        def gru(x_t, feat_t, h_tm1, pb):
            i_t = sigma(theano.dot(x_t, dropout(self.Wxi, self.ms[0], pb)) + theano.dot(h_tm1, self.Whi) +self.bi)
            f_t = sigma(theano.dot(x_t, dropout(self.Wxf, self.ms[1], pb)) + theano.dot(h_tm1, self.Whf) + self.bf)
            c_t = T.tanh(theano.dot(x_t, dropout(self.Wxc, self.ms[2], pb)) + theano.dot(h_tm1 * f_t, self.Whc) + self.bc)
            h_t = (T.ones_like(i_t) - i_t) * h_tm1 + i_t * c_t
            
            if self.featdim > 0:
                all_t = T.concatenate([h_t, feat_t])
            else:
                all_t = h_t
            
            #s_t = softmax(theano.dot(all_t, self.Why) + self.by)
            
            return [h_t, c_t]
        

        
        def ctx_win(l, win, seq_size):
            lpadded = T.concatenate([[seq_size], l])
            lpadded = T.concatenate([lpadded, [seq_size]])
            out, _ = theano.scan(lambda i, lpadded: lpadded[i: i+win], sequences=[T.arange(l.shape[0])], non_sequences=lpadded)
            #out = [ lpadded[i:i+win] for i in T.arange(l.shape[0]) ]
            return out
        
        
        #masks_ha = multimask([self.Ua,self.Va,self.Vao_1,self.Vao_2,self.Vao_3,self.Vao_4], p)
        def get_hidden_aspect(ma, mo, h, pb):
            #a, _ = theano.scan(lambda h_i, m: T.tanh(T.dot(h_i, self.Ua) + T.dot(m, self.Va)), sequences=[h], non_sequences=m)
            #best performance
            #a, _ = theano.scan(lambda h_i, ma, mo: T.tanh(T.dot(h_i, self.Ua) + T.dot(ma, self.Va) + T.tanh(T.dot(mo, self.Vao))), \
            #                    sequences=[h], non_sequences=[ma, mo])
            a, _ = theano.scan(lambda h_i, ma, mo: T.concatenate([tensor_product(h_i, ma, dropout(self.Ua, self.ms[3], pb)), \
                                tensor_product(h_i, mo, dropout(self.Va, self.ms[4], pb))], axis=0), sequences=[h], non_sequences=[ma, mo])
            
            #r, _ = theano.scan(lambda a_i, rm1: T.tanh(T.dot(a_i, self.Wa) + T.dot(rm1, self.Ra)), sequences=[a], outputs_info=self.r0_a)
            #a_pad = T.concatenate([a, self.pad.reshape((1, self.pad.shape[0]))], axis=0)
            #ctx_ind = ctx_win(T.arange(a.shape[0]), csv, a.shape[0])
            #ctx = a_pad[ctx_ind].reshape((a.shape[0], n_v * csv))
            r, _ = theano.scan(fn=gru_aspect, sequences=[a], outputs_info=self.r0_a, non_sequences=[pb])
            return r
        
        #masks_ho = multimask([self.Uo,self.Vo,self.Voa_1,self.Voa_2,self.Voa_3,self.Voa_4], p)
        def get_hidden_opinion(ma, mo, h, pb):
            #a, _ = theano.scan(lambda h_i, m: T.tanh(T.dot(h_i, self.Uo) + T.dot(m, self.Vo)), sequences=[h], non_sequences=m)
            #a, _ = theano.scan(lambda h_i, ma, mo: T.tanh(T.dot(h_i, self.Uo) + T.dot(mo, self.Vo) + T.tanh(T.dot(ma, self.Voa))), \
            #                    sequences=[h], non_sequences=[ma, mo])
            a, _ = theano.scan(lambda h_i, ma, mo: T.concatenate([tensor_product(h_i, ma, dropout(self.Uo, self.ms[5], pb)), \
                                tensor_product(h_i, mo, dropout(self.Vo, self.ms[6], pb))], axis=0), sequences=[h], non_sequences=[ma, mo])
            #r, _ = theano.scan(lambda a_i, rm1: T.tanh(T.dot(a_i, self.Wo) + T.dot(rm1, self.Ro)), sequences=[a], outputs_info=self.r0_o)
            
            #a_pad = T.concatenate([a, self.pad.reshape((1, self.pad.shape[0]))], axis=0)
            #ctx_ind = ctx_win(T.arange(a.shape[0]), csv, a.shape[0])
            #ctx = a_pad[ctx_ind].reshape((a.shape[0], n_v * csv))
            r, _ = theano.scan(fn=gru_opinion, sequences=[a], outputs_info=self.r0_o, non_sequences=[pb])
            return r
        
        
        #masks_grua = multimask([self.Wxa_1,self.Wha_1,self.Wxa_2,self.Wha_2,self.Wxa_3,self.Wha_3], p)
        def gru_aspect(a_i, rm1, pb):
            g_i = sigma(T.dot(a_i, dropout(self.Wxa_1, self.ms[7], pb)) + T.dot(rm1, self.Wha_1))
            f_i = sigma(T.dot(a_i, dropout(self.Wxa_2, self.ms[8], pb)) + T.dot(rm1, self.Wha_2))
            c_i = T.tanh(theano.dot(a_i, dropout(self.Wxa_3, self.ms[9], pb)) + theano.dot(rm1 * f_i, self.Wha_3))
            r_i = (T.ones_like(g_i) - g_i) * rm1 + g_i * c_i
            return r_i
        
        #masks_gruo = multimask([self.Wxo_1,self.Who_1,self.Wxo_2,self.Who_2,self.Wxo_3,self.Who_3], p)
        def gru_opinion(a_i, rm1, pb):
            g_i = sigma(T.dot(a_i, dropout(self.Wxo_1, self.ms[10], pb)) + T.dot(rm1, self.Who_1))
            f_i = sigma(T.dot(a_i, dropout(self.Wxo_2, self.ms[11], pb)) + T.dot(rm1, self.Who_2))
            c_i = T.tanh(theano.dot(a_i, dropout(self.Wxo_3, self.ms[12], pb)) + theano.dot(rm1 * f_i, self.Who_3))
            r_i = (T.ones_like(g_i) - g_i) * rm1 + g_i * c_i
            return r_i
            
        # attention model
        def attention_pool_aspect(h, ma, mo, pb):
            #s, _ = theano.scan(lambda h_t, m: T.dot(h_t, T.dot(self.Ua, T.transpose(m))), sequences=[h], non_sequences=m, n_steps=h.shape[0])
            #with GRU unit for computing attention weight
            #e, _ = theano.scan(lambda h_t, m: T.dot(self.va, T.transpose(T.tanh(T.dot(h_t, self.Ua) + T.dot(m, self.Va)))), \
            #                    sequences=[h], non_sequences=m, n_steps=h.shape[0])
            e, _ = theano.scan(lambda r_t, ma, mo: T.dot(self.va, r_t), sequences=[get_hidden_aspect(ma, mo, h, pb)], non_sequences=[ma, mo])
            alpha = softmax(e)[0]
            ctx_pool = T.dot(alpha, h)
            return ctx_pool, e

        def attention_pool_opinion(h, ma, mo, pb):
            #s, _ = theano.scan(lambda h_t, m: T.dot(h_t, T.dot(self.Uo, T.transpose(m))), sequences=[h], non_sequences=m, n_steps=h.shape[0])
            #with GRU unit for computing attention weight
            
            #e, _ = theano.scan(lambda h_t, m: T.dot(self.vo, T.transpose(T.tanh(T.dot(h_t, self.Uo) + T.dot(m, self.Vo)))), \
            #                    sequences=[h], non_sequences=m, n_steps=h.shape[0])
            e, _ = theano.scan(lambda r_t, ma, mo: T.dot(self.vo, r_t), sequences=[get_hidden_opinion(ma, mo, h, pb)], non_sequences=[ma, mo])
            alpha = softmax(e)[0]
            ctx_pool = T.dot(alpha, h)
            return ctx_pool, e
            


        def memory_iteration(ma_t, mo_t, h, pb):
            ca_tp1, _ = attention_pool_aspect(h, ma_t, mo_t, pb)
            co_tp1, _ = attention_pool_opinion(h, ma_t, mo_t, pb)
            #ma_tp1 = T.tanh(T.dot(mo_t, self.Ma) + T.dot(ca_tp1, self.Ca))
            #mo_tp1 = T.tanh(T.dot(ma_t, self.Mo) + T.dot(co_tp1, self.Co))
            #ma_tp1 = T.tanh(T.dot(mo_t, self.Ma)) + ca_tp1
            #mo_tp1 = T.tanh(T.dot(ma_t, self.Mo)) + co_tp1
            ma_tp1 = T.tanh(T.dot(ma_t, self.Ma)) + ca_tp1
            mo_tp1 = T.tanh(T.dot(mo_t, self.Mo)) + co_tp1
            
            #return [ma_tp1, mo_tp1]
            return [ma_tp1, mo_tp1]

        #[h, _], _ = theano.scan(fn=recurrence, sequences=[x,f], outputs_info=[self.h0, self.c0], n_steps=x.shape[0])
        [h, _], _ = theano.scan(fn=gru, sequences=[x,f], outputs_info=[self.h0, None], non_sequences=[p], n_steps=x.shape[0])
        #[bh, _],_ = theano.scan(fn=brecurrence, sequences=[xr,f], outputs_info=[self.bh0,self.bc0], n_steps=xr.shape[0])
        #h = T.concatenate([fh,bh[::-1]],axis=1)

        # compute memory state for each iteration
        [ma, mo], _ = theano.scan(fn=memory_iteration, outputs_info=[self.m0_a, self.m0_o], \
                                            non_sequences=[h, p], n_steps=iteration)
        #[ma, mo], _ = theano.scan(fn=memory_iteration, non_sequences=h, outputs_info=[self.m0_a, self.m0_o], n_steps=iteration)
        ma, mo = T.concatenate([self.m0_a.reshape((1, self.m0_a.shape[0])), ma], axis=0), T.concatenate([self.m0_o.reshape((1, self.m0_o.shape[0])), mo], axis=0)

        

        # p_y_given_x_lastword = s[-1,0,:]
        # p_y_given_x_sentence = s[:,0,:]
        # y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        #ya_pred = T.matrix('ya_pred')
        #yo_pred = T.matrix('yo_pred')
        #score_a, _ = theano.scan(lambda h_i, ca: T.dot(h_i, T.dot(self.Wa, T.transpose(ma))), sequences=[h], non_sequences=ma)
        hidden_a, _ = theano.scan(fn=get_hidden_aspect, sequences=[ma, mo], non_sequences=[h, p])
        hidden_o, _ = theano.scan(fn=get_hidden_opinion, sequences=[ma, mo], non_sequences=[h, p])
        '''
        def acc_final_aspect(h, m_t):
            Uha = T.dot(h, self.Ua)
            Vma = T.dot(m_t,self.Va)
            s_t, _ = theano.scan(lambda Uha_t, Vma: T.tanh(Uha_t + Vma), sequences=[Uha], non_sequences=Vma, n_steps=Uha.shape[0])
            return s_t
        def acc_final_opinion(h, m_t):
            Uho = T.dot(h, self.Uo)
            Vmo = T.dot(m_t,self.Vo)
            s_t, _ = theano.scan(lambda Uho_t, Vmo: T.tanh(Uho_t + Vmo), sequences=[Uho], non_sequences=Vmo, n_steps=Uho.shape[0])
            return s_t
        '''    
        #score_a, _ = theano.scan(fn=acc_final_aspect, non_sequences=h, sequences=[ma], n_steps=ma.shape[0])
        #score_a, _ = theano.scan(lambda sa_t, W, b: T.dot(sa_t, W) + b, sequences=[score_a], non_sequences=[self.Wy_a, self.by_a])
        #score_a = T.sum(score_a, axis=0)
        #ya_pred, _ = theano.scan(lambda sa_i: softmax(sa_i)[0], sequences=[score_a])
        ya_pred, _ = theano.scan(lambda ha_i, W, b: softmax(T.dot(ha_i, W) + b)[0], \
                                sequences=[T.sum(hidden_a, axis=0)], non_sequences=[self.Wy_a, self.by_a]) 

        #append lstm vector for prediction
        #ya_pred, _ = theano.scan(lambda sa_i, h_i, W, b: softmax(T.dot(T.concatenate([sa_i.dimshuffle('x'), h_i]), W) + b)[0], \
        #                         sequences=[T.sum(score_a, axis=1), h], non_sequences=[self.Wy_a, self.by_a])

        #score_o, _ = theano.scan(fn=acc_final_opinion, non_sequences=h, sequences=[mo], n_steps=mo.shape[0])
        #score_o, _ = theano.scan(lambda so_t, W, b: T.dot(so_t, W) + b, sequences=[score_o], non_sequences=[self.Wy_o, self.by_o])
        #score_o = T.sum(score_o, axis=0)
        #yo_pred, _ = theano.scan(lambda so_i: softmax(so_i)[0], sequences=[score_o])
        #yo_pred, _ = theano.scan(lambda so_i, W, b: softmax(T.dot(so_i, W) + b)[0], \
        #                        sequences=[T.sum(score_o, axis=1)], non_sequences=[self.Wy_o, self.by_o]) 
        yo_pred, _ = theano.scan(lambda ho_i, W, b: softmax(T.dot(ho_i, W) + b)[0], \
                                sequences=[T.sum(hidden_o, axis=0)], non_sequences=[self.Wy_o, self.by_o]) 
        #append lstm vector for prediction
        #yo_pred, _ = theano.scan(lambda so_i, h_i, W, b: softmax(T.dot(T.concatenate([so_i.dimshuffle('x'), h_i]), W) + b)[0], \
        #                         sequences=[T.sum(score_o, axis=1), h], non_sequences=[self.Wy_o, self.by_o])

        ya_label = T.argmax(ya_pred, axis=1)
        yo_label = T.argmax(yo_pred, axis=1)
        ya_pos = ya_pred[:, 1]
        yo_pos = yo_pred[:, 1]


        # cost and gradients and learning rate
        lr = T.scalar('lr')
        max_ya, _ = theano.scan(lambda v, l: T.log(v)[l], sequences=[ya_pred, ya_true])
        max_yo, _ = theano.scan(lambda v, l: T.log(v)[l], sequences=[yo_pred, yo_true])
        nll_a = -T.mean(max_ya)
        nll_o = -T.mean(max_yo)
        nll = nll_a + nll_o
        # nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        # max_y, _ = theano.scan(lambda v, i: T.log(v)[i], sequences=[p_y_given_x_sentence, y])
        # nll = -T.mean(max_y,axis=0)
    
        gradients = T.grad( nll, self.params )
        # rmsprop
        #allcache = [decay * cacheW + (1 - decay) * gradient ** 2 for cacheW, gradient in zip(self.allcache, gradients)]
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        #updates = OrderedDict([( p, p-lr*g/T.sqrt(cache+1e-6) ) for p, g, cache in zip( self.params , gradients, allcache)] \
        #                        + [(w, new_w) for w, new_w in zip(self.allcache, allcache)])
        emb_update = T.grad(nll, emb)

        # theano functions
        #self.classify = theano.function(inputs=[idxs, emb, f], outputs=[ya_pos, yo_pos], allow_input_downcast=True)
        self.classify = theano.function(inputs=[idxs, emb, f, p], outputs=[ya_label, yo_label], allow_input_downcast=True)
        self.train = theano.function(inputs=[idxs, emb, f, ya_true, yo_true, lr, p], outputs=[nll, emb_update], updates=updates, allow_input_downcast=True)
        # self.classify = theano.function(inputs=[idxs, emb, f], outputs=y_pred, allow_input_downcast=True)
        # self.train = theano.function(inputs=[idxs, emb, f, y, lr], outputs=nll, updates=updates,allow_input_downcast=True)
        #self.normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})


        #add returning gradients for embedding
        #self.grad = theano.function(inputs=[idxs, emb, f, ya_true, yo_true, p], outputs=emb_update, allow_input_downcast=True)
        # self.grad = theano.function(inputs=[idxs,emb,f,y],outputs=emb_update, allow_input_downcast=True)
        
        dropout_params = multimask(self.ms, p)
        dropout_updates = OrderedDict(( m, m_ ) for m, m_ in zip( self.ms , dropout_params))
        self.dropout_layer = theano.function(inputs=[p], outputs=None, updates=dropout_updates)

    def save(self, filename):   
        '''
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
        '''
        cPickle.dump([param.get_value() for param in self.params], filename)
