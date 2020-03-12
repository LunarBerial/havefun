#coding:utf-8
import mxnet as mx
from mxnet import gluon, nd
from U_Net import U_Net
import numpy as np


class Attention (gluon.HybridBlock):

    def __init__ (self, units, prefix):

        super (Attention, self).__init__ (prefix = prefix)

        with self.name_scope ():

            self.units = units
 
            self.dense = gluon.nn.Dense (units, activation = 'tanh', flatten = False, prefix = 'atten_dense_')

    def hybrid_forward (self, F, key, query, embedding, mask_enc):

        atten = F.batch_dot (query, key, transpose_b = True) / np.sqrt (self.units)
        atten_tmp=atten[:,:,:]
        mask_enc = F.expand_dims (mask_enc, axis = 1)

        atten = F.broadcast_minus (lhs = atten, rhs = (1.0 - mask_enc) * 1e5)
        return self.dense (F.batch_dot (F.softmax (atten * 10, axis = 2), embedding)),atten_tmp#越小越平滑


class AddRNN (gluon.rnn.HybridRecurrentCell):

    def __init__ (self, prefix):

        super (AddRNN, self).__init__ (prefix = prefix)

        self._hidden_size = 1

    def state_info (self, batch_size = 0):

        return [{'shape' : (batch_size, self._hidden_size), '__layout__' : 'NC'}]

    def hybrid_forward (self, F, x, s):

        return x + s[0], [x + s[0]]


class DynamicPosEnc (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (DynamicPosEnc, self).__init__ (prefix = prefix)

        with self.name_scope ():

            self.Add_RNN = AddRNN ('Add_RNN_')

    def hybrid_forward (self, F, r, values_L, T):

        self.Add_RNN.reset ()

        values_T = self.Add_RNN.unroll (T, r, merge_outputs = True)[0]

        values_T = values_T - 0.5 * r

        values_TL = F.batch_dot (values_T, values_L, transpose_b = True)

        values_sin = F.sin (values_TL)

        values_cos = F.cos (values_TL)

        return F.concat (values_sin, values_cos, dim = 2)  


class ConvBlock (gluon.HybridBlock):

    def __init__ (self, dp_rate, channels, kernel, padding, prefix, dilation = 1):

        super (ConvBlock, self).__init__ (prefix = prefix)

        self.dp_rate = dp_rate
       
        with self.name_scope ():

            self.id_bias = gluon.nn.Dense (channels * 2, activation = None, flatten = False, use_bias = False, prefix = 'id_bias_')

            self.Conv1D = gluon.nn.Conv1D (channels * 2, kernel, strides = 1, padding = padding, dilation = dilation, layout = 'NCW', use_bias = True, bias_initializer = 'zeros')  

    def hybrid_forward (self, F, x, id_emd):

        if self.dp_rate > 0:

            y = F.Dropout (x,p=self.dp_rate)

        else:

            y = x

        embedding = self.id_bias (id_emd)

        b1, b2 = F.split (embedding, num_outputs = 2, axis = 1)  

        b1 = F.expand_dims (b1, axis = 1)

        b2 = F.expand_dims (b2, axis = 1)

        y = F.swapaxes (y, 1, 2)

        y = self.Conv1D (y)

        l, r = F.split (y, num_outputs = 2, axis = 1)

        l = F.swapaxes (l, 1, 2)

        r = F.swapaxes (r, 1, 2)

        y = F.sigmoid (l + b1) * F.tanh (r + b2)

        return (x + y) * np.sqrt (0.5)

        

class End2End_1 (gluon.nn.HybridBlock):

    def __init__ (self, num_id, dim_id, num_phoneme, num_tone, num_order, num_down_enc, dim_embed_phoneme, dim_embed_tone, dim_embed_order,dim_embed_seg, size_enc, size_dec, size_output, dp_dec, prefix):

        super (End2End_1, self).__init__ (prefix = prefix)

        self.num_id = num_id
        self.dim_id = dim_id
        self.num_phoneme = num_phoneme
        self.num_tone = num_tone
        self.num_order = num_order
        self.num_down_enc = num_down_enc
        self.dim_embed_phoneme = dim_embed_phoneme
        self.dim_embed_tone = dim_embed_tone
        self.dim_embed_order = dim_embed_order
        self.dim_embed_seg = dim_embed_seg
        self.size_enc = size_enc
        self.size_dec = size_dec
        self.size_output = size_output
        self.dp_dec = dp_dec  

        with self.name_scope ():

            self.emd_id = gluon.nn.Embedding (self.num_id, self.dim_id) 

            self.emd_phoneme = gluon.nn.Embedding (self.num_phoneme, self.dim_embed_phoneme, prefix = 'embed_phoneme_') 
      
            self.emd_tone = gluon.nn.Embedding (self.num_tone, self.dim_embed_tone, prefix = 'embed_tone_')

            self.emd_order = gluon.nn.Embedding (self.num_order, self.dim_embed_order, prefix = 'embed_order_')

            self.emd_phoneme_rate = gluon.nn.Embedding (self.num_phoneme, self.dim_embed_phoneme, prefix = 'embed_phoneme_rate_') 

            self.emd_tone_rate = gluon.nn.Embedding (self.num_tone, self.dim_embed_tone, prefix = 'embed_tone_rate_')

            self.emd_order_rate = gluon.nn.Embedding (self.num_order, self.dim_embed_order, prefix = 'embed_order_rate_')

            self.emd_seg = gluon.nn.Dense (self.dim_embed_seg, flatten = False, prefix = 'dense_seg_')

            self.emd_seg_rate = gluon.nn.Dense (self.dim_embed_seg, flatten = False, prefix = 'dense_seg_rate_')

            self.U_Net_rate = U_Net (self.num_id, self.dim_id, 0, 256, 1, self.num_down_enc, [self.size_enc] * self.num_down_enc, [1] * self.num_down_enc, 'tanh', 'U_Net_rate_', dropout = 0.0)

            self.dense_enc_1 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, prefix = 'dense_enc_1_')

            self.id_bias_1 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, use_bias = False, prefix = 'id_bias_1_')

            self.dense_enc_2 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, prefix = 'dense_enc_2_')

            self.id_bias_2 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, use_bias = False, prefix = 'id_bias_2_')

            self.dense_enc_3 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, prefix = 'dense_enc_3_')
  
            self.id_bias_3 = gluon.nn.Dense (self.size_enc, activation = None, flatten = False, use_bias = False, prefix = 'id_bias_3_')

            self.Atten = Attention (self.size_dec, 'atten_')

            self.Dynamic = DynamicPosEnc ('DynamicPosEnc_')

            self.dec_conv_1 = ConvBlock (0.5, self.size_dec, 3, 1, 'dec_conv_1_') 

            self.dec_conv_2 = ConvBlock (0.5, self.size_dec, 3, 1, 'dec_conv_2_')

            self.dec_conv_3 = ConvBlock (0.5, self.size_dec, 3, 1, 'dec_conv_3_')

            self.final_dense = gluon.nn.Dense (self.size_output, flatten = False)


    def hybrid_forward (self, F, p, t, o, T, dy_dec, values_L, mask_enc, ratio, mask_dec,seg , id):      

        id_emd = self.emd_id (id)

        p_embed = self.emd_phoneme (p)
        p_embed_rate = self.emd_phoneme_rate (p)
        t_embed = self.emd_tone (t)
        t_embed_rate = self.emd_tone_rate (t)
        o_embed = self.emd_order (o)
        o_embed_rate = self.emd_order_rate (o)

        s_embed = self.emd_seg (seg)
        s_embed_rate = self.emd_seg_rate (seg)

        x_embed = F.concat (p_embed, t_embed, o_embed, s_embed, dim = 2)  
        x_embed_rate = F.concat (p_embed_rate, t_embed_rate, o_embed_rate,s_embed_rate, dim = 2)

        ratio = F.expand_dims (F.expand_dims (ratio, axis = 1), axis = 2)
        rate_x = self.U_Net_rate (x_embed_rate * F.expand_dims (mask_enc, axis = 2), id)
        rate_x = F.maximum (F.relu (rate_x + ratio), 2.0) 
        enc = F.tanh ((self.dense_enc_1 (x_embed) +  F.expand_dims (self.id_bias_1 (id_emd), axis = 1)) * F.expand_dims (mask_enc, axis = 2))
        enc = F.tanh ((self.dense_enc_2 (enc) + F.expand_dims (self.id_bias_2 (id_emd), axis = 1)) * F.expand_dims (mask_enc, axis = 2))

        enc = F.tanh ((self.dense_enc_3 (enc) + F.expand_dims (self.id_bias_3 (id_emd), axis = 1)) * F.expand_dims (mask_enc, axis = 2))

        dy_enc = self.Dynamic (rate_x, values_L, T)

        attention,atten_tmp = self.Atten (dy_enc, dy_dec, enc, mask_enc) 

        y = self.dec_conv_1 (attention * F.expand_dims (mask_dec, axis = 2), id_emd)

        y = self.dec_conv_2 (y, id_emd)

        y = self.dec_conv_3 (y, id_emd)

        return self.final_dense (y), rate_x,attention



class Loss_pred (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (Loss_pred, self).__init__ (prefix = prefix)

    def hybrid_forward (self, F, pred, truth, mask):

        mask_expand = F.expand_dims (mask, axis = 2)

        return F.sum (F.square (pred - truth) * mask_expand, axis = [1, 2], keepdims = False) / F.sum (mask, axis = 1, keepdims = False)    


class Loss_rate (gluon.HybridBlock):

    def __init__ (self, prefix):

        super (Loss_rate, self).__init__ (prefix = prefix)

    def hybrid_forward (self, F, pred_rate, true_rate, mask_enc):

        mask_enc = F.expand_dims (mask_enc, axis = 2)

        return F.maximum (F.abs (F.sum (pred_rate * mask_enc, axis = [1, 2]) - true_rate), 20.0)  























