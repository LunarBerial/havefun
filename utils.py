import mxnet as mx
from mxnet import nd
import numpy as np
import json
import random


class data_iterator (object):

    def __init__ (self, speaker_id, json_file, aco_path, batch_size, enc_num_down, file_rate_info, num_phonemes, num_tones, num_orders, aco_shape, if_sort = True, sort_rule = 'input'):


        self.speaker_id = speaker_id
        self.json_file = json_file
        self.aco_path = aco_path
        self.batch_size = batch_size
        self.enc_num_down = enc_num_down
        self.file_rate_info = file_rate_info    
        self.num_phonemes = num_phonemes
        self.num_tones = num_tones
        self.num_orders = num_orders
        self.aco_shape = aco_shape
        self.sort_rule = sort_rule
        self.encoderLenNeed = np.power (2, enc_num_down)
        f = open (self.json_file, 'r')
        self.data = json.load (f)
        f.close ()
        self.keys = self.data.keys ()

        if if_sort:

            if sort_rule == 'input':

                self.keys = sorted (self.keys, key = lambda x : self.data[x]['length_input'], reverse = True)

            elif sort_rule == 'output':

                self.keys = sorted (self.keys, key = lambda x : self.data[x]['length_output'], reverse = True)

            else:

                print 'sort_rule not existed'
                exit ()

        rate_info = np.load (file_rate_info).item () 

        self.maximal_rate = rate_info['max_rate']

    def reset (self):

        self.global_index = 0
        self.end_loading = False

        if self.sort_rule is None:
            print sort_rule
            random.shuffle (self.keys)
        #random.shuffle (self.keys)

    def load_one_batch (self):

        phoneme_tokens = []
       
        tone_tokens = []

        orders = []

        segs = []

        for i in range (self.global_index, self.global_index + self.batch_size):

            phoneme_token = np.array ([int (item) for item in self.data[self.keys[i]]['phoneme_token'].split (' ')]).astype (np.float32)  

            tone_token = np.array ([int (item) for item in self.data[self.keys[i]]['tone'].split (' ')]).astype (np.float32)

            order = np.array ([int (item) for item in self.data[self.keys[i]]['order'].split (' ')]).astype (np.float32)

            seg = np.array ([float (item) for item in self.data[self.keys[i]]['seg'].split (' ')]).astype (np.float32)

            reverse_seg = 1.0 - seg 

            seg = np.concatenate ((seg.reshape ((-1, 1)), reverse_seg.reshape ((-1, 1))), axis = 1)

            phoneme_tokens.append (phoneme_token)
 
            tone_tokens.append (tone_token)

            orders.append (order)

            segs.append (seg) 

        enc_masks = [np.ones_like (item) for item in phoneme_tokens]


        max_len_input = max ([item.shape[0] for item in phoneme_tokens]) 
        input_len_needed = ((max_len_input // self.encoderLenNeed) + 1) * self.encoderLenNeed
        #input_len_needed = np.power (2, self.enc_num_down) # Since the input length is restricted to be smaller than this value  
        phoneme_tokens = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_phonemes)) for item in phoneme_tokens] 
   
        tone_tokens = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_tones)) for item in tone_tokens]

        orders = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (self.num_orders)) for item in orders]

        segs = [np.pad (item, ((0, input_len_needed - item.shape[0]), (0, 0)), 'constant', constant_values = (0)) for item in segs]

        enc_masks = [np.pad (item, (0, input_len_needed - item.shape[0]), 'constant', constant_values = (0)) for item in enc_masks]

        output_len_needed = int (max_len_input * self.maximal_rate+0.5)

        acos = []

        dec_masks = []

        output_lengths = []


        ratios = []

        for i in range (self.global_index, self.global_index + self.batch_size):
            aco = np.fromfile (self.aco_path + self.data[self.keys[i]]['utter_id'] + '.cmp', dtype = np.float32).reshape ((-1, self.aco_shape)) 
            dec_mask = np.ones (shape = aco.shape[0]) 

            output_lengths.append (aco.shape[0])

            assert aco.shape[0] == self.data[self.keys[i]]['length_output']
            ratios.append (self.data[self.keys[i]]['length_output'] * 1.0 / self.data[self.keys[i]]['length_input'])
            #print output_len_needed, aco.shape[0],'output'
            aco = np.pad (aco, ((0, output_len_needed - aco.shape[0]), (0, 0)), 'constant')

            dec_mask = np.pad (dec_mask, (0, output_len_needed - dec_mask.shape[0]), 'constant' )

            acos.append (aco)

            dec_masks.append (dec_mask)

        self.global_index += self.batch_size

        if self.global_index + self.batch_size > len (self.keys):

            self.end_loading = True 

        sum0=np.sum(enc_masks[0]==0)
        sum1=np.sum(dec_masks[0]==0)

        return np.array (phoneme_tokens), np.array (tone_tokens), np.array (orders), np.array (enc_masks), np.array (acos), np.array (dec_masks), np.array (output_lengths), np.array (ratios), np.array (segs), self.speaker_id * np.ones (shape = (self.batch_size)),sum0,sum1

        
def make_values_L (range_min, range_max, L, batch_size):

    logs_L = np.linspace (0, np.log (range_max * 1.0 / range_min), num = L / 2)

    values_L = nd.array (1.0 / range_min * np.exp (-logs_L))

    values_L = nd.expand_dims (nd.expand_dims (values_L, axis = 0), axis = 2)

    return nd.broadcast_axis (values_L, axis = 0, size = batch_size)


def make_dynamic_dec (T, values_L):

    values_T = nd.array (np.linspace (1, T, num = T))

    values_T = nd.expand_dims (nd.expand_dims (values_T, axis = 0), axis = 2)

    values_T = nd.broadcast_axis (values_T, axis = 0, size = values_L.shape[0])

    values_TL = nd.batch_dot (values_T, values_L, transpose_b = True)

    values_sin = nd.sin (values_TL)
    values_cos = nd.cos (values_TL)

    return nd.concat (values_sin, values_cos, dim = 2)

 
