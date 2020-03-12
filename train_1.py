import mxnet as mx
from mxnet import nd, gluon, gpu, cpu, autograd
import numpy as np
from modules import *
from utils import *
import os

def plot_alignment(alignment, path, info=None):
  fig, ax = plt.subplots()
  im = ax.imshow(
    alignment,
    aspect='auto',
    origin='lower',
    interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
    xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()
  plt.savefig(path, format='png')
def make_info (train_losses_pred, valid_losses_pred, train_losses_rate, valid_losses_rate):

    print train_losses_rate

    info = 'train pred, '  

    keys = train_losses_pred.keys () 

    info += ' '.join (str (item) for item in keys) + ' : '

    for key in keys:

        info += str (np.mean (np.array (train_losses_pred[key]))) + ', '   

    info += ' valid pred : '

    for key in keys:

        info += str (np.mean (np.array (valid_losses_pred[key]))) + ', '

    info += ' train rate : '

    for key in keys:

        info +=  str (np.mean (np.array (train_losses_rate[key]))) + ', '

    info += ' valid rate : '

    for key in keys:

        info += str (np.mean (np.array (valid_losses_rate[key])))    
    info += '\n'

    return info

     


def train (num_speakers, model, loss_pred, loss_rate, trainer, batch_size_train, batch_size_valid, data_iters_train, data_iters_valid, ctx, maximal_epoch, epoch_save, path_save):

    curr_epoch = 0

    f = open ('record_laoluo', 'a+')

    loss_rate_weight = 0.01 

    while curr_epoch < maximal_epoch:

        train_epoch_losses_pred = {}
        train_epoch_losses_rate = {}
        for i in range (0, num_speakers):

            train_epoch_losses_pred.update ({i : []})
            
            train_epoch_losses_rate.update ({i : []})

            data_iters_train[i].reset ()

        end_count = 0
 
        current_data = 0
        batch_n=0
        while end_count < num_speakers: 
            #print end_count,batch_n,current_data
            batch_n+=1
            data_iter_train = data_iters_train[current_data]   

            values_L = make_values_L (1.0, 10000.0, 512, batch_size_train)

            phonemes, tones, orders, enc_masks, acos, dec_masks, aco_lengths, ratios, segs , ids,sum0,sum1 = data_iter_train.load_one_batch ()  

            T_enc = phonemes.shape[1]
            #print t2-t1,'t1'
            dy_dec = make_dynamic_dec (acos.shape[1], values_L)
  
            #print t3-t2,'t2'
            phonemes = gluon.utils.split_and_load (phonemes, ctx)
            tones = gluon.utils.split_and_load (tones, ctx)
            orders = gluon.utils.split_and_load (orders, ctx)
            enc_masks = gluon.utils.split_and_load (enc_masks, ctx)
            acos = gluon.utils.split_and_load (acos, ctx)
            dec_masks = gluon.utils.split_and_load (dec_masks, ctx)
            values_L_list = gluon.utils.split_and_load (values_L, ctx)
            dy_dec = gluon.utils.split_and_load (dy_dec, ctx)             
            aco_lengths = gluon.utils.split_and_load (aco_lengths, ctx)
            ratios = gluon.utils.split_and_load (ratios, ctx)
            segs = gluon.utils.split_and_load (segs, ctx)
            ids = gluon.utils.split_and_load (ids, ctx)
            #print t4-t3,'t3'
            losses = []
            losses_aco = []
            losses_rate = []

            with autograd.record ():

                for P, T, O, EM, A, DM, VL, DD, AL, R, S,ID in zip (phonemes, tones, orders, enc_masks, acos, dec_masks, values_L_list, dy_dec, aco_lengths, ratios,  segs, ids):
                    pred, pred_rate,atten_tmp = model (P, T, O, T_enc, DD, VL, EM, R, DM, S, ID)
                    #print t5-t4,'t4'
                    loss = loss_pred (pred, A, DM)
                    loss_r = loss_rate (pred_rate, AL, EM) * loss_rate_weight

                    losses.append (loss + loss_r)
                    losses_aco.append (loss)
                    losses_rate.append (loss_r)

            for item in losses:

                item.backward ()
            #if total_step%500==0:
                #align=align[0,:-sum1,:-sum0]
                #plot_alignment(align.asnumpy(),os.path.join('image/', 'step-%d-align.png' % total_step))
            trainer.step (batch_size_train)
            nd.waitall ()

            loss_avg_aco = 0
            loss_avg_rate = 0

            for item in losses_aco:
                loss_avg_aco += np.sum (item.asnumpy ()) / batch_size_train
            for item in losses_rate:
                loss_avg_rate += np.sum (item.asnumpy ()) / batch_size_train
            #print loss_avg_aco, loss_avg_rate

            train_epoch_losses_pred[current_data].append (loss_avg_aco)
            train_epoch_losses_rate[current_data].append (loss_avg_rate)

            if data_iter_train.end_loading:

                end_count += 1     

                data_iter_train.reset ()

            current_data += 1

      
            current_data %= num_speakers

        if curr_epoch % epoch_save == 0:
            model.collect_params ().save (path_save + 'model_' + str (curr_epoch))


        valid_epoch_losses_pred = {}
        valid_epoch_losses_rate = {}

        for i in range (0, num_speakers):

            valid_epoch_losses_pred.update ({i : []})

            valid_epoch_losses_rate.update ({i : []})

            data_iters_valid[i].reset ()

        for i in range (0, num_speakers):

            data_iter_valid = data_iters_valid[i] 

            values_L = make_values_L (1.0, 10000.0, 512, batch_size_valid)

            while not data_iter_valid.end_loading:

                phonemes, tones, orders, enc_masks, acos, dec_masks, aco_lengths, ratios, segs , ids ,sum0,sum1= data_iter_valid.load_one_batch ()  

                T_enc = phonemes.shape[1]

                dy_dec = make_dynamic_dec (acos.shape[1], values_L)
      
                phonemes = gluon.utils.split_and_load (phonemes, ctx)
                tones = gluon.utils.split_and_load (tones, ctx)
                orders = gluon.utils.split_and_load (orders, ctx)
                enc_masks = gluon.utils.split_and_load (enc_masks, ctx)
                acos = gluon.utils.split_and_load (acos, ctx)
                dec_masks = gluon.utils.split_and_load (dec_masks, ctx)
                values_L_list = gluon.utils.split_and_load (values_L, ctx)
                dy_dec = gluon.utils.split_and_load (dy_dec, ctx)             
                aco_lengths = gluon.utils.split_and_load (aco_lengths, ctx)
                ratios = gluon.utils.split_and_load (ratios, ctx)
                segs = gluon.utils.split_and_load (segs, ctx)
                ids = gluon.utils.split_and_load (ids, ctx)
                losses = []
                losses_aco = []

                
                for P, T, O, EM, A, DM, VL, DD, AL, R, S, ID in zip (phonemes, tones, orders, enc_masks, acos, dec_masks, values_L_list, dy_dec, aco_lengths, ratios, segs, ids):

                    pred, pred_rate,atten_tmp = model (P, T, O, T_enc, DD, VL, EM, R, DM, S, ID)

                    loss = loss_pred (pred, A, DM)
                    loss_r = loss_rate (pred_rate, AL, EM) * loss_rate_weight
                        
                    losses.append (loss + loss_r)
                    losses_aco.append (loss)
                    losses_rate.append (loss_r)

                loss_avg_aco = 0
                loss_avg_rate = 0

                #align=align[0,:-sum1,:-sum0]
                #plot_alignment(align.asnumpy(),os.path.join('image/', 'step-%d-align_test.png' % total_step))

                for item in losses_aco:
                    loss_avg_aco += np.sum (item.asnumpy ()) / batch_size_valid
    
                for item in losses_rate:
                    loss_avg_rate += np.sum (item.asnumpy ()) / batch_size_valid

                #valid_epoch_loss_pred.append (loss_avg_aco)
                #valid_epoch_loss_rate.append (loss_avg_rate)

                valid_epoch_losses_pred[i].append (loss_avg_aco)
                valid_epoch_losses_rate[i].append (loss_avg_rate)


        info = make_info (train_epoch_losses_pred, valid_epoch_losses_pred, train_epoch_losses_rate, valid_epoch_losses_rate) 

        print info
 
        f.writelines (info)

        curr_epoch += 1

    f.close ()

     
###########################################

if __name__ == '__main__':

    maximal_epoch = 1500

    epoch_save = 10
    
    path_save = 'parameters_laoluo/'
    
    if not os.path.exists(path_save):
        os.mkdir(path_save)

    #batch_size_train = 96
    batch_size_train = 48 

    batch_size_valid = 8

    ctx = [gpu(i) for i in range (0, 4)]

    #train_json_file_0 = './train_selected_ljh_pinyin_seg2.json'

    #valid_json_file_0 = './valid_selected_ljh_pinyin_seg2.json'

    train_json_file_0 = './train_selected_laoluo.json'
    valid_json_file_0 = './valid_selected_laoluo.json'

    #train_json_file_0 = './train_selected_emo_pinyin_seg.json' 

    #valid_json_file_0 = './valid_selected_emo_pinyin_seg.json'

    train_json_file_1 = './train_selected_xixi_pinyin_seg2.json' 

    valid_json_file_1 = './valid_selected_xixi_pinyin_seg2.json'

    train_json_file_2 = './train_selected_cartoonjing_pinyin_seg2.json' 

    valid_json_file_2 = './valid_selected_cartoonjing_pinyin_seg2.json'

    train_json_file_3 = './train_selected_emo_pinyin_seg.json'

    valid_json_file_3 = './valid_selected_emo_pinyin_seg.json'

    train_json_file_4 = './train_selected_cartoonjing_pinyin_seg.json'

    valid_json_file_4 = './valid_selected_cartoonjing_pinyin_seg.json'

    #aco_path_0 = '/home/data/ljh/out/normed_cmp/' 
    aco_path_0 = '/home/mdisk/data/laoluo/16k/out/normed_cmp/'

    aco_path_1 = '/home/data/xixi/out/normed_cmp/' 

    aco_path_2 = '/home/data/cartoonjing/out/normed_cmp/' 

    aco_path_3 = '/home/data/emo/out_dio16/normed_cmp/'

    aco_path_4 = '/home/data/cartoonjing/out/normed_cmp/'



    enc_num_down = 6

    #file_rate_info_0 = './ljh_rate_info.npy'
    file_rate_info_0 = './laoluo_rate_info.npy'

    file_rate_info_1 = './xixi_rate_info.npy'

    file_rate_info_2 = './cartoonjing_rate_info.npy'

    file_rate_info_3 = './emo_rate_info.npy'

    file_rate_info_4 = './cartoonjing_rate_info.npy'


    num_phonemes = 95

    num_tones = 7 

    num_orders = 2

    num_id = 4

    aco_shape = 187

    dim_embed_phoneme = 512

    dim_embed_tone = 128

    dim_embed_order = 32  

    dim_embed_seg = 512

    dim_id = 512 

    size_enc = 512

    size_dec = 512

    dp_dec = 0.15 


    data_iter_train_0 = data_iterator (0, train_json_file_0, aco_path_0, batch_size_train, enc_num_down, file_rate_info_0, num_phonemes, num_tones, num_orders, aco_shape) 
    data_iter_valid_0 = data_iterator (0, valid_json_file_0, aco_path_0, batch_size_valid, enc_num_down, file_rate_info_0, num_phonemes, num_tones, num_orders, aco_shape)

    data_iter_train_1 = data_iterator (1, train_json_file_1, aco_path_1, batch_size_train, enc_num_down, file_rate_info_1, num_phonemes, num_tones, num_orders, aco_shape) 
    data_iter_valid_1 = data_iterator (1, valid_json_file_1, aco_path_1, batch_size_valid, enc_num_down, file_rate_info_1, num_phonemes, num_tones, num_orders, aco_shape)

    data_iter_train_2 = data_iterator (2, train_json_file_2, aco_path_2, batch_size_train, enc_num_down, file_rate_info_2, num_phonemes, num_tones, num_orders, aco_shape) 
    data_iter_valid_2 = data_iterator (2, valid_json_file_2, aco_path_2, batch_size_valid, enc_num_down, file_rate_info_2, num_phonemes, num_tones, num_orders, aco_shape)


    data_iter_train_3 = data_iterator (3, train_json_file_3, aco_path_3, batch_size_train, enc_num_down, file_rate_info_3, num_phonemes, num_tones, num_orders, aco_shape)
    data_iter_valid_3 = data_iterator (3, valid_json_file_3, aco_path_3, batch_size_valid, enc_num_down, file_rate_info_3, num_phonemes, num_tones, num_orders, aco_shape)


    #data_iter_train_4 = data_iterator (4, train_json_file_4, aco_path_4, batch_size_train, enc_num_down, file_rate_info_4, num_phonemes, num_tones, num_orders, aco_shape)
    #data_iter_valid_4 = data_iterator (4, valid_json_file_4, aco_path_4, batch_size_valid, enc_num_down, file_rate_info_4, num_phonemes, num_tones, num_orders, aco_shape)


    data_iters_train = [data_iter_train_0,data_iter_train_1,  data_iter_train_2,data_iter_train_3]

    data_iters_valid = [data_iter_valid_0, data_iter_valid_1, data_iter_valid_2,data_iter_valid_3]
    
    model = End2End_1 (num_id, dim_id, num_phonemes, num_tones, num_orders, enc_num_down, dim_embed_phoneme, dim_embed_tone, dim_embed_order, dim_embed_seg, size_enc, size_dec, aco_shape, dp_dec, 'model_' )

    loss_pred = Loss_pred ('loss_pred_')

    loss_rate = Loss_rate ('loss_rate_') 

    model.collect_params ().initialize (ctx = ctx)

    trainer = gluon.Trainer (model.collect_params (), 'adam', {'learning_rate' : 1e-4, 'clip_gradient' : 1.0})

    train (num_id, model, loss_pred, loss_rate, trainer, batch_size_train, batch_size_valid, data_iters_train, data_iters_valid, ctx, maximal_epoch, epoch_save, path_save)

    


       

