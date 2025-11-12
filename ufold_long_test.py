import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

#from FCN import FCNNet
from network_small import UFoldXlong as FCNNet
##from Network_new import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset
#from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_two as Dataset_FCN_merge
import collections

import numpy as np
start_time = time.time()  
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess

def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq

def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file

def get_ct_dict(predict_matrix,batch_num,ct_dict):
    
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
    return ct_dict
    
def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    #seq = (torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1)).numpy().astype(int).reshape(predict_matrix.shape[-1]), torch.arange(predict_matrix.shape[-1]).numpy())
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict

#convert to beseq file
def contact_map_to_bpseq(contact_map, seq_ori, seq_name, seq_lens):
    nucleotide_map = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    seq_bases = [nucleotide_map[torch.argmax(seq_ori[0][i]).item()] for i in range(seq_lens.item())]
    seq_string = ''.join(seq_bases)
    seq_name_clean=seq_name[0].replace("/", "_")
    bpseq_filename = f"test_outputs/{seq_name_clean}.txt"    
    with open(bpseq_filename, 'w') as bpseq_file:
        bpseq_file.write(f">{seq_name_clean}\n")
        base_pairs = {i: 0 for i in range(seq_lens.item())}
        for i in range(seq_lens.item()):
            for j in range(i + 1, seq_lens.item()):
                if contact_map[0][0][i][j] > 0.5:
                    base_pairs[i] = j + 1
                    base_pairs[j] = i + 1
        for i in range(seq_lens.item()):
            bpseq_file.write(f"{i + 1} {seq_bases[i]} {base_pairs[i]}\n")
    #print(f"bpseq-file for {seq_name_clean} saved to {bpseq_filename}")

## convert to dot-bracket file
def contact_map_to_dot_bracket(contact_map, seq_ori, seq_name, seq_lens,predict_number):
    nucleotide_map = {0: 'A', 1: 'U', 2: 'C', 3: 'G'}
    seq_bases = [nucleotide_map[torch.argmax(seq_ori[0][i]).item()] for i in range(seq_lens.item())]

    dot_bracket = ['.'] * seq_lens.item() 

    for i in range(seq_lens.item()):
        for j in range(i + 1, seq_lens.item()):
            if contact_map[0][0][i][j] > 0.5:    
                dot_bracket[i] = '('  
                dot_bracket[j] = ')'  
    dot_bracket_str = ''.join(dot_bracket)
    seq_name_clean=seq_name[0].replace("/", "_")
    dot_bracket_filename = f"test_outputs/{seq_name_clean}_dot.txt"
    with open(dot_bracket_filename, 'w') as dot_file:
        dot_file.write(f">{seq_name[0]}\n")
        dot_file.write(f"{dot_bracket_str}\n")
        dot_file.write(f"{predict_number}\n")
    #print(f"Dot-bracket for {seq_name[0]} saved to {dot_bracket_filename}")
        
def model_eval_all_test(contact_net,test_generator):
    ##device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device(":1" if torch..is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    result_accuracy = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        nc_map_nc = nc_map.float() * contacts
        if batch_n%1000==0:
            print('Batch number: ', batch_n)
        
        batch_n += 1
        
        #if batch_n > 10:
        #    break

        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        # matrix_reps_batch = torch.unsqueeze(
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        tik = time.time()
        
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
            #seq_ori, 0.01, 0.1, 100, 1.6, True) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
        
        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_accuracy.append(result_no_train_tmp[0][2])
        result_no_train += result_no_train_tmp
        predict_number=result_no_train_tmp[0]
        #contact_map_to_dot_bracket(map_no_train, seq_ori, seq_name, seq_lens,predict_number)
        
        ###convert the predict map to bpseq file
        contact_map_to_bpseq(map_no_train, seq_ori, seq_name, seq_lens)
        
        if nc_map_nc.sum() != 0:
            #pdb.set_trace()
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                nc_map_nc.cpu().float()[i]), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])

    print(np.mean(run_time))

    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)  
    result_accuracy = [tensor.item() for tensor in result_accuracy]
    #print(len(result_accuracy))
    #np.savetxt(file_path, result_accuracy, fmt='%.6f')
    #pdb.set_trace()
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch..set_device(0)

    config_file = args.config
    test_file = args.test_files
    
    config = process_config(config_file)
    #print('Here is the configuration of this run: ')
    #print(config)
    MODEL_SAVED = 'models/UFold-X_long_train_50.pt'
    
    # os.environ["_VISIBLE_DEVICES"]= config.gpu
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    
    
    
    # if gpu is to be used
    ##device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device(":2" if torch..is_available() else "cpu")
    seed_torch()
    
    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':

        test_data = RNASSDataGenerator('data/', 'test_no_redundant_1800.pickle')

    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')
        
    seq_len = test_data.data_y.shape[-2]
    nums= test_data.data_y.shape[1]
    print(nums)
    print('Max seq length ', seq_len)
    print(test_data.data_y.shape)
    pdb.set_trace()
    
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    
    # test_set = Dataset(test_data)
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)

    ##contact_net = FCNNet(img_ch=17)
    contact_net = FCNNet()
    #contact_net = UNext(input_channels=17)
    #pdb.set_trace()
    
    print('==========Start Loading==========')
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cuda:0'))
    print('==========Finish Loading==========')
    
    contact_net.to(device)
    model_eval_all_test(contact_net,test_generator)
    

if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()

end_time = time.time()  
print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")

elapsed_time = end_time - start_time  
print(f"Duration: {elapsed_time:.2f} s")







