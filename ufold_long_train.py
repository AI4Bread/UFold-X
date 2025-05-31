import _pickle as pickle
import sys
import os
import tqdm 

import torch
import torch.optim as optim
from torch.utils import data

import pdb
import subprocess

# import sys
# sys.path.append('./..')


from network_small import UFoldXlong as FCNNet   ##for long seq

import losses
from torch.optim import lr_scheduler
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
from ufold.postprocess import postprocess_new as postprocess

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
#from ufold.data_generator import Dataset_Cut_concat_new_merge as Dataset_FCN_merge
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections


def train(contact_net,train_merge_generator,epoches_first,args):
    epoch = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")##
    #
    #pdb.set_trace()
    ## select loss
    if args.loss == 'BCEWithLogitsLoss':
        pos_weight = torch.Tensor([300]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(
                        pos_weight = pos_weight)
    else:
        criterion = losses.__dict__[args.loss]().to(device)
    ## end loss
    
    ## 
    u_optimizer = optim.Adam(contact_net.parameters())

    ## scheduler 
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            u_optimizer, T_max=epoches_first, eta_min=1e-5)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(u_optimizer, factor=0.1, patience=2,
                                                   verbose=1, min_lr=1e-5)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(u_optimizer, milestones=[int(e) for e in [1,2].split(',')], gamma=2/3.0)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    
    
    steps_done = 0
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    epoch_rec = []
    for epoch in range(100):
        contact_net.train()
        # num_batches = int(np.ceil(train_data.len / BATCH_SIZE))
        # for i in range(num_batches):
        #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            #seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
            # matrix_reps_batch = torch.unsqueeze(
            #     torch.Tensor(matrix_reps.float()).to(device), -1)
    
            # padding the states for supervised training with all 0s
            # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            #     seq_len, seq_len]).to(device)
    
    
            # PE_batch = get_pe(seq_lens, seq_len).float().to(device)
            # contact_masks = torch.Tensor(contact_map_masks(seq_lens, seq_len)).to(device)
            pred_contacts = contact_net(seq_embedding_batch)
            '''
            try:
                pred_contacts = contact_net(seq_embedding_batch)
                #print(seq_embeddings.shape)
            except:
                print('This fails..sel lens:',seq_embeddings.shape)
                #pdb.set_trace()
                continue
            '''
            #pred_contacts = contact_net(seq_embedding_batch,seq_embedding_batch_1)
            pred_contacts = pred_contacts.squeeze(1)
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
            #pdb.set_trace()
            # Compute loss
            
            #print("contact_masks shape:", contact_masks.shape)##
            #print("contacts_batch shape:", contacts_batch.shape)##
            #print("pred_contacts shape:", pred_contacts.shape)##
            # 

            loss_u = criterion(pred_contacts*contact_masks, contacts_batch)##
    
    
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1
        #pdb.set_trace()
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))
        
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            # scheduler.step(val_log['loss'])
            scheduler.step()
        #pdb.set_trace()
            # model_eval_all_test()
            # torch.save(contact_net.state_dict(), model_path)
            #torch.save(contact_net.state_dict(), model_path + f'unet_bpTR0_addsimmutate_addmoresimilar_finetune{epoch}.pt')
        if epoch > -1:
            torch.save(contact_net.state_dict(),  f'models/UFoldX_long_train_{epoch}.pt')

def main():
    #torch.cuda.device_count()
    #torch.cuda.set_device(0)
    
    args = get_args()
    
    config_file = args.config
    
    config = process_config(config_file)
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)
    
    #pdb.set_trace()
    
    os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    
    d = config.u_net_d
    ##BATCH_SIZE = config.batch_size_stage_1
    BATCH_SIZE = 1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    #model_path = './models_ckpt/'.format(model_type, data_type,d)
    #model_path = './models_ckpt/final_model/unet_train_on_RNAlign_99.pt'
    epoches_first = config.epoches_first

    train_files = ['train_1800']
    
    # if gpu is to be used
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")##
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    
    #pdb.set_trace()
    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ',file_item)
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_data_list.append(RNASSDataGenerator('data/','train.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
            ## train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))  ##format
    print('Data Loading Done!!!')
    #train_data = RNASSDataGenerator('data/{}/'.format(data_type), 'train.pickle', False)
    pdb.set_trace()
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    #pdb.set_trace()
    
    
    #contact_net = FCNNet(img_ch=17)
    contact_net = FCNNet()
    #contact_net = UNext(input_channels=17)
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    
    # contact_net.conv1d2.register_forward_hook(get_activation('conv1d2'))
    
    #if LOAD_MODEL and os.path.isfile(model_path):
    #    print('Loading u net model...')
    #    contact_net.load_state_dict(torch.load(model_path))
    
    
    
    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    train(contact_net,train_merge_generator,epoches_first,args)

        

#model_eval_all_test()
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()







