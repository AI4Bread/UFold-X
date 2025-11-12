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


# from FCN import FCNNet
from side_Network_for_UFold_X import U_Net_for_mamba as UNet
from side_Network_for_UFold_X import VSSMBranch as VSSMBranch  
from side_Network_for_UFold_X import UFoldXall as UFoldXall

import losses
from torch.optim import lr_scheduler


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



    # test_script()
    #t1 = subprocess.getstatusoutput('awk \'{if($1 ~ /^>/)print}\' /data2/darren/experiment/ufold/data/rnastralign_all/rnastralign_train_no_redundant.seq.cdhit')
    #all_cdhit_names = t1[1].split('\n')
    steps_done = 0
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    epoch_rec = []
    #for epoch in range(epoches_first):
    for epoch in range(100):
        contact_net.train()
        # num_batches = int(np.ceil(train_data.len / BATCH_SIZE))
        # for i in range(num_batches):
        #for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

            #pred_contacts = contact_net(seq_embedding_batch)          
            try:
                pred_contacts = contact_net(seq_embedding_batch)
            except:
                print('This fails..sel lens:',seq_embeddings.shape)
                #pdb.set_trace()
                continue
            
            pred_contacts = pred_contacts.squeeze(1)##
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1

            loss_u = criterion(pred_contacts*contact_masks, contacts_batch)##
    
    
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))
        
        if args.scheduler == 'CosineAnnealingLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            # scheduler.step(val_log['loss'])
            scheduler.step()
        if epoch > -1:
            torch.save(contact_net.state_dict(),  f'models/UFold-X_all_train_{epoch}.pt')

def main():
    #torch.cuda.device_count()
    #torch.cuda.set_device(0)
    
    args = get_args()
    
    config_file = args.config
    
    config = process_config(config_file)
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)
    os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    d = config.u_net_d
    ##BATCH_SIZE = config.batch_size_stage_1
    BATCH_SIZE = 1
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL
    data_type = config.data_type
    model_type = config.model_type
    epoches_first = config.epoches_first

    ## train_files = ['ArchiveII'] #args.train_files
    ## train_files = ['train_1800']
    train_files = ['train_1800','ArchiveII']
    
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
            ##train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))  ##train_file_format
    print('Data Loading Done!!!')
    
    #Merging the dataset
    if len(train_files) == 2:
        print(train_data_list[0].data_x.shape)
        print(train_data_list[1].data_x.shape)
        print(train_data_list[0].data_y.shape)
        print(train_data_list[1].data_y.shape)
        shape_0 = train_data_list[0].data_x.shape
        shape_1 = train_data_list[1].data_x.shape

        if shape_0[1] < shape_1[1]:
            pad_width = shape_1[1] - shape_0[1]
            train_data_list[0].data_x = np.pad(train_data_list[0].data_x, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            train_data_list[0].data_y = np.pad(train_data_list[0].data_y, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
        elif shape_0[1] > shape_1[1]:
            pad_width = shape_0[1] - shape_1[1]
            train_data_list[1].data_x = np.pad(train_data_list[1].data_x, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            train_data_list[1].data_y = np.pad(train_data_list[1].data_y, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
    
        print("Shape of train_data_list[0].x:", train_data_list[0].data_x.shape)
        print("Shape of train_data_list[1].x:", train_data_list[1].data_x.shape)
        print("Shape of train_data_list[0].y:", train_data_list[0].data_y.shape)
        print("Shape of train_data_list[1].y:", train_data_list[1].data_y.shape)
    pdb.set_trace()
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
  
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
   
    unet = UNet(img_ch=17)  
    vssm = VSSMBranch()
    alpha = 0.5  
    contact_net=UFoldXall(unet=unet, vssm=vssm, alpha=alpha)
    
    contact_net.to(device)
    
    train(contact_net,train_merge_generator,epoches_first,args)

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
#torch.save(con
