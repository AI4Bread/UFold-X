import pdb
import numpy as np
import os
import subprocess
import collections
import pickle as cPickle
import random
import sys

def one_hot(seq):
    RNN_seq = seq
    BASES = 'AUCG'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
            [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[0] * len(BASES)]) for base
            in RNN_seq])

    return feat

def clean_pair(pair_list,seq):
    for item in pair_list:
        if seq[item[0]] == 'A' and seq[item[1]] == 'U':
            continue
        elif seq[item[0]] == 'C' and seq[item[1]] == 'G':
            continue
        elif seq[item[0]] == 'U' and seq[item[1]] == 'A':
            continue
        elif seq[item[0]] == 'G' and seq[item[1]] == 'C':
            continue
        else:
            print('%s+%s'%(seq[item[0]],seq[item[1]]))
            #pdb.set_trace()
            pair_list.remove(item)
    return pair_list

if __name__=='__main__':

    file_dir = sys.argv[1]+'/'#'BPSeqFiles_PDB/'
    all_files = os.listdir(file_dir)
    random.seed(4)
    random.shuffle(all_files)
    
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    all_files_list = []
    ##add fasta file
    #pdb_file = open('pdb_yxc_list_train_672.fa','w')
    ## end add fasta file
    for index,item_file in enumerate(all_files):
        #t0 = subprocess.getstatusoutput('awk \'{if(NR <7 && $1 ~/^[AUCGNnaucg]/)print $0}\' bpRNA_dataset/TS0/'+item_file)
        t0 = subprocess.getstatusoutput('awk \'{print $2}\' '+file_dir+item_file)
        seq = ''.join(t0[1].split('\n'))
        #pdb_file.write('>%s\n%s\n'%(item_file,seq.upper()))
        #t0_0 = subprocess.getstatusoutput('cp %s pdb_bpseq_train/'%(file_dir+item_file))
        if t0[0] == 0:
            try:
                one_hot_matrix = one_hot(seq.upper())
            except:
                pdb.set_trace()
        t1 = subprocess.getstatusoutput('awk \'{print $1}\' '+file_dir+item_file)
        t2 = subprocess.getstatusoutput('awk \'{print $3}\' '+file_dir+item_file)
        if t1[0] == 0 and t2[0] == 0:
            pair_dict_all_list = [[int(item_tmp)-1,int(t2[1].split('\n')[index_tmp])-1] for index_tmp,item_tmp in enumerate(t1[1].split('\n')) if int(t2[1].split('\n')[index_tmp]) != 0]
        else:
            pair_dict_all_list = []
        seq_name = item_file
        seq_len = len(seq)
        pair_dict_all = dict([item for item in pair_dict_all_list if item[0]<item[1]])

        if index%1000==0:
            print('current processing %d/%d'%(index+1,len(all_files)))

        if seq_len > 0:
            ss_label = np.zeros((seq_len,3),dtype=int)
            ss_label[[*pair_dict_all.keys()],] = [0,1,0]
            ss_label[[*pair_dict_all.values()],] = [0,0,1]
            ss_label[np.where(np.sum(ss_label,axis=1) <= 0)[0],] = [1,0,0]

            ######Readers can change the length here to support longer sequences######
            ##Default =1800 nt
            one_hot_matrix_1800 = np.zeros((1800,4))
            one_hot_matrix_1800[:seq_len,] = one_hot_matrix
            ss_label_1800 = np.zeros((1800,3),dtype=int)
            ss_label_1800[:seq_len,] = ss_label
            ss_label_1800[np.where(np.sum(ss_label_1800,axis=1) <= 0)[0],] = [1,0,0]
            #pdb.set_trace()

            sample_tmp = RNA_SS_data(seq=one_hot_matrix_1800,ss_label=ss_label_1800,length=seq_len,name=seq_name,pairs=pair_dict_all_list)
            all_files_list.append(sample_tmp)
        
    print(len(all_files_list))
    cPickle.dump(all_files_list,open("data/newdataset.cPickle","wb"))
