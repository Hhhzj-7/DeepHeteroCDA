import json
import torch
import numpy as np
import random
import torch.nn.functional as F



def load_info_data(path):
    ori_data = np.load(path)
    circRNA_tensor = torch.tensor(ori_data['circ_arr'], dtype =torch.float) # [n_circRNA ,489]
    drug_tensor = torch.tensor(ori_data['drug_arr'], dtype =torch.float) # [n_dRUG, 489]
    circRNA_num = circRNA_tensor.shape[0]
    drug_num = drug_tensor.shape[0]
    node_num = circRNA_num + drug_num
    return circRNA_tensor, drug_tensor, node_num, circRNA_num

def load_pre_process(preprocess_path):
    with open(preprocess_path, 'r') as f:
        a = json.load(f)
        adj = torch.FloatTensor(a['adj'])
        cdi_inter_mat = torch.FloatTensor(a['cdi_inter_mat'])
        train_interact_pos = torch.tensor(a['train_interact_pos'])
        val_interact_pos = torch.tensor(a['val_interact_pos'])
    return adj, cdi_inter_mat, train_interact_pos, val_interact_pos







