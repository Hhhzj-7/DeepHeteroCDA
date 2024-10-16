import json
import torch
import os
import argparse
import random
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold
from tqdm import tqdm


def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError
    
def cmp_time(a, b):
    a_num = int(a.split('_')[1])
    b_num = int(b.split('_')[1])
    return a_num - b_num


def add_cdi_info(circRNA_num, drug_num, positive_sample_num, train_positive_inter_pos, val_positive_inter_pos, refer_val_interact_pos,
                 ):
    train_interact_pos = []
    train_label = []
    val_interact_pos = []
    val_label = []
    # generate negative sample
    circRNA_id_list = list(range(circRNA_num))
    drug_id_list = list(range(drug_num))
    negative_num = 0
    negative_interact_pos = []
    temp_refer_val_interact_pos = np.array(refer_val_interact_pos[0])
    for i in range(1, 5):
        temp_refer_val_interact_pos = np.concatenate((temp_refer_val_interact_pos, refer_val_interact_pos[i]), axis=0)
    temp_refer_val_interact_pos = temp_refer_val_interact_pos.tolist()
    while negative_num < positive_sample_num:
        circRNA_neg_id = random.choice(circRNA_id_list)
        drug_neg_id = random.choice(drug_id_list)
        neg_pos = [circRNA_neg_id, drug_neg_id]
        if neg_pos in temp_refer_val_interact_pos:
            # it is positive sample or the negative sample exist in final test set
            continue
        negative_interact_pos.append(neg_pos)
        negative_num += 1
    negative_interact_pos = np.array(negative_interact_pos)
    train_negative_inter_pos = []
    val_negative_inter_pos = []
    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    negative_index = list(range(len(negative_interact_pos)))
    kf = kf.split(negative_index)
    for i, (train_index, val_index) in enumerate(kf):
        train_negative_inter_pos.append(negative_interact_pos[train_index])
        val_negative_inter_pos.append(negative_interact_pos[val_index])
    # merge
    for i in range(5):
        train_interact_pos.append(np.concatenate((train_positive_inter_pos[i], train_negative_inter_pos[i]), axis=0))
        train_positive_label = np.ones(len(train_positive_inter_pos[i]))
        train_negative_label = np.zeros(len(train_negative_inter_pos[i]))
        train_label.append(np.concatenate((train_positive_label, train_negative_label), axis=0))

        val_interact_pos.append(np.concatenate((val_positive_inter_pos[i], val_negative_inter_pos[i]), axis=0))
        val_positive_label = np.ones(len(val_positive_inter_pos[i]))
        val_negative_label = np.zeros(len(val_negative_inter_pos[i]))
        val_label.append(np.concatenate((val_positive_label, val_negative_label), axis=0))

    # construct cdi
    cdi_list = []
    for i in range(5):
        cdi_inter_mat = -np.ones((circRNA_num, drug_num))  # [circRNA_num, drug_num]
        for j, inter in enumerate(train_interact_pos[i]):
            circRNA_id = inter[0]
            drug_id = inter[1]
            label = train_label[i][j]
            cdi_inter_mat[circRNA_id][drug_id] = label
        for j, inter in enumerate(val_interact_pos[i]):
            circRNA_id = inter[0]
            drug_id = inter[1]
            label = val_label[i][j]
            cdi_inter_mat[circRNA_id][drug_id] = label
        cdi_inter_mat = cdi_inter_mat.tolist()
        cdi_list.append(cdi_inter_mat)

    return cdi_list, train_interact_pos, val_interact_pos


def first_spilt_label(inter, groups):
    """
    :param
        protein [n, len, 20]
        drug [n, 881]
        inter [labelnum, 3] (p_id, d_id, label)

    :return:
        train_positive_interact_pos [5, n_train_p, 2]
        train_interact_pos [5, n_train~, 2]
        train_label [5, n_train~, 1]
        val_inter_pos [5, n_val~, 2]
        val_label [5, n_val~, 1]
    """
    inter_folds = [[],[],[],[],[]]
    label_folds = [[],[],[],[],[]]
    pos_inter_folds = [[],[],[],[],[]]
    pos_label_folds = [[],[],[],[],[]]
    neg_inter_folds = [[],[],[],[],[]]
    neg_label_folds = [[],[],[],[],[]]
    for i, inter_k in enumerate(inter):
        inter_type = inter_k[-1]  # 1 positive, 0 negative
        circRNA_node_id = inter_k[0]
        drug_node_id = inter_k[1]
        fold_id = int(groups[i])
        inter_folds[fold_id].append([circRNA_node_id, drug_node_id])
        label_folds[fold_id].append(inter_type)
        if inter_type == 1:
            #  positive sample
            pos_inter_folds[fold_id].append([circRNA_node_id, drug_node_id])
            pos_label_folds[fold_id].append(inter_type)
        elif inter_type == 0:
            # negative sample
            neg_inter_folds[fold_id].append([circRNA_node_id, drug_node_id])
            neg_label_folds[fold_id].append(inter_type)
        else:
            print("inter type has problem")

    train_positive_inter_pos = [[],[],[],[],[]]
    val_positive_inter_pos = [[],[],[],[],[]]
    train_negative_inter_pos = [[],[],[],[],[]]
    val_negative_inter_pos = [[],[],[],[],[]]

    train_interact_pos = [[],[],[],[],[]]
    val_interact_pos = [[],[],[],[],[]]
    train_label = [[],[],[],[],[]]
    val_label = [[],[],[],[],[]]

    for i in range(5):
        val_fold_id = i
        for j in range(5):
            if j != val_fold_id:
                train_positive_inter_pos[i] += pos_inter_folds[j]
                train_negative_inter_pos[i] += neg_inter_folds[j]

                train_interact_pos[i] += inter_folds[j]
                train_label[i] += label_folds[j]
            else:
                val_positive_inter_pos[i] += pos_inter_folds[j]
                val_negative_inter_pos[i] += neg_inter_folds[j]

                val_interact_pos[i] += inter_folds[j]
                val_label[i] += label_folds[j]

    return train_positive_inter_pos, val_positive_inter_pos, train_interact_pos, train_label, val_interact_pos, val_label, train_negative_inter_pos

def get_groups(train_ids, val_ids):
    """
    :param train_ids: (N1,)
    :param val_ids: (N2,)
    :return:
    groups: (N1+N2,) 0 is val, nonzero is train
    """
    dataset_len = len(train_ids) + len(val_ids)
    train_len = len(train_ids)
    groups = np.zeros(dataset_len)
    train_foldlen = int(train_len / 4)
    train_groups = np.ones(train_len)
    train_groups[train_foldlen:2 * train_foldlen] *= 2
    train_groups[2 * train_foldlen:3 * train_foldlen] *= 3
    train_groups[3 * train_foldlen:] *= 4
    groups[train_ids] = train_groups

    return groups


def load_data(data_root, dataset='enzyme', start_epoch=0, end_epoch=2000, common_neibor=3, neg_common_neibor=1, adj_norm=True, crossval=True):

    data_path = os.path.join(data_root, 'data_' + dataset + '.npz')
    root_save_dir = os.path.join(data_root, 'preprocess')
    if not os.path.exists(root_save_dir):
        os.mkdir(root_save_dir)
    dataset_save_dir = os.path.join(root_save_dir, dataset+'_com_'+str(common_neibor))
    if not os.path.exists(dataset_save_dir):
        os.mkdir(dataset_save_dir)
    data_file = np.load(data_path)
    circRNA_data = data_file['pssm_arr']
    drug_data = data_file['drug_arr']
    int_label = data_file['int_ids']
    if crossval:
        groups = data_file['folds']
    else:
        groups = get_groups(data_file['train_ids'], data_file['val_ids'])
    circRNA_num = len(circRNA_data)
    drug_num = len(drug_data)
    node_num = circRNA_num + drug_num
    positive_sample_num = len(int_label) // 2
    train_positive_inter_pos, val_positive_inter_pos, test_train_interact_pos, test_train_label, \
    test_val_interact_pos, test_val_label, train_negative_inter_pos = first_spilt_label(int_label, groups)

    test_adj_list = []
    # construct test file
    fold_num = 5 if crossval else 1
    for i in range(fold_num):
        test_cdi_inter_mat = -np.ones((circRNA_num, drug_num))  # [protein_num, drug_num]
        for j, inter in enumerate(test_train_interact_pos[i]):
            circRNA_id = inter[0]
            drug_id = inter[1]
            label = test_train_label[i][j]
            test_cdi_inter_mat[circRNA_id][drug_id] = label
        for j, inter in enumerate(test_val_interact_pos[i]):
            circRNA_id = inter[0]
            drug_id = inter[1]
            label = test_val_label[i][j]
            test_cdi_inter_mat[circRNA_id][drug_id] = label
        test_cdi_inter_mat = test_cdi_inter_mat.tolist()

        # construct adj
        test_adj_transform = constr_adj(node_num, train_positive_inter_pos[i], train_negative_inter_pos[i], circRNA_num, common_neibor, neg_common_neibor, adj_norm)
        test_adj_list.append(test_adj_transform)
        save_dict = {}
        save_dict['adj'] = test_adj_transform
        save_dict['cdi_inter_mat'] = test_cdi_inter_mat
        save_dict['train_interact_pos'] = test_train_interact_pos[i]
        save_dict['val_interact_pos'] = test_val_interact_pos[i]
        with open(os.path.join(dataset_save_dir, '0_' + str(i) + '.json'), 'w') as f:
            json.dump(save_dict, f, default=convert)
    for epoch in tqdm(range(start_epoch+1, end_epoch)):
        print("******epoch", epoch, "******")

        cdi_list, train_interact_pos, val_interact_pos = \
            add_cdi_info(circRNA_num, drug_num, positive_sample_num, train_positive_inter_pos,
                         val_positive_inter_pos, test_val_interact_pos)
        for i in range(fold_num):
            save_dict = {}
            save_dict['adj'] = test_adj_list[i]
            save_dict['cdi_inter_mat'] = cdi_list[i]
            save_dict['train_interact_pos'] = train_interact_pos[i].tolist()
            save_dict['val_interact_pos'] = val_interact_pos[i].tolist()
            with open(os.path.join(dataset_save_dir, str(epoch) + '_' + str(i) + '.json'), 'w') as f:
                json.dump(save_dict, f, default=convert)

def parallel_pos_transform_adj(node_num, adj, sample_type='positive',common_neibor=3):
    neighbor_mask = (adj.repeat(1, node_num).view(node_num * node_num, -1) + adj.repeat(node_num, 1))  # n^2, n
    ones_vec_0 = torch.ones_like(neighbor_mask)
    zeros_vec_0 = torch.zeros_like(neighbor_mask)
    if sample_type == 'positive':
        neighbor_mask = torch.where(neighbor_mask == 2, ones_vec_0, zeros_vec_0).sum(1)
    elif sample_type == 'negative':
        neighbor_mask = torch.where(neighbor_mask == -2, ones_vec_0, zeros_vec_0).sum(1)
    else:
        print("wrong_type")
    ones_vec_1 = torch.ones_like(neighbor_mask)
    zeros_vec_1 = torch.zeros_like(neighbor_mask)
    adj_transform = torch.where(neighbor_mask > common_neibor, ones_vec_1, zeros_vec_1).view(node_num, node_num)
    return adj_transform

def pos_transform_adj(node_num, adj, sample_type='positive',common_neibor=3):
    # neighbor_mask = (adj.repeat(1, node_num).view(node_num * node_num, -1) + adj.repeat(node_num, 1))  # n^2, n
    adj_transform = torch.zeros_like(adj)
    ones_vec_0 = torch.ones_like(adj[0])
    zeros_vec_0 = torch.zeros_like(adj[0])
    for row in tqdm(range(node_num)):
        row_adj = adj[row]
        for col in range(node_num):
            col_adj = adj[col]
            neighbor_mask = row_adj + col_adj
            if sample_type == 'positive':
                com_num = torch.where(neighbor_mask == 2, ones_vec_0, zeros_vec_0).sum(0).item()
            elif sample_type == 'negative':
                com_num = torch.where(neighbor_mask == -2, ones_vec_0, zeros_vec_0).sum(0).item()
            else:
                print("wrong_type")
            if com_num > common_neibor: adj_transform[row][col] = 1
    return adj_transform



def parallel_neg_transform_adj(node_num, pos_adj, neg_adj, common_neibor=3):
    trans_neg_adj = -neg_adj
    neighbor_mask = (pos_adj.repeat(1, node_num).view(node_num * node_num, -1) + trans_neg_adj.repeat(node_num, 1))  # n^2, n
    ones_vec_0 = torch.ones_like(neighbor_mask)
    zeros_vec_0 = torch.zeros_like(neighbor_mask)
    neighbor_mask = torch.where(neighbor_mask == 2, ones_vec_0, zeros_vec_0).sum(1)
    adj_transform = (-neighbor_mask / torch.max(neighbor_mask).item()).view(node_num, node_num)
    # adj_transform = torch.where(neighbor_mask > common_neibor, -neighbor_mask, zeros_vec_1).view(node_num, node_num)
    return adj_transform


def neg_transform_adj(node_num, pos_adj, neg_adj, common_neibor=3):
    trans_neg_adj = -neg_adj
    adj_transform = torch.zeros_like(pos_adj)
    ones_vec_0 = torch.ones_like(pos_adj[0])
    zeros_vec_0 = torch.zeros_like(pos_adj[0])
    for row in tqdm(range(node_num)):
        row_adj = pos_adj[row]
        for col in range(node_num):
            col_adj = trans_neg_adj[col]
            neighbor_mask = row_adj + col_adj
            adj_transform[row][col] = torch.where(neighbor_mask == 2, ones_vec_0, zeros_vec_0).sum(0).item()
    adj_transform = (-adj_transform / torch.max(adj_transform).item())
    return adj_transform

def constr_adj(node_num, interact_index, neg_inter_index, protein_num, common_neibor, neg_common_neibor, adj_norm):
    """
    """
    adj_transform = torch.zeros((node_num, node_num))
    positive_adj = torch.zeros((node_num, node_num))
    # add dti info to adj
    for inter_k in interact_index:
        protein_node_id = int(inter_k[0])
        drug_node_id = int(inter_k[1])
        positive_adj[protein_node_id][drug_node_id + protein_num] = 1
        positive_adj[drug_node_id + protein_num][protein_node_id] = 1
    negative_adj = torch.zeros((node_num, node_num))
    for inter_k in neg_inter_index:
        protein_node_id = int(inter_k[0])
        drug_node_id = int(inter_k[1])
        negative_adj[protein_node_id][drug_node_id + protein_num] = -1
        negative_adj[drug_node_id + protein_num][protein_node_id] = -1
        
    positive_adj_transform = pos_transform_adj(node_num, positive_adj, sample_type='positive',common_neibor=common_neibor)
    negative_adj_transform = pos_transform_adj(node_num, negative_adj, sample_type='negative',common_neibor=common_neibor)
    negone_adj_transform = neg_transform_adj(node_num, positive_adj, negative_adj, common_neibor=neg_common_neibor)
    
    adj_transform = F.normalize(positive_adj_transform + negative_adj_transform + negone_adj_transform)
    adj_transform = positive_adj + negative_adj + adj_transform
    if adj_norm:
        adj_transform = F.normalize(adj_transform)
    adj_transform = adj_transform.tolist()
    return adj_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='enzyme',
                        help='dataset')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='epoch_num')
    parser.add_argument('--end_epoch', type=int, default=2000,
                        help='epoch_num')
    parser.add_argument('--common_neighbor', type=int, default=3,
                        help='common neighbor threshold')
    parser.add_argument('--neg_common_neighbor', type=int, default=3,
                        help='common neighbor threshold')
    parser.add_argument('--adj_norm', type=bool, default=True,
                        help='adj_norm')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='data root')
    parser.add_argument('--crossval', type=int, default=1,
                        help='whether generate 5 fold')
    args = parser.parse_args()
    random.seed(2)
    load_data(args.data_root, dataset=args.dataset, start_epoch=args.start_epoch, end_epoch=args.end_epoch,
              common_neibor=args.common_neighbor, neg_common_neibor=args.neg_common_neighbor, adj_norm=args.adj_norm, crossval=args.crossval)
