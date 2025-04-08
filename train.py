import json
import os
import re
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model import DeepHeteroCDA
from dataloader import load_info_data, load_pre_process
from utils import accuracy, precision, recall, specificity, auc, aupr, f1
from drug_gcn import smile_to_graph, TestbedDataset, GCNNet, DataLoader

parser = argparse.ArgumentParser(description='CDI-GRAPH')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, help='Random seed.') 
parser.add_argument('--epochs', type=int, default=3500)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--model_dir', type=str, default='./enzyme_model_com10')
parser.add_argument('--crossvalidation', type=int, default=1)
parser.add_argument('--drug_ninput', type=int, default=256)
parser.add_argument('--dnn_nlayers', type=int, default=1,
                    help='dnn_nlayers num')
parser.add_argument('--dnn_nhid', type=str, default='[]')
parser.add_argument('--gat_type', type=str, default='PyG')
parser.add_argument('--gat_ninput', type=int, default=256)
parser.add_argument('--gat_nhid', type=int, default=256)
parser.add_argument('--gat_noutput', type=int, default=256)
parser.add_argument('--gat_nheads', type=int, default=3)
parser.add_argument('--gat_negative_slope', type=float, default=0.5)
parser.add_argument('--CDA_nn_nlayers', type=int, default=3)
parser.add_argument('--CDA_nn_nhid', type=str, default='[512,512,512]')

parser.add_argument('--dataset', type=str, default='enzyme')
parser.add_argument('--common_neighbor', type=int, default=1)
parser.add_argument('--data_path', type=str, default='./five_ten_cv_data/five',
                    help='dataset root path')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# nn layers
p1 = re.compile(r'[[](.*?)[]]', re.S)
if args.dnn_nhid == '[]':
    args.dnn_nhid = []
else:
    args.dnn_nhid = [int(i) for i in re.findall(p1, args.dnn_nhid)[0].replace(' ', '').split(',')]
args.CDA_nn_nhid = [int(i) for i in re.findall(p1, args.CDA_nn_nhid)[0].replace(' ', '').split(',')]

smiles = np.load("drug_smile.npy")

compound_iso_smiles = []
compound_iso_smiles += list(smiles)
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}

for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

df_smiles = pd.DataFrame(smiles)

drugdata = TestbedDataset( xd=df_smiles[0], y= [0 for _ in range(218)], smile_graph=smile_graph)
drugdata = DataLoader(drugdata, batch_size=32, shuffle=None)


smiles_gcn = GCNNet
smiles_gcn = smiles_gcn().to(device)

# Hyper Setting
dnn_hyper = [args.drug_ninput, args.dnn_nhid, args.gat_ninput, args.dnn_nlayers]
GAT_hyper = [args.gat_ninput, args.gat_nhid, args.gat_noutput, args.gat_negative_slope, args.gat_nheads]
Deco_hyper = [args.gat_noutput, args.CDA_nn_nhid, args.CDA_nn_nlayers]

def train(epoch, link_cdi_id_train, edge_index, edge_weight, train_cdi_inter_mat):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    row_cdi_id = link_cdi_id_train.permute(1, 0)[0]
    col_cdi_id = link_cdi_id_train.permute(1, 0)[1]
    circRNA_index = row_cdi_id
    drug_index = col_cdi_id + train_cdi_inter_mat.shape[0]
    for batch_idx, data in enumerate(drugdata):
        data1 = data.to(device)
        output, _ = model(epoch, circRNA_tensor, drug_tensor, edge_index, circRNA_index, drug_index, edge_weight, data1)
    Loss = nn.BCELoss()
    loss_train = Loss(output, train_cdi_inter_mat[row_cdi_id, col_cdi_id])
    acc_cdi_train = accuracy(output, train_cdi_inter_mat[row_cdi_id, col_cdi_id])
    loss_train.backward()
    optimizer.step()
    print('Epoch {:04d} Train '.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_cdi_train: {:.4f}'.format(acc_cdi_train),
          'time: {:.4f}s'.format(time.time() - t))

def test(link_cdi_id_test, edge_index, edge_weight, test_cdi_inter_mat):
    model.eval()
    row_cdi_id = link_cdi_id_test.permute(1, 0)[0]
    col_cdi_id = link_cdi_id_test.permute(1, 0)[1]
    circRNA_index = row_cdi_id
    drug_index = col_cdi_id + test_cdi_inter_mat.shape[0]
    
    
    for batch_idx, data in enumerate(drugdata): 
        data1 = data.to(device)   
        output, emb = model(epoch, circRNA_tensor, drug_tensor, edge_index, circRNA_index, drug_index, edge_weight, data1)
    Loss = nn.BCELoss()
    predicts = output
    targets = test_cdi_inter_mat[row_cdi_id, col_cdi_id]
    loss_test = Loss(predicts, targets)
    acc_cdi_test = accuracy(output, test_cdi_inter_mat[row_cdi_id, col_cdi_id])
    return acc_cdi_test, loss_test, predicts, targets, emb

# Train model
t_total = time.time()

fold_num = 5 if args.crossvalidation else 1

for train_times in range(fold_num):
    predict_list = []
    label_list = []
    # load data
    data_Path = os.path.join(args.data_path, 'data_'+args.dataset+str(train_times+1)+'.npz')
    preprocess_path = os.path.join(args.data_path, args.dataset+str(train_times+1)+'_com_'+str(args.common_neighbor))
    # save dir
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    circRNA_tensor, drug_tensor, node_num, circRNA_num = load_info_data(data_Path)
    
    model = DeepHeteroCDA(GAT_hyper=GAT_hyper, DNN_hyper=dnn_hyper, DECO_hyper=Deco_hyper,
                      CircRNA_num=circRNA_tensor.shape[0], Drug_num=drug_tensor.shape[0], dropout=args.dropout, smiles_gcn= smiles_gcn)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_test = 0
    preprocess_oripath = os.path.join(preprocess_path, '0_'+str(train_times+1)+'.json')  
    adj, ori_cdi_inter_mat, ori_train_interact_pos, ori_val_interact_pos = load_pre_process(preprocess_oripath)
    edge_index = torch.nonzero(adj > 0).permute(1, 0)
    edge_weight = adj[np.array(edge_index)]
    if args.cuda:
        model = model.cuda()
        circRNA_tensor = circRNA_tensor.cuda()
        drug_tensor = drug_tensor.cuda()
        edge_index = edge_index.cuda()
        edge_weight = edge_weight.cuda() 
        ori_cdi_inter_mat = ori_cdi_inter_mat.cuda()
        ori_train_interact_pos = ori_train_interact_pos.cuda()
        ori_val_interact_pos = ori_val_interact_pos.cuda()
    save_time_fold = os.path.join(args.model_dir, str(train_times))

    max_auc = 0
    max_aupr = 0
    max_precision = 0
    max_recall = 0
    max_specificity = 0
    max_f1 = 0
    max_acc = 0
    
    preprocess_generate_path = os.path.join(preprocess_path, '0_'+str(train_times+1) + '.json')
    adj, cdi_inter_mat, train_interact_pos, val_interact_pos = load_pre_process(preprocess_generate_path)
    if args.cuda:
        cdi_inter_mat = cdi_inter_mat.cuda()
        train_interact_pos = train_interact_pos.cuda()
    
    epoch = 0
    test_score, test_loss, predicts, targets, emb = test(ori_val_interact_pos, edge_index, edge_weight, ori_cdi_inter_mat)


    for epoch in range(args.epochs):

        train(epoch, train_interact_pos, edge_index, edge_weight, cdi_inter_mat)
        test_score, test_loss, predicts, targets, emb = test(ori_val_interact_pos, edge_index, edge_weight, ori_cdi_inter_mat)

        auc_score = round(auc(predicts, targets), 4)
        aupr_score = round(aupr(predicts, targets), 4)
        precision_score = round(precision(predicts, targets), 4)
        recall_score = round(recall(predicts, targets), 4)
        specificity_score = round(specificity(predicts, targets), 4)
        f1_score = round(f1(predicts, targets), 4)
        if auc_score > max_auc:
            predict_list = predicts
            targets_list = targets
            predict_target = torch.cat((predicts, targets), dim=0).detach().cpu().numpy()
            precision_score = round(precision(predicts, targets), 4)
            recall_score = round(recall(predicts, targets), 4)
            specificity_score = round(specificity(predicts, targets), 4)
            f1_score = round(f1(predicts, targets), 4)
            auc_score = round(auc(predicts, targets), 4)
            aupr_score = round(aupr(predicts, targets), 4)
            acc_score = round(accuracy(predicts, targets), 4)

            max_auc = auc_score
            max_aupr = aupr_score
            max_recall = recall_score
            max_precision = precision_score
            max_specificity = specificity_score
            max_f1 = f1_score
            max_acc = acc_score
            


    print("acc Score:", max_acc)
    print("precision Score:", max_precision)
    print("recall score", max_recall)
    print("specificity score", max_specificity)
    print("f1 score", max_f1)
    print("auc socre", max_auc)
    print("aupr score", max_aupr)
    print("Best Ave Test: {:.4f}".format(np.mean(acc_score)))

