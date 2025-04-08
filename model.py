
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot

from torch_geometric.nn import GATConv, GCNConv, MessagePassing


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, noutput, dropout, negative_slope, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = GATConv(nfeat, nhid, nheads, True, negative_slope=negative_slope, dropout=self.dropout)

        self.out_att = GATConv(nhid*nheads, noutput, 1, False, negative_slope=negative_slope, dropout=self.dropout)
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=noutput)
        self.LayerNorm = torch.nn.LayerNorm(noutput)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.attentions(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, edge_index))
        x = self.BatchNorm(x)
        x = self.LayerNorm(x)
        return x


class NN(nn.Module):
    def __init__(self, ninput, nhidden, noutput, nlayers, dropout=0.3):

        super(NN, self).__init__()
        self.dropout = dropout
        self.encode = torch.nn.ModuleList([
            torch.nn.Linear(ninput if l == 0 else nhidden[l - 1], nhidden[l] if l != nlayers - 1 else noutput) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l] if l != nlayers-1 else noutput) for l in range(nlayers)])
        self.LayerNormList = torch.nn.ModuleList([
            torch.nn.LayerNorm(nhidden[l] if l != nlayers - 1 else noutput) for l in range(nlayers)])

    def forward(self, x):
        # x [B, 220] or [B, 881]
        for l, linear in enumerate(self.encode):
            x = F.relu(linear(x))
            x = self.BatchNormList[l](x)
            x = self.LayerNormList[l](x)
            x = F.dropout(x, self.dropout)
        return x

class CDA_Decoder(nn.Module):
    def __init__(self, circRNA_num, Drug_num, Nodefeat_size, nhidden, nlayers, dropout=0.3):
        super(CDA_Decoder, self).__init__()
        self.circRNA_num = circRNA_num
        self.Drug_num = Drug_num
        self.dropout = dropout
        self.nlayers = nlayers
        self.decode = torch.nn.ModuleList([
            torch.nn.Linear(Nodefeat_size if l == 0 else nhidden[l - 1], nhidden[l]) for l in
            range(nlayers)])
        self.BatchNormList = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=nhidden[l]) for l in range(nlayers)])
        self.linear = torch.nn.Linear(nhidden[nlayers-1], 1)
        self.max_aoc = 0
        self.max_aupr = 0
        self.max_epoch = 0

        self.w_drug = nn.Parameter(torch.ones(2))
        self.w_rna = nn.Parameter(torch.ones(2))
        self.drug_linear1 = torch.nn.Linear(2048, 1024)
        self.drug_linear2 = torch.nn.Linear(1024, 128)

    def forward(self,nodes_features, circRNA_index, drug_index):
        circRNA_features = nodes_features[circRNA_index]
        drug_features = nodes_features[drug_index]
        pair_nodes_features0 = torch.cat((circRNA_features, drug_features), 1)


        for l, dti_nn in enumerate(self.decode):
            pair_nodes_features = F.dropout(pair_nodes_features0, self.dropout)
            pair_nodes_features1 = F.relu(dti_nn(pair_nodes_features))
            pair_nodes_features = self.BatchNormList[l](pair_nodes_features1)
        pair_nodes_features2 = F.dropout(pair_nodes_features, self.dropout)
        output = self.linear(pair_nodes_features)
        return torch.sigmoid(output), pair_nodes_features2
    
class DeepHeteroCDA(nn.Module):

    def __init__(self, GAT_hyper, DNN_hyper, DECO_hyper, CircRNA_num, Drug_num, dropout, smiles_gcn):
        super(DeepHeteroCDA, self).__init__()
        self.drug_nn = NN(DNN_hyper[0], DNN_hyper[1], DNN_hyper[2], DNN_hyper[3], dropout)
        self.gat = GAT(489, GAT_hyper[1], 256, dropout, GAT_hyper[3], GAT_hyper[4]) #256
        self.CDA_Decoder = CDA_Decoder(CircRNA_num, Drug_num, 512, DECO_hyper[1],DECO_hyper[2], dropout)

        self.CircRNA_num = CircRNA_num
        self.Drug_num = Drug_num
        self.BatchNorm = torch.nn.BatchNorm1d(num_features=489)
        self.LayerNorm = torch.nn.LayerNorm(489)

        self.smiles_gcn = smiles_gcn



    def forward(self, epoch, CircRNAs, Drugs, edge_index, circRNA_index, drug_index, edge_weight,drugdata):
        # CircRNA and Drug embeding
        Drugs_1 = Drugs[:, 0:271]
        Drugs = Drugs[:, :489]
        Rna_1 = CircRNAs[:, 0:218]
        CircRNAs = CircRNAs[:,:489]
        Drugs = self.smiles_gcn(drugdata, Drugs)

        Nodes_features_ori = torch.cat((CircRNAs, Drugs), 0)
        Nodes_features_ori = self.BatchNorm(Nodes_features_ori)
        Nodes_features_ori = self.LayerNorm(Nodes_features_ori)
        Nodes_features = self.gat(Nodes_features_ori, edge_index)
        # Decoder
        output, emb = self.CDA_Decoder(epoch, CircRNAs, Rna_1, Drugs, Drugs_1, Nodes_features, circRNA_index, drug_index)
        output = output.view(-1)
        return output, emb




