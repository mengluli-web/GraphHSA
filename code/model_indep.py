import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel  
from parameters import args_parser
import random
from torch_geometric.nn import GCNConv
import numpy as np
args = args_parser()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class GraphHSA(BaseModel):
    def __init__(self, 
                 protein_num, 
                 symptom_num,
                 drug_num,
                 emb_dim, 
                 n_hop,
                 l1_decay):
        super(GraphHSA, self).__init__()
        self.protein_num = protein_num
        self.symptom_num = symptom_num
        self.drug_num = drug_num
        self.emb_dim = emb_dim
        self.n_hop = n_hop
        self.l1_decay = l1_decay
        
        self.protein_embedding = nn.Embedding(self.protein_num, self.emb_dim)
        self.symptom_embedding = nn.Embedding(self.symptom_num, self.emb_dim)
        self.drug_embedding = nn.Embedding(self.drug_num, self.emb_dim)
        
        self.batch_drug = nn.BatchNorm1d(self.emb_dim)
        self.batch_sym = nn.BatchNorm1d(self.emb_dim)
        
        self.aggregation_function = nn.Linear(self.emb_dim*self.n_hop, self.emb_dim)
        
        
        self.combine_embedding_pro = torch.nn.Linear(self.emb_dim*2, self.emb_dim)
        self.batch_pro = nn.BatchNorm1d(self.emb_dim)
        self.combine_embedding_gcn = torch.nn.Linear(self.emb_dim*2, self.emb_dim)
        self.batch_gcn = nn.BatchNorm1d(self.emb_dim)
        
        self.gcn_layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        for layer in range(args.gcn_layer):
            self.gcn_layers.append(GCNConv(self.emb_dim,self.emb_dim))
            self.batch_norms.append(nn.BatchNorm1d(self.emb_dim))
            self.dropouts.append(nn.Dropout(0.2))
            
        self.p_MLP = MLP(self.emb_dim*2, 16, 1)
        self.MLP_pro = MLP_pro(self.emb_dim, 16, 1)
        self.MLP_gcn = MLP_gcn(self.emb_dim, 16, 1)
        self.weight = torch.nn.Parameter(torch.Tensor(self.emb_dim*2, self.emb_dim*2))
        
        torch.nn.init.xavier_uniform_(self.weight)
        
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.attention_mlp = nn.Sequential(
            nn.Linear(self.emb_dim*2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 2)
        )
        
    def discriminate(self, z, summary, sigmoid=True):
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value
        
    def constrative_loss(self,embeds_all,target):
        EPS = torch.tensor(1e-15)
        pos_z=embeds_all[target.squeeze()==1]
        neg_z=embeds_all[target.squeeze()==0]
        if pos_z.numel() == 0 or neg_z.numel() == 0:
            return 0
        else:
            pos_centers = pos_z.mean(0)
            dgi_loss = -torch.log(self.discriminate(pos_z, pos_centers.squeeze(0)) + EPS).mean() \
                        -torch.log(1 - self.discriminate(neg_z, pos_centers) + EPS).mean()
            
            return dgi_loss
    def multiple_loss(self,text_embeddings, image_embeddings):
        logits_per_text = text_embeddings @ image_embeddings.T / self.temperature
        logits_per_image = image_embeddings @ text_embeddings.T / self.temperature
        batch_size = logits_per_text.shape[0]
        labels = torch.arange(batch_size, device=logits_per_text.device)
        loss_text = nn.CrossEntropyLoss()(logits_per_text, labels)
        loss_image = nn.CrossEntropyLoss()(logits_per_image, labels)
        return torch.log((loss_text + loss_image) / 2  + 1e-15)
            
    def forward(self, symptoms: torch.LongTensor, drug: torch.LongTensor,
                sym_feat: torch.LongTensor, dru_feat: torch.LongTensor,
                symptom_neighbors: list, drug_neighbors: list, edge_index: torch.LongTensor):
        
        x_drug = self.drug_embedding(dru_feat)
        x_drug = self.batch_drug(F.relu(x_drug))
        x_symptom = self.symptom_embedding(sym_feat)
        x_symptom = self.batch_sym(F.relu(x_symptom))
        
        symptom_embeddings=x_symptom[symptoms]
        drug_embeddings=x_drug[drug]
        
        embeds_all=torch.cat((x_drug,x_symptom))
        for layer in range(args.gcn_layer):
            embeds_all=self.gcn_layers[layer](embeds_all,edge_index)
            if torch.sum(embeds_all)!=0:
                embeds_all = self.batch_norms[layer](embeds_all)
            embeds_all=F.relu(self.dropouts[layer](embeds_all))
            
        symptom_embeddings_gcn=embeds_all[self.drug_num:][symptoms]
        drug_embeddings_gcn=embeds_all[:self.drug_num][drug]
        
        symptom_neighbors_emb_list = self._get_neighbor_emb(symptom_neighbors)
        drug_neighbors_emb_list = self._get_neighbor_emb(drug_neighbors)
        
        symptom_i_list,contributions_sym = self._interaction_aggregation(symptom_embeddings, symptom_neighbors_emb_list)
        drug_i_list,contributions_drug = self._interaction_aggregation(drug_embeddings, drug_neighbors_emb_list)
        
        symptom_embeddings = self._aggregation(symptom_i_list)
        drug_embeddings = self._aggregation(drug_i_list)
        
        embeds_pro=F.relu(self.combine_embedding_pro(torch.cat([drug_embeddings,symptom_embeddings], dim=1)))
        embeds_pro = self.batch_pro(embeds_pro)
        embeds_gcn=F.relu(self.combine_embedding_gcn(torch.cat([drug_embeddings_gcn,symptom_embeddings_gcn], dim=1)))
        embeds_gcn = self.batch_gcn(embeds_gcn)
        if args.lambda_sim==0:
            loss_multi=0
        else:
            loss_multi=self.multiple_loss(embeds_pro, embeds_gcn)
            
        combined = torch.cat([embeds_pro, embeds_gcn], dim=-1)
        attention_logits = self.attention_mlp(combined)
        attention_weights = F.softmax(attention_logits, dim=-1)  
        
        x1_weighted = embeds_pro * attention_weights[:, 0].unsqueeze(-1)  
        x2_weighted = embeds_gcn * attention_weights[:, 1].unsqueeze(-1)
        combined_emb = torch.cat([x1_weighted, x2_weighted], dim=1)   
        
        score = self.p_MLP(combined_emb)
        score = torch.squeeze(score, 1)
        if args.lambda_pred==0:
            score_pro=[]
            score_gcn=[]
        else:
            score_pro = self.MLP_pro(embeds_pro)
            score_pro = torch.squeeze(score_pro, 1)
            
            score_gcn = self.MLP_gcn(embeds_gcn)
            score_gcn = torch.squeeze(score_gcn, 1)
        return score, combined_emb, loss_multi, score_pro, score_gcn, contributions_sym, contributions_drug


    def _get_neighbor_emb(self, neighbors):
        neighbors_emb_list = []
        for hop in range(self.n_hop):
            features=self.protein_embedding(neighbors[hop])
            neighbors_emb_list.append(features)  
        return neighbors_emb_list
    
    def _interaction_aggregation(self, item_embeddings, neighbors_emb_list):
        interact_list = []
        contribution_list=[]
        for hop in range(self.n_hop):
            neighbor_emb = neighbors_emb_list[hop]  
            item_embeddings_expanded = torch.unsqueeze(item_embeddings, dim=2)  
            contributions = torch.squeeze(torch.matmul(neighbor_emb, item_embeddings_expanded))
            contributions_normalized = F.softmax(contributions, dim=1)   
            contributions_expaned = torch.unsqueeze(contributions_normalized, dim=2)  
            i = (neighbor_emb * contributions_expaned).sum(dim=1)  
            item_embeddings = i
            interact_list.append(i)
            contribution_list.append(contributions_normalized)
        return interact_list,contribution_list

    def _aggregation(self, item_i_list):
        # [batch_size, n_hop+1, emb_dim]
        item_i_concat = torch.cat(item_i_list, 1)  
        # [batch_size, emb_dim]
        item_embeddings = self.aggregation_function(item_i_concat) 
        return item_embeddings

class MLP(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.fc23 = nn.Linear(n_hidden, n_output)
 
    def forward(self,x):
        h_1 = torch.tanh(self.fc1(x))
        h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x
class MLP_pro(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLP_pro,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.fc23 = nn.Linear(n_hidden, n_output)
 
    def forward(self,x):
        h_1 = torch.tanh(self.fc1(x))
        h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x

class MLP_gcn(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super(MLP_gcn,self).__init__()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.batch1 = nn.BatchNorm1d(n_hidden)
        self.fc23 = nn.Linear(n_hidden, n_output)
 
    def forward(self,x):
        h_1 = torch.tanh(self.fc1(x))
        h_1 = self.batch1(h_1)
        x = self.fc23(h_1)
        return x