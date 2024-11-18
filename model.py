import time
import pickle
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MLP

import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import accuracy_score

from utils import evaluate

#################### Vanila GCN #################### 

# Vanila MLP
class MLPClassifier(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=16):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, node_features):
        x = node_features
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class GCN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, edge_weight):
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        
        return F.log_softmax(x, dim=1)

class DAWN:
    
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_loss = float('inf')
        self.weights = None
        self.estimator = None
        self.estimator_weight = None 
        self.model = model.to(device)
    
    def fit(self, train_data, valid_data):
        
        args = self.args
        
        train_mask = train_data.train_mask
        valid_mask = valid_data.valid_mask
                
        train_edge_index = train_data.edge_index.to(self.device)
        valid_edge_index = valid_data.edge_index.to(self.device)
        
        features = train_data.x.to(self.device)
        num_node_feat = features.shape[1]
        self.features = features
        
        train_edge_features = train_data.edge_feat.to(self.device)
        valid_edge_features = valid_data.edge_feat.to(self.device)
        num_edge_feat = train_edge_features.shape[1]

        y_train = train_data.y_train.to(self.device)
        y_valid = valid_data.y_valid.to(self.device)

        hom_save_dir = args.main_dir + 'homophily/' + args.dataset_name + '/'
        hom_dir = hom_save_dir + args.dataset_name + '_t' + str(args.u_thres) + '_train_ratio' + str(args.train_ratio) + '_valid_ratio' + str(args.valid_ratio) + '_' 
        hom_mask_dir = hom_save_dir + args.dataset_name + '_t' + str(args.u_thres) + '_train_ratio' + str(args.train_ratio) + '_valid_ratio' + str(args.valid_ratio) + '_' 
        
        hom_dir = hom_dir + 'ori_homophily.pkl'
        hom_mask_dir = hom_mask_dir + 'hom_mask.pkl'
        hom_mask = pickle.load(open(hom_mask_dir, 'rb'))
        self.hom_mask = hom_mask
        
        self.estimator = Edge_estimator(num_node_feat, num_edge_feat, args, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.decay)
        self.optimizer_adj = torch.optim.Adam(self.estimator.parameters(), lr=args.lr_adj, weight_decay=args.decay)
        
        # Training Process    
        
        t_total = time.time()
        
        for epoch in range(args.epochs):
            
            self.args.cur_epoch = epoch
            
            for i in range(args.adj_steps):
                self.train_adj(epoch, features, train_edge_features, valid_edge_features, train_edge_index, y_train, valid_edge_index, y_valid, train_mask, valid_mask, hom_mask)
                
            for i in range(args.gnn_steps):
                loss_train, loss_val = self.train_gnn(epoch, features, train_edge_features, valid_edge_features, train_edge_index, y_train, valid_edge_index, y_valid, train_mask, valid_mask)
        
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        
        self.model.load_state_dict(self.weights)
        self.estimator.load_state_dict(self.estimator_weight)
        
        print("Best Model updated done!")
            
    def train_gnn(self, epoch, features, train_edge_features, valid_edge_features, train_edge_index,  y_train, valid_edge_index, y_valid, train_mask, valid_mask, train_loader=None):
        
        # Training GNN
        
        t = time.time()
        
        self.model.train()
        self.optimizer.zero_grad()
        
        train_refined_weights = self.estimator.estimate(train_edge_features)
        out = self.model( features, train_edge_index, train_refined_weights )
        loss_train = F.nll_loss(out[train_mask], y_train)
                
        pred, _ = self.get_pred(out, train_mask)
        acc_train = accuracy_score(pred, y_train.detach().cpu().numpy())
        
        loss_train.backward()
        self.optimizer.step()
            
        # Evaluate 
        self.estimator.eval()
        self.model.eval()
        valid_refined_weights = self.estimator.estimate(valid_edge_features)
        valid_out = self.model( features, valid_edge_index, valid_refined_weights )
        loss_val = F.nll_loss(valid_out[valid_mask], y_valid)
        pred_valid, _ = self.get_pred(valid_out, valid_mask)
        acc_val = accuracy_score(pred_valid, y_valid.detach().cpu().numpy())
        
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())
            self.estimator_weight = deepcopy(self.estimator.state_dict())
            print('\tSaving current gcn | best_val_loss: %s' % self.best_val_loss.item())
        
        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val),
                    'time: {:.4f}s'.format(time.time() - t))

        return loss_train, loss_val
        
        
    def train_adj(self, epoch, features, train_edge_features, valid_edge_features, train_edge_index, y_train, valid_edge_index, y_valid, train_mask, valid_mask, hom_mask, rank_criterion=None, train_loader=None):
        
        args = self.args
        t = time.time()
        
        # Training edge weight estimator
        
        self.estimator.train()
        self.optimizer_adj.zero_grad()
        
        added_loss, train_refined_weights = self.estimator(features, train_edge_features, train_edge_index, hom_mask['train'])
        
        out = self.model(features, train_edge_index, train_refined_weights)
        loss_train = F.nll_loss(out[train_mask], y_train)
        pred, _ = self.get_pred(out, train_mask)
        
        loss = loss_train + args.alpha * added_loss
        loss.backward()
        self.optimizer_adj.step()
        
        total_loss = loss_train.item()
        acc_train = accuracy_score(pred, y_train.detach().cpu().numpy())
            
        # Evaluate 
        self.estimator.eval()
        self.model.eval()
        valid_refined_weights = self.estimator.estimate(valid_edge_features)
        valid_out = self.model( features, valid_edge_index, valid_refined_weights )
        loss_val = F.nll_loss(valid_out[valid_mask], y_valid)
        pred_valid, _ = self.get_pred(valid_out, valid_mask)
        acc_val = accuracy_score(pred_valid, y_valid.detach().cpu().numpy())
        
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())
            self.estimator_weight = deepcopy(self.estimator.state_dict())
            print('\t Saving current gcn | best_val_loss: %s' % self.best_val_loss.item())
        
        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(total_loss),
                    'acc_train: {:.4f}'.format(acc_train),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val),
                    'time: {:.4f}s'.format(time.time() - t))
    
    def test(self, test_data):
        
        # Evaluate the test perforamnce
        features = self.features
        test_edge_features = test_data.edge_feat.to(self.device)
        y_test = test_data.y_test.to(self.device)               # TODO
        test_edge_index = test_data.edge_index.to(self.device)
        test_mask = test_data.test_mask
        
        self.model.eval()
        self.estimator.eval()
        test_refined_weights = self.estimator.estimate(test_edge_features)

        test_out = self.model( features, test_edge_index, test_refined_weights )
        pred_test, pred_test_auc = self.get_pred(test_out, test_mask)

        acc, f1, auc, prec, rec, correct_mask = evaluate(pred_test, pred_test_auc, y_test.detach().cpu().numpy())
        
        return acc, f1, auc, prec, rec
        
    def get_pred(self, out, mask):
        
        with torch.no_grad():
            pred = out.argmax(dim=1)[mask].detach().cpu().numpy()
            pred_auc = out[:,1][mask].detach().cpu().numpy()
            pred_auc = np.exp(pred_auc)

        return pred, pred_auc
        
class Edge_estimator(nn.Module):
    
    def __init__(self, num_node_features, num_edge_features, args, device):
        super(Edge_estimator, self).__init__()

        self.module = args.module
        self.activation = args.activation
        self.est_hidden = args.est_hidden
        self.estimator = nn.Sequential(nn.Linear(num_edge_features, self.est_hidden), nn.ReLU(), nn.Linear(self.est_hidden, 1), nn.Sigmoid() )
                            
        self.device = device
        self.args = args
    
    # Estimate edge weights
    def estimate(self, edge_features):
        
        refined_weights = self.estimator(edge_features)
            
        return refined_weights
    
    # For training loss
    def forward(self, features, edge_features, edge_index, hom_mask):
        
        refined_weights = self.estimate(edge_features)
            
        rank_loss = self.all_rank_loss(refined_weights, hom_mask)
            
        return rank_loss, refined_weights
    
    # Rank Loss computation part
    def all_rank_loss(self, refined_weights, hom_mask):
        
        hom_indices = torch.nonzero(hom_mask).squeeze().tolist()
        het_indices = torch.nonzero(~hom_mask).squeeze().tolist()
        
        weights_tensor = refined_weights.reshape(-1)
        sample_size = self.args.sampling_num
        
        # 1) use all pairs
        if self.args.use_all:
            
            x1 = weights_tensor[hom_indices]
            x2 = weights_tensor[het_indices]
            
            x1_repeated = x1.repeat(x2.shape[0], 1).view(-1)
            x2_repeated = x2.repeat_interleave(x1.shape[0], dim=0)
            
            target = torch.ones_like(x1_repeated)
            random_rank_loss = nn.MarginRankingLoss(margin=self.args.margin)(x1_repeated, x2_repeated, target)
        
        # 2) default use k^2 pairs
        else:
            hom_sampled_indices = random.sample(hom_indices, sample_size)
            het_sampled_indices = random.sample(het_indices, sample_size)
            x1 = weights_tensor[hom_sampled_indices]
            x2 = weights_tensor[het_sampled_indices]
            x1_repeated = x1.repeat(sample_size, 1).view(-1)
            x2_repeated = x2.repeat_interleave(sample_size, dim=0)
            target = torch.ones_like(x1_repeated)
            random_rank_loss = nn.MarginRankingLoss(margin=self.args.margin)(x1_repeated, x2_repeated, target)
    
        rank_loss = random_rank_loss
        
        return rank_loss