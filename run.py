import pickle
import torch
import argparse
import numpy as np
import pandas as pd
import time
import random
import sys,os
sys.path.append(os.getcwd())

from tqdm import tqdm

from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from model import *
from utils import *

def graph_construct(args, base_dir, full_dir, vari_hyper):

    datasetname = args.dataset_name
    main_dir = args.main_dir
    u_thres = args.u_thres
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    get_hom = args.get_hom
    
    print("Data download started!")
    start_time = time.time()
    full_data = pickle.load(open(full_dir, 'rb'))
    
    news_features = full_data['news_features']
    y_train, y_valid, y_test = full_data['y_train'], full_data['y_valid'], full_data['y_test']
    train_mask, valid_mask, test_mask = full_data['train_mask'], full_data['valid_mask'], full_data['test_mask']
    
    train_active_users = full_data['train_active_users']
    valid_active_users = full_data['valid_active_users']
    test_active_users = full_data['final_active_users']
    active_users_dict = { "train" : train_active_users, "valid" : valid_active_users, "test" : test_active_users }
    
    temp_df = full_data['temp_df']
    temp_uid_dict = full_data['temp_uid_dict']
    temp_sid_dict = { sid : idx  for idx, sid in enumerate(full_data['temp_df']['sid'].tolist()) }
    
    train_eng_mat, valid_eng_mat, test_eng_mat = full_data['train_eng_mat'], full_data['valid_eng_mat'], full_data['test_eng_mat']
    
    train_end, valid_end = full_data['train_end'], full_data['valid_end']
    
    train_wadj, train_co_eng = make_adj(train_eng_mat)
    valid_wadj, valid_co_eng = make_adj(valid_eng_mat)
    test_wadj, test_co_eng = make_adj(test_eng_mat)
    
    # original edge idx
    train_edge_index, train_edge_weights = adj2edgeindex(train_wadj)
    valid_edge_index, valid_edge_weights = adj2edgeindex(valid_wadj)
    test_edge_index, test_edge_weights = adj2edgeindex(test_wadj)
    
    ################################### Edge feature construction start ###################################
    
    data_types = ['train', 'valid', 'test']
    
    for data_type in data_types:
    
        if data_type == 'train':
            edge_index, eng_mat = train_edge_index, train_eng_mat
        elif data_type == 'valid':
            edge_index, eng_mat = valid_edge_index, valid_eng_mat
        else:
            edge_index, eng_mat = test_edge_index, test_eng_mat
        
        edge_feat = add_edge_feat(args, base_dir, vari_hyper, data_type, active_users_dict, train_end, valid_end, temp_df, temp_sid_dict, temp_uid_dict, edge_index, eng_mat)

        if data_type == 'train':
            train_data = Data(x = news_features, edge_index=edge_index, edge_feat=edge_feat, train_mask = train_mask, y_train = y_train )
        elif data_type == 'valid':
            valid_data = Data(x = news_features, edge_index=edge_index, edge_feat=edge_feat, valid_mask = valid_mask, y_valid = y_valid)
        else:
            test_data = Data(x = news_features, edge_index=edge_index, edge_feat=edge_feat, test_mask = test_mask, y_test = y_test  )
    
    end_time = time.time()

    print(f"Data Construction Execution time: {end_time - start_time} seconds")
    
    return train_data, valid_data, test_data
        
################################### Data Construction done ###################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='politifact', type=str)
    parser.add_argument('--model_name', default='DAWN', type=str)        
    parser.add_argument('--train_ratio', default=0.7, type=float)
    parser.add_argument('--valid_ratio', default=0.1, type=float)
    parser.add_argument('--u_thres', default=3, type=int)
    parser.add_argument('--main_dir', default='./')            # default 'data_union/'
    
    parser.add_argument('--gpu', default=5, type=int)
    parser.add_argument('--iters', default=1, type=int)                 # model initalization, data split
    parser.add_argument('--epochs', default=1000, type=int)             
    parser.add_argument('--hidden_dim', default=16, type=int)           # GNN hidden dim
    parser.add_argument('--lr', default=0.001, type=float)              # for Adam
    parser.add_argument('--lr_adj', default=0.001, type=float)
    parser.add_argument('--decay', default=5e-4, type=float)            # for Adam
    parser.add_argument('--patience', default=20, type=int)             # for early stopping
    parser.add_argument('--adj_steps', default=2, type=int)
    parser.add_argument('--gnn_steps', default=1, type=int)
    
    # For edge feature
    parser.add_argument('--dead_criterion', default='48hrs', type=str)  # deadline criterion # To calculate earliness of user        # ['48hrs' ]
    parser.add_argument('--early_thres', default=0.3, type=float)                                                                    # [ 0.1, 0.3 ]
    parser.add_argument('--final_type', default='number', type=str)     # Feature type                                               # ['number']
    parser.add_argument('--activation', default='sigmoid', type=str)    # Activation function                                        # ['sigmoid']
    parser.add_argument('--feat_comb', default='eeellell', type=str)
    parser.add_argument('--est_hidden', default=16, type=int)           # Hidden layer dimension of edge weight estimator            # [16, 8]
    parser.add_argument('--normalize', default=True, type=str2bool)     
    
    # For saving 
    parser.add_argument('--use_saved_ft', default=True, type=str2bool)  # decide whether we use saved_feature
    parser.add_argument('--save_feature', default=True, type=str2bool)
    parser.add_argument('--get_hom', default=False, type=str2bool)
    
    parser.add_argument('--module', default='gcn_ef', type=strlower)    # Model                                                      
    parser.add_argument('--mlp_hidden', default=16, type=int)           # For feature embedding
    
    # For rank loss
    parser.add_argument('--margin', default=0.1, type=float)
    parser.add_argument('--sampling_num', default=1000, type=int)
    parser.add_argument('--alpha', default=0.1, type=float)             # for rank loss
    parser.add_argument('--use_all', default=False, type=str2bool)
    
    # for saving
    parser.add_argument('--exp_name', default="240815-test", type=str)
    
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
    main_dir = args.main_dir
    
    # For learning framework
    datasetname = args.dataset_name
    u_thres = args.u_thres
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    model_name = args.model_name
    patience = args.patience
    hidden_dim = args.hidden_dim
    iterations = args.iters
    epochs = args.epochs
    alpha = args.alpha
    
    # For determine EE, EL, LE, LL
    dead_criterion = args.dead_criterion
    early_thres = args.early_thres
    final_type = args.final_type
    
    use_saved_ft = args.use_saved_ft
    save_feature = args.save_feature
    
    result_hps = { "uthr" : u_thres, "iter" : iterations, "epochs" : epochs, "alpha" : alpha,
    "margin": args.margin, "numsample" : args.sampling_num, "useall" : args.use_all
    }
    
    vari_hyper = { "dcr" : args.dead_criterion, "ethrs" : args.early_thres, "type" : args.final_type, "ft" : args.feat_comb }
    module_hyper = { "mod" : args.module}
    
    result_hps.update(vari_hyper)
    result_hps.update(module_hyper)
    args.vari_hyper = vari_hyper
    
    # For load data files,
    
    # data should be in the main folder.
    
    data_dir = main_dir + 'main/' + datasetname + '/'
    os.makedirs(data_dir, exist_ok=True)
    
    base_dir = data_dir + datasetname + '_base_data.pkl'
    full_dir = data_dir + datasetname + '_t' + str(u_thres) + '_train_ratio' + str(train_ratio) + '_valid_ratio' + str(valid_ratio) + '_full.pkl'

    #################### Data Setting ####################
    
    adj_type = False
    
    train_data, valid_data, test_data = graph_construct( args, base_dir, full_dir, vari_hyper )
    
    num_features = train_data.x.shape[1]
    num_classes = 2
    
    #################### Training & Validating & Testing ####################

    test_accs, test_f1s, test_aucs= [], [], []
    test_precs, test_recs = [], []
    #test_correct_masks = {}
    
    train_homophily_result = []
    valid_homophily_result = []
    test_homophily_result = []
    
    model_dir = main_dir + 'model/' + datasetname + '/' + 'module' + str(args.module) + '/'
    os.makedirs(model_dir, exist_ok=True)
    
    for idx, key in enumerate(list(result_hps.keys())):
        
        value = str(result_hps[key])
        model_dir = model_dir + key + value + '_'
    
    model_dir += 'best_model.pth'

    for iter in range(iterations):
        
        set_seed(iter)
        print("------------------------{}th iteration started!------------------------".format(iter))
        model = GCN(num_node_features=num_features, num_classes=num_classes, hidden_dim=hidden_dim)
        
        dawn = DAWN(model, args, device)
        dawn.fit(train_data, valid_data)
                
        print("------------------------Testing with Best model!!!------------------------")
        
        acc, f1, auc, prec, rec = dawn.test(test_data)
        print(f"{iter}th iter : Test_Acc : {acc:.4f}  | Test_F1 : {f1:.4f} ")
        
        test_accs.append(acc), test_f1s.append(f1), test_aucs.append(auc)
        test_precs.append(prec), test_recs.append(rec)
        #test_correct_masks[iter] = correct_mask

    test_data = test_data.cpu()
        
    test_accs, test_f1s,test_aucs  = np.array(test_accs), np.array(test_f1s), np.array(test_aucs)
    test_precs, test_recs = np.array(test_precs), np.array(test_recs)
    
    results = { 'acc' : test_accs, 'prec' : test_precs, 'rec' : test_recs, 
            'f1' : test_f1s, 'auc' : test_aucs }
    #results['test_correct_mask'] = test_correct_masks
    
    save_main_dir = main_dir + 'result/' + datasetname + '/' + args.exp_name + '/module' + str(args.module) + '/'
    os.makedirs(save_main_dir, exist_ok=True)
    result_save_csv(args, save_main_dir, results, result_hps)

    print("------------Result save done!------------")