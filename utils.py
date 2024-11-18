import torch
import numpy as np
import pandas as pd
import argparse
import re
import sys, os
import os.path as osp
import scipy.sparse as sp
import scipy.io
import ast
import pickle
import math
import gc
import random

from copy import deepcopy

from tqdm import tqdm

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from torch_geometric.io import read_txt_array
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())

##################### set seed #####################

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##################### userful python tools #####################

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def strlower(v):
    return v.lower()

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list " % (s))
    return v

def find_key_for_value(my_dict, search_value):
    
    for key, value in my_dict.items():
        if value == search_value:
            return key
        
    return None

##################### for evaluating #####################

def evaluate(y_pred, y_logit, y_true):

    correct_mask = (y_true == y_pred).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_logit)
    
    return acc, fscore, auc, precision, recall, correct_mask


##################### utils for calculating edge features #####################

def parse_utc_time(time_string):
    try:
        # Try parsing with microseconds
        return datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        # If it fails, parse without microseconds
        return datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

def timestring2utc(time_str):
    
    utc = parse_utc_time(time_str)
    
    return utc

def utc_delta_min(utc, str_mins_to_add):
    
    mins_to_add = re.findall(r'\d+', str_mins_to_add)[0]
    
    new_utc_time = utc + timedelta(minutes=int(mins_to_add))
    
    return new_utc_time

def utc_delta(utc, str_hours_to_add):
    
    hours_to_add = re.findall(r'\d+', str_hours_to_add)[0]

    new_utc_time = utc + timedelta(hours=int(hours_to_add))
    
    return new_utc_time

def read_file(folder, name, dtype=None):
	path = osp.join(folder, '{}.txt'.format(name))
	return read_txt_array(path, sep=',', dtype=dtype)

def unixtotime(unix_int):
    
    real_time = datetime.utcfromtimestamp(unix_int)
    
    return str(real_time)

def make_adj(eng_mat):
    
    eng_mat_thres1 = torch.where(eng_mat < 1, eng_mat, torch.tensor(1., dtype=eng_mat.dtype))
    wadj = eng_mat.transpose(0, 1).matmul(eng_mat)
    wadj = wadj
    co_eng = eng_mat_thres1.transpose(0, 1).matmul(eng_mat_thres1)
    
    return wadj, co_eng

def adj2edgeindex(adj):
    
    adj_sp = sp.coo_matrix(adj.detach().cpu().numpy())
    
    source_nodes = torch.tensor(adj_sp.row, dtype=torch.long)
    target_nodes = torch.tensor(adj_sp.col, dtype=torch.long)
    
    edge_weights = torch.tensor(adj_sp.data, dtype=torch.float)
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)
    
    return edge_index, edge_weights

def get_news_early_uid(eng_mat, temp_uid_dict, users_early_ratios, e_thres):
    
    nonzero_indices_tuple = torch.nonzero(eng_mat, as_tuple=True)
    row_indices, col_indices = nonzero_indices_tuple
    news_uid = { i : row_indices[col_indices == i].tolist() for i in range(eng_mat.shape[1])  }
    
    early_user_list = []
    
    for uid in list(users_early_ratios.keys()):
        uidx = temp_uid_dict[uid]
        
        if users_early_ratios[uid] >= e_thres:
            early_user_list.append(uidx)
    
    news_early_uid = { i : list ( set(row_indices[col_indices == i].tolist()) & set(early_user_list) ) for i in range(eng_mat.shape[1])  }

    return news_uid, news_early_uid

def early_find( active_users, tweet, temp_df, dead_criterion, save_dir):

    early_ratio_list = []       
    users_early_ratios = {}     # key : user id / value : user early ratio
    users_early_tweets = {}     # key : user id / value : user early tid dictionary
    users_late_tweets = {}      # key : user id / value : user late tid dictionary
    users_time_deltas = {}      # key : user id / value : user tweet delay dictionary
    
    for user in tqdm(list(active_users.keys())):
        
        tweets = tweet[tweet['uid'] == user]        
        tweet_index_list = tweets.index.tolist()
        sid_list = tweets['sid'].tolist()           
        time_list = tweets['real_time'].tolist()    
        
        tweet_num = len(time_list)                  
        early_tweet = 0
        
        users_early_tweets[user] = {}
        users_late_tweets[user] = {}
        users_time_deltas[user] = {}
        
        for idx in range(len(sid_list)):
            tweet_idx = tweet_index_list[idx]                                               
            sid = sid_list[idx]
            news_time = temp_df[temp_df['sid'] == sid]['real_time'].tolist()[0]             
            
            tweet_time = timestring2utc(time_list[idx])                                     
            if dead_criterion[-1] == 'm':                                                   
                deadline_time = utc_delta_min(timestring2utc(news_time), dead_criterion)
            else:
                deadline_time = utc_delta(timestring2utc(news_time), dead_criterion)
            
            if sid not in list(users_early_tweets[user].keys()):                            
                users_early_tweets[user][sid] = []
            
            if sid not in list(users_late_tweets[user].keys()):                             
                users_late_tweets[user][sid] = []
            
            if tweet_time > deadline_time:                                                  
                users_late_tweets[user][sid].append(tweet_idx)
            else:
                early_tweet +=1                                                             
                users_early_tweets[user][sid].append(tweet_idx)

            news_time = timestring2utc(news_time)                                           
            
            time_difference = tweet_time - news_time                                        
            delta_t = int(time_difference.total_seconds() / 60)                             
            
            if sid not in list(users_time_deltas[user].keys()):        
                users_time_deltas[user][sid] = []
                users_time_deltas[user][sid].append(delta_t)
            else:
                users_time_deltas[user][sid].append(delta_t)
                
        early_ratio = early_tweet / tweet_num                                               
        early_ratio_list.append(early_ratio)                                                
        
        users_early_ratios[user] = early_ratio

    result = { "early_ratio" : users_early_ratios, "early_tweets" : users_early_tweets, "late_tweets" : users_late_tweets , "time_deltas" : users_time_deltas }
    
    with open(save_dir, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    return users_early_ratios, users_early_tweets, users_late_tweets, users_time_deltas

def make_co_features(save_feature, save_dir, temp_uid_dict, temp_sid_dict, edge_index, eng_mat, news_uid, news_early_uid, users_early_tweets, users_late_tweets, final_type='ratio'):

    ee_features = torch.zeros(edge_index.shape[1])
    el_features = torch.zeros(edge_index.shape[1])
    le_features = torch.zeros(edge_index.shape[1])
    ll_features = torch.zeros(edge_index.shape[1])
    
    ue_features = torch.zeros(edge_index.shape[1])
    te_features = torch.zeros(edge_index.shape[1])
    
    rev_temp_uid_dict = { idx : user for idx, user in enumerate(list(temp_uid_dict.keys())) }
    rev_temp_sid_dict = { idx : sid for idx, sid in enumerate(list(temp_sid_dict.keys())) }
    
    for idx in tqdm(range(edge_index.shape[1])):
        
        if idx % 10000 == 0:
            print("{}th edge_index is retreived".format(idx))
        
        src, tgt = edge_index[0, idx].item(), edge_index[1, idx].item()             
        src_users, tgt_users = news_uid[src], news_uid[tgt]                         
        co_users = list( set(src_users) & set(tgt_users) )                          
        co_users_tweets = int(eng_mat[co_users][:, [src, tgt]].sum().item())        
        
        src_early_users, tgt_early_users = news_early_uid[src], news_early_uid[tgt] 
        co_early_users = list( set(src_early_users) & set(tgt_early_users) )        
        co_late_users = list ( set(co_users) - set(co_early_users) )                
        
        ee_list = [ len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[src]]) + len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[tgt]]) for uidx in co_early_users ]
        el_list = [ len(users_late_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[src]]) + len(users_late_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[tgt]]) for uidx in co_early_users ]
        le_list = [ len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[src]]) + len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[tgt]]) for uidx in co_late_users ]
        ll_list = [ len(users_late_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[src]]) + len(users_late_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[tgt]]) for uidx in co_late_users ]
        
        if final_type == 'ratio':
            ee = np.sum(ee_list) / co_users_tweets
            el = np.sum(el_list) / co_users_tweets
            le = np.sum(le_list) / co_users_tweets
            ll = np.sum(ll_list) / co_users_tweets
        else:
            ee = np.sum(ee_list)
            el = np.sum(el_list)
            le = np.sum(le_list)
            ll = np.sum(ll_list)
        
        ee_features[idx] = ee
        el_features[idx] = el
        le_features[idx] = le
        ll_features[idx] = ll
    
        src_early_users, tgt_early_users = news_early_uid[src], news_early_uid[tgt]
        co_early_users = list( set(src_early_users) & set(tgt_early_users) )
        user_earliness = len(co_early_users) / len(co_users)
                
        co_users_early_tweets_list = [ len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[src]]) + len(users_early_tweets[rev_temp_uid_dict[uidx]][rev_temp_sid_dict[tgt]]) for uidx in co_users ]
        co_users_early_tweets = np.sum(co_users_early_tweets_list)
        tweet_earliness = co_users_early_tweets / co_users_tweets
        
        ue_features[idx] = user_earliness
        te_features[idx] = tweet_earliness
    
    if save_feature:
        print("co_features save started!")
        co_features = { "ee" : ee_features, "el" : el_features, "le" : le_features, "ll" : ll_features } #"ue" : ue_features, "te" : te_features
        
        with open(save_dir, 'wb') as f:
            pickle.dump(co_features, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("co_features save done!")
        
    return ee_features, el_features, le_features, ll_features


def get_early_ratio(main_dir, base_dir, datasetname, u_thres, train_ratio, valid_ratio, active_users_dict, train_end, valid_end, temp_df, dead_criterion):

    data_types = ['train', 'valid', 'test']
    users_early_ratios_dict = {}
    users_early_tweets_dict = {}
    users_late_tweets_dict = {}
    users_time_deltas_dict = {}
    
    for data_type in data_types:
        
        save_dir = main_dir + 'main/' + datasetname + '/'
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + datasetname + '_t' + str(u_thres) + '_train_ratio' + str(train_ratio) + '_valid_ratio' + str(valid_ratio) +'_dcr' + str(dead_criterion)
            
        save_dir = save_dir + data_type + '_td_el_ratio_tweet.pkl'
        
        if os.path.exists(save_dir):
            results = pickle.load(open(save_dir, 'rb'))
            users_early_ratios = results['early_ratio']
            users_early_tweets = results['early_tweets']
            users_late_tweets = results['late_tweets']
            users_time_deltas = results['time_deltas']
        else:
            base_data = pickle.load(open(base_dir, 'rb'))
            
            tweet_sorted = base_data['tweet_sorted']        # same as test_tweet
            train_tweet = tweet_sorted.iloc[:train_end,:]
            valid_tweet = tweet_sorted.iloc[:valid_end,:]
            
            train_active_users = active_users_dict['train']
            valid_active_users, test_active_users = active_users_dict['valid'], active_users_dict['test']
            
            if data_type == 'train':
                active_users, tweet = train_active_users, train_tweet
            elif data_type == 'valid':
                active_users, tweet = valid_active_users, valid_tweet
            else:
                active_users, tweet = test_active_users, tweet_sorted
                
            users_early_ratios, users_early_tweets, users_late_tweets, users_time_deltas = early_find(active_users, tweet, temp_df, dead_criterion, save_dir)
        
        users_early_ratios_dict[data_type] = users_early_ratios
        users_early_tweets_dict[data_type] = users_early_tweets
        users_late_tweets_dict[data_type] = users_late_tweets
        users_time_deltas_dict[data_type] = users_time_deltas

    return users_early_ratios_dict, users_early_tweets_dict, users_late_tweets_dict, users_time_deltas_dict

def add_edge_feat(args, base_dir, vari_hyper, data_type, active_users_dict, train_end, valid_end, temp_df, temp_sid_dict, temp_uid_dict, edge_index, eng_mat, prune_hyper=None):

    ft_save_dir = args.main_dir + 'features/' + args.dataset_name + '/'
    os.makedirs(ft_save_dir, exist_ok=True)
    feat_dir = ft_save_dir + args.dataset_name + '_'
    
    for key in list(vari_hyper.keys()):
    
        value = str(vari_hyper[key])
        
        feat_dir = feat_dir + key + '-' + value + '_'
    
    feat_dir = feat_dir + 't' + str(args.u_thres) + '_train_ratio' + str(args.train_ratio) + '_valid_ratio' + str(args.valid_ratio) + '_'
    feat_save_dir = feat_dir + 'fts_' + str(data_type) + '.pkl'
    
    if os.path.exists(feat_save_dir):
        
        print("Preprocessed co-feature download started!") 

        if os.path.exists(feat_save_dir) :
            co_features = pickle.load(open(feat_save_dir, 'rb'))
        else:
            raise Exception("No such features!")
        
        ee, el, le, ll = co_features['ee'], co_features['el'], co_features['le'], co_features['ll'] 
        edge_feat = torch.cat([ee.reshape(-1, 1), el.reshape(-1, 1), le.reshape(-1, 1), ll.reshape(-1, 1)], dim=1)
        
        if args.normalize:
            
            if data_type == 'train':
                scaler = StandardScaler()
                edge_feat = scaler.fit_transform(edge_feat)
                edge_feat = torch.Tensor(edge_feat)
                args.scaler = scaler
            else:
                edge_feat = args.scaler.transform(edge_feat)
                edge_feat = torch.Tensor(edge_feat)

        return edge_feat
        
    else:
        
        users_early_ratios_dict, users_early_tweets_dict, users_late_tweets_dict, users_time_deltas_dict = get_early_ratio(args.main_dir, base_dir, args.dataset_name, args.u_thres, args.train_ratio, args.valid_ratio, active_users_dict, train_end, valid_end, temp_df, args.dead_criterion)
                    
        users_early_ratios, users_early_tweets, users_late_tweets, users_time_deltas = users_early_ratios_dict[data_type], users_early_tweets_dict[data_type], users_late_tweets_dict[data_type], users_time_deltas_dict[data_type]
        
        news_uid, news_early_uid = get_news_early_uid(eng_mat, temp_uid_dict, users_early_ratios, args.early_thres)
        ee, el, le, ll = make_co_features(args.save_feature, feat_save_dir, temp_uid_dict, temp_sid_dict, edge_index, eng_mat, news_uid, news_early_uid, users_early_tweets, users_late_tweets, args.final_type)
        edge_feat = torch.cat([ee.reshape(-1, 1), el.reshape(-1, 1), le.reshape(-1, 1), ll.reshape(-1, 1)], dim=1)
        
        if args.normalize:
            if data_type == 'train':
                scaler = StandardScaler()
                edge_feat = scaler.fit_transform(edge_feat)
                edge_feat = torch.Tensor(edge_feat)
                args.scaler = scaler
            else:
                edge_feat = args.scaler.transform(edge_feat)
                edge_feat = torch.Tensor(edge_feat)

    return edge_feat

##################### utils for saving #####################

def result_save_csv(args, save_main_dir, results, hps):

    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_main_dir, exist_ok=True)

    accs = results['acc']
    precs = results['prec']
    recs = results['rec']
    f1s = results['f1']
    aucs = results['auc']
    #correct_masks = results['test_correct_mask']
    save_dir = save_main_dir + datetime_now + '_'
    
    for key in list(hps.keys()):
        
        value = str(hps[key])
        
        save_dir = save_dir + key + '-' + value + '_'
    
    save_dir += '.txt'
    
    main_metric = { 'Acc(mean)' : round(np.mean(accs),3) , 'F1(mean)' : round(np.mean(f1s),3), 'Acc(std)' : round(np.std(accs),3), 'F1(std)' : round(np.std(f1s), 3) }
    
    main_metric_df = pd.DataFrame(main_metric, index=[0]) 
    main_metric_df.to_csv(save_dir, index=False, sep='\t')
    
    # correct mask save
    """
    correct_mask_dict = {}
    #correct_mask_dict['test'] = correct_masks
    
    cor_save_folder = args.main_dir + 'correct/' + args.dataset_name + '/' + args.exp_name + '/'
    os.makedirs(cor_save_folder, exist_ok=True)
    cor_save_dir = cor_save_folder + args.dataset_name + '_'
    
    for key in list(hps.keys()):
            
        value = str(hps[key])
        
        cor_save_dir = cor_save_dir + key + '-' + value + '_'
        
    cor_save_dir += '.pkl'
    
    with open(cor_save_dir, 'wb') as f:
        pickle.dump(correct_mask_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    """

    print("Save well done!")