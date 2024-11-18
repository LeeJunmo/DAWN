import torch
import pandas as pd
import pickle
import random
import time
import re

import os.path as osp
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import numpy as np
import scipy.sparse as sp
from TimestampEstimator import tidtotime
from datetime import datetime, timedelta
from copy import deepcopy
from tqdm import tqdm

from utils import *

def make_eng_mat(active_users, news_num, temp_uid_dict, temp_sid_dict):
        
    num_active_users = len(list(active_users.keys()))
    obs_articles_len = news_num
    
    eng_mat = torch.zeros(num_active_users, obs_articles_len)
    
    for uid in list(active_users.keys()):
        tweets = active_users[uid]
        row_num = temp_uid_dict[uid]
        
        for sid in tweets:
            col_num = temp_sid_dict[sid]    # article timeline 순으로 배열되어 있으니
            eng_mat[row_num, col_num] += 1
        
    return eng_mat
   
def construct_full(obj, main_dir, train_ratio, valid_ratio, u_thres, sampling_type=None, random_ratio=None, start=None, contain_ratio=None ):

    if sampling_type == None:
        base_file = main_dir + obj + "_base_data.pkl"
        base_data = pickle.load(open(base_file, 'rb'))
    else:
        if sampling_type == 'random':
            base_dir = main_dir + obj + "_base_data_rand" + str(random_ratio) +".pkl"
        else:
            base_dir = main_dir + obj + "_base_data_start" + str(start) + "_contain" + str(contain_ratio) +".pkl"
        base_data = pickle.load(open(base_dir, 'rb'))
    
    news_features, news_labels = base_data['news_features'], base_data['news_labels']
    tweet_sorted = base_data['tweet_sorted']
    temp_df, temp_sid_dict = base_data['temp_df'], base_data['temp_sid_dict']
    
    # 1. Data Construction
    
    news_sorted = list(temp_sid_dict.keys())
    news_num = len(news_sorted)
    tweet_num = tweet_sorted.shape[0]
    
    train_split_idx = round(news_num * train_ratio)
    train_end_time = temp_df.loc[train_split_idx,'real_time'] 
    train_end_list = tweet_sorted[tweet_sorted['real_time'] == train_end_time].index.tolist()       
    train_end = train_end_list[0]                                                               # valid/test가 시작되는 index for tweet
    
    # For validation
    
    if valid_ratio > 0.0:
        train_valid_ratio = train_ratio + valid_ratio
        train_valid_split_idx = round(news_num * train_valid_ratio)
        train_valid_end_time = temp_df.loc[train_valid_split_idx,'real_time']
        train_valid_end_list = tweet_sorted[tweet_sorted['real_time'] == train_valid_end_time].index.tolist()
        train_valid_end = train_valid_end_list[0]
    
    user_engagement = {}
    user_tweet_indices = {}
    obs_news = []
    current_active_user_set = {}
    current_user_tweet_indices = {}
    
    for tweet_idx in range(tweet_num):              # full_tweet
        sid = tweet_sorted.loc[tweet_idx, 'sid']     # article name
        uid = tweet_sorted.loc[tweet_idx, 'uid']     # user

        ############## For data splitting ############

        # train information
        if tweet_idx == train_end:
            train_obs_news = deepcopy(obs_news)
            train_active_users = deepcopy(current_active_user_set)
            train_active_users_tweet_indices = deepcopy(current_user_tweet_indices)
            
        # valid information
        if valid_ratio > 0.0:
            
            if tweet_idx == train_valid_end:
                valid_obs_news = deepcopy(obs_news) 
                valid_active_users = deepcopy(current_active_user_set)
                valid_active_users_tweet_indices = deepcopy(current_user_tweet_indices)
        
        ############## For data splitting ############
        
        if sid not in obs_news:
            obs_news.append(sid)
        
        if uid not in list(user_engagement.keys()):
            user_engagement[uid], user_tweet_indices[uid] = [], []
            user_engagement[uid].append(sid)
            user_tweet_indices[uid].append(tweet_idx)
            
        else:
            user_engagement[uid].append(sid)
            user_tweet_indices[uid].append(tweet_idx)
        
        # Current Active User updating
        if len(user_engagement[uid]) > u_thres-1 :
            current_active_user_set[uid] = user_engagement[uid]
            current_user_tweet_indices[uid] = user_tweet_indices[uid]
            
        if tweet_idx % 10000 == 0:
            print("{}th tweet is retrieved".format(tweet_idx+1))
        
    final_active_users = current_active_user_set
    final_active_users_tweet_indices = current_user_tweet_indices
    
    temp_uid_dict = {}
    for idx, uid in enumerate(list(final_active_users.keys())):        
        temp_uid_dict[uid] = idx
    
    train_eng_mat = make_eng_mat(train_active_users, len(train_obs_news), temp_uid_dict, temp_sid_dict )
    test_eng_mat = make_eng_mat(final_active_users, news_num, temp_uid_dict, temp_sid_dict)
    if valid_ratio > 0.0:
        valid_eng_mat = make_eng_mat(valid_active_users, len(valid_obs_news), temp_uid_dict, temp_sid_dict )
        
    # Train/Valid/Test News label
    y_train = news_labels[:train_split_idx]
    y_test = news_labels[train_split_idx:]
    if valid_ratio > 0.0:
        y_valid = news_labels[train_split_idx:train_valid_split_idx]
        y_test = news_labels[train_valid_split_idx:]
    
    # Train/Valid/Test mask
    train_mask = torch.zeros(news_num)
    train_mask[:train_split_idx] = 1
    train_mask = train_mask.to(torch.bool)
    test_mask = torch.zeros(news_num)
    test_mask[train_split_idx:] = 1
    test_mask = test_mask.to(torch.bool)
    
    if valid_ratio > 0.0:
        valid_mask = torch.zeros(news_num)
        valid_mask[train_split_idx:train_valid_split_idx] = 1
        valid_mask = valid_mask.to(torch.bool)
        test_mask = torch.zeros(news_num)
        test_mask[train_valid_split_idx:] = 1
        test_mask = test_mask.to(torch.bool)
    
    save_dir = main_dir
    
    #tweet_text = tweet_sorted['text']
    #, "tweet_text" : tweet_text
    
    file_name_base = save_dir + obj + "_t" + str(u_thres) + "_train_ratio" + str(train_ratio)
    
    if valid_ratio > 0.0:
        if sampling_type == None:
            file_name = file_name_base + "_valid_ratio" + str(valid_ratio) + "_full.pkl"
        elif sampling_type == 'random':
            file_name = file_name_base + "_valid_ratio" + str(valid_ratio) + '_rand' + str(random_ratio) + "_full.pkl"
        else:
            file_name = file_name_base + "_valid_ratio" + str(valid_ratio) + "_start" + str(start) + "_contain" + str(contain_ratio) +"_full.pkl"
        
        result = { "train_mask" : train_mask, "y_train" : y_train, "train_eng_mat" : train_eng_mat, "train_active_users" : train_active_users, "train_end" : train_end,
                "valid_mask" : valid_mask, "y_valid" : y_valid, "valid_eng_mat" : valid_eng_mat, "valid_active_users" : valid_active_users, "valid_end" : train_valid_end,
                "test_mask" : test_mask, "y_test" : y_test, "test_eng_mat" : test_eng_mat,
                "news_features" : news_features, "temp_df" : temp_df, "temp_uid_dict" : temp_uid_dict, "final_active_users" : final_active_users,
                "train_active_users_tweet_indices" : train_active_users_tweet_indices, "valid_active_users_tweet_indices" : valid_active_users_tweet_indices, "final_active_users_tweet_indices" : final_active_users_tweet_indices
                }
    else:
        if sampling_type == None:
            file_name = file_name_base + "_full.pkl"
        elif sampling_type == 'random':
            file_name = file_name_base + '_rand' + str(random_ratio) + "_full.pkl"
        else:
            file_name = file_name_base + "_start" + str(start) + "_contain" + str(contain_ratio) +"_full.pkl"
        
        result = { "train_mask" : train_mask, "y_train" : y_train, "train_eng_mat" : train_eng_mat, "train_active_users" : train_active_users, "train_end" : train_end,
                "test_mask" : test_mask, "y_test" : y_test, "test_eng_mat" : test_eng_mat,
                "news_features" : news_features, "temp_df" : temp_df, "temp_uid_dict" : temp_uid_dict, "final_active_users" : final_active_users,
                "train_active_users_tweet_indices" : train_active_users_tweet_indices, "final_active_users_tweet_indices" : final_active_users_tweet_indices
                }
        
    with open(file_name, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', default='politifact', type=str)
    parser.add_argument('--main_dir', default='./', type=str)                   # dataset 저장 되는 장소
    parser.add_argument('--train_ratio', default=0.7, type=float)               # train ratio
    parser.add_argument('--u_thres', default=3, type=int)                       # active user limit
    parser.add_argument('--valid_ratio', default=0.1, type=float)               # valid data가 전체에서 얼마를 차지하고 있는지에 대한 정보 ( 0.1 이면 전체 중 10% 가 validation )
    parser.add_argument('--gpu',default=5, type=int)                            # 사용하게 될 GPU
    args = parser.parse_args()  
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")
    
    obj = args.obj
    obj_3 = obj[:3]
    train_ratio = args.train_ratio
    u_thres = args.u_thres
    valid_ratio = args.valid_ratio
    main_dir = args.main_dir

    # Save folder
    main_dir = main_dir + 'main/' + obj + '/'
    os.makedirs(main_dir, exist_ok=True)
    
    print("Construct total file")
    test_ratio = round(1.0-train_ratio-valid_ratio, 1)
    print("dataset : {} | train_ratio : {} | valid_ratio : {} | test_ratio : {} | u_thres : {} (full access)".format(obj, train_ratio, valid_ratio, test_ratio, u_thres))
    construct_full(obj=obj, main_dir=main_dir, train_ratio=train_ratio, valid_ratio=valid_ratio, u_thres=u_thres)

    print("Successfully done!")