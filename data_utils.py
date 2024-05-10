import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path): # [[0 0], [0 1], [0 2] ... ] -> (1014486, 2)
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
 
    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
   
    # sparse matrix -> train_list[:, 0] = user_id, train_list[:, 1] = item_id
    
    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]),
        (train_list[:, 0], train_list[:, 1])), dtype='float64',
        shape=(n_user, n_item))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
    

class DataDiffusion_method(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item, index
    def __len__(self):
        return len(self.data)
    
class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

    def __len__(self):
        return max(len(d) for d in self.datasets)
    
class implicit_CF_dataset(Dataset):
    def __init__(self, user_count, item_count, rating_mat, num_interactions, num_ns):
        super(implicit_CF_dataset, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        
        self.num_interactions = num_interactions
        self.num_ns = num_ns
        self.length = num_interactions * num_ns
        
        self.train_arr = None
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        assert self.train_arr
               
        return {'user' : self.train_arr[idx][0], 
                'pos_item' : self.train_arr[idx][1], 
                'neg_item' : self.train_arr[idx][2]}
    
    def negative_sampling(self):
        
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10 * self.length)
                
        sample_idx = 0
        for user, pos_items in self.rating_mat.items():
            filtering_items = pos_items
            
            for pos_item in pos_items:
                ns_count = 0
                
                while True:
                    neg_item = sample_list[sample_idx]
                    sample_idx += 1
                    
                    if neg_item not in filtering_items:
                        self.train_arr.append((user, pos_item, neg_item))
                        ns_count += 1
                        
                        if ns_count == self.num_ns:
                            break
                        
class implicit_CF_dataset_AE(Dataset):
    def __init__(self, user_count, item_count, R):
        super(implicit_CF_dataset_AE, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.R = R
       
    def __len__(self):
        return self.user_count
        
    def __getitem__(self, idx):
        return {'user': idx, 'rating_vec': self.R[idx]}

    def negative_sampling(self):
        pass