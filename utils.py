import torch
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from requests import get
import pickle
from datetime import datetime
import sys
import gc
import evaluate_utils

def get_predict_items_masking(score_mat, mask_mat, k = 1000):
    
    score_mat[mask_mat.nonzero()] = -np.inf
    sorted_mat = torch.topk(score_mat, k).indices.detach().cpu().numpy().tolist()
    
    return sorted_mat

def get_score_mat_for_LGCN(model):
    user_emb, item_emb = model.get_embedding()
    score_mat = torch.matmul(user_emb, item_emb.T)    
    return score_mat

def get_score_mat_for_VAE(model, train_loader, gpu):
    model = model.to(gpu)
    score_mat = torch.zeros(model.user_count, model.item_count).to(gpu)
    with torch.no_grad():
        for mini_batch in train_loader:
            mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}
            output = model.forward_eval(mini_batch)
            score_mat[mini_batch['user'], :] = output
    return score_mat


def get_SNM(total_user, total_item, R, gpu):
    Zero_top = torch.zeros(total_user, total_user)
    Zero_under = torch.zeros(total_item, total_item)
    upper = torch.cat([Zero_top, R], dim = 1)
    lower = torch.cat([R.T, Zero_under], dim = 1)
    Adj_mat = torch.cat([upper,lower])
    Adj_mat = Adj_mat.to_sparse().to(gpu)

    interactions = torch.cat([torch.sum(R,dim = 1), torch.sum(R,dim = 0)])
    D = torch.diag(interactions)
    half_D = torch.sqrt(1/D)
    half_D[half_D == float("inf")] = 0
    half_D = half_D.to_sparse().to(gpu)
    
    SNM = torch.spmm(torch.spmm(half_D, Adj_mat), half_D).detach()
    SNM.requires_grad = False

    del Adj_mat, D, half_D
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return SNM

def print_command_args(args):
    # Get the current datetime
    print("[Time]", str(datetime.now()))
    ip = get("https://api.ipify.org").text
    print()
    print(f"IP = {ip}")
    command = "python " + " ".join(sys.argv)
    print(f"command = {command}")
    print(f"args = {vars(args)}")
    print()
    
def set_random_seed(random_seed):
    
    # Random Seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# 각 행의 norm을 계산하여 정규화
def get_cos_similarity_mat(M):
    row_norms = np.sqrt((M.multiply(M)).sum(axis=1))
    norm_M = M / np.clip(row_norms, a_min=1e-8, a_max=None)

    # 코사인 유사도 계산
    similarity = cosine_similarity(norm_M)

    # 대각선 원소를 0으로 설정
    np.fill_diagonal(similarity, 0.0)
    
    return torch.tensor(similarity)

def get_cos_similarity_pair(M):
    
    norms = torch.norm(M, dim=1, keepdim = True)
    norm_M = M / torch.clamp(norms, min = 1e-8)
    similarity = torch.matmul(norm_M, norm_M.t())
    similarity.fill_diagonal_(0.0)
    
    return similarity

def get_fill_value(agg, topk):
    
    assert agg is not None
    
    if agg == "att":
        fill_value = 1
        
    elif agg == "avg":
        fill_value = 1 / topk
        
    elif agg == "sim":
        fill_value = None
    return fill_value

def evaluate(model, diffusion, data_loader, data_te, 
             r_topk_UU, f_topk_UU, f_train_data_A,
             mask_his, topN, n_user, n_item, f_n_user, args):
    
    device = torch.device("cuda:0" if args.cuda else "cpu")

    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    r_info = f_info = torch.tensor(0.0).to(device)
    
    r_predictions = torch.zeros((n_user, n_item)).double().to(device)
    
    if args.using_pseudo:
        f_predictions = torch.zeros((f_n_user, n_item)).double().to(device)

    print("Start evaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            
            if args.training_pseudo:
                r_batch_user, r_idxs = batch[0][0], batch[0][1]
                f_batch_user, f_idxs = batch[1][0], batch[1][1]
                
                f_batch_user = f_batch_user.to(device)
                f_prediction = diffusion.p_sample(model, f_batch_user, args.sampling_steps, args.sampling_noise)
                f_predictions[f_idxs] = f_prediction.double()
            else:
                r_batch_user, r_idxs = batch[0], batch[1]
            
            r_batch_user = r_batch_user.to(device)
            r_prediction = diffusion.p_sample(model, r_batch_user, args.sampling_steps, args.sampling_noise)
            r_predictions[r_idxs] = r_prediction.double()
        
        if args.using_real:
            if args.r_agg == "att":
                r_info = torch.spmm((r_topk_UU / args.topk).to_sparse().to(device), r_predictions)
            else:
                r_info = torch.spmm(r_topk_UU.to_sparse().to(device), r_predictions)
        
        if args.using_pseudo:
            if args.training_pseudo:
                if args.f_agg == "att":
                    f_info = torch.spmm((f_topk_UU / args.topk).to_sparse().to(device), f_predictions)
                else:
                    f_info = torch.spmm(f_topk_UU.to_sparse().to(device), f_predictions)
            else:
                try:
                    f_info = torch.spmm(f_topk_UU.to_sparse().to(device), f_train_data_A)
                except:
                    f_info = torch.matmul(f_topk_UU.to(device), f_train_data_A.to_dense())
            del f_predictions
            
        r_predictions = args.alpha * r_predictions + args.beta * r_info + args.gamma * f_info
        r_predictions[mask_his.nonzero()] = -np.inf
        _, indices = torch.topk(r_predictions, topN[-1])
        predict_items = indices.cpu().numpy().tolist()
        del r_predictions, r_info, f_info
    
    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    
    # get gpu memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return test_results

def keep_topk_values(mat, k, fill_value = None):

    top_values, top_indices = torch.topk(mat, k = k, dim = 1)

    
    result_mat = torch.zeros_like(mat)
    
    if fill_value is None:
        top_values = F.softmax(top_values, dim = 1)
        result_mat.scatter_(1, top_indices, top_values)
    else:
        result_mat.scatter_(1, top_indices, fill_value)
    
    result_mat = result_mat.detach().cpu()
    result_mat.requires_grad = False

    return result_mat

def save_pickle(file_path, file):
    with open(file_path,"wb") as f:
        pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path):
    with open(file_path,"rb") as f:
        return pickle.load(f)