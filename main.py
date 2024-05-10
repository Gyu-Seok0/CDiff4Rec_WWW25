from ast import parse
from copy import deepcopy
from tqdm import tqdm
import random
import gc
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np

import models.gaussian_diffusion_method as gd
from models.DNN import DNN
from models.Attention import Attention

import evaluate_utils
import data_utils
from utils import *
from config import get_config

def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print("device: ", device)
    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ######################################### Data #########################################

    dataset_path = os.path.join(args.data_path, args.dataset) + "/"
    print("dataset_path", dataset_path)

    train_path = dataset_path + 'train_list.npy'
    valid_path = dataset_path + 'valid_list.npy'
    test_path = dataset_path + 'test_list.npy'
    pseudo_path = dataset_path + 'pseudo_data.pickle'

    train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path) #  sp.csr_matrix
    train_data_A = torch.FloatTensor(train_data.A)
    train_dataset = data_utils.DataDiffusion_method(train_data_A) # A = .toarray() -> N x M mat

    print("num of train_data :", train_data.getnnz())
    print("num of valid_y_data :", valid_y_data.getnnz())
    print("num of test_y_data :", test_y_data.getnnz())
    print("total data :", train_data.getnnz() + valid_y_data.getnnz() + test_y_data.getnnz())

    print("n_user :", n_user)
    print("n_item :", n_item)
    
    ######################################### Real User #########################################
    
    print(f"Using_Real = {args.using_real}, r_agg = {args.r_agg}")
    
    if args.using_real:
        r_cosine_sim = cosine_similarity(train_data)
        np.fill_diagonal(r_cosine_sim, 0.0)
        r_cosine_sim = torch.tensor(r_cosine_sim) / args.tau

        r_fill_value = get_fill_value(args.r_agg, args.topk)
        r_topk_UU = keep_topk_values(r_cosine_sim.to(device), k = args.topk, fill_value = r_fill_value)
        r_topk_UU = r_topk_UU.double()

    ######################################### Fake User #########################################

    print(f"Using_Fake = {args.using_fake}, f_agg = {args.f_agg}")
    
    if args.using_fake:
        
        fake_data = load_pickle(pseudo_path) # keys: train_data, valid_y_data, test_y_data, n_user, n_item
        f_train_data, f_valid_y_data, f_test_y_data, f_n_user, f_n_item = fake_data["train_data"], fake_data["valid_y_data"], fake_data["test_y_data"], fake_data["n_user"], fake_data["n_item"]
        
        print("num of f_train_data :", f_train_data.getnnz())
        print("num of f_valid_y_data :", f_valid_y_data.getnnz())
        print("num of f_test_y_data :", f_test_y_data.getnnz())
        
        f_n_user = args.topk_fake_user
        print(f"topk_fake_user = {args.topk_fake_user}")
        
        f_train_data_A = torch.FloatTensor(f_train_data.A)[:args.topk_fake_user]
        f_train_dataset = data_utils.DataDiffusion_method(f_train_data_A)
        f_train_data_A = f_train_data_A.double().to_sparse().to(device)

        f_cosine_sim = cosine_similarity(train_data, f_train_data)[:, :args.topk_fake_user] # User x topk_pu
        np.fill_diagonal(f_cosine_sim, 0.0)
        f_cosine_sim = torch.tensor(f_cosine_sim) / args.tau

        f_fill_value = get_fill_value(args.f_agg, args.topk)
        f_topk_UU = keep_topk_values(f_cosine_sim.to(device), k = args.topk, fill_value = f_fill_value)
        f_topk_UU = f_topk_UU.double()#.to(device)
            
        print(f"Training_Fake = {args.training_fake}")
        if args.training_fake:
            train_dataset = data_utils.ConcatDataset(train_dataset, f_train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion_method(train_data_A + torch.FloatTensor(valid_y_data.A))
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    mask_tv = train_data + valid_y_data

    print('data ready.')
    ######################################### Global Attention #########################################

    if args.using_real and args.using_fake:
        alpha = args.alpha
        beta = args.beta
        gamma = args.gamma
        
    elif args.using_real:
        alpha = args.alpha
        beta = 1 - args.alpha
        gamma = 0.0
        
    elif args.using_fake:
        alpha = args.alpha
        beta = 0.0
        gamma = 1 - args.alpha
        
    else:
        alpha = 1.0
        beta = 0.0
        gamma = 0.0
        
    print(f"alpha = {alpha:.2f}, beta = {beta:.2f}, gamma = {gamma:.2f}")
    args.alpha = alpha
    args.beta = beta
    args.gamma = gamma

    ######################################### Model #########################################

    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, args.noise_scale, args.noise_min, args.noise_max, args.steps, device, args.alpha, args.random_seed).to(device)

    ### Build MLP ###
    out_dims = eval(args.dims) + [n_item] # [1000, 2810]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    print("\n[model]\n", model)

    ### Attention layer ###
    if args.r_agg == "att" or args.f_agg == "att":
        att_model = Attention(n_item, args.topk).to(device)
        print("\n[att_model]\n", att_model)
        param = [{"params" : model.parameters()}, {"params" : att_model.parameters()}]

        optimizer = optim.AdamW(param, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("models ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num

    att_num = 0
    if args.r_agg == "att" or args.f_agg == "att":
        att_num = sum([param.nelement() for param in att_model.parameters()])
        param_num += att_num

    print(f"Number of all parameters : {param_num}, mlp_num = {mlp_num}, diff_num = {diff_num}, att_num = {att_num}")

    

    ######################################### Training #########################################

    best_recall, best_epoch = -100, 0
    best_test_result = None
    print("args.random_seed", args.random_seed)
    set_random_seed(args.random_seed)

    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        if epoch - best_epoch >= 20:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        total_r_loss = 0.0
        total_f_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            with torch.cuda.amp.autocast():
                if args.training_fake:
                    r_batch_user, r_idxs = batch[0][0], batch[0][1]
                    f_batch_user, f_idxs = batch[1][0], batch[1][1]
                    f_batch_user = f_batch_user.to(device)
                    f_target, f_model_output, f_ts, f_x_start, f_x_t, f_pt = diffusion.get_output(model, f_batch_user)
                    f_losses = diffusion.get_losses(f_target, f_model_output, args.reweight, f_ts, f_x_start, f_x_t, f_pt)
                    f_loss = f_losses["loss"].mean()
                else:
                    r_batch_user, r_idxs = batch[0], batch[1]
                    f_loss = torch.tensor(0.0).to(device)
                
                r_batch_user = r_batch_user.to(device)
                r_target, r_model_output, r_ts, r_x_start, r_x_t, r_pt = diffusion.get_output(model, r_batch_user)
                
                r_batch_info = torch.zeros_like(r_model_output).to(device)
                f_batch_info = torch.zeros_like(r_model_output).to(device)
                            
                if args.using_real:
                    r_batch_weight = r_topk_UU[r_idxs, :][:, r_idxs].to(device)
                    
                    if args.r_agg in ["avg", "sim"]:
                        r_batch_info = torch.spmm(r_batch_weight.to_sparse(), r_model_output)
                    
                    elif args.r_agg == "att":
                        r_batch_info = att_model.forward(x1 = r_model_output.float(), label_map = r_batch_weight)
                        
                if args.using_fake:
                    if args.training_fake:
                        f_batch_weight = f_topk_UU[r_idxs, :][:, f_idxs].to(device) # batch_real_user x batch_fake_user
                        
                        if args.f_agg in ["avg", "sim"]:
                            f_batch_info = torch.spmm(f_batch_weight.to_sparse(), f_model_output) # batch_real_user x Item
                            
                        elif args.f_agg == "att":
                            f_batch_info = att_model.forward(x1 = r_model_output.float(),
                                                            x2 = f_model_output.float(),
                                                            label_map = f_batch_weight)
                        
                    else:
                        f_batch_weight = f_topk_UU[r_idxs, :].to(device) # real_user x total_fake_user
                        
                        assert args.f_agg != "att"

                        if args.f_agg in ["avg", "sim"]:
                            f_batch_info = torch.spmm(f_batch_weight.to_sparse(), f_train_data_A) # real_user x Item

                r_model_output = alpha * r_model_output + beta * r_batch_info + gamma * f_batch_info
                r_losses = diffusion.get_losses(r_target, r_model_output, args.reweight, r_ts, r_x_start, r_x_t, r_pt)
                r_loss = r_losses["loss"].mean()
            
                batch_loss = r_loss + args.lamda * f_loss
                
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            total_r_loss += r_loss.item()
            total_f_loss += args.lamda * f_loss.item()
        
        if epoch % 1 == 0:
            valid_results = evaluate(model, diffusion, test_loader, valid_y_data, r_topk_UU, f_topk_UU, f_train_data_A, train_data, eval(args.topN), n_user, n_item, f_n_user, args)
            if args.tst_w_val:
                test_results = evaluate(model, diffusion, test_twv_loader, test_y_data, r_topk_UU, f_topk_UU, f_train_data_A, mask_tv, eval(args.topN), n_user, n_item, f_n_user, args)
            else:
                test_results = evaluate(model, diffusion, test_loader, test_y_data, r_topk_UU, f_topk_UU, f_train_data_A, mask_tv, eval(args.topN), n_user, n_item, f_n_user, args)
            evaluate_utils.print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall: # recall@20 as selection
                best_recall, best_epoch, test_recall = valid_results[1][1], epoch, test_results[1][1]
                best_model_state_dict = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
                
                if args.r_agg == "att" or args.f_agg == "att":
                    best_att_model_state_dict = deepcopy({k: v.cpu() for k, v in att_model.state_dict().items()})
                    
                best_results = valid_results
                best_test_results = test_results
        
        print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f} '.format(total_loss) + 
            'real_user_loss {:.4f} '.format(total_r_loss) + 'fake_user_loss {:.4f} '.format(total_f_loss) +
            "time costs: " + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)
        
    if not args.save:
        save_dir_path = args.save_path + f"_valid_recall_{best_recall:.4f}_test_recall_{test_recall:.4f}"
    else:
        save_dir_path = args.save_path
        
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        
    if args.save:
        save_path = os.path.join(save_dir_path, "model.pth")
        torch.save(best_model_state_dict, save_path)
        print(f"Model saved in {save_path}")
        
    print('==='*18)
    print("End. Best Epoch {:03d} ".format(best_epoch))
    evaluate_utils.print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print_command_args(args)

if __name__ == "__main__":
    args = get_config()
    main(args)
