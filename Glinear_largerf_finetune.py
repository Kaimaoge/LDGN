# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:48:44 2023

@author: AA
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
import torch
import torch.optim as optim
import random
import copy
import numpy as np
from utils.metrics import metric
from utils.tools import gaussian_likelihood_loss, masked_mse, laplacian_likelihood_loss, cauchy_likelihood_loss
from Larger_data.dataloader import load_num, ForecastDataset, _get_time_features, de_normalized, load_dist_adj
from models.GCN_linear import NG_Model_Gaussian_Re
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt

if __name__ ==  '__main__':
    '''
    Parameter
    '''
    device = torch.device('cuda')
    num_nodes = 150
    seq_len = 63
    pred_len = 63
    layers = 4
    c_in = 2
    batch_size = 16
    lrate = 0.001
    dropout = 0.3
    wdecay = 0.0001
    epochs = 100
    clip = 5
    model_params_root = "pth/GWAVE_fine_largerf63_cau.pth"
    pretrain_model = "pth/GWAVE_pre_largerf63.pth"
    
    '''
    Load Data
    '''
    data = load_num()
    data[np.isnan(data)] = 0
    sch = load_num()
    time = _get_time_features(data)
    feature = np.tile(np.expand_dims(time, axis = 0), (150, 1, 1))
    norm_statistic = dict(mean=np.mean(data[:, :int(data.shape[1] * 0.7), :], axis=1), std=np.std(data[:, :int(data.shape[1] * 0.7), :], axis=1))
    train_set = ForecastDataset(data[:, :int(data.shape[1] * 0.7), :], sch[:, :int(data.shape[1] * 0.7), :], feature[:, :int(data.shape[1] * 0.7), :], seq_len, pred_len, norm_statistic) 
    test_set = ForecastDataset(data[:, int(data.shape[1] * 0.8):, :], sch[:, int(data.shape[1] * 0.8):, :], feature[:, int(data.shape[1] * 0.8):, :], seq_len, pred_len, norm_statistic)
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True,
                                        num_workers=1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)
    adj_mx = load_dist_adj()
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    
    
    '''
    Model Definition
    '''
    model = NG_Model_Gaussian_Re(seq_len, pred_len, layers, supports, num_nodes, c_in, device)
    model = model.to(device)
    model = torch.load(pretrain_model)
    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=wdecay)

    MAE_list = []
    
#     print("start training...",flush=True)
    
#     for ep in range(1,1+100):
#         for i, (inputs, target, feature, mask) in enumerate(train_loader):
#             trainx = torch.Tensor(inputs).to(device)
#             # trainx_ = torch.Tensor(feature).to(device)
#             # trainx = torch.concat([trainx, trainx_], axis = -1)
#             trainy = torch.Tensor(target).to(device)
#             trainm = torch.ones(target.shape).to(device)
#             model.train()
#             optimizer.zero_grad()
#             mean, std, _ = model(trainx)
#             #print(output[0, 0, 0])
#             loss = cauchy_likelihood_loss(trainy, mean, std, trainm)
# #            loss = masked_mse(mean, trainy, trainm)
#             #print(loss)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()  
            
#         yhat = []
#         label = []
#         eval_m = []
#         model.eval()
#         for i, (inputs, target, feature, mask) in enumerate(val_loader):
#             testx = torch.Tensor(inputs).to(device)
#             testy = torch.Tensor(target).to(device)
#             output, _, _ = model(testx)
#             output = output.detach().cpu().numpy()
#             yhat.append(de_normalized(output, norm_statistic))
#             label.append(de_normalized(testy.cpu().numpy(), norm_statistic))
#         yhat = np.concatenate(yhat)
#         label = np.concatenate(label)
#         res = metric(yhat, label)
#         print('episode', ep, 'mae', res[0], 'mse', res[1], 'rmse', res[2], 'rse', res[3], 'r2', res[6])
#         MAE_list.append(res[0])
#         if res[0] == min(MAE_list):
#             best_model = copy.deepcopy(model.state_dict())
#             torch.save(model, model_params_root)
            
    model = torch.load(model_params_root)
    outputs = []
    targets = []
    masks = []
    stds = []
    model.eval()
    for i, (inputs, target, feature, mask) in enumerate(test_loader):
        testx = torch.Tensor(inputs).to(device)
        testy = torch.Tensor(target).to(device)
        testm = torch.Tensor(mask).to(device)
        output, std, _ = model(testx)
        output = output.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        outputs.append(output)
        targets.append(testy.cpu().numpy())
        stds.append(std)
    yhat = np.concatenate(outputs)
    label = np.concatenate(targets)
    stds = np.concatenate(stds)
    
    time_steps = np.arange(126)  # Time steps (e.g., days, hours)

    #Calculate upper and lower bounds for the variance
    upper_bound = yhat + 1.96 * stds
    lower_bound = yhat - 1.96 * stds
    
    '''
    laplacian distribution
    '''
    # mnll = np.mean(-np.log(2 * stds) - np.abs(yhat - label) / stds)
    # upper_bound = yhat -  stds * np.sign(0.975-0.5)* np.log10(1-2*np.abs(0.975-0.5))
    # lower_bound = yhat -  stds * np.sign(0.025-0.5)* np.log10(1-2*np.abs(0.025-0.5))
    # picp = np.mean(((label >= lower_bound) & (label <= upper_bound)))   
    # mpiw = np.mean(upper_bound - lower_bound)
    
    '''
    Cauchy distribution
    '''
    # mnll = np.mean(np.log(np.pi*stds) + np.log(1 + (yhat - label) ** 2/ stds**2))
    # upper_bound = yhat + stds * np.tan(np.pi * (0.975 - 0.5))
    # lower_bound = yhat + stds * np.tan(np.pi * (0.025 - 0.5))
    # picp = np.mean(((label >= lower_bound) & (label <= upper_bound)))   
    # mpiw = np.mean(upper_bound - lower_bound)
    
    # # Plotting
    # plt.figure(figsize=(6, 4))
    # plt.plot(time_steps, true_values, label='True Delays', color='blue')
    # plt.plot(time_steps, predicted_means, label='Predicted Means', color='green')
    # plt.fill_between(time_steps, lower_bound, upper_bound, color='green', alpha=0.2, label='2std area')
    # plt.title('LDGN')
    # plt.xlabel('Time Step')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.savefig('LDGN_flow.pdf')
