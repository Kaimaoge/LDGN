# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:22:53 2023

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
from utils.tools import masked_mse
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
    layers = 6
    c_in = 2
    batch_size = 32
    lrate = 0.001
    dropout = 0.3
    wdecay = 0.0001
    epochs = 100
    clip = 5
    model_params_root = "pth/GWAVE_pre_largerf63_6.pth"
    
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

    optimizer = optim.Adam(model.parameters(), lr=lrate, weight_decay=wdecay)

    MAE_list = []
    MSE = nn.MSELoss()
    print("start training...",flush=True)
    
    for ep in range(1,1+40):
        for i, (inputs, target, feature, mask) in enumerate(train_loader):
            trainx = torch.Tensor(inputs).to(device)
            # trainx_ = torch.Tensor(feature).to(device)
            # trainx = torch.concat([trainx, trainx_], axis = -1)
            trainy = torch.Tensor(target).to(device)
            mask = torch.rand((trainy.shape)).to(device)
            mask[mask <= 0.3] = 0  # masked
            mask[mask > 0.3] = 1  # remained
            inp = trainy.masked_fill(mask == 0, 0)
            model.train()
            optimizer.zero_grad()
            _, _, rec = model(inp)
            #print(output[0, 0, 0])
            loss = MSE(rec, trainy)
            #print(loss)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()  
        print(loss.item())
            
        torch.save(model,  model_params_root)