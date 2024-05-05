# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:55:53 2023

@author: AA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nvlc,vw->nwlc',(x,A))
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = nn.Linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=3)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
    
class NG_Model(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, layers, supports, num_nodes, c_in, device):
        super(NG_Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.layers = layers
        self.linears = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.supports_len = len(supports)
        self.supports = supports
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
        for i in range(layers):
            self.linears.append(nn.Linear(self.seq_len, self.pred_len))
        for i in range(layers - 1):
            self.gcns.append(gcn(c_in, c_in, 0, self.supports_len))
        self.adaptive = nn.Parameter(torch.randn(layers).to(device), requires_grad=True).to(device)    
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        seq_last = x[:, :,-1:,:].detach()
        x = x - seq_last
        x_ = []
        x_.append(self.linears[0](x.permute(0, 1, 3, 2)).permute(0,1, 3, 2))
        for i in range(self.layers - 1):
            x += self.gcns[i](x, new_supports)
            x_.append(self.linears[i+1](x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        res = torch.stack(x_, dim=-1)
        period_weight = F.softmax(self.adaptive)
        res = torch.sum(res * period_weight, -1)
        res = res + seq_last
        return res # [Batch, Output length, Channel]
    
    
class NG_Model_Gaussian(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, layers, supports, num_nodes, c_in, device):
        super(NG_Model_Gaussian, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.layers = layers
        self.linears = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.supports_len = len(supports)
        self.supports = supports
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
        for i in range(layers):
            self.linears.append(nn.Linear(self.seq_len, self.pred_len))
        for i in range(layers - 1):
            self.gcns.append(gcn(c_in, c_in, 0, self.supports_len))
        self.adaptive = nn.Parameter(torch.randn(layers).to(device), requires_grad=True).to(device)   
        
        self.std_linear = nn.Linear(self.seq_len, self.pred_len)
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        seq_last = x[:, :,-1:,:].detach()
        x = x - seq_last
        x_ = []
        x_.append(self.linears[0](x.permute(0, 1, 3, 2)).permute(0,1, 3, 2))
        for i in range(self.layers - 1):
            x += self.gcns[i](x, new_supports)
            x_.append(self.linears[i+1](x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        std = self.std_linear(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        std = torch.log(1 + torch.exp(std)) + 1e-6
        res = torch.stack(x_, dim=-1)
        period_weight = F.softmax(self.adaptive)
        res = torch.sum(res * period_weight, -1)
        res = res + seq_last
        return res, std # [Batch, Output length, Channel]
    
    
class NG_Model_Gaussian_Re(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, layers, supports, num_nodes, c_in, device):
        super(NG_Model_Gaussian_Re, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.layers = layers
        self.linears = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.supports_len = len(supports)
        self.supports = supports
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
        self.supports_len +=1
        for i in range(layers):
            self.linears.append(nn.Linear(self.seq_len, self.pred_len))
        for i in range(layers - 1):
            self.gcns.append(gcn(c_in, c_in, 0, self.supports_len))
        self.adaptive = nn.Parameter(torch.randn(layers).to(device), requires_grad=True).to(device)   
        
        self.std_linear = nn.Linear(self.seq_len, self.pred_len)
        self.re_head = nn.Linear(self.seq_len, self.seq_len)
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports + [adp]
        seq_last = x[:, :,-1:,:].detach()
        x = x - seq_last
        x_ = []
        x_.append(self.linears[0](x.permute(0, 1, 3, 2)).permute(0,1, 3, 2))
        for i in range(self.layers - 1):
            x += self.gcns[i](x, new_supports)
            x_.append(self.linears[i+1](x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        std = self.std_linear(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        std = torch.log(1 + torch.exp(std)) + 1e-6
        rec = self.re_head(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        res = torch.stack(x_, dim=-1)
        period_weight = F.softmax(self.adaptive)
        res = torch.sum(res * period_weight, -1)
        res = res + seq_last
        rec = rec + seq_last
        return res, std, rec # [Batch, Output length, Channel]
    
    
class NG_Model_Gaussian_Re_Woadp(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, layers, supports, num_nodes, c_in, device):
        super(NG_Model_Gaussian_Re_Woadp, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.layers = layers
        self.linears = nn.ModuleList()
        self.gcns = nn.ModuleList()
        self.supports_len = len(supports)
        self.supports = supports
        for i in range(layers):
            self.linears.append(nn.Linear(self.seq_len, self.pred_len))
        for i in range(layers - 1):
            self.gcns.append(gcn(c_in, c_in, 0, self.supports_len))
        self.adaptive = nn.Parameter(torch.randn(layers).to(device), requires_grad=True).to(device)   
        
        self.std_linear = nn.Linear(self.seq_len, self.pred_len)
        self.re_head = nn.Linear(self.seq_len, self.seq_len)
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        #adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        new_supports = self.supports #+ [adp]
        seq_last = x[:, :,-1:,:].detach()
        x = x - seq_last
        x_ = []
        x_.append(self.linears[0](x.permute(0, 1, 3, 2)).permute(0,1, 3, 2))
        for i in range(self.layers - 1):
            x += self.gcns[i](x, new_supports)
            x_.append(self.linears[i+1](x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2))
        # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        std = self.std_linear(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        std = torch.log(1 + torch.exp(std)) + 1e-6
        rec = self.re_head(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        res = torch.stack(x_, dim=-1)
        period_weight = F.softmax(self.adaptive)
        res = torch.sum(res * period_weight, -1)
        res = res + seq_last
        rec = rec + seq_last
        return res, std, rec # [Batch, Output length, Channel]