import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .Transformer_EncDec import Encoder, EncoderLayer
# from .SelfAttention_Family import FullAttention, AttentionLayer

import lib.utils as utils
from lib.evaluation import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

class nconv(nn.Module):
	def __init__(self):
		super(nconv,self).__init__()

	def forward(self, x, A):
		# x (B, F, N, M)
		# A (B, M, N, N)
		x = torch.einsum('bfnm,bmnv->bfvm',(x,A)) # used
		# print(x.shape)
		return x.contiguous() # (B, F, N, M)

class linear(nn.Module):
	def __init__(self, c_in, c_out):
		super(linear,self).__init__()
		# self.mlp = nn.Linear(c_in, c_out)
		self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1,1), padding=(0,0), stride=(1,1), bias=True)

	def forward(self, x):
		# x (B, F, N, M)

		# return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
		return self.mlp(x)
		
class gcn(nn.Module):
	def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
		super(gcn,self).__init__()
		self.nconv = nconv()
		c_in = (order*support_len+1)*c_in
		# c_in = (order*support_len)*c_in
		self.mlp = linear(c_in, c_out)
		self.dropout = dropout
		self.order = order

	def forward(self, x, support):
		# x (B, F, N, M)
		# a (B, M, N, N)
		out = [x]
		for a in support:
			x1 = self.nconv(x,a)
			out.append(x1)
			for k in range(2, self.order + 1):
				x2 = self.nconv(x1,a)
				out.append(x2)
				x1 = x2

		h = torch.cat(out, dim=1) # concat x and x_conv
		h = self.mlp(h)
		return F.relu(h)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        """
        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x