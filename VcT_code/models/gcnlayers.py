import math
import numpy as np
import torch
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.adj = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):  # input: torch.Size([B, N, C]), adj: torch.Size([B, N, N])
        support = torch.matmul(input, self.weight)  
        output = torch.matmul(adj, support)  # torch.Size([B, N, C'])

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
def Adj_Normalize(A):
    '''
    :param S: (B, N, N), a similar matrix
    :param knng: K-nearnest-neighbor relationship graph
    Aij = Sij when Vj in KNN(Vi), else 0
    :return: the row-normalize adj (D^-1 * A)
    '''
    D = torch.pow(A.sum(2), -1)
    D = torch.where(torch.isnan(D), torch.zeros_like(D), D).diag_embed()
    out = torch.bmm(D, A)
    return out

def gen_adj(A):
    # A: [B, N, N]
    D = torch.pow(A.sum(2).float(), -0.5)  # [B, N]
    D = torch.diag_embed(D)  # [B, N, N]
    adj = torch.bmm(torch.bmm(A, D).transpose(1,2), D)
    return adj
    
def normalize_adj(A):
    # A: [B, N, N]
    #A = A+torch.eye(A.size(0))
    D = torch.pow(A.sum(2).float(), -0.5)  # [B, N]
    D = torch.diag_embed(D)  # [B, N, N]
    adj = torch.bmm(D, torch.bmm(A, D))
    return adj
