import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding


    
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

def default_conv_my_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)
    
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(), res_scale=0.1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(3):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size,padding=1, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i != 2:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)
	    self.tail=nn.ReLU()

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return self.tail(x) * torch.sigmoid(g)

class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))  
        self.softmax = Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)

        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy

        attention = self.softmax(energy_new)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, dim, head_count, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.head_count = head_count
        self.dim = dim
        self.gamma = Parameter(torch.zeros(1))

        self.keys = nn.Conv2d(in_channels, dim, 1)
        self.queries = nn.Conv2d(in_channels, dim, 1)
        self.values = nn.Conv2d(in_channels, dim, 1)
        if dim != out_channels:
            self.reprojection = nn.Conv2d(dim, out_channels, 1)
        else:
            self.reprojection = None

    def forward(self, input_, mask=None, return_scores=False):
        n, _, h, w = input_.size()
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_channels = self.dim // self.head_count

        if mask is not None:
            # [b,1,h,w]
            mask = F.interpolate(mask, size=[h, w], mode='nearest')
            keys += (mask * -10000.0)
            queries += (mask * -10000.0)

        keys = keys.reshape((n, self.dim, h * w))  # [b,d,h*w]
        queries = queries.reshape(n, self.dim, h * w)
        values = values.reshape((n, self.dim, h * w))

        attended_values = []
        scores = 0
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_channels: (i + 1) * head_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_channels: (i + 1) * head_channels, :], dim=1)
            value = values[:, i * head_channels: (i + 1) * head_channels, :]
            context = key @ value.transpose(1, 2)  # [b, d, d]
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_channels, h, w)
            attended_values.append(attended_value)
            if return_scores:
                score = torch.matmul(query.transpose(1, 2), key)  # [b, hw, hw]
                score = torch.mean(score, dim=1).reshape([n, h, w])
                scores += score

        aggregated_values = torch.cat(attended_values, dim=1)
        if self.reprojection is not None:
            reprojected_value = self.reprojection(aggregated_values)
        else:
            reprojected_value = aggregated_values

        attention = self.gamma * reprojected_value + input_

        if return_scores:
            max_value, _ = torch.max(scores.reshape([n, h * w]), dim=1)
            max_value = max_value[:, None, None]
            scores = scores / (max_value + 1e-5)
            scores = scores.unsqueeze(1)
            scores = scores.detach()
            return attention, scores
        else:
            return attention


if __name__ == '__main__':
    net = MBblock(in_channels=4).cuda()
    data = torch.FloatTensor(size=(2,4,16,16)).cuda()
    print(net(data).shape)

