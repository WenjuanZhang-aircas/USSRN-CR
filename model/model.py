from numpy.lib.function_base import delete
import model.common as common
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import spectral.io.envi as envi
import cv2
from skimage.feature import canny
import math
import os
import pandas as pd
from tqdm import tqdm


def make_model(args, parent=False):
    return RR(args)


class RR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RR, self).__init__()
        # define basic setting
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3

        my_head1_ms = [common.GateConv(3, 72, kernel_size, padding=1)]
        my_head1_sar = [conv(3, 72, 3)]  
        my_head2_M3_ms=[common.GateConv(72, 24, 3, stride=1, padding=1)]
        my_head2_M5_ms=[common.GateConv(72, 24, 5, stride=1, padding=2)]
        my_head2_M7_ms=[common.GateConv(72, 24, 7, stride=1, padding=3)]
        my_head3_M3_sar=[nn.Conv2d(72, 24, 3, stride=1, padding=1)]
        my_head3_M5_sar=[nn.Conv2d(72, 24, 5, stride=1, padding=2)]
        my_head3_M7_sar=[nn.Conv2d(72, 24, 7, stride=1, padding=3)]
        my_head1_canny = [conv(2, 36, 3)]
        my_head4 = [conv(180, n_feats, 3)]
        my_head5 = [nn.ReLU(inplace=True)]

        n_resblocks = 8
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, nn.ReLU(), res_scale=args.res_scale 
            ) for _ in range(n_resblocks // 2)
        ]
        for i in range(1):
            m_body.append(
                SACA(in_channels=n_feats,res_scale=args.res_scale))  

        for i in range(n_resblocks // 2):
            m_body.append(
                common.ResBlock(conv, n_feats, kernel_size, nn.ReLU(), res_scale=args.res_scale))  
        m_tail = [
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.my_head1_ms = nn.Sequential(*my_head1_ms)
        self.my_head1_sar = nn.Sequential(*my_head1_sar)
        self.my_head2_M3_ms = nn.Sequential(*my_head2_M3_ms)
        self.my_head2_M5_ms = nn.Sequential(*my_head2_M5_ms)
        self.my_head2_M7_ms = nn.Sequential(*my_head2_M7_ms)
        self.my_head3_M3_sar = nn.Sequential(*my_head3_M3_sar)
        self.my_head3_M5_sar = nn.Sequential(*my_head3_M5_sar)
        self.my_head3_M7_sar = nn.Sequential(*my_head3_M7_sar)
        self.head4 = nn.Sequential(*my_head4)
        self.head5 = nn.Sequential(*my_head5)
        self.my_head1_canny = nn.Sequential(*my_head1_canny)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        trainer_width = x.shape[2]
        trainer_band = x.shape[1]
        batch = x.shape[0]
        x1 = x[:, 0:trainer_band-3]  
        x2 = x[:, trainer_band-3:trainer_band]  
        x_ms = x1.reshape(batch, trainer_band-3, trainer_width, trainer_width)

        Canny_1 = extract_canny(x2[:,0,:,:],batch)
        Canny_2 = extract_canny(x2[:,1,:,:],batch)
        Canny_1=torch.from_numpy(Canny_1)
        Canny_2=torch.from_numpy(Canny_2)
        canny = torch.cat((Canny_1,Canny_2), 1)
        canny=canny.cuda()
        x_sar = x2.reshape(batch, 3, trainer_width, trainer_width)
        res1_ms = self.my_head1_ms(x_ms)
        res1_sar = self.my_head1_sar(x_sar)
        res2_ms3 = self.my_head2_M3_ms(res1_ms)
        res2_ms5 = self.my_head2_M5_ms(res1_ms)
        res2_ms7 = self.my_head2_M7_ms(res1_ms)
        res2_sar3 = self.my_head3_M3_sar(res1_sar)
        res2_sar5 = self.my_head3_M5_sar(res1_sar)
        res2_sar7 = self.my_head3_M7_sar(res1_sar)
        res2_contact_ms = torch.cat((res2_ms3, res2_ms5, res2_ms7), 1)
        res2_contact_sar = torch.cat((res2_sar3, res2_sar5, res2_sar7), 1)
        res2_all_ms = res1_ms+res2_contact_ms
        res2_all_sar = res1_sar+res2_contact_sar
        canny=self.my_head1_canny(canny)
        res=torch.cat((res2_all_ms, res2_all_sar,canny), 1)
        res=self.head4(res)
        res = self.head5(res)
        res = self.body(res)
        res = self.tail(res)
        return res + x1


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

class SACA(nn.Module):
    def __init__(self, in_channels,res_scale, num=4,conv=common.default_conv):
        super(SACA, self).__init__()
        n_feats = in_channels
        kernel_size = 3
        self.RBS1 = common.ResBlock(conv, n_feats, kernel_size, nn.ReLU(), res_scale=res_scale)
        self.attention = common.EfficientAttention(n_feats, n_feats, head_count=num, out_channels=n_feats)
        self.attention2 = common.CAM_Module(n_feats)
        self.RBS2 = common.ResBlock(conv, n_feats, kernel_size, nn.ReLU(), res_scale=res_scale)
    def forward(self, x):

        rb1=self.RBS1(x)
        a1 = self.attention(rb1)
        a2 = self.attention2(rb1)
        rb2 = self.RBS2(a1+a2)
        return rb2

"""
Fundamental functions
"""
def extract_canny(images,batch):
    images=images.cpu()
    img_edge2out = np.zeros_like(images)
    img_edge2out = img_edge2out[:, np.newaxis, :, :]
    for i in range(batch):
        LC = images[i,:,:].cpu().detach().numpy()
        out = np.zeros_like(LC)
        LC = LC.astype(np.float)
        out=cv2.normalize(LC, None, 0, 255, cv2.NORM_MINMAX)
        out = out.astype(np.uint8)
        img_edge2 = canny(out, sigma=2).astype(np.float)
        img_edge2 = img_edge2.astype(np.float32)
        img_edge2out[i,0,:,:] = cv2.normalize(img_edge2, None, 0, 1, cv2.NORM_MINMAX)
    return img_edge2out
