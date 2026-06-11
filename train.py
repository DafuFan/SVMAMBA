#!/usr/bin/python
# -*- encoding: utf-8 -*-
from PIL import Image
import numpy as np
from torch.autograd import Variable
from net.net10 import VSSM
from Dataset import Fusion_dataset
import argparse
import datetime
import time
import logging
import os
import pickle
from utils import setup_logger
from losses import Fusionloss
from net.u2net import U2NETP
import torch
import warnings
from datetime import datetime as dt

from torchvision import transforms
from torch.utils.data import DataLoader
import imageio
import torch.nn as nn
import torchvision
import kornia

warnings.filterwarnings('ignore')

def parse_args():
    parse = argparse.ArgumentParser()
    return parse.parse_args()


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out
class Sobel(nn.Module):
    def __init__(self, channels):
        super(Sobel, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

        sobel_filter = np.expand_dims(sobel_filter, axis=0)  
        sobel_filter = np.expand_dims(sobel_filter, axis=0)  
        sobel_filter = np.repeat(sobel_filter, channels, axis=0)  

        sobel_filter_T = np.transpose(sobel_filter, (0, 1, 3, 2))  

        self.convx = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.convy = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter).float())
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter_T).float())

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x




def train_fusion(i, logger=None, save_every=10, resume_from=''):
    lr_start = 0.001
    modelpth = ''
    Method = ''
    modelpth = os.path.join(modelpth, Method)
    if not os.path.exists(modelpth):
        os.makedirs(modelpth)

    fusionmodel = VSSM().cuda()

    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    model_tag = 'MAMBAFusion'

    transform = transforms.ToTensor()
    crop_size = 256
    train_dataset = Fusion_dataset(
        split='train',
        ir_path='',
        vi_path='',
        transform=transform,
        crop_size=crop_size
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    train_loader.n_iter = len(train_loader)

    criteria_fusion = Fusionloss().cuda()
    epoch = 100
    st = glob_st = time.time()
    logger.info('Training Fusion Model start~')

    for epo in range(epoch):
        epoch_loss_sum = 0.0
        epoch_steps = 0

        if epo < 50:
            lr_current = 1e-3
        elif epo < 80:
            lr_current = 1e-4
        else:
            lr_current = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_current

        for it, (image_vis, image_ir, name) in enumerate(train_loader):
            fusionmodel.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            logits = fusionmodel(image_vis_ycrcb, image_ir)
            fusion_ycrcb = torch.cat(
                (logits, image_vis_ycrcb[:, 1:2, :, :], image_vis_ycrcb[:, 2:, :, :]),
                dim=1,
            )
            fusion_image = YCrCb2RGB(fusion_ycrcb)
            fusion_image = torch.clamp(fusion_image, 0, 1)

            loss_total, logs = criteria_fusion(fusion_ycrcb, image_vis, image_ir)
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            epoch_loss_sum += float(logs['total'])
            epoch_steps += 1


            if (it + 1) % 10 == 0:
                ed = time.time()
                t_intv = ed - st
                st = ed
                logger.info(
                    f"epoch:{epo}/{epoch}, step:{it + 1}/{len(train_loader)}, "
                    f"L_pix:{logs['L_pix']:.4f}, L_grad:{logs['L_grad']:.4f}, "
                    f"S_struct:{logs['S_struct']:.4f}, total:{logs['total']:.4f}, "
                    f"time:{t_intv:.3f}s, lr:{lr_current:.6f}"
                )

        if save_every > 0 and ((epo + 1) % save_every == 0):
            train_avg = epoch_loss_sum / max(1, epoch_steps)

            fname = f"{model_tag}_epoch_{epo + 1}_train_{train_avg:.4f}.pth"
            save_path = os.path.join(modelpth, fname)

            to_save = {
                'epoch': epo + 1,
                'model_state_dict': fusionmodel.state_dict(),
                'train_loss': float(train_avg),
            }
            torch.save(to_save, save_path)


            logger.info(f"Saved: {save_path} (avg total={train_avg:.4f})")

    final_path = os.path.join(modelpth, f'fusion_model_final_{i+1}.pth')
    torch.save(fusionmodel.state_dict(), final_path)
    logger.info(f"Final model saved to: {final_path}")

def main():
    parser = argparse.ArgumentParser(description='Train with pytorch')
    parser.add_argument('--model_name', '-M', type=str, default='Fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=4)
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--num_workers', '-j', type=int, default=2)
    parser.add_argument('--save_every', type=int, default=10, help='Save a full checkpoint every N epochs.')
    parser.add_argument('--resume_from', type=str, default='', help='Resume from a checkpoint path (optional).')

    args = parser.parse_args()

    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)

    train_fusion(1, logger)

    print("training Done!")

if __name__ == "__main__":
    main()
