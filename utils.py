import os.path as osp
import time
import sys
import os
import logging
import torch.distributed as dist
import torch
import numpy as np
from PIL import Image


def setup_logger(logpth):
    logfile = 'Net-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                model,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                it,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = it
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        param_list = [
                {'params': wd_params},
                {'params': nowd_params, 'weight_decay': 0},
                {'params': lr_mul_wd_params, 'lr_mul': True},
                {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr_mul': True}]
        self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_mul', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_mul', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()


def RGB2YCrCb(rgb_image):  
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr): 
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out


def tensor2img(img, is_norm=True):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  if is_norm:
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

def save_img_single(img, name, is_norm=True):
  img = tensor2img(img, is_norm=is_norm)  
  img = Image.fromarray(img) 
  img.save(name)  


