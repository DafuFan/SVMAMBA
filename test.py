# coding:utf-8
import os
import argparse
import torch
from torch.utils.data import DataLoader
from Dataset import Fusion_dataset
from net.net10 import VSSM
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np
import imageio
import time

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
    out = out.clamp(0, 1.0)
    return out

def RGB2YCrCb1(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)
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

def main(save_dir, fusion_model_path):
    fusionmodel = VSSM(u2net_weight_path="", freeze_u2net=True)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(fusion_model_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    fusionmodel.load_state_dict(sd, strict=False)

    fusionmodel = fusionmodel.to(device)
    fusionmodel.eval()

    print('fusionmodel load done!')
    transform = transforms.ToTensor()
    test_dataset = Fusion_dataset('val', ir_path='', vi_path='',transform=transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    tic = time.time()
    with torch.no_grad():
        start = time.time()
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)
            vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vis)
            vi_ycrcb = RGB2YCrCb1(img_vis).to(device)

            vi_Y = vi_Y.to(device)
            vi_Cb = vi_Cb.to(device)
            vi_Cr = vi_Cr.to(device)
            fused_img = fusionmodel(vi_ycrcb, img_ir)

            fused_image = fused_img.cpu().numpy()
            fused_image = fused_image.transpose((0, 2, 3, 1))
            fused_image = (fused_image - np.min(fused_image)) / (
                    np.max(fused_image) - np.min(fused_image)
            )

            fused_img = np.uint8(255.0 * fused_image)
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                gray_img = fused_img[k].squeeze()
                imageio.imwrite(save_path, gray_img)
                test_bar.set_description('Fusion {0} Sucessfully!'.format(name[k]))
        toc = time.time()
        print('Inference done, average time per batch: {:.4f}s'.format((toc - tic) / len(test_loader)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run MAMBA with pytorch')
    parser.add_argument('--gpu', '-G', type=int, default=0)
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--num_workers', '-j', type=int, default=1)
    args = parser.parse_args()

    test_models = [
        {
        },

    for m in test_models:
        os.makedirs(m["save_dir"], exist_ok=True)
        print(f"\n🧩 正在测试模型: {m['model_path']}")
        main(save_dir=m["save_dir"], fusion_model_path=m["model_path"])
