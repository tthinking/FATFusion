from PIL import Image
import numpy as np
import os
import torch
import cv2
import time
import imageio
import math

import torchvision.transforms as transforms

from Networks.networks import MODEL as net
from thop import profile

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda:0')


model = net(in_channel=1)

model_path = "./model.pth"
use_gpu = torch.cuda.is_available()

if use_gpu:
    print('GPU Mode Acitavted')
    model = model.cuda()
    model.cuda()

    model.load_state_dict(torch.load(model_path))
    print(model)
else:
    print('CPU Mode Acitavted')
    state_dict = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state_dict)


def fusion_gray():



    path1 = './images/SPECT.bmp'

    path2 = './images/MRI.bmp'


    img1 = Image.open(path1).convert('L')
    img2 = Image.open(path2).convert('L')

    img1_read = np.array(img1)
    img2_read = np.array(img2)
    h = img1_read.shape[0]
    w = img1_read.shape[1]
    img1_org = img1
    img2_org = img2
    tran = transforms.ToTensor()
    img1_org = tran(img1_org)
    img2_org = tran(img2_org)

    if use_gpu:
        img1_org  = img1_org.cuda()
        img2_org = img2_org.cuda()
    else:
        img1_org = img1_org
        img2_org = img2_org
    img1_org = img1_org.unsqueeze(0)
    img2_org = img2_org.unsqueeze(0)

    model.eval()
    out = model(img1_org, img2_org )

    d_map_1_4 = np.squeeze(out.detach().cpu().numpy())

    decision_1_4 = (d_map_1_4 * 255).astype(np.uint8)


    imageio.imwrite(
        './result/result.bmp',decision_1_4)





if __name__ == '__main__':

    fusion_gray()
