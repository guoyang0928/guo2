import argparse
from PIL import Image
import os
import sys
import cv2
import json
import time
import shutil
import logging
import numpy as np

import matplotlib.cm as cm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from dataset_thyroid import Thyroid
# from model import get_model

from model import model_new,model3,model5,model_new

from image_loader import TripletImageLoader, MetaLoader
from torch import nn
import torch

from torch.nn import functional as F
if __name__ == '__main__':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testset = Thyroid(root='./data_new', name1='draw.txt', is_train=False, data_len=1)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=True, num_workers=0, drop_last=False)
    backbone = model3.Res()
    net = model3.DoubleNet(backbone)
    backbone.cuda()
    net.cuda()
    backbone.eval()
    checkpoint = torch.load('./runs/no_sub/model_best_acc.pth')
    net.load_state_dict(checkpoint['state_dict'])
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('./data_new/', "thyroid", 1, "draw.txt",
                           transform=transforms.Compose([
                               #transforms.Resize(224,interpolation=Image.BICUBIC),
                               transforms.Resize((224,224),interpolation=Image.BICUBIC),
                               #transforms.CenterCrop(224),
                               # transforms.RandomHorizontalFlip(),
                               # transforms.RandomHorizontalFlip(p=0.5),
                               # transforms.RandomVerticalFlip(p=0.5),
                               #transforms.ColorJitter(brightness=(1,1.5),contrast=(1,1.5)),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=1, shuffle=True)

    for batch_idx, (data1, label1, data2, label2) in enumerate(train_loader):
        with torch.no_grad():
            # img, label = data[0], data[1]
            data1, data2 = data1.cuda(), data2.cuda()
            label1, label2 = label1.cuda(), label2.cuda()
            #img, label = data[0].cuda(), data[1].cuda()
            net(data1,data2)
            #backbone(img)
            break
