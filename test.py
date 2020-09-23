#!/usr/bin/python3
#coding=utf-8

import os
import sys
#sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from lib import dataset
from network  import Segment
import time
import logging as logger
import argparse

#TAG = "rgbd_v0"
#SAVE_PATH = TAG



DATASETS = ['./data/RGBD_sal/test/DUT',
           './data/RGBD_sal/test/RGBD135',
           './data/RGBD_sal/test/SSD100',
           './data/RGBD_sal/test/LFSD',
           './data/RGBD_sal/test/SIP',
           './data/RGBD_sal/test/NJUD', './data/RGBD_sal/test/NLPR', './data/RGBD_sal/test/STEREO797']

class Test(object):
    def __init__(self, conf, Dataset, datapath, Network):
        ## dataset
        #self.cfg    = Dataset.Config(datapath='../data/SOD', snapshot='./out/model-30', mode='test')

        self.datapath = datapath.split("/")[-1]
        print("Testing on %s"%self.datapath)
        self.cfg = Dataset.Config(datapath = datapath, snapshot=conf.model, mode='test')
        self.tag = conf.tag
        self.data   = Dataset.RGBDData(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=True, num_workers=0)
        ## network
        self.net    = Network(cfg=self.cfg, norm_layer=nn.BatchNorm2d)
        self.net.train(False)
        self.net.cuda()
        self.net.eval()

    def accuracy(self):
        with torch.no_grad():
            mae, fscore, cnt, number   = 0, 0, 0, 256
            mean_pr, mean_re, threshod = 0, 0, np.linspace(0, 1, number, endpoint=False)
            cost_time = 0
            for image, d, mask, (H,W), name in self.loader:
                image, d, mask            = image.cuda().float(), d.cuda().float(), mask.cuda().float()
                start_time = time.time()
                out, gate  = self.net(image, d)
                pred                   = torch.sigmoid(out)
                torch.cuda.synchronize()
                end_time = time.time()
                cost_time += end_time - start_time

                ## MAE
                #pred     = F.interpolate(pred, size=(H,W), mode='bilinear')
                #mask     = F.interpolate(mask, size=(H,W), mode='bilinear')

                cnt += 1
                mae += (pred-mask).abs().mean()
                ## F-Score
                precision = torch.zeros(number)
                recall    = torch.zeros(number)
                for i in range(number):
                    temp         = (pred >= threshod[i]).float()
                    precision[i] = (temp*mask).sum()/(temp.sum()+1e-12)
                    recall[i]    = (temp*mask).sum()/(mask.sum()+1e-12)

                mean_pr += precision
                mean_re += recall
                fscore   = mean_pr*mean_re*(1+0.3)/(0.3*mean_pr+mean_re+1e-12)
                if cnt % 20 == 0:
                    fps = image.shape[0] / (end_time - start_time)
                    print('MAE=%.6f, F-score=%.6f, fps=%.4f'%(mae/cnt, fscore.max()/cnt, fps))
            fps = len(self.loader.dataset) / cost_time
            msg = '%s MAE=%.6f, F-score=%.6f, len(imgs)=%s, fps=%.4f'%(self.datapath, mae/cnt, fscore.max()/cnt, len(self.loader.dataset), fps)
            print(msg)
            logger.info(msg)

    def show(self):
        with torch.no_grad():
            for image, d, mask, (H, W), maskpath in self.loader:
                image, d, mask  = image.cuda().float(), d.cuda().float(), mask.cuda().float()
                out, out2_1, _, _, _, out2_2, _, _, _, gate = self.net(image, d)
                pred         = torch.sigmoid(out)
                out2_1 = torch.sigmoid(out2_1)
                out2_2 = torch.sigmoid(out2_2)
                plt.subplot(231)
                plt.title("image")
                image = image[0].permute(1,2,0).cpu().numpy()*255
                plt.imshow(np.uint8(image))
                plt.subplot(232)
                plt.title("gt")
                mask  = mask[0, 0].cpu().numpy()
                plt.imshow(mask, cmap='gray')
                plt.subplot(233)
                plt.title("pred-final")
                tmp  = pred[0, 0].cpu().numpy()
                plt.imshow(tmp, cmap='gray')
                plt.subplot(234)
                plt.title("pred-out1")
                out2_1 = out2_1[0].permute(1,2,0).cpu().squeeze().numpy()*255
                plt.imshow(np.uint8(out2_1), cmap='gray')
                plt.subplot(235)
                plt.title("pred-out2")
                out2_2 = out2_2[0].permute(1,2,0).cpu().squeeze().numpy()*255
                plt.imshow(np.uint8(out2_2), cmap='gray')

                plt.show()
                input()

    def save(self):
        with torch.no_grad():
            for image, d, mask, (H, W), name in self.loader:
                image, d = image.cuda().float(), d.cuda().float()
                out,  gate = self.net(image, d)
                out     = F.interpolate(out, size=(H,W), mode='bilinear')
                pred     = (torch.sigmoid(out[0,0])*255).cpu().numpy()
                head     = './rgbd_pred/{}/'.format(self.tag) + self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head+'/'+name[0],np.uint8(pred))


if __name__=='__main__':
    conf = argparse.ArgumentParser(description="train model")
    conf.add_argument("--tag", type=str)
    conf.add_argument("--gpu", type=int, default=0)
    conf.add_argument("--model", type=str)

    args = conf.parse_args()
    logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="test_%s.log"%(args.tag), filemode="w")
    logger.info("Configuration:{}".format(args))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for e in DATASETS:
        t =Test(args, dataset, e, Segment)
#        t.accuracy() # this is not accurate due to the resize operation, please use the matlab code to eval the performance
#        t.show()
        t.save()

