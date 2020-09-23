#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset, DataLoader
import pickle

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))
        """
        if 'LFSD' in self.kwargs['datapath']:
            self.mean = np.array([[[128.67, 117.24, 107.97]]])
            self.std = np.array([[[66.14, 58.32, 56.37]]])
        elif 'NJUD' in self.kwargs['datapath']:
            self.mean = np.array([[[104.89, 101.66, 92.15]]])
            self.std = np.array([[[55.89, 53.03, 53.95]]])
        elif 'NLPR' in self.kwargs['datapath']:
            self.mean = np.array([[[126.74, 123.91, 123.04]]])
            self.std = np.array([[[52.91, 52.31, 50.61]]])
        elif 'STEREO797' in self.kwargs['datapath']:
            self.mean = np.array([[[113.17, 110.05, 98.60]]])
            self.std = np.array([[[58.60, 55.89, 58.32]]])
        """
        #else:
            #raise ValueError

        """
        self.mean = np.array([[[0.485, 0.456, 0.406]]])*255.0
        self.std = np.array([[[0.229, 0.224, 0.225]]])*255.0

        """
        self.mean = np.array([[[128.67, 117.24, 107.97]]])
        self.std = np.array([[[66.14, 58.32, 56.37]]])
        self.d_mean = 116.09
        self.d_std = 56.61

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None



class RGBDData(Dataset):
    def __init__(self, cfg):
        # NJUD: depth:*.jpg, gt:*.png, rgb:*.jpg
        # NLPR: depth:*.jpg, gt:*.jpg, rgb:*.jpg
        self.samples = []
        self.mode = cfg.mode
        if cfg.mode == "train":
            with open(osp.join(cfg.datapath, "NLPR_score.pkl"), "rb") as fin:
                nlpr_data = pickle.load(fin)
            with open(osp.join(cfg.datapath, "NJUD_score.pkl"), "rb") as fin:
                njud_data = pickle.load(fin)
            with open(osp.join(cfg.datapath, "NLPR", cfg.mode+'.txt'), 'r') as lines:
                for line in lines:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "NLPR/rgb", line+".jpg")
                    depth_name = osp.join(cfg.datapath, "NLPR/depth", line+".jpg")
                    ostu_rgb_name = osp.join(cfg.datapath, "NLPR/ostu_rgb", line+".jpg")
                    mask_name = osp.join(cfg.datapath, "NLPR/gt", line+".jpg")
                    #self.samples.append([image_name, ostu_rgb_name, mask_name])
                    key = nlpr_data[line]['f_beta']
                    self.samples.append([key, image_name, depth_name, mask_name])
            with open(osp.join(cfg.datapath, "NJUD", cfg.mode+'.txt'), 'r') as lines:
                for line in lines:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "NJUD/rgb", line+".jpg")
                    depth_name = osp.join(cfg.datapath, "NJUD/depth", line+".jpg")
                    ostu_rgb_name = osp.join(cfg.datapath, "NJUD/ostu_rgb", line+".jpg")
                    mask_name = osp.join(cfg.datapath, "NJUD/gt", line+".png")
                    #self.samples.append([image_name, ostu_rgb_name, mask_name])
                    key = njud_data[line]['f_beta']
                    self.samples.append([key, image_name, depth_name, mask_name])
            """
            with open(osp.join(cfg.datapath, "train.txt"), "r") as fin:
                for line in fin:
                    line = line.strip()
                    image_name = osp.join(cfg.datapath, "input_train", line+".jpg")
                    depth_name = osp.join(cfg.datapath, "depth_train", line+".png")
                    mask_name = osp.join(cfg.datapath, "gt_train", line+".png")
                    self.samples.append([image_name, depth_name, mask_name])
            """
            print("train mode: len(samples):%s"%(len(self.samples)))
        else:
            #LFSD,NJUD,NLPR,STEREO797
            #image, depth: *.jpg, mask:*.png
            def read_test(name):
                samples = []
                with open(osp.join(cfg.datapath, "test.txt"), "r") as lines:
                    for line in lines:
                        line = line.strip()
                        image_name = osp.join(cfg.datapath,  "image", line+".jpg")
                        depth_name = osp.join(cfg.datapath,  "depth", line+".jpg")
                        ostu_rgb_name = osp.join(cfg.datapath,  "ostu_rgb", line+".jpg")
                        mask_name  = osp.join(cfg.datapath,  "mask", line+".png")
                        samples.append([line, image_name, depth_name, mask_name])
                return samples
            db_name = cfg.datapath.rstrip().split("/")[-1]
            self.samples = read_test(db_name)
            print("test mode name:%s, len(samples):%s"%(db_name, len(self.samples)))

        if cfg.mode == 'train':
            if cfg.train_scales is None:
                cfg.train_scales = [224, 256, 320]
            print("Train_scales:", cfg.train_scales)
            self.transform = transform.Compose(
                                                transform.MultiResize(cfg.train_scales),
                                                transform.MultiRandomHorizontalFlip(),
                                                transform.MultiNormalize(),
                                                transform.MultiToTensor()
                                                )
        elif cfg.mode == 'test':
            self.transform = transform.Compose(
                                                transform.Resize((256, 256)),
                                                transform.Normalize(mean=cfg.mean, std=cfg.std, d_mean=cfg.d_mean, d_std=cfg.d_std),
                                                transform.ToTensor(depth_gray=True))
        else:
            raise ValueError

    def __getitem__(self, idx):
        key, image_name, depth_name, mask_name = self.samples[idx]
        image               = cv2.imread(image_name).astype(np.float32)[:,:,::-1]
        depth               = cv2.imread(depth_name).astype(np.float32)[:,:, ::-1]
        mask                = cv2.imread(mask_name).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        image, depth, mask         = self.transform(image, depth, mask)
        if self.mode == "train":
            gate_gt = torch.zeros(1)
            gate_gt[0] = key
            return image, depth, mask, gate_gt
        else:
            mask_name = mask_name.split("/")[-1]
            return image, depth, mask, (H,W), mask_name

    def __len__(self):
        return len(self.samples)


""" for train loader """
def train_collate_fn(batch):
    images, depths, masks, gate_gt = zip(*batch)
    l = len(images[0])
    images_t, depths_t, masks_t = {}, {}, {}
    gates_t = {}
    gate_gt = torch.stack(gate_gt)
    for i in range(l):
        images_t[i] = []
        depths_t[i] = []
        masks_t[i] = []
        gates_t[i] = gate_gt

    for i in range(len(images)):
        for j in range(l):
            images_t[j].append(images[i][j])
            depths_t[j].append(depths[i][j])
            masks_t[j].append(masks[i][j])

    for i in range(l):
        images_t[i] = torch.stack(images_t[i])
        depths_t[i] = torch.stack(depths_t[i])
        masks_t[i] = torch.stack(masks_t[i])


    return images_t, depths_t, masks_t, gates_t






if __name__=='__main__':
    import time
    def plot_3x3(images, depths, masks):
        l = len(images)
        for j in range(min(l, 3)):
            image, depth, mask = images[j], depths[j], masks[j]

            image = image.permute(1,2,0).numpy()*255.0
            depth = depth.permute(1,2,0).numpy()*255.0
            mask  = mask.permute(1,2,0).numpy()*255.0
            H, W, _ = image.shape

            mask = mask.squeeze()
            depth = depth.squeeze()
            plt.subplot(331 + 3*j)
            plt.imshow(np.uint8(image))
            plt.subplot(332 + 3*j)
            plt.imshow(np.uint8(depth), cmap='gray')
            plt.subplot(333 + 3*j)
            plt.imshow(np.uint8(mask), cmap='gray')
    from data_prefetcher import *
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='../data/RGBD_sal/train')
    data = RGBDData(cfg)
    loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0, drop_last=True, collate_fn=train_collate_fn)
    #images, depths, masks, gate = next(iter(loader))
    #import pdb; pdb.set_trace()

    for i in range(3):
        prefetcher = DataPrefetcher(loader)
        rgb, depth, mask, gt = prefetcher.next()
        while rgb is not None:
            print("shape:", rgb.shape, " is_cuda:", rgb.is_cuda)
            rgb, depth, mask  = rgb.cpu(), depth.cpu(), mask.cpu()
            plot_3x3([rgb[0], rgb[1], rgb[2]], [depth[0], depth[1], depth[2]], [mask[0], mask[1], mask[2]])
            input()
            #time.sleep(0.1)
            rgb, depth, mask, gt = prefetcher.next()




    #for i, (images, depths, masks, gate) in enumerate(loader):
    #    k = 0
    #    plot_3x3([images[0][k], images[1][k], images[2][k]], [depths[0][k], depths[1][k], depths[2][k]], [masks[0][k], masks[1][k], masks[2][k]])
    #    input()



    #for i in range(100):
    #    images, depths, masks, gt = data[i]
    #    plot_3x3(images, depths, masks)

    #    input()


