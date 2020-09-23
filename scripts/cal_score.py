import cv2
import os
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle

datasets = ['NLPR', 'NJUD']
#datasets = ['NLPR', 'NJUD', 'LFSD', 'RGBD135', 'SSD100', 'STEREO797', 'DUT', 'SIP']


def cal_score(img, gt, beta=0.3):
    img = np.float32(img)
    gt = np.float32(gt)
    gt *= 1/255.0
    img *= 1/255.0
    gt[gt >= 0.5] =1.
    gt[gt < 0.5] = 0.
    img[img >= 0.5] = 1.
    img[img < 0.5] = 0.

    over = (img*gt).sum()
    union = ((img+gt)>=1).sum()
    sum_gt = gt.sum()

    iou = over / (1e-7 + union);
    cover = over / (1e-7 + sum_gt);

    f_beta = (1.+beta) * iou * cover / (1e-7 + iou + beta*cover)

    return iou, cover, f_beta


S = [0] * len(datasets)
for dataset in datasets:
    path = osp.join('train', dataset)
    #path = osp.join('test', dataset)
    imgs = [line.rstrip() for line in open(osp.join(path, "test.txt"))]
    scores = {}
    sum_s = 0
    for f in tqdm(imgs):
        depth = cv2.imread(osp.join(path, "ostu_depth", f+".jpg"), 0)
        if depth is None:
            print("depth is None, check:", dataset)
        assert depth is not None
        try:
            gt    = cv2.imread(osp.join(path, "mask", f+".png"), 0) # gt->mask
        except:
            gt    = cv2.imread(osp.join(path, "mask", f+".jpg"), 0)
        if gt is None:
            print("gt:{} is None!".format(os.path.join(path, "mask", f)))
        assert gt is not None

        iou, cover, f_beta = cal_score(depth, gt, 0.3)
        #print("name:", f, " iou:", iou, " cover:", cover, " f_beta:", f_beta)
        scores[f] = {"iou":iou, "cover":cover, "f_beta":f_beta}
        sum_s += f_beta
        """
        plt.subplot(121)
        plt.imshow(depth, cmap="gray")
        plt.subplot(122)
        plt.imshow(gt, cmap="gray")
        plt.show()
        """
    sum_s /= len(imgs)
    idx = datasets.index(dataset)
    S[idx] = sum_s
    print("Dataset:", dataset, "Mean f:", S[idx])
    with open(osp.join("train", dataset+"_score.pkl"), "wb") as fout:
        pickle.dump(scores, fout)




