import cv2
import matplotlib.pyplot as plt

from transform import *


#func1 = Resize(256, 256)
#func = RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(3./5, 5./3))
func1 = Resize(256)
func  = RandomCrop(224, 224)

transform = Compose(
                                MultiResize([256, 320, 352]),
                                MultiNormalize(),
                                RandomHorizontalFlip(),
                                MultiToTensor())

for j in range(30):
    keys = []
    with open("../data/RGBD_sal/train/NJUD/train.txt", "r") as fin:
        for line in fin:
            line = line.rstrip()
            keys.append(line)

    for i in range(1, 10+1):
        img = cv2.imread("../data/RGBD_sal/train/NJUD/rgb/%s.jpg"%keys[i]).astype(np.float32)

        mask = cv2.imread("../data/RGBD_sal/train/NJUD/gt/%s.png"%keys[i]).astype(np.float32)
        depth = cv2.imread("../data/RGBD_sal/train/NJUD/depth/%s.jpg"%keys[i]).astype(np.float32)

        img, depth, mask = func1(img, depth, mask)
        out, depth_1, mask_1 = func(img, depth, mask)


        imgs, depths, masks = transform(img, depth, mask)

        import pdb; pdb.set_trace()


        plt.subplot(231)
        plt.title("image")
        plt.imshow(img[:, :, ::-1])
        plt.subplot(232)
        plt.title("depth")
        plt.imshow(depth, cmap='gray')
        plt.subplot(233)
        plt.title("gt")
        plt.imshow(mask, cmap='gray')

        plt.subplot(234)
        plt.imshow(out[:, :, ::-1])
        plt.subplot(235)
        plt.imshow(depth_1, cmap='gray')
        plt.subplot(236)
        plt.imshow(mask_1, cmap='gray')

        plt.show()
