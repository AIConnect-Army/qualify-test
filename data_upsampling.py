import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os
from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from modules.augmentation import DataAugmentation
import torch 

if __name__ == '__main__':
    dataaug = DataAugmentation(img_size=754,
            with_random_hflip=False,  # Horizontally flip
            with_random_vflip=False,  # Vertically flip
            with_random_rot=True,  # rotation
            with_random_crop=False,  # transforms.RandomResizedCrop
            with_scale_random_crop=False,  # rescale & crop
            with_random_blur=False,  # GaussianBlur
            random_color_tf=True)  # colorjitter


    df = pd.read_csv("./augment_paths.csv")
    for path in df['paths']:
        x = os.path.join('data', 'x', path)
        y = os.path.join('data', 'y', path)


        image = Image.open(x)
        label = Image.open(y)

        image = np.asarray(np.expand_dims(image, axis=0))
        label = np.asarray(np.expand_dims(label, axis=0)).transpose(1, 2, 0)
        label = np.asarray(np.expand_dims(label, axis=0))

        # Augment an image
        trans_imgs, trans_labels = dataaug.transform(image, label)
        trans_img = trans_imgs[0].permute(1,2,0)
        trans_label = trans_labels[0].permute(1,2,0)

    # # visualize
    # f, ax = plt.subplots(2, 2)

    # ax[0, 0].axis('off')
    # ax[0, 1].axis('off')
    # ax[1, 0].axis('off')
    # ax[1, 1].axis('off')

    # ax[0,0].set_title('original image')
    # ax[0,1].set_title('original label')
    # ax[1,0].set_title('transformed image')
    # ax[1,1].set_title('transformed label')

    # ax[0, 0].imshow(image[0])
    # ax[0, 1].imshow(label[0])
    # ax[1, 0].imshow(trans_img)
    # ax[1, 1].imshow(trans_label)

    # plt.show()