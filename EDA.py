import cv2
from glob import glob
import os
from PIL import Image
import numpy as np
import torchvision
import collections
from modules.datasets import SegDataset
from modules.utils import load_yaml
from modules.scalers import get_image_scaler
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

prj_dir = os.path.dirname(os.path.abspath(__file__))
train_dirs = os.path.join(prj_dir, 'data', 'train')
train_img_paths = glob(os.path.join(train_dirs, 'y', '*.png'))


config_path = os.path.join(prj_dir, 'config', 'train.yaml')
config = load_yaml(config_path)

class_info = {i: 0 for i in range(4)}

for i, img_path in enumerate(train_img_paths):

    img = Image.open(img_path)
    img = np.array(img)

    for j in range(4):
        if j in img:
            class_info[j] += 1
            continue
    if i % 100 == 0:
        print(i, class_info)



# class_info = {0: 12000, 1: 1609, 2: 821, 3: 10605}
x = class_info.keys()
y = class_info.values()

# {0: 12000, 1: 1609, 2: 821, 3: 10605}

plt.bar(x, y)
plt.xlabel('class')
plt.ylabel('count')

# for i, v in enumerate(x):
#     plt.text(v, y[i], y[i],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
#              fontsize = 9,
#              color='blue',
#              horizontalalignment='center',  # horizontalalignment (left, center, right)
#              verticalalignment='bottom')

plt.show()

print(class_info)



