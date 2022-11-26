import pandas as pd
import numpy as np
import os
import glob
import cv2

from tqdm import tqdm

import warnings

warnings.filterwarnings(action='ignore')

x_img_paths = sorted(glob.glob('D:/x/*.png'))
y_img_paths = sorted(glob.glob('D:/y/*.png'))

df_simul = pd.DataFrame({'x': x_img_paths, 'y': y_img_paths})

from os.path import join as opj

removeFile = True

Train_dir_1 = 'D:/baselinegu/data/add_cut/train/x'
os.makedirs(Train_dir_1, exist_ok=True)
Train_dir_2 = 'D:/baselinegu/data/add_cut/train/y'
os.makedirs(Train_dir_2, exist_ok=True)

for idx in tqdm(range(len(df_simul))):
    img = cv2.imread(df_simul.iloc[idx, 0])
    imgname = df_simul.iloc[idx, 0].split('/')[-1]

    im1 = img[:, :754]
    im2 = img[:, 754:]

    im1 = cv2.resize(im1, (320, 320), interpolation=cv2.INTER_NEAREST)
    im2 = cv2.resize(im2, (320, 320), interpolation=cv2.INTER_NEAREST)

    im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
    im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.hconcat([im1, im2])

    im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
    im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
    img3 = cv2.hconcat([im1, im2])

    im1 = cv2.rotate(im1, cv2.ROTATE_90_CLOCKWISE)
    im2 = cv2.rotate(im2, cv2.ROTATE_90_CLOCKWISE)
    img4 = cv2.hconcat([im1, im2])

    label = cv2.imread(df_simul.iloc[idx, 1], cv2.IMREAD_GRAYSCALE)
    labelname = df_simul.iloc[idx, 1].split('/')[-1]

    lab1 = label[:, :754]
    lab2 = label[:, 754:]

    lab1 = cv2.resize(lab1, (320, 320), interpolation=cv2.INTER_NEAREST)
    lab2 = cv2.resize(lab2, (320, 320), interpolation=cv2.INTER_NEAREST)

    lab1 = cv2.rotate(lab1, cv2.ROTATE_90_CLOCKWISE)
    lab2 = cv2.rotate(lab2, cv2.ROTATE_90_CLOCKWISE)
    label2 = cv2.hconcat([lab1, lab2])

    lab1 = cv2.rotate(lab1, cv2.ROTATE_90_CLOCKWISE)
    lab2 = cv2.rotate(lab2, cv2.ROTATE_90_CLOCKWISE)
    label3 = cv2.hconcat([lab1, lab2])

    lab1 = cv2.rotate(lab1, cv2.ROTATE_90_CLOCKWISE)
    lab2 = cv2.rotate(lab2, cv2.ROTATE_90_CLOCKWISE)
    label4 = cv2.hconcat([lab1, lab2])

    save_path_1 = opj(Train_dir_1, f'r0_{imgname}')
    save_path_2 = opj(Train_dir_2, f'r0_{labelname}')
    save_path_3 = opj(Train_dir_1, f'r1_{imgname}')
    save_path_4 = opj(Train_dir_2, f'r1_{labelname}')
    save_path_5 = opj(Train_dir_1, f'r2_{imgname}')
    save_path_6 = opj(Train_dir_2, f'r2_{labelname}')
    save_path_7 = opj(Train_dir_1, f'r3_{imgname}')
    save_path_8 = opj(Train_dir_2, f'r3_{labelname}')

    cv2.imwrite(save_path_1, img)
    cv2.imwrite(save_path_2, label)
    cv2.imwrite(save_path_3, img2)
    cv2.imwrite(save_path_4, label2)
    cv2.imwrite(save_path_5, img3)
    cv2.imwrite(save_path_6, label3)
    cv2.imwrite(save_path_7, img4)
    cv2.imwrite(save_path_8, label4)

    if removeFile:
        os.remove(df_simul.iloc[idx, 0])
        os.remove(df_simul.iloc[idx, 1])