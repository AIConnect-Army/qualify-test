import os
import sys
import random
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from collections import deque
from PIL import Image
import csv


def dfs(cls, pt, visited, img, bbox):
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]]  # 상,하,좌,우로 살펴봄
    q = deque([pt])
    tl_x, tl_y, br_x, br_y = bbox  # 좌상단 x, 좌상단 y, 우하단 x, 우하단 y

    while q:
        pt = q.pop()

        for d in dir:
            x, y = pt[0] + d[0], pt[1] + d[1]
            if 0 <= x < img.shape[0] and 0 <= y < img.shape[1]:
                if img[x, y] == cls and visited[x][y] == 0:

                    visited[x][y] = 1

                    tl_x = min(tl_x, x)
                    tl_y = min(tl_y, y)
                    br_x = max(br_x, x)
                    br_y = max(br_y, y)
                    q.append([x, y])

                else:
                    visited[x][y] = 1
    # 홀수 값을 짝수로 바꿔줌
    if tl_x % 2 != 0:
        tl_x += 1
    if tl_y % 2 != 0:
        tl_y += 1
    if br_x % 2 != 0:
        br_x += 1
    if br_y % 2 != 0:
        br_y += 1
    return tl_x, tl_y, br_x, br_y



def find_cls_bbox(cls, img):
    """
        img에서 cls에 맞는 영역을 구함

        ex)
            args:
                cls: 1(int), img: label(np.array)

            return:
                cls_bbox: list() ex) [[tlx, tly, brx, bry, sum()], [tlx, tly, brx, bry, sum()],,]
        """
    cls_bbox = []
    visited = [[0 for _ in range(img.shape[1])] for _ in range(img.shape[0])]
    tl_x, tl_y = sys.maxsize, sys.maxsize
    br_x, br_y = 0, 0

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):

            if img[i, j] == cls and visited[i][j] == 0:

                visited[i][j] = 1
                tl_x, tl_y, br_x, br_y = dfs(cls, [i, j], visited, img, [tl_x, tl_y, br_x, br_y])

                cls_bbox.append([tl_x, tl_y, br_x, br_y, img[tl_x:br_x, tl_y:br_y].sum()])   # 해당 box의 픽셀 값 sum()

                tl_x, tl_y = sys.maxsize, sys.maxsize
                br_x, br_y = 0, 0
            else:
                visited[i][j] = 1

    cls_bbox = sorted(cls_bbox, key=lambda x: x[-1])  # cls가 많이 포함된 순으로 정렬
    return cls_bbox


def cutmix(img1, img2, bbox1, label1, label2, label_info, save_path, filename):
    """
    img1에서 bbox1를 crop하여 img2의 백그라운드 영역에 붙임

    ex)
        args:
            img1: crop할 이미지 (np.array)
            img2: crop한 이미지를 붙일 이미지 (np.array)
            bbox1: img1에서의 find_cls_bbox(cls, img)를 거친 bbox 정보들 (list)
            label1: img1의 label 이미지 (np.array)
            label2: img2의 label 이미지 (np.array)
            label_info: 'left' or 'right'(라벨을 어느쪽에 붙일지 정보가 포함된 변수)
            filename: img 이름
    """

    h, w, c = img1.shape
    patch_l = h // 4

    # origin2 = img2.copy()
    # origin2_label = label2.copy()

    index = []
    for i in range(4):
        for j in range(4):
            index.append([i, j])

    random.shuffle(index)

    for [i, j] in index:
        # 랜덤 패치 위치 및 크기
        tl_x, tl_y = patch_l * i, patch_l * j
        br_x, br_y = tl_x + patch_l, tl_y + patch_l
        half_patch_l = patch_l // 2

        if label2[tl_x:br_x, tl_y:br_y].sum() == 0:  # 붙이는 곳이 백그라운드일때
                rand_tlx, rand_tly, rand_brx, rand_bry, _ = bbox1[-1]
                # rand_tlx, rand_tly, rand_brx, rand_bry = random.choice(bbox1)  # label1에서 가져올 bbox
                rand_cx, rand_cy = (rand_tlx+rand_brx)//2, (rand_tly+rand_bry)//2

                label_tlx, label_tly, label_brx, label_bry = rand_cx-half_patch_l, rand_cy-half_patch_l, rand_cx+half_patch_l, rand_cy+half_patch_l

                # 박스를 패치의 크기로 만들어줌
                if label_tlx < 0:
                    label_tlx = 0
                    label_brx = patch_l

                if label_tly < 0:
                    label_tly = 0
                    label_bry = patch_l

                if label_brx > h:
                    label_brx = h
                    label_tlx = h-patch_l

                if label_bry > w:
                    label_bry = w
                    label_tly = w-patch_l

                # cutmix img
                if tl_y <= w // 2:  # 패치가 왼쪽
                    img2[tl_x:br_x, tl_y:br_y] = img1[tl_x:br_x, tl_y:br_y]
                    img2[tl_x:br_x, w//2+tl_y:w//2+br_y] = img1[tl_x:br_x, w//2+tl_y:w//2+br_y]
                else:
                    img2[tl_x:br_x, tl_y-w//2:br_y-w//2] = img1[tl_x:br_x, tl_y-w//2:br_y-w//2]
                    img2[tl_x:br_x, tl_y:br_y] = img1[tl_x:br_x, tl_y:br_y]

                # cutmix label
                if label_info == 'left':  # 왼쪽에 라벨 저장
                    if tl_y < w//2:
                        label2[tl_x:br_x, tl_y: br_y] = label1[label_tlx:label_brx, label_tly:label_bry]
                    else:
                        label2[tl_x:br_x, tl_y-w//2: br_y-w//2] = label1[label_tlx:label_brx, label_tly:label_bry]
                else:
                    if tl_y < w//2:
                        label2[tl_x:br_x, tl_y+w//2: br_y+w//2] = label1[label_tlx:label_brx, label_tly:label_bry]
                    else:
                        label2[tl_x:br_x, tl_y: br_y] = label1[label_tlx:label_brx, label_tly:label_bry]

                # save
                cv2.imwrite(os.path.join(save_path, 'x', filename), img2)
                cv2.imwrite(os.path.join(save_path, 'y', filename), label2)

                # visualize
                # f, ax = plt.subplots(3, 2, figsize=(15, 15))
                #
                # ax[0, 0].axis('off')
                # ax[0, 1].axis('off')
                # ax[1, 0].axis('off')
                # ax[1, 1].axis('off')
                # ax[2, 0].axis('off')
                # ax[2, 1].axis('off')
                #
                # ax[0, 0].set_title('image1')
                # ax[0, 1].set_title('label1')
                # ax[1, 0].set_title('image2')
                # ax[1, 1].set_title('label2')
                # ax[2, 0].set_title('cutmix image')
                # ax[2, 1].set_title('cutmix label')
                #
                # ax[0, 0].imshow(img1)
                # ax[0, 1].imshow(label1)
                # ax[1, 0].imshow(origin2)
                # ax[1, 1].imshow(origin2_label)
                # ax[2, 0].imshow(img2)
                # ax[2, 1].imshow(label2)
                # plt.show()
                return
    return

def check_num():
    from glob import glob

    x_path = '****/data/cutmix/x/*.png'
    y_path = '****/data/cutmix/y/*.png'

    x_paths = glob(x_path)
    y_paths = glob(y_path)

    answer = len(x_paths)
    cnt = 0

    for x, y in zip(x_paths, y_paths):
        if os.path.basename(x) != os.path.basename(y):
            print(f'filename {os.path.basename(x)}, {os.path.basename(y)}')
        else:
            cnt += 1

    print(f'answer: {answer}, correct: {cnt}')


if __name__ == '__main__':
    x1_paths = []
    with open('****/cls1.csv', newline='') as f:  # cls 라벨 정보
        reader = csv.reader(f)
        for r in reader:
            x1_paths.append(os.path.join('****/data/train/x', r[0]))  # 원본 데이터 경로
    y1_paths = list(map(lambda x: x.replace('x', 'y'), x1_paths))

    x2_paths = []
    with open('****/cls2.csv', newline='') as f:
        reader = csv.reader(f)
        for r in reader:
            x2_paths.append(os.path.join('****/data/train/x', r[0]))
    y2_paths = list(map(lambda x: x.replace('x', 'y'), x2_paths))

    # over sampling
    n = len(x1_paths) - len(x2_paths)
    x2_paths.extend(x2_paths[:n])
    y2_paths.extend(y2_paths[:n])

    change_ = ['', 'left', 'right']
    c = 1
    diff = 0

    for ith, [x1_path, x2_path, y1_path, y2_path] in enumerate(zip(x1_paths, x2_paths, y1_paths, y2_paths)):
        print(f'{ith} cutmix processing..')

        img1 = cv2.imread(x1_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(x2_path, cv2.IMREAD_COLOR)

        # shape 안맞을때 skip
        if img1.shape != (754, 1508, 3) or img2.shape != (754, 1508, 3):
            print(f'shape different! img1:{img1.shape}, img2:{img2.shape}')
            diff += 1
            continue

        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        label1 = cv2.imread(y1_path, cv2.IMREAD_GRAYSCALE)
        label2 = cv2.imread(y2_path, cv2.IMREAD_GRAYSCALE)

        # shape 안맞을때 skip
        if label1.shape != (754, 1508) or label2.shape != (754, 1508):
            print(f'shape different! label1:{label1.shape}, label2:{label2.shape}')
            diff += 1
            continue

        # save 경로
        save_path = f'C:/Users/dudtj/contest/qualify-test/data/cutmix'
        # filename = f'y1_{ith}_{os.path.basename(y2_path)}'
        filename = f'y2_{ith}_{os.path.basename(y2_path)}'
        print(f'filename: {filename}')

        label_info = change_[c]
        c *= -1

        # 1 class에 대해.. cutmix
        # bbox = find_cls_bbox(1, label1)
        # cutmix(img1, img2, bbox, label1, label2, label_info, save_path, filename)

        # 2 class에 대해.. cutmix
        bbox = find_cls_bbox(2, label2)
        cutmix(img2, img1, bbox, label2, label1, label_info, save_path, filename)

# shape 다른 개수 출력
print(f'different shape num: {diff}')  # 5
