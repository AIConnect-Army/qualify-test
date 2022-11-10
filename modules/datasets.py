"""Datasets
"""

from torch.utils.data import Dataset
import numpy as np
import cv2
import os

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, cache, mode='train', logger=None, verbose=False, transform=None):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.cache = cache
        self.mode = mode
        self.transform = transform


    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):

        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        cache_data = self.cache.get(self.x_paths[id_], None)

        if cache_data:
            x, orig_size = cache_data['data'],cache_data['size']
        else:
            x, orig_size = None, None

        if x is None:
            x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
            orig_size = x.shape
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = cv2.resize(x, self.input_size)
            # x = self.scaler(x)
            # x = np.transpose(x, (2, 0, 1))

        if self.mode in ['train', 'valid']:
            y = self.cache.get(self.y_paths[id_], None)

            if y is None:
                y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
                y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

                if self.transform:
                    x, y = np.expand_dims(x, axis=0), np.expand_dims(y, axis=0).transpose(1, 2, 0)
                    x, y = self.transform.transform(x, np.expand_dims(y, axis=0))
                    x, y = x[0], y[0][0]
                else:
                    x = np.transpose(x, (2, 0, 1))

                x = self.scaler(x)

                self.cache[self.x_paths[id_]] = {"data":x, "size":orig_size}
                self.cache[self.y_paths[id_]] = y

            return x, y, filename

        elif self.mode in ['test']:
            x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))
            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"
