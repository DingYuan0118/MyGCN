import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple

import numpy as np
import scipy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

# 用于保存处理好的数据
Data = namedtuple("Data", ["x", "y", "adjacency", "train_mask", "val_mask", "test_mask"])

class CoraData(object):
    download_url = "https://github.com/kimiyoung/planetoid/raw/master/data"
    filenames = ["ind.cora.{}".format(name) for name in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]]

    def __init__(self, data_root="cora", rebuild=False):
        """
        包括数据下载、处理、加载等功能，当缓存文件存在时，将使用缓存文件。
        Arg:
            data_root: string, optional
                存放数据的目录，原始数据路径:{data_root}/raw
                缓存数据路径:{data_root}/processed_cora.pkl
            rebuild: boolean, optional
                是否需要重新构建数据集,当设为True时，即使有缓存，也会重建数据
        """
        self.data_root = data_root
        save_file = osp.join(self.data_root, "processed_cora.pkl")
        if osp.exists(save_file) and not rebuild:
            print("Using Cache file:{}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self.maybe_download()
            self._data = self.process_data()
            with open(save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Cached file: {}".format(save_file))

    @property
    def data(self):
        """返回Data数据对象，包括x, y, adjacency, train_mask, val_mask, test_mask"""
        return self._data

    def maybe_download(self):
        save_path = osp.join(self.data_root, "raw")
        for name in self.filenames:
            if not osp.exists(osp.join(save_path, name)):
                self.download_data("{}/{}".format(self.download_url, name), save_path)
            
    @staticmethod
    def download_data(url, save_path):
        """数据下载工具，当原始数据不存在时将会进行下载"""
        if not osp.exists(save_path):
            os.makedirs(save_path)
        data = urllib.request.urlopen(url)
        filename = osp.basename(url)

        with open(osp.join(save_path, filename), "wb") as f:
            f.write(data.read())

        return True

Data_Source = CoraData()