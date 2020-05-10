#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

#from medpy.io import load
import os
import numpy as np
import nibabel as nib

#from datasets.utils import reshape
from util.file_and_folder_operations import subfiles


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os


class dataset_heart17(Dataset):

    def __init__(self, folder_dataset, image_list, label_list, transform=None):
        self.transform = transform
        self.__xs = []
        self.__ys = []
        self.file_name_list = []
        self.num_batches = len(image_list)

        for i in range(0, len(image_list)):
            pathT_data = os.path.join(folder_dataset, image_list[i])
            pathT_label = os.path.join(folder_dataset, label_list[i])
            temp_image_path = image_list[i].replace("image","label")
            assert label_list[i]==temp_image_path, 'image and its label name are not matched'
            if not os.path.isfile(pathT_data):
                raise AssertionError("file name wrong, not exist")
            if not os.path.isfile(pathT_label):
                raise AssertionError("file name wrong, not exist")
            self.__xs.append(pathT_data)
            self.__ys.append(pathT_label)
            self.file_name_list.append(image_list[i])


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # Convert image and label to torch tensors
        #print(self.__xs[index][-3:])
        if self.__xs[index][-3:] == "npy":
            data = np.load(self.__xs[index])
            data = torch.from_numpy(1.0*data)
            #print('read data')
            #print(torch.max(data))
            #data1 = data.float()
            data = torch.unsqueeze(data, 0)
            label = np.load(self.__ys[index])
            label = torch.from_numpy(1.0*label)
            file_name = self.file_name_list[index]
        elif self.__xs[index][-2:] == "gz":
            data = nib.load(self.__xs[index])
            #print(self.__xs[index])
            data = data.get_fdata()
            data = torch.from_numpy(1.0 * data)
            #print('read data')
            #print(torch.max(data))
            # data1 = data.float()
            data = torch.unsqueeze(data, 0)
            label = nib.load(self.__ys[index])
            label = label.get_fdata()
            label = torch.from_numpy(1.0 * label)
            file_name = self.file_name_list[index]
        else:
            assert("data type is not nii.gz or npy.")
        #print("label.shape,", label.shape)
        #print("data.shape,", data.shape)
        return data, label,file_name

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

class dataset_heart17_multiChannel(Dataset):

    def __init__(self, folder_dataset, image_list, label_list, transform=None):
        self.transform = transform
        self.__xs = []
        self.__ys = []
        self.file_name_list = []
        self.num_batches = len(image_list)

        for i in range(0, len(image_list)):
            pathT_data = os.path.join(folder_dataset, image_list[i])
            pathT_label = os.path.join(folder_dataset, label_list[i])
            temp_image_path = image_list[i].replace("image","label")
            assert label_list[i]==temp_image_path, 'image and its label name are not matched'
            if not os.path.isfile(pathT_data):
                raise AssertionError("file name wrong, not exist")
            if not os.path.isfile(pathT_label):
                raise AssertionError("file name wrong, not exist")
            self.__xs.append(pathT_data)
            self.__ys.append(pathT_label)
            self.file_name_list.append(image_list[i])


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # Convert image and label to torch tensors
        #print(self.__xs[index][-3:])
        if self.__xs[index][-3:] == "npy":
            data = np.load(self.__xs[index])
            data = torch.from_numpy(1.0*data)
            #data1 = data.float()
            #data = torch.unsqueeze(data, 0)
            label = np.load(self.__ys[index])
            label = torch.from_numpy(1.0*label)
            file_name = self.file_name_list[index]
        elif self.__xs[index][-2:] == "gz":
            data = nib.load(self.__xs[index])
            data = data.get_fdata()
            data = torch.from_numpy(1.0 * data)
            data = data.permute(2, 0, 1)
            # data1 = data.float()
            #data = torch.unsqueeze(data, 0)
            label = nib.load(self.__ys[index])
            label = label.get_fdata()
            label = torch.from_numpy(1.0 * label)
            #label.permute(2, 0, 1)
            file_name = self.file_name_list[index]
        else:
            data = None
            label = None
            file_name = None
        #print("label.shape,", label.shape)
        #print("data.shape,", data.shape)
        return data, label,file_name

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)


'''  previous version
class dataset_heart17(Dataset):

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        self.__xs = []
        self.__ys = []
        self.file_name_list = []
        # Open and load text file including the whole training data
        path_data = os.path.join(folder_dataset, 'data')
        path_label = os.path.join(folder_dataset, 'label')
        list = os.listdir(path_label)  # 列出文件夹下所有的目录与文件
        self.num_batches = len(list)

        for i in range(0, len(list)):
            pathT_data = os.path.join(path_data, list[i])
            pathT_label = os.path.join(path_label, list[i])
            if not os.path.isfile(pathT_data):
                raise AssertionError("file name wrong, not exist")
            if not os.path.isfile(pathT_label):
                raise AssertionError("file name wrong, not exist")
            self.__xs.append(pathT_data)
            self.__ys.append(pathT_label)
            self.file_name_list.append(list[i])


    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        # Convert image and label to torch tensors
        data = np.load(self.__xs[index])
        data = torch.from_numpy(1.0*data)
        #data1 = data.float()
        data = torch.unsqueeze(data, 0)
        label = np.load(self.__ys[index])
        label = torch.from_numpy(1.0*label)
        file_name = self.file_name_list[index]
        return data, label,file_name

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)

'''
