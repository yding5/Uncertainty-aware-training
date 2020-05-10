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

import pickle
from util.file_and_folder_operations import subfiles

import os
import numpy as np


def create_splits_heart17(output_dir, fileName, image_dir, cross_vali_number):

    ## there are 20 volumes in total.
    ## index_asValidate is the one as the validation
    ## index_asTest is the one as the test
    NallData = 20
    step = NallData/cross_vali_number

    splits = []
    for split in range(0, cross_vali_number):
        trainset = []
        trainset_label = []
        valset = []
        valset_label = []
        testset = []
        testset_label = []
        for i in range(NallData):
            npy_files_onePatient_data = subfiles(image_dir, prefix=str(i+1)+"_",suffix="image.nii.gz", join=False)
            npy_files_onePatient_label = subfiles(image_dir, prefix=str(i + 1) + "_", suffix="label.nii.gz", join=False)
            if i > step*split-1 and i < (split+1)*step:
                valset = valset + npy_files_onePatient_data
                valset_label = valset_label + npy_files_onePatient_label
                testset = testset + npy_files_onePatient_data
                testset_label = testset_label + npy_files_onePatient_label
            else:
                trainset = trainset + npy_files_onePatient_data
                trainset_label = trainset_label + npy_files_onePatient_label
        split_dict = dict()
        split_dict['train'] = trainset
        split_dict['train_label'] = trainset_label
        split_dict['val'] = valset
        split_dict['val_label'] = valset_label
        split_dict['test'] = testset
        split_dict['test_label'] = testset_label

        splits.append(split_dict)

    with open(os.path.join(output_dir, fileName+'_splits.pkl'), 'wb') as f:
        pickle.dump(splits, f)
