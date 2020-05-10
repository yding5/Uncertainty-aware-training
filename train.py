#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import exists
import configs.config_unet3d as allConfig 
from experiment_setup.UNetExperiment3D import UNetExperiment3D
import data_prepare.data_preprocess.preprocess_heart17 as AllPreprocessFunc
from data_prepare.validation_splits.create_splits_heart17 import create_splits_heart17
import datetime

def normalTraining(c):

    print('exp_ID: {}'.format(c.exp_ID))
    exp = UNetExperiment3D(config=c, name='unet_3d', n_epochs=c.n_epochs,
                           seed=42, globs=globals(), exp_ID=c.exp_ID)
    print('initialized')
    exp.run_train()

if __name__ == "__main__":

    print('exp starts at:')
    print(datetime.datetime.now().strftime("_%Y%m%d-%H%M%S"))
    ## list all the available experiment settings
    AllConfigs = []
    AllPreprocess = []

    AllConfigs.append(allConfig.get_config_3d_heart17_8class_largerFit)  # 1
    AllPreprocess.append(AllPreprocessFunc.preprocess_heart17_8class)
    c = AllConfigs[0]()
    # create a file to store all the resulting images
    ## step 1: preprocess the dataset, and store the dataset in mpy format in specific folders
    #AllPreprocess[ExperiN](root_dir=c.data_dir, output_dir=c.preprocess_dir)
    if not exists(c.preprocess_dir):
        print('Preprocessing data. [STARTED]')
        AllPreprocess[0](root_dir=c.data_dir, output_dir=c.preprocess_dir)
        print('Preprocessing data. [DONE]')
    else:
        print('The data has already been preprocessed. It will not be preprocessed again. Delete the folder to enforce it.')
    c.cross_vali_result_all_dir = '/afs/crc.nd.edu/user/y/yding5/Private/Project/XAI/segmentation/pycharm_proj_simple/data/checkpoint'
    cross_vali_result_all_dir = os.path.join(c.base_dir, c.cross_vali_result_all_dir + datetime.datetime.now().strftime(
        "_%Y%m%d-%H%M%S"))
    if not os.path.exists(cross_vali_result_all_dir):
        os.makedirs(cross_vali_result_all_dir)
        print('Created' + cross_vali_result_all_dir + '...')
        c.base_dir = cross_vali_result_all_dir
        c.cross_vali_result_all_dir = os.path.join(cross_vali_result_all_dir, "all_results")
        os.makedirs(c.cross_vali_result_all_dir)

    ## step 2: create splits for cross validation
    if not exists(os.path.join(c.cross_vali_dir, 'splits.pkl')):
        create_splits_heart17(output_dir=c.cross_vali_dir, fileName=c.expName, image_dir=c.preprocess_dir, cross_vali_number = c.cross_vali_N)
    else:
        print('The data has already been splited. It will not be preprocessed again. Delete the file to enforce it.')
    for valiIndex in range(c.cross_vali_N):
        c.cross_vali_index = valiIndex
        normalTraining(c)

    print('exp ends at:')
    print(datetime.datetime.now().strftime("_%Y%m%d-%H%M%S"))
