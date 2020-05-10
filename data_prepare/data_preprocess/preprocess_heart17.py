#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict
import random
#from medpy.io import load
import os
import numpy as np
import nibabel as nib
#import matplotlib.pyplot as plt
from PIL import Image
import cv2
import skimage
import matplotlib.image as mpimg
import torch.utils.data as data_utils
import torch
#from util.util import reshape
from util.file_and_folder_operations import subfiles


def preprocess_heart17_8_class(root_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    image_label_dir = os.path.join(root_dir, 'data')
    #label_dir = os.path.join(root_dir, 'mask')
    output_dir = os.path.join(root_dir, 'preprocessed_8_class')
    classes = 8

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0

    img_files = subfiles(image_label_dir, suffix="image.nii.gz", join=False)
    #label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(img_files)
    #trainingN = np.int(file_num*extractPatchPerImg*0.8)
    #patch_training = np.zeros([trainingN,patchSize,patchSize,3])
    #patch_training_label = np.zeros([trainingN, patchSize, patchSize])
    #patch_vali = np.zeros([file_num*extractPatchPerImg-trainingN,patchSize,patchSize,3])
    #patch_vali_label = np.zeros([file_num*extractPatchPerImg-trainingN, patchSize, patchSize])

    temp_idx = 0
    TrainN = file_num*0.8

    for img_name in img_files:
        #image, _ = load(os.path.join(image_dir, f))
        label_name = img_name.replace("image","label")
        image_nii = nib.load(os.path.join(image_label_dir, img_name))
        image = image_nii.get_fdata()
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()

        #print(img_name, 'size:',image.shape)
        for i in range(classes):
            print(np.sum((label == i*1.0)))
            class_stats[i] += np.sum(label == i*1.0 )
            total += np.sum(label == i*1.0 )

        #image = (image - image.min())/(image.max()-image.min())
        #image_size = image.shape

        #temp_label = np.zeros([patchSize,patchSize])
        #img_name_new = img_name.replace("image","")
        #label_name_new = label_name.replace("label", "")
        np.save(os.path.join(output_dir, img_name + '.npy'), image)
        np.save(os.path.join(output_dir, label_name + '.npy'), label)

    print(total)
    for i in range(classes):
        print(class_stats[i], class_stats)

def preprocess_heart17_2_class(root_dir):
    image_label_dir = os.path.join(root_dir, 'data')
    #label_dir = os.path.join(root_dir, 'mask')
    output_dir = os.path.join(root_dir, 'preprocessed_2_class')
    classes = 2  ## object, and background  3 ## bounary, object, and background
    #extractPatchPerImg = 100
    #patchSize = 192

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0

    img_files = subfiles(image_label_dir, suffix="image.nii.gz", join=False)
    #label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(img_files)
    #trainingN = np.int(file_num*extractPatchPerImg*0.8)
    #patch_training = np.zeros([trainingN,patchSize,patchSize,3])
    #patch_training_label = np.zeros([trainingN, patchSize, patchSize])
    #patch_vali = np.zeros([file_num*extractPatchPerImg-trainingN,patchSize,patchSize,3])
    #patch_vali_label = np.zeros([file_num*extractPatchPerImg-trainingN, patchSize, patchSize])

    #temp_idx = 0
    #TrainN = file_num*0.8

    for img_name in img_files:
        #image, _ = load(os.path.join(image_dir, f))
        label_name = img_name.replace("image","label")
        image_nii = nib.load(os.path.join(image_label_dir, img_name))
        image = image_nii.get_fdata()
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        label_2class = (label > 0)

        #print(img_name, 'size:',image.shape)
        for i in range(classes):
            print(np.sum((label_2class == i*1.0)))
            class_stats[i] += np.sum(label_2class == i*1.0 )
            total += np.sum(label_2class == i*1.0 )
        for i in range(8):
            print(np.sum((label == i*1.0)))
            class_stats[i] += np.sum(label == i*1.0 )
            total += np.sum(label == i*1.0 )

        np.save(os.path.join(output_dir, img_name + '.npy'), image)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_2class)

    print(total)
    for i in range(classes):
        print(class_stats[i], class_stats)

def saveImage2D(label_1, label_2, output_dir):
    d1, d2, d3 = label_1.shape
    min1 = np.min(label_1)
    min2 = np.min(label_2)
    max1 = np.max(label_1)
    max2 = np.max(label_2)
    for i in range(d3):
        im = Image.fromarray(((label_1[:,:,i]-min1)*255/(max1-min1)).astype(np.uint8))
        im = im.convert("L")
        im.save(os.path.join(output_dir,str(i)+"_1.png"))
        im = Image.fromarray(((label_2[:,:,i]-min2)*255/(max2-min2)).astype(np.uint8))
        im = im.convert("L")
        im.save(os.path.join(output_dir, str(i)+"_2.png"))


def findAround(data ,i1 ,i2 ,i3 ,d1, d2, d3, number):
    sizeSearch = 1
    found = False
    i1_min = max(i1 - sizeSearch,0)
    i2_min = max(i2 - sizeSearch, 0)
    i3_min = max(i3 - sizeSearch, 0)
    i1_max = min(i1+sizeSearch+1, d1)
    i2_max = min(i2+sizeSearch+1, d2)
    i3_max = min(i3+sizeSearch+1, d3)
    #for j1 in range(1, 3):
    #    print(j1)
    for j1 in range(i1_min, i1_max):
        for j2 in range(i2_min, i2_max):
            for j3 in range(i3_min, i3_max):
                if data[j1,j2,j3] == number:
                    found = True
                    return found
    return found

def preprocess_heart17_detection_connection(root_dir):
    ## this function extracts the original datasets with 5 classes
    ## 0 background,
    # 1 the connection between LV and LA,
    # 2 the connection between LV and AO
    # 3 the connection between RV and RA
    # 4 the connection between RV and PA

    image_label_dir = os.path.join(root_dir, 'data')
    #label_dir = os.path.join(root_dir, 'mask')
    output_dir = os.path.join(root_dir, 'preprocessed_detect_connection')
    classes = 5  ## object, and background  3 ## bounary, object, and background
    #extractPatchPerImg = 100
    #patchSize = 192

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    #label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        label_2 = np.zeros(label.shape)
        d1, d2, d3 = label.shape
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if label[i1,i2,i3] == 1:  # LV  to  LA or AO
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,3): # try to find LA
                            label_2[i1, i2, i3] = 1  #8
                        if findAround(label, i1, i2, i3, d1, d2, d3, 6):  # try to find AO
                            label_2[i1, i2, i3] = 2  #9
                    if label[i1,i2,i3] == 2:  # RV to RA or PA
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,4): # try to find RA
                            label_2[i1, i2, i3] = 3  #10
                        if findAround(label, i1, i2, i3, d1, d2, d3, 7):  # try to find PA
                            label_2[i1, i2, i3] = 4  #11
                    if label[i1,i2,i3] == 3:  # LA
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,1): # try to find LV
                            label_2[i1, i2, i3] = 1  #8
                    if label[i1,i2,i3] == 4:  # RA
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,2): # try to find RV
                            label_2[i1, i2, i3] = 3  #10
                    if label[i1,i2,i3] == 6:  # AO
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,1): # try to find LV
                            label_2[i1, i2, i3] = 2  #9
                    if label[i1,i2,i3] == 7:  # PA
                        if findAround(label, i1 ,i2 ,i3, d1, d2, d3 ,2): # try to find RA
                            label_2[i1, i2, i3] = 4  #11
        sum1 = np.sum(label_2 == 1)
        print(sum1)
        print(np.sum(label_2 == 2))
        print(np.sum(label_2 == 3))
        print(np.sum(label_2 == 4))

        '''
        for i in range(d3):
            im = Image.fromarray((label[:,:,i]*255).astype(np.uint8))
            im = im.convert("L")
            im.save(os.path.join(output_dir,str(i)+".png"))
            im = Image.fromarray((label_2[:,:,i]*255).astype(np.uint8))
            im = im.convert("L")
            im.save(os.path.join(output_dir, str(i)+"_detect.png"))
        '''
        label_new_name = label_name.replace("label","image")
        label_new_2_name = label_name
        np.save(os.path.join(output_dir, label_new_name + '.npy'), label)
        np.save(os.path.join(output_dir, label_new_2_name + '.npy'), label_2)

def preprocess_heart17_bloodBoundryA(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood   (RA RV PA) and (LA LV AO)
    # 2 boundry  : the boundry of blood
    # 3 myo

    image_label_dir =  root_dir
    #output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 4  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    #label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label== 5.0
        blood_AO = label == 6.0
        blood_PA = label == 7.0
        print(np.sum(blood_PA))

        blood_A = blood_RA | blood_RV | blood_PA
        blood_A = blood_A*1.0
        blood_A_boundry = np.zeros(label.shape)
        print(np.sum(blood_A))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_A[i1,i2,i3] == 1:
                        blood_A_boundry[i1,i2,i3] = findAround(blood_A, i1, i2, i3, d1, d2, d3, 0)
        blood_A_area = blood_A - blood_A_boundry
        print(np.sum(blood_A_boundry))
        print(np.sum(blood_A_area))
        print(np.sum(blood_A_area)+np.sum(blood_A_boundry)-np.sum(blood_A))
        #saveImage2D(blood_A, blood_A_boundry, output_dir)

        blood_B = blood_LA | blood_LV | blood_AO
        blood_B = blood_B*1.0
        blood_B_boundry = np.zeros(label.shape)
        print(np.sum(blood_B))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_B[i1,i2,i3] == 1:
                        blood_B_boundry[i1,i2,i3] = findAround(blood_B, i1, i2, i3, d1, d2, d3, 0)
        blood_B_area = blood_B - blood_B_boundry
        print(np.sum(blood_B_boundry))
        print(np.sum(blood_B_area))
        print(np.sum(blood_B_area)+np.sum(blood_B_boundry)-np.sum(blood_B))

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_A_area*1.0 + blood_A_boundry * 2.0 + blood_B_area*1.0 + blood_B_boundry*2.0 + blood_myo*3.0
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label","image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_all_class)

def preprocess_heart17_bloodBoundryB(root_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 the connection between LV and LA,
    # 2 the connection between LV and AO
    # 3 the connection between RV and RA
    # 4 the connection between RV and PA
    # 5 blood A (RA RV PA)
    # 6 boundry A: the boundry of blood A
    # 7 blood B (LA LV AO)
    # 8 boundry B: the boundry of blood B
    # 9 myo

    connection_folder = os.path.join(root_dir, 'preprocessed_detect_connection')
    image_label_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 10  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    #label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:
        # connection part: 1-4
        label_10class = np.load(os.path.join(connection_folder, label_name.replace(".nii.gz", ".nii.gz.npy")))
        connection_mask = label_10class > 0
        ## label 5-9
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label== 5.0
        blood_AO = label == 6.0
        blood_PA = label == 7.0
        print(np.sum(blood_PA))

        blood_A = blood_RA | blood_RV | blood_PA
        blood_A = blood_A*1.0
        blood_A_boundry = np.zeros(label.shape)
        print(np.sum(blood_A))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_A[i1,i2,i3] == 1:
                        blood_A_boundry[i1,i2,i3] = findAround(blood_A, i1, i2, i3, d1, d2, d3, 0)
        blood_A_area = blood_A - blood_A_boundry
        print(np.sum(blood_A_boundry))
        print(np.sum(blood_A_area))
        print(np.sum(blood_A_area)+np.sum(blood_A_boundry)-np.sum(blood_A))
        and_A_area_with_connection = (blood_A_area >0) & (connection_mask>0)
        and_A_boundry_with_connection = (blood_A_boundry>0) & (connection_mask>0)
        blood_A_area = blood_A_area - and_A_area_with_connection*1.0
        blood_A_boundry = blood_A_boundry - and_A_boundry_with_connection*1.0
        #saveImage2D(blood_A, blood_A_boundry, output_dir)

        blood_B = blood_LA | blood_LV | blood_AO
        blood_B = blood_B*1.0
        blood_B_boundry = np.zeros(label.shape)
        print(np.sum(blood_B))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_B[i1,i2,i3] == 1:
                        blood_B_boundry[i1,i2,i3] = findAround(blood_B, i1, i2, i3, d1, d2, d3, 0)
        blood_B_area = blood_B - blood_B_boundry
        print(np.sum(blood_B_boundry))
        print(np.sum(blood_B_area))
        print(np.sum(blood_B_area)+np.sum(blood_B_boundry)-np.sum(blood_B))
        and_B_area_with_connection = (blood_B_area>0) & (connection_mask>0)
        and_B_boundry_with_connection = (blood_B_boundry>0) & (connection_mask>0)
        blood_B_area = blood_B_area - and_B_area_with_connection*1.0
        blood_B_boundry = blood_B_boundry - and_B_boundry_with_connection*1.0

        label_10class = label_10class + blood_A_area*5 + blood_A_boundry * 6 + blood_B_area*7 + blood_B_boundry*8 + blood_myo*9
        #saveImage2D(label, label_10class, output_dir)

        image_name = label_name.replace("label","image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_10class)

def preprocess_heart17_bloodBoundryC(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood A (RA RV PA)
    # 2 boundry A: the boundry of blood A
    # 3 blood B (LA LV AO)
    # 4 boundry B: the boundry of blood B
    # 5 myo

    image_label_dir = root_dir
    # output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 6  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    # label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label == 5.0
        blood_AO = label == 6.0
        blood_PA = label == 7.0
        print(np.sum(blood_PA))

        blood_A = blood_RA | blood_RV | blood_PA
        blood_A = blood_A * 1.0
        blood_A_boundry = np.zeros(label.shape)
        print(np.sum(blood_A))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_A[i1, i2, i3] == 1:
                        blood_A_boundry[i1, i2, i3] = findAround(blood_A, i1, i2, i3, d1, d2, d3, 0)
        blood_A_area = blood_A - blood_A_boundry
        print(np.sum(blood_A_boundry))
        print(np.sum(blood_A_area))
        print(np.sum(blood_A_area) + np.sum(blood_A_boundry) - np.sum(blood_A))
        # saveImage2D(blood_A, blood_A_boundry, output_dir)

        blood_B = blood_LA | blood_LV | blood_AO
        blood_B = blood_B * 1.0
        blood_B_boundry = np.zeros(label.shape)
        print(np.sum(blood_B))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_B[i1, i2, i3] == 1:
                        blood_B_boundry[i1, i2, i3] = findAround(blood_B, i1, i2, i3, d1, d2, d3, 0)
        blood_B_area = blood_B - blood_B_boundry
        print(np.sum(blood_B_boundry))
        print(np.sum(blood_B_area))
        print(np.sum(blood_B_area) + np.sum(blood_B_boundry) - np.sum(blood_B))

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_A_area * 1.0 + blood_A_boundry * 2.0 + blood_B_area * 3.0 + blood_B_boundry * 4.0 + blood_myo * 5.0
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_all_class)

def preprocess_heart17_bloodBoundryCplus(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood A (RA RV PA)
    # 2 boundry A: the boundry of blood A
    # 3 blood B (LA LV AO)
    # 4 boundry B: the boundry of blood B

    image_label_dir = root_dir
    # output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 6  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    # label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label == 5.0
        blood_AO = label == 6.0
        blood_PA = label == 7.0
        print(np.sum(blood_PA))

        blood_A = blood_RA | blood_RV | blood_PA
        blood_A = blood_A * 1.0
        blood_A_boundry = np.zeros(label.shape)
        print(np.sum(blood_A))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_A[i1, i2, i3] == 1:
                        blood_A_boundry[i1, i2, i3] = findAround(blood_A, i1, i2, i3, d1, d2, d3, 0)
        blood_A_area = blood_A - blood_A_boundry
        print(np.sum(blood_A_boundry))
        print(np.sum(blood_A_area))
        print(np.sum(blood_A_area) + np.sum(blood_A_boundry) - np.sum(blood_A))
        # saveImage2D(blood_A, blood_A_boundry, output_dir)

        blood_B = blood_LA | blood_LV | blood_AO
        blood_B = blood_B * 1.0
        blood_B_boundry = np.zeros(label.shape)
        print(np.sum(blood_B))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_B[i1, i2, i3] == 1:
                        blood_B_boundry[i1, i2, i3] = findAround(blood_B, i1, i2, i3, d1, d2, d3, 0)
        blood_B_area = blood_B - blood_B_boundry
        print(np.sum(blood_B_boundry))
        print(np.sum(blood_B_area))
        print(np.sum(blood_B_area) + np.sum(blood_B_boundry) - np.sum(blood_B))

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_A_area * 1.0 + blood_A_boundry * 2.0 + blood_B_area * 3.0 + blood_B_boundry * 4.0
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_all_class)


def preprocess_heart17_TwoBlood(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood A (RA RV PA)  no boundary
    # 2 blood B (LA LV AO)  no boundary
    # 3 myo

    image_label_dir = root_dir
    # output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 4  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    # label_files = subfiles(label_dir, suffix=".png", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label == 5.0
        blood_AO = label == 6.0
        blood_PA = label == 7.0
        print(np.sum(blood_PA))

        blood_A = blood_RA | blood_RV | blood_PA
        blood_A = blood_A * 1.0
        blood_A_boundry = np.zeros(label.shape)
        print(np.sum(blood_A))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_A[i1, i2, i3] == 1:
                        blood_A_boundry[i1, i2, i3] = findAround(blood_A, i1, i2, i3, d1, d2, d3, 0)
        blood_A_area = blood_A - blood_A_boundry
        print(np.sum(blood_A_boundry))
        print(np.sum(blood_A_area))
        print(np.sum(blood_A_area) + np.sum(blood_A_boundry) - np.sum(blood_A))
        # saveImage2D(blood_A, blood_A_boundry, output_dir)

        blood_B = blood_LA | blood_LV | blood_AO
        blood_B = blood_B * 1.0
        blood_B_boundry = np.zeros(label.shape)
        print(np.sum(blood_B))
        for i1 in range(d1):
            for i2 in range(d2):
                for i3 in range(d3):
                    if blood_B[i1, i2, i3] == 1:
                        blood_B_boundry[i1, i2, i3] = findAround(blood_B, i1, i2, i3, d1, d2, d3, 0)
        blood_B_area = blood_B - blood_B_boundry
        print(np.sum(blood_B_boundry))
        print(np.sum(blood_B_area))
        print(np.sum(blood_B_area) + np.sum(blood_B_boundry) - np.sum(blood_B))

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_A_area * 1.0  + blood_B_area * 2.0 + blood_myo * 3.0
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_all_class)

'''
def preprocess_heart17_bloodMyo(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood   (RA RV PA) and (LA LV AO)
    # 2 myo

    image_label_dir =  root_dir
    #output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 3  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1, d2, d3 = label.shape

        blood_LV = label == 1.0
        print(np.sum(blood_LV))
        blood_RV = label == 2.0
        print(np.sum(blood_RV))
        blood_LA = label == 3.0
        print(np.sum(blood_LA))
        blood_RA = label == 4.0
        print(np.sum(blood_RA))
        blood_myo = label== 5.0
        print(np.sum(blood_myo))
        blood_AO = label == 6.0
        print(np.sum(blood_AO))
        blood_PA = label == 7.0
        print(np.sum(blood_PA))
        bcgrd = label == 0
        print(np.sum(bcgrd))

        blood_all = blood_RA | blood_RV | blood_PA | blood_LA | blood_LV | blood_AO
        blood_all = blood_all*1.0

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_all*1.0 + blood_myo*2.0
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label","image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        np.save(os.path.join(output_dir, image_name + '.npy'), image_data)
        np.save(os.path.join(output_dir, label_name + '.npy'), label_all_class)
    '''

def getBBox(real_label):
    real_position_list = np.where(real_label > 0)
    shape_real = real_position_list[0].shape

    real_position = np.zeros([shape_real[0], 3])
    real_position[:, 0] = real_position_list[0][:]
    real_position[:, 1] = real_position_list[1][:]
    real_position[:, 2] = real_position_list[2][:]

    real_min = real_position.min(0)
    real_max = real_position.max(0)
    return real_min, real_max

def preprocess_heart17_8class(root_dir, output_dir):
    ## this function extracts the original tasets with 10  as  classes

    image_label_dir = root_dir
    #output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 8  ## object, hehe and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        cur_num = int(label_name[11])*10 + int(label_name[12])

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1_origin, d2_origin, d3_origin = label.shape

        blood_LV = label == 500.0
        print(np.sum(blood_LV))
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        print(np.sum(blood_LA))
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        print(np.sum(blood_myo))
        blood_AO = label == 820.0
        print(np.sum(blood_AO))
        blood_PA = label == 850.0
        print(np.sum(blood_PA))
        bcgrd = label == 0
        print(np.sum(bcgrd))

        label_all_class = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        ## new shape and new crop
        label_bbox_min, label_bbox_max = getBBox(label > 0)
        crop_x_min = int(np.max([label_bbox_min[0] - 8*3, 0]))
        crop_y_min = int(np.max([label_bbox_min[1] - 8*3, 0]))
        crop_z_min = int(np.max([label_bbox_min[2] - int(d3_origin/64*3), 0]))
        crop_x_max = int(np.min([label_bbox_max[0] + 8*3, d1_origin]))
        crop_y_max = int(np.min([label_bbox_max[1] + 8*3, d2_origin]))
        crop_z_max = int(np.min([label_bbox_max[2] + int(d3_origin/64*3), d3_origin]))

        #label_crop = label_all_class[crop_x_min:crop_x_max,crop_y_min:crop_y_max,crop_z_min:crop_z_max]
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label","image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        #image_crop = image_data[crop_x_min:crop_x_max,crop_y_min:crop_y_max,crop_z_min:crop_z_max]

        # resize to 64*64*64
        new_size = 64
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        interval = 20
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    temp_x_min = int(np.min([label_bbox_min[0], crop_x_min + interval * i1]))
                    temp_x_max = int(np.max([label_bbox_max[0], crop_x_max - interval * i1]))
                    temp_y_min = int(np.min([label_bbox_min[1], crop_y_min + interval * i2]))
                    temp_y_max = int(np.max([label_bbox_max[1], crop_y_max - interval * i2]))
                    temp_z_min = int(np.min([label_bbox_min[2], crop_z_min + int(d3_origin / 64 * i3)]))
                    temp_z_max = int(np.max([label_bbox_max[2], crop_z_max - int(d3_origin / 64 * i3)]))
                    image_data_crop = image_data[     temp_x_min:temp_x_max, temp_y_min:temp_y_max, temp_z_min:temp_z_max ]
                                      # version 1:  crop_z_min+int(d3_origin/64*5):crop_z_max-int(d3_origin/64*5)]
                    label_data_crop = label_all_class[temp_x_min:temp_x_max, temp_y_min:temp_y_max, temp_z_min:temp_z_max ]
                                      #version 1:  crop_z_min+int(d3_origin/64*5):crop_z_max-int(d3_origin/64*5)]
                    d1, d2, d3 = image_data_crop.shape
                    if temp_x_min > label_bbox_min[0] or temp_x_max < label_bbox_max[0] or \
                            temp_y_min > label_bbox_min[1] or temp_y_max < label_bbox_max[1] or \
                            temp_z_min > label_bbox_min[2] or temp_z_max < label_bbox_max[2] :
                        print("error may happen__________________________________")

                    z_scale = d3/new_size
                    for i in range(new_size):
                        z_temp = int(i*z_scale)+int(z_scale) ####  mapping change
                        z_temp = np.min([z_temp, d3-1])
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(new_size,new_size))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = np.zeros([newSize3d[0], newSize3d[1]])
                        for classIndex in range(1, 8):
                            temp_class = temp_label == classIndex
                            temp_temp = cv2.resize(temp_class*1.0, dsize=(newSize3d[0], newSize3d[1]))
                            new_slice_label = new_slice_label + classIndex*(temp_temp>0.5)
                            if (np.max(new_slice_label)) > 7:
                                print((np.max(new_slice_label)))
                        new_label[:, :, i] = new_slice_label[:, :]

                    print(np.max(new_label))
                    if (np.max(new_label)) > 7:
                        print((np.max(new_label)))
                    # normalization image
                    mean = np.mean(new_image)
                    std = np.std(new_image)
                    new_image = (new_image - mean) / std

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)
                                #saveImage2D(flip_image, flip_label, output_dir)
                            flip_image_tmp = nib.Nifti1Image(flip_image, np.eye(4))
                            nib.save(flip_image_tmp, os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + image_name))
                            flip_label_tmp = nib.Nifti1Image(flip_label, np.eye(4))
                            nib.save(flip_label_tmp, os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + label_name))
                            print(os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + label_name))
                            print(np.max(flip_label))
                            if np.max(flip_label) > 7:
                                assert (np.max(flip_label) < 7), '(np.max(flip_label) > 2'




def preprocess_heart17_bloodMyo(root_dir, output_dir):
    ## this function extracts the original datasets with 10 classes
    ## 0 background,
    # 1 blood   (RA RV PA) and (LA LV AO)
    # 2 myo

    image_label_dir = root_dir
    #output_dir = os.path.join(root_dir, 'preprocessed_bloodBoundryB')
    classes = 3  ## object, and background  3 ## bounary, object, and background

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        cur_num = int(label_name[11])*10 + int(label_name[12])

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        d1_origin, d2_origin, d3_origin = label.shape

        blood_LV = label == 500.0
        print(np.sum(blood_LV))
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        print(np.sum(blood_LA))
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        print(np.sum(blood_myo))
        blood_AO = label == 820.0
        print(np.sum(blood_AO))
        blood_PA = label == 850.0
        print(np.sum(blood_PA))
        bcgrd = label == 0
        print(np.sum(bcgrd))

        blood_all = blood_RA | blood_RV | blood_PA | blood_LA | blood_LV | blood_AO
        blood_all = blood_all*1.0

        ## new shape and new crop
        label_bbox_min, label_bbox_max = getBBox(label > 0)
        crop_x_min = int(np.max([label_bbox_min[0] - 8*3, 0]))
        crop_y_min = int(np.max([label_bbox_min[1] - 8*3, 0]))
        crop_z_min = int(np.max([label_bbox_min[2] - int(d3_origin/64*3), 0]))
        crop_x_max = int(np.min([label_bbox_max[0] + 8*3, d1_origin]))
        crop_y_max = int(np.min([label_bbox_max[1] + 8*3, d2_origin]))
        crop_z_max = int(np.min([label_bbox_max[2] + int(d3_origin/64*3), d3_origin]))

        label_all_class = np.zeros(label.shape)
        label_all_class = label_all_class + blood_all*1.0 + blood_myo*2.0
        #label_crop = label_all_class[crop_x_min:crop_x_max,crop_y_min:crop_y_max,crop_z_min:crop_z_max]
        #saveImage2D(label, label_all_class, output_dir)

        image_name = label_name.replace("label","image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        #image_crop = image_data[crop_x_min:crop_x_max,crop_y_min:crop_y_max,crop_z_min:crop_z_max]

        # resize to 64*64*64
        new_size = 64
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        interval = 20
        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    temp_x_min = int(np.min([label_bbox_min[0], crop_x_min + interval * i1]))
                    temp_x_max = int(np.max([label_bbox_max[0], crop_x_max - interval * i1]))
                    temp_y_min = int(np.min([label_bbox_min[1], crop_y_min + interval * i2]))
                    temp_y_max = int(np.max([label_bbox_max[1], crop_y_max - interval * i2]))
                    temp_z_min = int(np.min([label_bbox_min[2], crop_z_min + int(d3_origin / 64 * i3)]))
                    temp_z_max = int(np.max([label_bbox_max[2], crop_z_max - int(d3_origin / 64 * i3)]))
                    image_data_crop = image_data[     temp_x_min:temp_x_max, temp_y_min:temp_y_max, temp_z_min:temp_z_max ]
                                      # version 1:  crop_z_min+int(d3_origin/64*5):crop_z_max-int(d3_origin/64*5)]
                    label_data_crop = label_all_class[temp_x_min:temp_x_max, temp_y_min:temp_y_max, temp_z_min:temp_z_max ]
                                      #version 1:  crop_z_min+int(d3_origin/64*5):crop_z_max-int(d3_origin/64*5)]
                    d1, d2, d3 = image_data_crop.shape
                    if temp_x_min > label_bbox_min[0] or temp_x_max < label_bbox_max[0] or \
                            temp_y_min > label_bbox_min[1] or temp_y_max < label_bbox_max[1] or \
                            temp_z_min > label_bbox_min[2] or temp_z_max < label_bbox_max[2] :
                        print("error may happen__________________________________")

                    z_scale = d3/new_size
                    for i in range(new_size):
                        z_temp = int(i*z_scale)+int(z_scale) ####  mapping change
                        z_temp = np.min([z_temp, d3-1])
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(new_size,new_size))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = np.zeros([newSize3d[0], newSize3d[1]])
                        for classIndex in range(1,3):
                            temp_class = temp_label == classIndex
                            temp_temp = cv2.resize(temp_class*1.0, dsize=(newSize3d[0], newSize3d[1]))
                            new_slice_label = new_slice_label + classIndex*(temp_temp>0.5)
                            if (np.max(new_slice_label)) > 2:
                                print((np.max(new_slice_label)))
                        new_label[:, :, i] = new_slice_label[:, :]

                    print(np.max(new_label))
                    if (np.max(new_label)) > 2:
                        print((np.max(new_label)))
                    # normalization image
                    mean = np.mean(new_image)
                    std = np.std(new_image)
                    new_image = (new_image - mean) / std

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)
                                #saveImage2D(flip_image, flip_label, output_dir)
                            flip_image_tmp = nib.Nifti1Image(flip_image, np.eye(4))
                            nib.save(flip_image_tmp, os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + image_name))
                            flip_label_tmp = nib.Nifti1Image(flip_label, np.eye(4))
                            nib.save(flip_label_tmp, os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + label_name))
                            print(os.path.join(output_dir, str(cur_num) + '_p1_r' + str(rr) + '_f' + str(ff) + '_c' + str( \
                            (i1 + 1) * 100 + i2 * 10 + i3) + '_' + label_name))
                            print(np.max(flip_label))
                            if np.max(flip_label) > 2:
                                assert (np.max(flip_label) < 2), '(np.max(flip_label) > 2'




def preprocess_heart17_localization64(root_dir, output_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    #label_dir = os.path.join(root_dir, 'mask')
    classes = 2
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        blood_all = blood_RA | blood_RV | blood_PA | blood_LA | blood_LV | blood_AO
        label_all = blood_all*1.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias 60 60 60
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    image_data_crop = image_data[i1*20:, i2*20:, i3*60:]
                    label_data_crop = label_all[i1 * 20:, i2 * 20:, i3 * 60:]
                    d1, d2, d3 = image_data_crop.shape

                    z_scale = d3/newSize3d[2]
                    for i in range(newSize3d[2]):
                        z_temp = int(i*z_scale)
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(newSize3d[0], newSize3d[1]))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = cv2.resize(temp_label, dsize=(newSize3d[0], newSize3d[1]))
                        new_label[:, :, i] = new_slice_label[:, :]

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)

                            #saveImage2D(flip_image, flip_label, output_dir)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+image_name + '.npy'), flip_image)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+label_name + '.npy'), flip_label)

def preprocess_heart17_localization64_mri(root_dir, output_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    #label_dir = os.path.join(root_dir, 'mask')
    classes = 2
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        label_all = label>0
        label_all = label_all*1.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias 60 60 60
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    image_data_crop = image_data[i1*10:, i2*10:, i3*10:]
                    label_data_crop = label_all[i1 * 10:, i2 * 10:, i3 * 10:]
                    d1, d2, d3 = image_data_crop.shape

                    z_scale = d3/newSize3d[2]
                    for i in range(newSize3d[2]):
                        z_temp = int(i*z_scale)
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(newSize3d[0], newSize3d[1]))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = cv2.resize(temp_label, dsize=(newSize3d[0], newSize3d[1]))
                        new_label[:, :, i] = new_slice_label[:, :]

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)

                            saveImage2D(flip_image, flip_label, output_dir)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+image_name + '.npy'), flip_image)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+label_name + '.npy'), flip_label)


def preprocess_heart17_localization64_2blood(root_dir, output_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    #label_dir = os.path.join(root_dir, 'mask')
    classes = 3
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        blood_others = blood_RA | blood_RV | blood_LA | blood_LV | blood_AO
        label_all = blood_others*1.0 + blood_PA*2.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias 60 60 60
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    image_data_crop = image_data[i1*20:, i2*20:, i3*60:]
                    label_data_crop = label_all[i1 * 20:, i2 * 20:, i3 * 60:]
                    d1, d2, d3 = image_data_crop.shape

                    z_scale = d3/newSize3d[2]
                    for i in range(newSize3d[2]):
                        z_temp = int(i*z_scale)
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(newSize3d[0], newSize3d[1]))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = cv2.resize(temp_label, dsize=(newSize3d[0], newSize3d[1]))
                        new_label[:, :, i] = new_slice_label[:, :]

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)

                            #saveImage2D(flip_image, flip_label, output_dir)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+image_name + '.npy'), flip_image)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+label_name + '.npy'), flip_label)


def resizeLabel(arrayLabel, classN):
    print("v")

def preprocess_heart17_localization64_allClass(root_dir, output_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    #label_dir = os.path.join(root_dir, 'mask')
    classes = 3
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias 60 60 60
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    image_data_crop = image_data[i1*20:, i2*20:, i3*60:]
                    label_data_crop = label_all[i1 * 20:, i2 * 20:, i3 * 60:]
                    d1, d2, d3 = image_data_crop.shape

                    z_scale = d3/newSize3d[2]
                    for i in range(newSize3d[2]):
                        z_temp = int(i*z_scale)
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(newSize3d[0], newSize3d[1]))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = np.zeros([newSize3d[0], newSize3d[1]])
                        for classIndex in range(1,8):
                            temp_class = temp_label == classIndex
                            temp_temp = cv2.resize(temp_class*1.0, dsize=(newSize3d[0], newSize3d[1]))
                            new_slice_label = new_slice_label + classIndex*(temp_temp>0.5)
                        new_label[:, :, i] = new_slice_label[:, :]

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)

                            #saveImage2D(flip_image, flip_label, output_dir)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+image_name + '.npy'), flip_image)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+label_name + '.npy'), flip_label)

def preprocess_heart17_localization64_mri_allClass(root_dir, output_dir):
    ## this function extracts the original datasets with 8 classes
    ## 0 background, 1 LV, 2 RV, 3 LA, 4 RA, 5 myo, 6 AO, 7 PA.
    #label_dir = os.path.join(root_dir, 'mask')
    classes = 3
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias 60 60 60
        newSize3d = (int(new_size), int(new_size), int(new_size))
        new_label = np.zeros((newSize3d))
        new_image = np.zeros((newSize3d))
        for i1 in range(2):
            for i2 in range(2):
                for i3 in range(2):
                    image_data_crop = image_data[i1*20:, i2*20:, i3*60:]
                    label_data_crop = label_all[i1 * 20:, i2 * 20:, i3 * 60:]
                    d1, d2, d3 = image_data_crop.shape

                    z_scale = d3/newSize3d[2]
                    for i in range(newSize3d[2]):
                        z_temp = int(i*z_scale)
                        temp_img = image_data_crop[:, :, z_temp]
                        new_slice_img = cv2.resize(temp_img, dsize=(newSize3d[0], newSize3d[1]))
                        new_image[:, :, i] = new_slice_img[:, :]
                        temp_label = label_data_crop[:, :, z_temp]
                        new_slice_label = np.zeros([newSize3d[0], newSize3d[1]])
                        for classIndex in range(1,8):
                            temp_class = temp_label == classIndex
                            temp_temp = cv2.resize(temp_class*1.0, dsize=(newSize3d[0], newSize3d[1]))
                            new_slice_label = new_slice_label + classIndex*(temp_temp>0.5)
                        new_label[:, :, i] = new_slice_label[:, :]

                    for rr in range(4):# rotation
                        rot_image = np.rot90(new_image, rr, (0, 1))
                        rot_label = np.rot90(new_label, rr, (0, 1))
                        for ff in range(2): #flip
                            if ff == 0:
                                flip_label = rot_label
                                flip_image = rot_image
                            else:
                                flip_label = np.flip(rot_label, ff)
                                flip_image = np.flip(rot_image, ff)

                            #saveImage2D(flip_image, flip_label, output_dir)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+image_name + '.npy'), flip_image)
                            np.save(os.path.join(output_dir, str(cur_num)+'_p1_r'+str(rr)+'_f'+str(ff)+'_c'+str((i1+1)*100+i2*10+i3)+'_'+label_name + '.npy'), flip_label)

def preprocess_2d_heart17_3class(root_dir, output_dir):
    classes = 3  ## blood, boundary, and background
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        print(label_name)
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        d1,d2,d3 = image_data.shape

        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        #label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        blood_A = blood_LV*1.0 + blood_RV*1.0+ blood_LA*1.0+ blood_RA*1.0+blood_AO*1.0+blood_PA*1.0
        print(np.sum(blood_A))
        bbox_min, bbox_max = getBBox(blood_A)
        z_min = int(bbox_min[2])
        z_max = int(bbox_max[2])

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias
        newSize2d = (int(d1), int(d2))
        new_label = np.zeros((newSize2d))
        new_image = np.zeros((newSize2d))
        covN = 7
        scharr = np.ones([covN,covN])
        for ii in range(z_min, z_max+1):
            #saveImage2D(flip_image, flip_label, output_dir)
            #blood_A_boundry_temp = np.zeros(newSize2d)
            new_label[:,:] = blood_A[:,:,ii]*1.0
            all_blood = signal.convolve2d(new_label, scharr, boundary='symm', mode='same')
            all_blood = all_blood == covN*covN
            blood_A_boundry_temp = new_label - all_blood
            new_label = all_blood * 1.0 + blood_A_boundry_temp * 2.0

            new_image[:, :] = image_data[:, :, ii]
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+image_name + '.npy'), new_image)
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+label_name + '.npy'), new_label)
            '''
            fig = plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(new_image, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.subplot(1, 2, 2)
            plt.imshow(new_label, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.savefig(os.path.join(output_dir, label_name + "_" + str(ii) + ".png"))
            plt.close(fig)
            '''

def preprocess_2d_heart17_4class(root_dir, output_dir):
    classes = 3  ## blood, boundary, and background
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        print(label_name)
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        d1,d2,d3 = image_data.shape

        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        #label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        blood_A = blood_LV*1.0 + blood_RV*1.0+ blood_LA*1.0+ blood_RA*1.0+blood_AO*1.0+blood_PA*1.0
        print(np.sum(blood_A))
        bbox_min, bbox_max = getBBox(label)
        z_min = int(bbox_min[2])
        z_max = int(bbox_max[2])

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias
        newSize2d = (int(d1), int(d2))
        new_label = np.zeros((newSize2d))
        new_image = np.zeros((newSize2d))
        covN = 7
        scharr = np.ones([covN,covN])
        for ii in range(z_min, z_max+1):
            #saveImage2D(flip_image, flip_label, output_dir)
            #blood_A_boundry_temp = np.zeros(newSize2d)
            new_label[:,:] = blood_A[:,:,ii]*1.0
            all_blood = signal.convolve2d(new_label, scharr, boundary='symm', mode='same')
            all_blood = all_blood == covN*covN
            blood_A_boundry_temp = new_label - all_blood
            new_label = all_blood * 1.0 + blood_A_boundry_temp * 2.0 +blood_myo[:,:,ii] * 3.0

            new_image[:, :] = image_data[:, :, ii]
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+image_name + '.npy'), new_image)
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+label_name + '.npy'), new_label)
            '''
            fig = plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(new_image, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.subplot(1, 2, 2)
            plt.imshow(new_label, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.savefig(os.path.join(output_dir, label_name + "_" + str(ii) + ".png"))
            plt.close(fig)
            '''

def preprocess_2d_heart17_8class(root_dir, output_dir):
    classes = 3  ## blood, boundary, and background
    new_size = 64

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)

    for label_name in label_files:

        print(label_name)
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        d1,d2,d3 = image_data.shape

        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        bbox_min, bbox_max = getBBox(label)
        z_min = int(bbox_min[2])
        z_max = int(bbox_max[2])

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        ## add some random bias
        newSize2d = (int(d1), int(d2))
        new_label = np.zeros((newSize2d))
        new_image = np.zeros((newSize2d))
        for ii in range(z_min, z_max+1):
            #saveImage2D(flip_image, flip_label, output_dir)
            new_label = label_all[:, :, ii]
            new_image[:, :] = image_data[:, :, ii]
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+image_name + '.npy'), new_image)
            np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+label_name + '.npy'), new_label)
            '''
            fig = plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.imshow(new_image, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.subplot(1, 2, 2)
            plt.imshow(new_label, vmin=0, vmax=2)  # , cmap='Greys_r')
            plt.savefig(os.path.join(output_dir, label_name + "_" + str(ii) + ".png"))
            plt.close(fig)
            '''

def preprocess_2d_heart17_8class_8input(root_dir, output_dir):
    classes = 3  ## blood, boundary, and background
    new_size = 64

    segDir = os.path.join("data/heart17_sameSize", "segResult")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    image_label_dir =  root_dir
    label_files = subfiles(image_label_dir, suffix="label.nii.gz", join=False)
    file_num = len(label_files)
    crop_para_file =os.path.join(segDir, "save_npy_bbox.npy")
    cropInfo =np.load(crop_para_file)

    for label_name in label_files:

        print(label_name)
        label_nii = nib.load(os.path.join(image_label_dir, label_name))
        label = label_nii.get_fdata()
        cur_num = int(label_name[11])*10 + int(label_name[12])
        image_name = label_name.replace("label", "image")
        image_nii = nib.load(os.path.join(image_label_dir, image_name))
        image_data = image_nii.get_fdata()
        d1,d2,d3 = image_data.shape

        blood_LV = label == 500.0
        blood_RV = label == 600.0
        print(np.sum(blood_RV))
        blood_LA = label == 420.0
        blood_RA = label == 550.0
        print(np.sum(blood_RA))
        blood_myo = label== 205.0
        blood_AO = label == 820.0
        blood_PA = label == 850.0
        print(np.sum(blood_myo))
        label_all = blood_LV*1.0 + blood_RV*2.0+ blood_LA*3.0+ blood_RA*4.0+blood_myo*5.0+blood_AO*6.0+blood_PA*7.0

        # normalization image
        mean = np.mean(image_data)
        std = np.std(image_data)
        image_data = (image_data-mean)/std

        pred_name = str(1000+cur_num)+"Labels.nii.gz"
        pred_label = nib.load(os.path.join(segDir, pred_name))
        pred_label = pred_label.get_fdata()
        originalCrop = (cropInfo[cur_num - 1, :])
        pred_label_Origin = np.zeros([int(originalCrop[0]), int(originalCrop[1]), int(originalCrop[2])])
        pred_label_Origin[int(originalCrop[3]):int(originalCrop[4]), int(originalCrop[5]):int(originalCrop[6]), \
            int(originalCrop[7]):int(originalCrop[8])] = pred_label[0:-1, 0:-1, 0:-1]
        bbox_min, bbox_max = getBBox(pred_label_Origin)
        z_min = int(bbox_min[2])
        z_max = int(bbox_max[2])

        ## add some random bias
        newSize2d = (int(d1), int(d2))
        new_label = np.zeros((newSize2d))
        new_image = np.zeros((int(d1), int(d2), 8))
        class_radis = np.array([8, 8, 16, 16, 8, 8, 8])
        z_up_down = int(d3/64)*2
        for ii in range(int(np.max([0, z_min-z_up_down])), int(np.min([d3, z_max+1+z_up_down]))):
            #saveImage2D(flip_image, flip_label, output_dir)
            new_label = label_all[:, :, ii]

            new_image[:, :, 0] = image_data[:, :, ii]
            pred_label_slice = pred_label_Origin[:, :, ii]
            for classIndex in range(1,8):
                class_mask = pred_label_slice == classIndex
                covN = class_radis[classIndex-1]
                scharr = np.ones([covN, covN])
                boundary_extract = signal.convolve2d(class_mask, scharr, boundary='symm', mode='same')
                boundary_extract = boundary_extract > 0  #== covN * covN
                print(classIndex, np.sum(boundary_extract))
                new_image[:,:,classIndex] = boundary_extract
            #np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+image_name + '.npy'), new_image)
            #np.save(os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+label_name + '.npy'), new_label)
            newimage = nib.Nifti1Image(new_image, np.eye(4))
            nib.save(newimage, os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+image_name))
            newimage = nib.Nifti1Image(new_label, np.eye(4))
            nib.save(newimage, os.path.join(output_dir, str(cur_num)+'_'+str((ii))+'_'+label_name))

            fig = plt.figure(figsize=(8, 8))
            plt.subplot(3, 3, 1)
            plt.imshow(new_label, vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 2)
            plt.imshow(new_image[:,:,0])  # , cmap='Greys_r')
            plt.subplot(3, 3, 3)
            plt.imshow(new_image[:,:,1], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 4)
            plt.imshow(new_image[:,:,2], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 5)
            plt.imshow(new_image[:,:,3], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 6)
            plt.imshow(new_image[:,:,4], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 7)
            plt.imshow(new_image[:,:,5], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 8)
            plt.imshow(new_image[:,:,6], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.subplot(3, 3, 9)
            plt.imshow(new_image[:,:,7], vmin=0, vmax=7)  # , cmap='Greys_r')
            plt.savefig(os.path.join(output_dir, label_name + "_" + str(ii) + ".png"))
            plt.close(fig)
