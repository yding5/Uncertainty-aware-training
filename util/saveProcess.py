import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import nibabel as nib

# PyTroch version

def SaveAllCrossVali2D(self,imgName, savePath, realLabel, predLabel):
    realLabel = np.squeeze(realLabel)
    predLabel = np.squeeze(predLabel)
    label_pre = nib.Nifti1Image(predLabel, affine=np.eye(4))
    NewName = imgName.replace('image', 'label')
    print(imgName, NewName)
    fileSavePath = os.path.join(savePath, NewName)
    nib.save(label_pre, fileSavePath)

def SaveAllCrossVali(self,imgName, savePath, realLabel, predLabel, saveResult):
    realLabel = np.squeeze(realLabel)
    predLabel = np.squeeze(predLabel)
    label_pre = nib.Nifti1Image(predLabel, affine=np.eye(4))
    NewName = imgName.replace('image', 'label')
    fileSavePath = os.path.join(savePath, NewName)
    nib.save(label_pre, fileSavePath)

    # save cca results
    '''
    str_temp = "cca,"
    for j in range(1, 2):
        dataTemp = predLabel == j
        retval, numLabel = ndimage.label(dataTemp)  # cv2.connectedComponents(binary_data)
        mask_area = np.zeros((5))
        area_sum = 0
        largest_index = 0
        for k in range(0, numLabel):
            temp_mask = retval == k + 1
            area_sum_temp = np.sum(temp_mask)
            mask_area[k] = area_sum_temp
            if area_sum < area_sum_temp:
                area_sum = area_sum_temp
                largest_index = k
        mask_area = np.sort(mask_area)
        mask_area = np.flip(mask_area)
        for k in range(0, numLabel):
            str_temp = str_temp + str((np.sum(mask_area[k]))) + ","
        ## iou with the largest
        largest_one =  retval == 1 + largest_index
        real_one = realLabel == j
        iou_and = np.sum(largest_one & real_one)
        iou_or = np.sum(largest_one ^ real_one)
        IoU = 2*iou_and/iou_or
        str_temp = str_temp + "IoU,"+str(IoU) + ","
        ## crop position
        largest_position_list = np.where( retval == 1 + largest_index)
        real_position_list = np.where(realLabel == j)
        shape_largest = largest_position_list[0].shape
        shape_real = real_position_list[0].shape

        largest_position = np.zeros([shape_largest[0],3])
        largest_position[:, 0] = largest_position_list[0][:]
        largest_position[:, 1] = largest_position_list[1][:]
        largest_position[:, 2] = largest_position_list[2][:]
        real_position = np.zeros([shape_real[0], 3])
        real_position[:, 0] = real_position_list[0][:]
        real_position[:, 1] = real_position_list[1][:]
        real_position[:, 2] = real_position_list[2][:]

        largest_min = largest_position.min(0)
        largest_max = largest_position.max(0)
        real_min = real_position.min(0)
        real_max = real_position.max(0)
        str_temp = str_temp+"realMax,"+str(real_max)+"realMin,"+str(real_min)+"predMax,"+str(largest_max)+"predMax,"+str(largest_min)

    print(str_temp)

    # save str (dice result)
    saveStr = 'test file name: %s dice: %s,' % (imgName, str(saveResult))
    saveFile = os.path.join(savePath, "test_results.txt")
    f = open(saveFile, 'a')
    f.writelines(saveStr+str_temp+"\n")
    f.close()
    '''


    '''
    if np.min(saveResult) < 0.9:
        realLabel = realLabel.squeeze()
        predLabel = predLabel.squeeze()
        realLabel = realLabel / np.max(realLabel)
        predLabel = predLabel / np.max(predLabel)

        diff = (realLabel) == (predLabel)
        shape = realLabel.shape

        for i in range(1, shape[2]):
            fig = plt.figure(figsize=(12, 8))
            plt.subplot(1, 3, 1)
            plt.imshow(diff[:, :, i], vmin=0, vmax=1)  # , cmap='Greys_r')
            plt.subplot(1, 3, 2)
            plt.imshow(realLabel[:, :, i], vmin=0, vmax=1)
            plt.subplot(1, 3, 3)
            plt.imshow(predLabel[:, :, i], vmin=0, vmax=1)
            plt.savefig(os.path.join(savePath, imgName + "_" + str(i) + ".png"))
            plt.close(fig)
    '''