import torch
import numpy as np

# PyTroch version

SMOOTH = 1e-5


def global_accuracy(outputs, labels):
    outputs = outputs.squeeze().float()
    labels = labels.squeeze().float()
    outputs = torch.flatten(outputs)
    labels = torch.flatten(labels)
    total = len(outputs)
    correct = torch.sum(torch.eq(outputs,labels))
    return correct/total
    


def dice_pytorch(outputs: torch.Tensor, labels: torch.Tensor, N_class):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze().float()
    labels = labels.squeeze().float()
    dice = torch.ones(N_class-1).float()
    ## for test
    #outputs = torch.tensor([[1,1],[3,3]]).float()
    #labels = torch.tensor([[0, 1], [2, 3]]).float()

    for iter in range(1,N_class): ## ignore the background
        predict_temp = torch.eq(outputs, iter)
        label_temp = torch.eq(labels, iter)
        intersection = predict_temp & label_temp
        intersection = intersection.float().sum()
        union = (predict_temp.float().sum() + label_temp.float().sum())

        if intersection>0 and union>0:
            dice_temp = (2*intersection)/(union)
        else:
            dice_temp = 0
        dice[iter-1] = dice_temp #(intersection + SMOOTH) / (union + SMOOTH)

    return dice  # Or thresholded.mean()

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


################ Numpy version ################
# Well, it's the same function, so I'm going to omit the comments

def iou_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze()

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10

    return thresholded  # Or thresholded.mean()


# Numpy version
# Well, it's the same function, so I'm going to omit the comments

def dice_numpy(outputs: np.array, labels: np.array):
    outputs = outputs.squeeze()

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    dice = (intersection + SMOOTH) / (union + SMOOTH)

    return dice  # Or thresholded.mean()
