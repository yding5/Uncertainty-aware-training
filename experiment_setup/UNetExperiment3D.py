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

import os
import pickle
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from torch.utils.data import DataLoader
from data_prepare.data_loader.dataset_heart17 import dataset_heart17
from util.make_dot import make_dot
from subprocess import check_call

from network_models.RecursiveUNet3D import UNet3D
from util.metrics import *
from util.saveProcess import SaveAllCrossVali

import sys
#sys.path.append("..")
from experiment_setup.experiment_base.pytorch_experiment import PytorchExperiment
import datetime



SMOOTH = 1e-5



class UNetExperiment3D(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        pkl_dir = self.config.cross_vali_dir
        with open(os.path.join(pkl_dir, self.config.expName+"_splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys,tr_keys_label = splits[self.config.cross_vali_index]['train'], splits[self.config.cross_vali_index]['train_label']
        val_keys,val_keys_label = splits[self.config.cross_vali_index]['val'], splits[self.config.cross_vali_index]['val_label']
        test_keys,test_keys_label = splits[self.config.cross_vali_index]['test'], splits[self.config.cross_vali_index]['test_label']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.dset_train = dataset_heart17(folder_dataset=self.config.data_train_dir, image_list=tr_keys,label_list=tr_keys_label)
        self.train_data_loader = DataLoader(self.dset_train, batch_size=self.config.batch_size, shuffle=True, num_workers=1)
        self.dset_vali = dataset_heart17(folder_dataset=self.config.data_vali_dir, image_list=val_keys,label_list=val_keys_label)
        self.val_data_loader = DataLoader(self.dset_vali, batch_size=self.config.batch_size, shuffle=True, num_workers=1)
        self.testset = dataset_heart17(folder_dataset=self.config.data_test_dir, image_list=test_keys,label_list=test_keys_label)
        self.test_data_loader = DataLoader(self.testset, batch_size=1, shuffle=False, num_workers=1)
        print(len(self.test_data_loader))
        self.model = UNet3D(num_classes=self.config.num_classes, in_channels=self.config.in_channels,initial_filter_size=self.config.initial_filter_size)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of para:{}'.format(pytorch_total_params))
        self.model.to(self.device)

        self.ce_loss = torch.nn.CrossEntropyLoss(weight = torch.Tensor(self.config.segWeights).to(self.device))  # Kein Softmax für CE Loss -> ist in torch schon mit drin!
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_file, save_types=("model"), path=self.config.checkpoint_dir)

        self.elog.print('Experiment set up.')

    def get_loss(self, pred, pred_softmax, target, ranking_criterion, ranking_criterion_m1, ranking_criterion_m1_sum, epoch):
        if self.config.loss_type == 0:
            if self.config.batch_size > 1:
                loss = self.ce_loss(pred, target.squeeze())
            else:
                loss = self.ce_loss(pred, target)
        elif self.config.loss_type == 1:
            rankingLoss = 0

            temp = torch.max(pred_softmax,dim=1,keepdim=False)
            max_prob = torch.flatten(temp[0])
            pred_ind = torch.flatten(temp[1])
            flattened_target = torch.flatten(target.squeeze())

            correctInd = torch.eq(pred_ind,flattened_target)
            wrongInd = ~correctInd 
            correctPred = max_prob[correctInd]
            wrongPred = max_prob[wrongInd]

            if len(correctPred) > 0 and len(correctPred) < len(pred_ind):
                numCorr = correctPred.size()[0]
                numWrong = wrongPred.size()[0]
                rankingPairListC = []
                rankingPairListW = []

                correct_idx = torch.randperm(len(correctPred))
                correct_idx = correct_idx.cuda()
                wrong_idx = torch.randperm(len(wrongPred))
                wrong_idx = wrong_idx.cuda()

                num_sub_Wrong = 300
                num_sub_Corr = 6600

                if len(correctPred) > num_sub_Corr:
                    correctPred = correctPred.index_select(0, correct_idx[:num_sub_Corr])
                if len(wrongPred) > num_sub_Wrong:
                    wrongPred = wrongPred.index_select(0, wrong_idx[:num_sub_Wrong])

                for i in range(len(correctPred)):
                    rankingPairListC.append(correctPred[i].repeat(len(wrongPred)))
                    rankingPairListW.append(wrongPred)

                rankingPairC = torch.cat(rankingPairListC,0)
                rankingPairW = torch.cat(rankingPairListW,0)

                rankingTarget = torch.FloatTensor(rankingPairC.size()).fill_(1)
                rankingTarget = rankingTarget.cuda()
                rankingLoss = ranking_criterion_m1(rankingPairC, rankingPairW, rankingTarget)
            ceLoss = self.ce_loss(pred, target.squeeze())
            loss = ceLoss + self.config.loss_weight*rankingLoss
        else:
            raise NotImplementedError
        return loss

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        print('self.exp_ID: {}'.format(self.exp_ID))
        self.model.train()

        ranking_criterion = torch.nn.MarginRankingLoss().cuda()
        ranking_criterion_m1 = torch.nn.MarginRankingLoss(margin=0.1).cuda()
        ranking_criterion_m1_sum = torch.nn.MarginRankingLoss(margin=0.1, reduction='sum').cuda()

        batch_counter = 0
        for data, target, filenames in self.train_data_loader:

            self.optimizer.zero_grad()

            # Move data and target to the GPU
            data = data.float().to(self.device)
            target = target.long().to(self.device)

            pred = self.model(data)
            pred_softmax = F.softmax(pred, dim=1) 

            loss = self.get_loss(pred, pred_softmax, target, ranking_criterion, ranking_criterion_m1, ranking_criterion_m1_sum, epoch=epoch)
                
            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            self.elog.print('Epoch: %d Batch %d in %d Loss: %.4f' % (self._epoch_idx, batch_counter, len(self.train_data_loader.dataset)/self.config.batch_size, loss))

            '''
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                #self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))
                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss',
                                counter=epoch + (batch_counter / self.dset_train.num_batches))
                self.clog.show_image_grid(data[:,:,30].float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target[:,:,:].float(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True)[:,:,30], name="unt_argmax", title="Unet", n_iter=epoch)
            '''
            batch_counter += 1

    def validate(self, epoch):
        if epoch % 1 != 0:
            return
        self.elog.print('VALIDATE')
        self.model.eval()

        ranking_criterion = torch.nn.MarginRankingLoss().cuda()
        ranking_criterion_m1 = torch.nn.MarginRankingLoss(margin=0.1).cuda()
        ranking_criterion_m1_sum = torch.nn.MarginRankingLoss(margin=0.1, reduction='sum').cuda()

        data = None
        loss_list = []
        with torch.no_grad():
            for data, target, filenames in self.val_data_loader:
                data = data.float().to(self.device)
                target = target.long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.
                max_prob, max_index = torch.max(pred_softmax,1)

                loss = self.get_loss(pred, pred_softmax, target, ranking_criterion, ranking_criterion_m1, ranking_criterion_m1_sum, epoch=epoch)
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))
        print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

