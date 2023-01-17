import copy
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
from tensorboardX import SummaryWriter
#import cv2
from random import randint
import numpy as np
import json
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from models.model_memory_IRM import model_memory_IRM
import io
from PIL import Image
from torchvision.transforms import ToTensor

import tqdm
from data.loader import data_loader
from utils import (
    displacement_error,
    final_displacement_error,
    mse_error,
    get_dset_path,
    int_tuple,
    l2_loss,
    relative_to_abs,
)
import logging
def cal_de_mse(pred_traj_gt, pred_traj_fake):
    ade = displacement_error(pred_traj_fake, pred_traj_gt, mode="raw")
    de = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1], mode="raw")
    mse = mse_error(pred_traj_fake, pred_traj_gt)
    return ade, de, mse
def evaluate_helper(error, seq_start_end):
    sum_ = 0

    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_
class Trainer:
    def __init__(self, config):
        """
        Trainer class for training the Iterative Refinement Module (IRM)
        :param config: configuration parameters (see train_IRM.py)
        """

        self.name_run = 'runs/runs-TP/'


        self.num_prediction = config.best_k
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": self.num_prediction,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "th": config.th
        }
        self.max_epochs = config.max_epochs

        # load pretrained model and create memory_model
        self.model = torch.load(config.model)
        self.mem_n2n = model_memory_IRM(self.settings, self.model)
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config
        self.pred_len=config.future_len
        # Write details to file
        self.best_ade = 100
        self.previous_memory_len = {'ETH':0,"STU":0,'ZARA':0}
        self.previous_traj_len = {'ETH':0,"STU":0,'ZARA':0}
        self.dataset_name = 0
        self.dest_path =""

        self.cl_flag=config.CL_flag
        self.task_order=config.task_order

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """
        self.file.write("points of past track: " + str(self.config.past_len) + '\n')
        self.file.write("points of future track: " + str(self.config.future_len) + '\n')
        self.file.write("train size: " + str(len(self.data_train)) + '\n')
        self.file.write("test size: " + str(len(self.data_test)) + '\n')
        self.file.write("batch size: " + str(self.config.batch_size) + '\n')
    def create_dataset(self,dataset_name,config):
        print('creating dataset...')
        train_path = get_dset_path(dataset_name, "train")  # train
        val_path = get_dset_path(dataset_name, "val")  ##test
        val_ = get_dset_path(dataset_name, "val")#val
        logging.info("Initializing train dataset")
        train_dset, train_loader = data_loader(config, train_path)
        logging.info("Initializing test dataset")
        test_set, val_loader = data_loader(config, val_path)

        self.data_train = train_dset

        self.data_test = test_set
        self.train_loader = train_loader

        self.test_loader = val_loader
        print('dataset created')
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'training/training_TP/' +dataset_name+'_'+ self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.name_run + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: ' + self.mem_n2n.name_model, 0)
        self.writer.add_text('Training Configuration', 'dataset train: ' + str(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: ' + str(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'number of prediction: ' + str(self.num_prediction), 0)
        self.writer.add_text('Training Configuration', 'batch_size: ' + str(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: ' + str(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: ' + str(self.settings["dim_embedding_key"]),
                             0)

    def replay_memory(self,dataset_name,replay=40):

        future = self.mem_n2n.pred_gt[21:self.previous_traj_len[dataset_name]]
        past = self.mem_n2n.past_gt[21:self.previous_traj_len[dataset_name]]
        future_rel = self.mem_n2n.pred_rel_gt[21:self.previous_traj_len[dataset_name]]
        past_rel = self.mem_n2n.past_rel_gt[21:self.previous_traj_len[dataset_name]]

        memory_len = past_rel.size(0)
        if memory_len<replay:
            replay = memory_len
        # = replay
        index =torch.randint(memory_len,(replay,1))
        ### delete repeat
        past = past[index].squeeze().reshape(-1, 8, 2)
        future = future[index].squeeze().reshape(-1, 12, 2)

        past_rel = past_rel[index].squeeze().reshape(-1, 8, 2)
        future_rel = future_rel[index].squeeze().reshape(-1, 12, 2)

        # return past[sample_random].squeeze(),future[sample_random].squeeze(), past_rel[sample_random].squeeze(),future_rel[sample_random].squeeze()

        return past, future, past_rel, future_rel

    def fit(self):
        """
        Iterative refinement model training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad =False

        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.decoder_fut.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.FC_output_fut.parameters():
            param.requires_grad = True

        for param in self.mem_n2n.Lin_Q.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_K.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_V.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.Lin_Sgmoid.parameters():
            param.requires_grad = True#True

        data_name = self.task_order #['STU','ETH','ZARA'] # ,,'ETH''ST',

        task_id=0
        dict_task={}
        for data_nm in data_name:
            task_id = task_id+1
            self.best_ade=100
            self.create_dataset(data_nm,config)
            self.dataset_name = data_nm

            print('train:',data_nm)

            start = time.time()
            # Write the memory of current task
            if data_nm!=data_name[0]:
                self.previous_traj_len[data_nm]=int(self.mem_n2n.pred_gt.shape[0])
                self._memory_writing(self.config.saved_memory)
            else:
                # save memory
                torch.save(self.mem_n2n.memory_past, str(self.dataset_name) + '_memory_past.pt')
                torch.save(self.mem_n2n.memory_fut, str(self.dataset_name) + '_memory_fut.pt')

            self.writer.add_text('Training Configuration', 'memory size: ' + str(len(self.mem_n2n.memory_past)), 0)
            end = time.time()
            print('writing time: ' + str(end-start))
            dict_out = {}
            for epoch in range(self.start_epoch, config.max_epochs):
                self.mem_n2n.train()

                print('epoch: ' + str(epoch))
                start = time.time()
                loss = self._train_single_epoch()
                end = time.time()
                print('Epoch took: {} Loss: {}'.format(end - start, loss))

                #if (epoch + 1) in step_results:
                # Test model while training
                print('start test')
                start_test = time.time()

                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)
                #self.testloader


                end_test = time.time()
                print('Test took: {}'.format(end_test - start_test))

                # Save model checkpoint
                if dict_metrics_test['euclMean'].item()<self.best_ade:
                    dict_all={}
                    for id in range(task_id-1):
                        val_path = get_dset_path(data_name[id], "val")  ##test
                        test_set, val_loader = data_loader(config, val_path)  # self.testloader
                        dict_metrics_val = self.evaluate(val_loader, epoch + 1)
                        dict_all[id]=dict_metrics_val
                        self.save_results('task '+str(id+1)+'val', dict_metrics_val, epoch=epoch + 1)

                    for i in range(task_id-1):
                        dict_out[i] = dict_all[i]
                        print('task '+str(i+1)+' val ade:', dict_all[i]['euclMean'], 'fde:', dict_all[i]['horizon40s'])
                    dict_out[task_id-1]=dict_metrics_test
                    print('task ' + str(task_id) + ' ade:', dict_metrics_test['euclMean'], 'fde:',
                          dict_metrics_test['horizon40s'])

                    self.best_ade=dict_metrics_test['euclMean'].item()
                    torch.save(self.mem_n2n, self.folder_test + 'model_IRM_epoch_' + str(epoch) + '_' + self.name_test)
                    self.save_results('test',dict_metrics_test, epoch=epoch + 1)
                    #self.save_results('val',dict_metrics_val, epoch=epoch + 1)



                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

            # Save final trained model
            torch.save(self.mem_n2n, self.folder_test + 'model_mantra_' + self.name_test)
            #print(task_id,dict_out)
            dict_task[task_id-1]=dict_out
        print(dict_task)
        #
        aer_ade=0
        aer_fde=0
        fgt_ade=0
        fgt_fde=0
        task_num=len(data_name)

        for t in range(task_num-1):

                for j in range(t+1,task_num):
                    fgt_ade = fgt_ade+dict_task[j][t]['euclMean']-dict_task[t][t]['euclMean']
                    fgt_fde = fgt_fde + dict_task[j][t]['horizon40s'] - dict_task[t][t]['horizon40s']

        for t in range(task_num):
            cur_task=dict_task[t]

            for jj in range( len(cur_task)):
                aer_ade = aer_ade + cur_task[jj]['euclMean']
                aer_fde = aer_fde + cur_task[jj]['horizon40s']
        aer_ade=aer_ade/(task_num*(task_num+1)/2)
        fgt_ade = fgt_ade / (task_num * (task_num - 1) / 2)
        aer_fde = aer_fde / (task_num * (task_num + 1) / 2)
        fgt_fde = fgt_fde / (task_num * (task_num - 1) / 2)
        print(task_num,task_num * (task_num - 1) / 2)
        print('FGT: ade: ',fgt_ade,' fde: ',fgt_fde)
        print('AER: ade: ',aer_ade,' fde: ',aer_fde)


    def save_results(self,name, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test +name+ "results.txt", "w")
        self.file.write("TEST:" + '\n')
        #self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.best_k) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')


        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')

        self.file.write("ADE 4s: " + str(dict_metrics_test['euclMean'].item()) + '\n')

        self.file.close()



    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """

        self.mem_n2n.eval()
        data_len=0
        with torch.no_grad():
            dict_metrics = {}
            ade = utils.AverageMeter("ADE", ":.6f")
            fde = utils.AverageMeter("FDE", ":.6f")
            ade_outer, de_outer, mse_outer = [], [], []
            progress = utils.ProgressMeter(len(loader), [ade, fde], prefix="Test: ")

            total_traj = 0
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            for step, batch in enumerate(tqdm.tqdm(loader)):
                batch = [tensor.cuda() for tensor in batch]
                (
                    obs_traj,
                    pred_traj_gt,
                    obs_traj_rel,
                    pred_traj_gt_rel,
                    non_linear_ped,
                    loss_mask,
                    seq_start_end,
                ) = batch

                past = Variable(obs_traj)
                past = past.transpose(1, 0)
                future = Variable(pred_traj_gt)
                future = future.transpose(1, 0)

                past_rel = Variable(obs_traj_rel)
                past_rel = past_rel.transpose(1, 0)
                future_rel = Variable(pred_traj_gt_rel)
                future_rel = future_rel.transpose(1, 0)

                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    past_rel = past_rel.cuda()
                    future_rel = future_rel.cuda
                pred = self.mem_n2n(past_rel,obs_traj=past)
                ade1, de1, mse1 = [], [], []

                total_traj += pred_traj_gt.size(1)

                for topki in range(self.config.best_k):  # topk=20( num_samples is topk )
                    # multi-modal
                    pred_traj_fake = pred[:, topki]  # ([729, 20, 12, 2])
                    # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                    pred_traj_fake = pred_traj_fake.transpose(1,0)  # relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
                    ade_, de_, mse_ = cal_de_mse(pred_traj_gt, pred_traj_fake)
                    ade1.append(ade_)
                    de1.append(de_)
                    mse1.append(mse_)
                # print('ff',mem.size())
                ade_sum = evaluate_helper(ade1, seq_start_end)
                de_sum = evaluate_helper(de1, seq_start_end)
                mse_sum = evaluate_helper(mse1, seq_start_end)

                ade_outer.append(ade_sum)
                de_outer.append(de_sum)
                mse_outer.append(mse_sum)

            ade1 = sum(ade_outer) / (total_traj * self.pred_len)
            de1 = sum(de_outer) / (total_traj)
            mse1 = sum(mse_outer) / (total_traj)

            '''ade.update(ade1, obs_traj.shape[1])
            fde.update(de1, obs_traj.shape[1])'''


            dict_metrics['euclMean'] = torch.tensor(ade1,dtype=torch.float)#ade.avg
            dict_metrics['horizon40s'] = torch.tensor(de1,dtype=torch.float)

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        self.mem_n2n.train()
        self.iterations=0
        repaly_freq=int(len(self.train_loader)/3)#/3
        for step, batch in enumerate(tqdm.tqdm(self.train_loader)):#self.train_loader

            batch_cp=copy.deepcopy(batch)
            if self.cl_flag:
                if self.dataset_name!=self.task_order[0] and step%repaly_freq==0: #step+1
                    self.opt.zero_grad()
                    # memory replay : from preivous data include(trajecory)
                    past_replay, future_replay, past_rel_replay, future_rel_replay = self.replay_memory(self.dataset_name,replay=80)#80
                    past_replay = Variable(past_replay).cuda()
                    future_replay = Variable(future_replay).cuda()
                    past_rel_replay = Variable(past_rel_replay).cuda()

                    output_replay = self.mem_n2n(past_rel_replay, obs_traj=past_replay)
                    # args.pred_len, mem, epoch,
                    loss_sum_replay = torch.zeros(1).cuda()
                    rmse_all = []
                    for topki in range(config.best_k):
                        pred_traj_fake_train = output_replay[:, topki]  # ([729, 20, 12, 2])
                        # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]

                        # rmse
                        batch, seq_len, _ = pred_traj_fake_train.size()

                        # equation below , the first part do noing, can be delete

                        loss_f = (pred_traj_fake_train - future_replay) ** 2

                        rmse =(loss_f.sum(dim=2).sum(dim=1) / seq_len).reshape(-1, 1)

                        rmse_all.append(rmse)

                    rmse_ = torch.stack(rmse_all, dim=1).squeeze()
                    #print(.shape)

                    best, _ = torch.min(rmse_, dim=1)
                    loss_sum_replay = best.sum()
                    loss_sum_replay.backward()
                    # torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
                    self.opt.step()

            batch = [tensor.cuda() for tensor in batch_cp]
            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            self.iterations += 1
            past = Variable(obs_traj)
            past = past.transpose(1, 0)
            future = Variable(pred_traj_gt)
            future = future.transpose(1, 0)

            past_rel = Variable(obs_traj_rel)
            past_rel = past_rel.transpose(1, 0)
            future_rel = Variable(pred_traj_gt_rel)
            future_rel = future_rel.transpose(1, 0)

            if self.config.cuda:
                past = past.cuda()
                future = future.cuda()
                past_rel = past_rel.cuda()
                future_rel = future_rel.cuda
            self.opt.zero_grad()

            output = self.mem_n2n(past_rel,obs_traj=past)
            loss = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = []
            loss_mask = loss_mask[:, config.past_len:]


            # args.pred_len, mem, epoch,
            for topki in range(config.best_k):
                pred_traj_fake= output[:, topki]  # ([729, 20, 12, 2])
                # pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                pred_traj_fake = pred_traj_fake.transpose(1, 0)
                l2_loss_rel.append(
                    l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode="raw")
                )

            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1) #(bs,20)

            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)

                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]

                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake.shape[0]) * (end - start)
                )
                l2_loss_sum_rel = l2_loss_sum_rel + _l2_loss_rel


            loss = loss + l2_loss_sum_rel

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        if not self.cl_flag:
             self.mem_n2n.init_memory(self.data_train)
        if saved_memory:
            print('memories of pretrained model')
            print('old memory size: ', self.mem_n2n.memory_past.shape)
            with torch.no_grad():
                #### train set
                for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
                    self.iterations += 1
                    batch = [tensor.cuda() for tensor in batch]
                    (
                        obs_traj,
                        pred_traj_gt,
                        obs_traj_rel,
                        pred_traj_gt_rel,
                        non_linear_ped,
                        loss_mask,
                        seq_start_end,
                    ) = batch

                    past = Variable(obs_traj)
                    past = past.transpose(1, 0)
                    future = Variable(pred_traj_gt)
                    future = future.transpose(1, 0)

                    past_rel = Variable(obs_traj_rel)
                    past_rel = past_rel.transpose(1, 0)
                    future_rel = Variable(pred_traj_gt_rel)
                    future_rel = future_rel.transpose(1, 0)

                    if self.config.cuda:
                        past = past.cuda()
                        future_ = future.cuda()
                        past_rel = past_rel.cuda()
                        future_rel = future_rel.cuda()

                    self.mem_n2n.write_in_memory(past_rel, future=future_rel, obs_traj=past, pred_gt=future_)
            print('old memory size: ', self.mem_n2n.memory_past.shape)
        else:
            self.mem_n2n.init_memory(self.data_train)
            #config = self.config
            with torch.no_grad():
                #### train set

                for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
                    self.iterations += 1
                    batch = [tensor.cuda() for tensor in batch]
                    (
                        obs_traj,
                        pred_traj_gt,
                        obs_traj_rel,
                        pred_traj_gt_rel,
                        non_linear_ped,
                        loss_mask,
                        seq_start_end,
                    ) = batch

                    past = Variable(obs_traj)
                    past = past.transpose(1, 0)
                    future = Variable(pred_traj_gt)
                    future = future.transpose(1, 0)

                    past_rel = Variable(obs_traj_rel)
                    past_rel = past_rel.transpose(1, 0)
                    future_rel = Variable(pred_traj_gt_rel)
                    future_rel = future_rel.transpose(1, 0)

                    if self.config.cuda:
                        past = past.cuda()
                        future_ = future.cuda()
                        past_rel = past_rel.cuda()
                        future_rel = future_rel.cuda()

                    self.mem_n2n.write_in_memory(past_rel, future=future_rel,obs_traj=past,pred_gt=future_)

        # save memory
        #torch.save(self.mem_n2n.memory_past, str(self.dataset_name) +'_memory_past.pt')
        #torch.save(self.mem_n2n.memory_fut,str(self.dataset_name) +'_memory_fut.pt')

