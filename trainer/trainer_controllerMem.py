import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import datetime
import numpy as np

import json
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from models.model_controllerMem import model_controllerMem

from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
import time
import tqdm
import utils
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
class Trainer():
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the memory writing controller.
        :param config: configuration parameters (see train_controllerMem.py)
        """

        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_tensorboard = 'runs/runs-createMem/'
        self.folder_test = 'training/training_controller/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('creating dataset...')

        self.dim_clip = 180
        self.pred_len=config.future_len
        train_path = get_dset_path(config.dataset_name, "train")  # train
        val_path = get_dset_path(config.dataset_name, "val")  ##test


        logging.info("Initializing train dataset")
        train_dset, train_loader = data_loader(config, train_path)
        logging.info("Initializing val dataset")
        test_set, val_loader = data_loader(config, val_path)

        self.data_train=train_dset
        self.data_test=test_set
        self.train_loader = train_loader
        self.test_loader = val_loader



        print('dataset created')
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.best_k,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "th": config.th
        }
        self.max_epochs = config.max_epochs
        # load pretrained model and create memory model
        self.model_ae = torch.load(config.model_ae)
        self.mem_n2n = model_controllerMem(self.settings, self.model_ae)
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()
        self.best_ade=100

        self.mem_past = 0
        self.mem_fut = 0

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: {}'.format(self.mem_n2n.name_model), 0)
        self.writer.add_text('Training Configuration', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'batch_size: {}'.format(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: {}'.format(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: {}'.format(self.config.dim_embedding_key), 0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """

        self.file.write('points of past track: {}'.format(self.config.past_len) + '\n')
        self.file.write('points of future track: {}'.format(self.config.future_len) + '\n')
        self.file.write('train size: {}'.format(len(self.data_train)) + '\n')
        self.file.write('test size: {}'.format(len(self.data_test)) + '\n')
        self.file.write('batch size: {}'.format(self.config.batch_size) + '\n')
        self.file.write('learning rate: {}'.format(self.config.learning_rate) + '\n')
        self.file.write('embedding dim: {}'.format(self.config.dim_embedding_key) + '\n')

    def fit(self):
        """
        Writing controller training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = True

        # Memory Initialization
        self.mem_n2n.init_memory(self.data_train)####?
        self.mem_past=self.mem_n2n.memory_past
        self.mem_fut=self.mem_n2n.memory_fut


        from torch.nn import init

        self.save_plot_controller(0)
        init.uniform_( self.mem_n2n.linear_controller.weight, a=0.99, b=1.0)
        init.constant_(self.mem_n2n.linear_controller.bias, val=0.0)
        print(self.mem_n2n.linear_controller.weight)
        # Main training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            #self.mem_n2n.init_memory(self.data_train)############
            self.mem_n2n.memory_past= self.mem_past
            self.mem_n2n.memory_fut = self.mem_fut
            self.mem_n2n.pred_gt = torch.zeros((21, 12, 2)).cuda()
            self.mem_n2n.past_gt = torch.zeros((21, 8, 2)).cuda()
            self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
            self.mem_n2n.past_rel_gt = torch.zeros((21, 8, 2)).cuda()

            print('epoch: ' + str(epoch))
            start = time.time()
            loss = self._train_single_epoch()
            end = time.time()

            print('Epoch took: {} Loss: {}'.format(end - start, loss))
            #self.save_plot_controller(epoch)

            #if (epoch + 1) % 5 != 0:
            # Test model while training
            print('start test')
            start_test = time.time()
            dict_metrics_test,flag = self.evaluate(self.test_loader, epoch + 1)#self.test_loader
            end_test = time.time()
            print('Test took: {}'.format(end_test - start_test))

            # Tensorboard summary: test
            #self.writer.add_scalar('accuracy_test/ade', dict_metrics_test['eucl_mean'], epoch)
            #self.writer.add_scalar('accuracy_test/fde', dict_metrics_test['horizon40s'], epoch)


            # print memory on tensorboard
            mem_size = self.mem_n2n.memory_past.shape[0]

            '''for i in range(mem_size):
                track_mem = self.mem_n2n.check_memory(i).squeeze(0).cpu().detach().numpy()
                plt.plot(track_mem[:, 0], track_mem[:, 1], marker='o', markersize=1)
            plt.axis('equal')
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)
            self.writer.add_image('memory_content/memory', image.squeeze(0), epoch)
            plt.close()'''
            # Save model checkpoint
            if flag:
                torch.save(self.mem_n2n,
                           self.folder_test + 'model_controller_best_' + str(epoch) + '_' + self.name_test)

                # save results in a file .txt
                self.save_results(dict_metrics_test, epoch=epoch + 1)

            for name, param in self.mem_n2n.named_parameters():

                self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_controller_' + self.name_test)

    def save_plot_controller(self, epoch):
        """
        plot the learned threshold bt writing controller
        :param epoch: epoch index (default: 0)
        :return: None
        """

        fig = plt.figure()
        x = torch.Tensor(np.linspace(0, 1, 100))
        weight = self.mem_n2n.linear_controller.weight.cpu()
        bias = self.mem_n2n.linear_controller.bias.cpu()
        y = torch.sigmoid(weight * x + bias).squeeze()
        plt.plot(x.data.numpy(), y.data.numpy(), '-r', label='y=' + str(weight.item()) + 'x + ' + str(bias.item()))
        plt.plot(x.data.numpy(), [0.5] * 100, '-b')
        plt.title('controller')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        self.writer.add_image('controller_plot/function', image.squeeze(0), epoch)
        plt.close(fig)

    def save_results(self, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')
        #self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.best_k) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')
        self.file.write("TRAIN size: " + str(len(self.data_train)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')

        #self.file.write("FDE : " + str(dict_metrics_test['horizon40s'].item()) + '\n')
        #self.file.write("ADE : " + str(dict_metrics_test['eucl_mean'].item()) + '\n')

        self.file.close()


    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """
        self._memory_writing()
        data_len=0
        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0
            ade = utils.AverageMeter("ADE", ":.6f")
            fde = utils.AverageMeter("FDE", ":.6f")
            ade_outer, de_outer, mse_outer = [], [], []
            progress = utils.ProgressMeter(len(loader), [ade, fde], prefix="Test: ")

            total_traj = 0
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

                data_len = data_len+past.size(0)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    past_rel = past_rel.cuda()
                    future_rel = future_rel.cuda()
                pred = self.mem_n2n(past_rel,obs_traj=past,pred_gt=future)
                ade1, de1, mse1 = [], [], []

                total_traj += pred_traj_gt.size(1)


                for topki in range(self.config.best_k):  # topk=20( num_samples is topk )
                    # multi-modal
                    pred_traj_fake = pred[:,topki]#([729, 20, 12, 2])
                    #pred_traj_fake_rel = pred_traj_fake_rel[-self.pred_len:]
                    pred_traj_fake = pred_traj_fake.transpose(1,0) #relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
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
            fde1 = sum(de_outer) / (total_traj)
            mse1 = sum(mse_outer) / (total_traj)

            #ade.update(ade1, obs_traj.shape[1])
            #fde.update(de1, obs_traj.shape[1])
            if self.best_ade > ade1:
                self.best_ade=ade1
                flag=True
                print('best:', self.best_ade, "memory size: " + str(self.mem_n2n.memory_past.size()))
                print('pred:', self.mem_n2n.pred_rel_gt.shape)

            else:
                flag=False
            print(" *  ADE  :", ade1, " FDE  :", fde1)

            dict_metrics['eucl_mean'] = ade1
            dict_metrics['horizon40s'] = fde1
            self.writer.add_scalar('memory_size/memory_size_test', len(self.mem_n2n.memory_past), epoch)

        return dict_metrics,flag

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        data_len=0

        for step,batch in enumerate(tqdm.tqdm(self.train_loader)):

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



            data_len = data_len + past.size(0)
            if self.config.cuda:
                past = past.cuda()
                future = future.cuda()
                past_rel = past_rel.cuda()
                future_rel = future_rel.cuda()


            self.opt.zero_grad()
            prob, sim = self.mem_n2n(past_rel, future_rel,obs_traj=past,pred_gt=future)

            loss = self.ControllerLoss(prob, sim) # (bs,1)

            loss.backward()
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)
        print('s',data_len)

        return loss.item()

    def ControllerLoss(self, prob, sim):
        """
        Loss to train writing controller:
        :param prob: writing probability generated by controller
        :param sim: similarity (between 0 and 1) between better prediction and ground-truth.
        :return: loss
        """
        loss = prob * sim + (1 - prob) * (1 - sim) # prob * (1 - sim) + (1 - prob)* sim #prob * (1 - sim) + (1 - prob)* sim #

        return sum(loss)

    def _memory_writing(self):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        #self.mem_n2n.init_memory(self.data_train)###############????
        self.mem_n2n.memory_past = self.mem_past
        self.mem_n2n.memory_fut = self.mem_fut
        self.mem_n2n.pred_gt = torch.zeros((21,12,2)).cuda() #
        self.mem_n2n.past_gt = torch.zeros((21, 8, 2)).cuda()
        self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
        self.mem_n2n.past_rel_gt= torch.zeros((21, 8, 2)).cuda()



        self.mem_n2n.past_rel_gt = torch.zeros((21, 8, 2)).cuda()
        self.mem_n2n.pred_rel_gt = torch.zeros((21, 12, 2)).cuda()
        config = self.config
        with torch.no_grad():
            for step, batch in enumerate(tqdm.tqdm(self.train_loader)):
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
                    future_rel = future_rel.cuda()

                _, _ = self.mem_n2n(past_rel, future_rel,obs_traj=past,pred_gt=future)
            # save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

    def load(self, directory):
        pass
