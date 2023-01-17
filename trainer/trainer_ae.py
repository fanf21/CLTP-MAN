import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import logging
import torch.nn as nn
#from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_encdec import model_encdec

from torch.autograd import Variable
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


class Trainer:
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        # test folder creating
        self.name_test = str(datetime.datetime.now())[:13]
        self.folder_tensorboard = 'runs/runs-ae/'
        self.folder_test = 'training/training_ae/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('Creating dataset...')
        self.dim_clip = 180
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
        print('Dataset created')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
        }
        self.max_epochs = config.max_epochs

        # model
        self.mem_n2n = model_encdec(self.settings)

        # loss
        self.criterionLoss = nn.MSELoss()

        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

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
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config
        # Training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            if (epoch + 0) % 2 == 0: # 20
                print('test on train dataset')
                dict_metrics_train = self.evaluate(self.train_loader, epoch + 1)

                print('test on TEST dataset')
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)

                # Tensorboard summary: learning rate
                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                # Tensorboard summary: train
                self.writer.add_scalar('accuracy_train/eucl_mean', dict_metrics_train['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon10s', dict_metrics_train['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon20s', dict_metrics_train['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon30s', dict_metrics_train['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon40s', dict_metrics_train['horizon40s'], epoch)

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/eucl_mean', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)

                # Tensorboard summary: model weights
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        dict_metrics = {}

        # Loop over samples
        data_len =0
        with torch.no_grad():
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

                past_rel = Variable(obs_traj_rel)
                past_rel = past_rel.transpose(1, 0)
                future_rel = Variable(pred_traj_gt_rel)
                future_rel = future_rel.transpose(1, 0)
                future = Variable(pred_traj_gt)
                future = future.transpose(1, 0)
                past = Variable(obs_traj)
                past = past.transpose(1, 0)
                # [batch_size,len,2]

                if self.config.cuda:
                    past = past.cuda()
                    past_rel = past_rel.cuda()
                    future_rel = future_rel.cuda()
                    future = future.cuda()

                pred = self.mem_n2n(past_rel, future_rel, past)
                data_len=data_len+past.size(0)

                distances = torch.norm(pred -future , dim=2)
                eucl_mean += torch.sum(torch.mean(distances, 1))
                horizon10s += torch.sum(distances[:, 3])
                horizon20s += torch.sum(distances[:, 6])
                horizon30s += torch.sum(distances[:, 9])
                horizon40s += torch.sum(distances[:, 11])

            dict_metrics['eucl_mean'] = eucl_mean / data_len
            dict_metrics['horizon10s'] = horizon10s / data_len
            dict_metrics['horizon20s'] = horizon20s / data_len
            dict_metrics['horizon30s'] = horizon30s / data_len
            dict_metrics['horizon40s'] = horizon40s / data_len

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        self.iterations += 1
        loss_all=0
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

            past_rel = Variable(obs_traj_rel)
            past_rel=past_rel.transpose(1,0)
            future_rel = Variable(pred_traj_gt_rel)
            future_rel = future_rel.transpose(1, 0)

            past = Variable(obs_traj)
            past = past.transpose(1, 0)

            if config.cuda:
                past = past.cuda()
                past_rel = past_rel.cuda()
                future_rel = future_rel.cuda()
            self.opt.zero_grad()
            output = self.mem_n2n(past_rel, future_rel,past)

            ##############################
            loss = torch.zeros(1).to(pred_traj_gt).cuda()
            l2_loss_rel = []
            loss_mask = loss_mask[:,config.past_len:]
            # args.pred_len, mem, epoch,
            pred_traj_fake=output.transpose(1,0)
                # print(loss)
            l2_loss_rel.append(
                    l2_loss(pred_traj_fake, pred_traj_gt, loss_mask, mode="raw")
                )

            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake.shape[0]) * (end - start)
                )
                l2_loss_sum_rel = l2_loss_sum_rel + _l2_loss_rel

            loss_all = loss_all+l2_loss_sum_rel
            loss =loss+ l2_loss_sum_rel
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

        # Tensorboard summary: loss
        self.writer.add_scalar('loss/loss_total', loss_all/len(self.train_loader), self.iterations)

        return loss.item()
