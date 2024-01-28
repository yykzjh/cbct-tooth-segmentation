import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import lib.utils as utils





class TensorboardWriter():

    def __init__(self, tensorboard_dir, log_txt_path):
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        self.log_file = log_txt_path
        index_to_class_dict = utils.load_json_file(
            os.path.join(r"./lib/dataloaders/index_to_class", config.dataset_name + ".json"))
        self.label_names = list(index_to_class_dict.values())
        self.data = self.create_data_structure()


    def save_log(self, path, line):
        with open(path, 'a') as log_f:
            log_f.write(line)


    def create_data_structure(self, ):
        data = {"train": dict((label, 0.0) for label in self.label_names),
                "val": dict((label, 0.0) for label in self.label_names)}
        data['train']['loss'] = 0.0
        data['val']['loss'] = 0.0
        data['train']['count'] = 1.0
        data['val']['count'] = 1.0
        data['train']['dsc'] = 0.0
        data['val']['dsc'] = 0.0
        return data

    def display_terminal(self, epoch, step=None, per_epoch_total_step=None, mode='train', summary=False):
        """

        :param epoch: epoch of training
        :param step: step of current epoch
        :param loss: any loss numpy
        :param mode: train or val ( for training and validation)
        :param summary: to print total statistics at the end of epoch
        """
        if summary:
            info_print = "\nSummary {} Epoch {:4d}:  Loss:{:.4f} \t DSC:{:.4f}  ".format(mode, epoch,
                                                                                         self.data[mode]['loss'] /
                                                                                         self.data[mode]['count'],
                                                                                         self.data[mode]['dsc'] /
                                                                                         self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{} : {:.4f}".format(self.label_names[i],
                                                     self.data[mode][self.label_names[i]] / self.data[mode]['count'])

            print(info_print)
        else:

            info_print = "\nEpoch: {:4d} [{}/{}] Loss:{:.4f} \t DSC:{:.4f}".format(epoch, step, per_epoch_total_step,
                                                                                   self.data[mode]['loss'] /
                                                                                   self.data[mode]['count'],
                                                                                   self.data[mode]['dsc'] /
                                                                                   self.data[mode]['count'])

            for i in range(len(self.label_names)):
                info_print += "\t{}:{:.4f}".format(self.label_names[i],
                                                   self.data[mode][self.label_names[i]] / self.data[mode]['count'])
            print(info_print)


    def reset(self, mode):
        self.data[mode]['dsc'] = 0.0
        self.data[mode]['loss'] = 0.0
        self.data[mode]['count'] = 1
        for i in range(len(self.label_names)):
            self.data[mode][self.label_names[i]] = 0.0


    def update_scores(self, iter, loss, channel_score, mode, writer_step):
        """
        :param iter: iteration or partial epoch
        :param loss: any loss torch.tensor.item()
        :param channel_score: per channel score or dice coef
        :param mode: train or val ( for training and validation)
        :param writer_step: tensorboard writer step
        """
        # WARNING ASSUMING THAT CHANNELS IN SAME ORDER AS DICTIONARY

        dice_coeff = np.mean(channel_score)

        num_channels = len(channel_score)
        self.data[mode]['dsc'] += dice_coeff
        self.data[mode]['loss'] += loss
        self.data[mode]['count'] = iter + 1

        for i in range(num_channels):
            self.data[mode][self.label_names[i]] += channel_score[i]
            if self.writer is not None:
                self.writer.add_scalar(mode + '/' + self.label_names[i], channel_score[i], global_step=writer_step)
        self.writer.flush()

    def write_end_of_epoch(self, epoch):

        self.writer.add_scalars('epoch/dsc',
                                {'train': self.data['train']['dsc'] / self.data['train']['count'],
                                 'val': self.data['val']['dsc'] / self.data['val']['count']},
                                epoch)
        self.writer.add_scalars('epoch/loss',
                                {'train': self.data['train']['loss'] / self.data['train']['count'],
                                 'val': self.data['val']['loss'] / self.data['val']['count']},
                                epoch)
        for i in range(len(self.label_names)):
            self.writer.add_scalars("epoch/" + self.label_names[i],
                                    {'train': self.data['train'][self.label_names[i]] / self.data['train']['count'],
                                     'val': self.data['val'][self.label_names[i]] / self.data['train']['count']},
                                    epoch)
        self.writer.flush()

        train_csv_line = 'Epoch:{:4d} Train Loss:{:.4f} Train DSC:{:.4f}'.format(
            epoch,
            self.data['train']['loss'] / self.data['train']['count'],
            self.data['train']['dsc'] / self.data['train']['count'])
        val_csv_line = 'Epoch:{:4d} Val Loss:{:.4f} Val DSC:{:.4f}'.format(
            epoch,
            self.data['val']['loss'] / self.data['val']['count'],
            self.data['val']['dsc'] / self.data['val']['count'])

        self.save_log(self.log_file, '\n' + train_csv_line + '\n')
        self.save_log(self.log_file, val_csv_line + '\n')


    def close_writer(self):
        self.writer.close()
