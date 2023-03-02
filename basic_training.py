import os
import yaml
import torch.nn as nn
import torch
import numpy as np
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from model.embedding import LossFunctions
from model.utils import seed_everything
from model.metrics import compute_binary_metrics, compute_duration_metrics, compute_categorical_metrics


class BasicTraining(pl.LightningModule):
    def __init__(self, config, train_set, val_set, test_set, label_names, feature_names, embedding_model):
        super(BasicTraining, self).__init__()
        self.config = config
        self.model = embedding_model
        self.duration_tasks = config['duration_prediction_tasks']
        self.binary_tasks = config['binary_prediction_tasks']
        self.categorical_tasks = config['categorical_prediction_tasks']
        self.num_duration_tasks = len(self.duration_tasks)
        self.num_binary_tasks = len(self.binary_tasks)
        self.binary_reconstruction_tasks = config['binary_reconstruction_tasks']
        self.continuous_reconstruction_tasks = config['continuous_reconstruction_tasks']
        self.num_binary_reconstruction_tasks = len(self.binary_reconstruction_tasks)
        self.num_continuous_reconstruction_tasks = len(self.continuous_reconstruction_tasks)
        self.last_timestep_reconstruction_tasks = config['last_timestep_reconstruction_tasks']
        self.forecast_reconstruction_tasks = config['forecast_reconstruction_tasks']
        self.num_last_timestep_reconstruction_tasks = len(self.last_timestep_reconstruction_tasks)
        self.num_forecast_reconstruction_tasks = len(self.forecast_reconstruction_tasks)
        self.max_seq_len = config['max_seq_len']
        self.embedding_dim = config['embedding_dim']
        self.class_weights = config['class_weights']
        self.gpu = config['gpus']
        self.trainset = train_set
        self.validset = val_set
        self.testset = test_set
        self.label_names = label_names
        self.feature_names = feature_names
        self.binary_weights, self.categorical_weights = self.get_weights()
        self.lossfunctions = LossFunctions(self.binary_weights, self.categorical_weights)
        self.on_train_start()

    def on_train_start(self):
        seed_everything(self.config['seed'])

    def get_weights(self):
        _, _, y_data = self.trainset.tensors
        duration_data, binary_data, categorical_data = self.split_labels(y_data)
        num_train_patients = binary_data.shape[0]
        binary_weights = num_train_patients / binary_data[:, 0, :].sum(axis=0)
        categorical_weights = {}
        for task_name in self.categorical_tasks:
            weights = num_train_patients / categorical_data[task_name][:, 0, :].int().flatten().bincount()
            # standardise the weights a bit so they don't massively affect the overall weighting of the categorical tasks
            if self.gpu:
                categorical_weights[task_name] = (weights / weights.mean()).cuda()
            else:
                categorical_weights[task_name] = (weights / weights.mean())
        return binary_weights, categorical_weights

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'], shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config['batch_size'],
                          num_workers=self.config['num_workers'], shuffle=False)

    def load_model(self, log_dir, **hconfig):
        """
        :param log_dir: str, path to the directory that must contain a .yaml file containing the model hyperparameters and a .ckpt file as saved by pytorch-lightning;
        :param config: list of named arguments, used to update the model hyperparameters
        """
        assert os.path.exists(log_dir)
        # load hparams
        with open(list(Path(log_dir).glob('**/*yaml'))[0]) as fp:
            config = yaml.load(fp, Loader=yaml.Loader)
            config.update(hconfig)

        model_path = list(Path(log_dir).glob('**/*ckpt'))[0]
        print(f'Loading model {model_path.parent.stem}')
        model = self.load_from_checkpoint(checkpoint_path=str(model_path), **config)

        return model, config

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['l2'])
        if self.config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        return {'optimizer': optimizer,
                'scheduler': scheduler,
                'monitor': 'val_loss'}

    def split_labels(self, y_true):
        duration_true = y_true[:, :, :self.num_duration_tasks]
        ind = self.num_duration_tasks + self.num_binary_tasks
        binary_true = y_true[:, :, self.num_duration_tasks:ind]
        categorical_true = {}
        for task_name in self.categorical_tasks:
            categorical_true[task_name] = y_true[:, :, ind:ind+1]
            ind += 1
        return duration_true, binary_true, categorical_true

    def get_reconstruction_labels(self, ts_data):
        return

    def training_step(self, batch, batch_idx):
        return self.step('train', batch)

    def validation_step(self, batch, batch_idx):
        return self.step('val', batch)

    def test_step(self, batch, batch_idx):
        return self.step('test', batch)

    # for some reason that makes no sense to me, these methods are necessary to get it to run smoothly (I get errors without them),
    # but they don't do anything
    def training_step_end(self, outputs):
        return outputs

    def validation_step_end(self, outputs):
        return outputs

    def test_step_end(self, outputs):
        return outputs

    def get_prediction_metrics(self, metrics_dict, phase, outputs):
        for i, task in enumerate(self.duration_tasks):
            y_true = np.concatenate([self.remove_padding(output['duration_true'][:, :, i], output['ts_mask'][:, :, 0]) for output in outputs])
            preds = np.concatenate([self.remove_padding(output['duration_pred'][:, :, i], output['ts_mask'][:, :, 0]) for output in outputs])
            if self.config['verbose']:
                print('{} {} statistics:'.format(task, phase))
            metrics_dict[task] = compute_duration_metrics(y_true, preds, self.config['verbose'])

        for i, task in enumerate(self.binary_tasks):
            y_true = np.concatenate([self.remove_padding(output['binary_true'][:, :, i], output['binary_mask'][:, :, i]) for output in outputs])
            preds = np.concatenate([self.remove_padding(output['binary_pred'][:, :, i], output['binary_mask'][:, :, i]) for output in outputs])
            if self.config['verbose']:
                print('{} {} statistics:'.format(task, phase))
            metrics_dict[task] = compute_binary_metrics(y_true, preds, self.config['verbose'])

        for i, task in enumerate(self.categorical_tasks):
            y_true = np.concatenate([self.remove_categorical_padding(output['categorical_true'][task], output['ts_mask']) for output in outputs])
            preds = np.concatenate([self.remove_categorical_padding(output['categorical_pred'][task], output['ts_mask']) for output in outputs])
            if self.config['verbose']:
                print('{} {} statistics:'.format(task, phase))
            metrics_dict[task] = compute_categorical_metrics(y_true, preds, self.config['verbose'])

        for task in metrics_dict.keys():
            for metric in metrics_dict[task]:
                prog_bar = True if metric in ['auroc', 'auprc', 'msle', 'log_loss'] and phase == 'val' else False  # only put certain metrics on the progress bar
                self.log('{}_{}_{}'.format(phase, task, metric), metrics_dict[task][metric], on_epoch=True, prog_bar=prog_bar, logger=True)
        return metrics_dict

    def get_reconstruction_metrics(self, metrics_dict, phase, outputs):
        return metrics_dict

    def training_epoch_end(self, outputs):
        self.epoch_end('train', outputs)
        return

    def validation_epoch_end(self, outputs):
        self.epoch_end('val', outputs)
        return

    def test_epoch_end(self, outputs):
        metrics_dict = self.epoch_end('test', outputs)
        return metrics_dict

    def duration_loss(self, duration_pred, duration_true, ts_mask):
        duration_loss = self.lossfunctions.msle_loss(duration_pred, duration_true)
        # some of these "duration losses" correspond to positions where we have no data, so we apply the mask to these positions
        duration_loss[ts_mask.repeat(1, 1, self.num_duration_tasks) == 0] = 0
        # then we take the mean of the remaining losses, getting us a loss per task
        duration_loss = duration_loss.sum(axis=1).sum(axis=0) / (ts_mask.sum() * self.num_duration_tasks)
        return duration_loss

    def binary_loss(self, binary_pred, binary_true, ts_mask):
        # sometimes the binary task is missing, which is indicated with a -1 (note that 0 means the negative class,
        # not a missing label) therefore, we need to add an extra "labels" mask, which is 0 in the absence of data
        labels_mask = (binary_true >= 0).int()
        binary_mask = labels_mask * ts_mask
        if self.class_weights:  # apply class weights if indicated
            binary_loss = self.lossfunctions.bce_loss_weighted(binary_pred, binary_true)
        else:
            binary_loss = self.lossfunctions.bce_loss(binary_pred, binary_true)
        binary_loss[binary_mask == 0] = 0
        binary_loss = binary_loss.sum(axis=1).sum(axis=0) / binary_mask.sum()
        return binary_loss, binary_mask

    def categorical_loss(self, categorical_pred, categorical_true, ts_mask):
        categorical_loss = {}
        total_cat_loss = 0
        for task_name in self.categorical_tasks:
            num_cats = len(self.categorical_weights[task_name])
            if self.class_weights:  # apply class weights if indicated
                categorical_loss[task_name] = self.lossfunctions.ce_loss_weighted[task_name](categorical_pred[task_name].view(-1, num_cats),
                                               categorical_true[task_name].long().view(-1)).view(-1, self.max_seq_len, 1)
            else:
                categorical_loss[task_name] = self.lossfunctions.ce_loss(categorical_pred[task_name].view(-1, num_cats),
                                               categorical_true[task_name].long().view(-1)).view(-1, self.max_seq_len, 1)
            categorical_loss[task_name][ts_mask == 0] = 0
            categorical_loss[task_name] = categorical_loss[task_name].sum() / ts_mask.sum()
            total_cat_loss += categorical_loss[task_name]
        return categorical_loss, total_cat_loss

    def ts_reconstruction_loss(self, last_timestep_reconstruction, ts_data, ts_mask):
        ts_data = ts_data[:, :, self.last_timestep_reconstruction_indices]
        # minus 1 because the first column "time" is not reconstructed in the model
        last_timestep_reconstruction = last_timestep_reconstruction[:, :, [ind - 1 for ind in self.last_timestep_reconstruction_indices]]
        ts_reconstruction_loss = self.lossfunctions.mse_loss(last_timestep_reconstruction, ts_data)
        ts_reconstruction_loss[ts_mask.repeat(1, 1, self.num_last_timestep_reconstruction_tasks) == 0] = 0
        ts_reconstruction_loss = ts_reconstruction_loss.sum(axis=1).sum(axis=0) / (ts_mask.sum() * self.num_last_timestep_reconstruction_tasks)
        return ts_reconstruction_loss

    def forecast_loss(self, forecast, ts_data, ts_mask):
        # remove the first timepoint as this isn't possible to use as a label for a forecasting task
        # also constrain to the original features only (no masking variables, or "time")
        ts_data = ts_data[:, 1:, self.forecast_reconstruction_indices]
        ts_mask = ts_mask[:, 1:, :]
        # remove the last forecast as we have no label for this
        forecast = forecast[:, :-1, [ind - 1 for ind in self.forecast_reconstruction_indices]]
        forecast_loss = self.lossfunctions.mse_loss(forecast, ts_data)
        forecast_loss[ts_mask.repeat(1, 1, self.num_forecast_reconstruction_tasks) == 0] = 0
        forecast_loss = forecast_loss.sum(axis=1).sum(axis=0) / (ts_mask.sum() * self.num_forecast_reconstruction_tasks)
        return forecast_loss

    def binary_reconstruction_loss(self, static_binary_reconstruction, s_data, ts_mask):
        s_data = s_data[:, :, self.binary_reconstruction_indices].repeat(1, self.max_seq_len, 1)
        binary_reconstruction_loss = self.lossfunctions.bce_loss(static_binary_reconstruction, s_data)
        binary_reconstruction_loss[ts_mask.repeat(1, 1, self.num_binary_reconstruction_tasks) == 0] = 0
        binary_reconstruction_loss = binary_reconstruction_loss.sum(axis=1).sum(axis=0) / (ts_mask.sum() * self.num_binary_reconstruction_tasks)
        return binary_reconstruction_loss

    def continuous_reconstruction_loss(self, static_continuous_reconstruction, s_data, ts_mask):
        s_data = s_data[:, :, self.continuous_reconstruction_indices].repeat(1, self.max_seq_len, 1)
        continuous_reconstruction_loss = self.lossfunctions.mse_loss(static_continuous_reconstruction, s_data)
        continuous_reconstruction_loss[ts_mask.repeat(1, 1, self.num_continuous_reconstruction_tasks) == 0] = 0
        continuous_reconstruction_loss = continuous_reconstruction_loss.sum(axis=1).sum(axis=0) / (ts_mask.sum() * self.num_continuous_reconstruction_tasks)
        return continuous_reconstruction_loss

    def remove_padding(self, y, mask):
        """
            Filters out padding from tensor of predictions or labels

            Args:
                y: tensor of predictions or labels
                mask (bool_type): tensor showing which values are padding (0) and which are data (1)
        """
        # note it's fine to call .cpu() on a tensor already on the cpu
        y = y.flatten().detach().cpu().numpy()
        mask = mask.flatten().detach().cpu().numpy()
        y = y[mask > 0]
        return y

    def remove_categorical_padding(self, y, mask):
        """
            Filters out padding from tensor of predictions or labels

            Args:
                y: tensor of predictions or labels
                mask (bool_type): tensor showing which values are padding (0) and which are data (1)
        """
        # note it's fine to call .cpu() on a tensor already on the cpu
        if y.shape[2] == 1:
            y = y[mask > 0].detach().cpu().numpy()
        else:
            num_cats = y.shape[2]
            y = y.view(-1, num_cats)[mask.view(-1, 1).repeat(1, num_cats) > 0].view(-1, num_cats).detach().cpu().numpy()
        return y


def import_data(data_path, debug=False):
    '''
        Output:
            - data_x: [N, max_length, 1+x_dim] tensor (where N: number of samples, max_length: max sequence length, x_dim: feature dimension)
                      the first feature is the time difference.
            - data_s: [N, 1, 1+s_dim] tensor (where N: number of samples, 1 because these are the static features, s_dim: feature dimension)
                      the first feature is the time difference.
            - data_y: [N, max_length, y_dim] tensor (where N: number of samples, max_length: max sequence length, y_dim: output dimension)
    '''
    print (data_path)
    npz = np.load(data_path, allow_pickle=True)
    data_x = npz['data_x']
    data_s = npz['data_s']
    data_y = npz['data_y']

    if debug:
        data_x = data_x[:100, :, :]
        data_s = data_s[:100, :, :]
        data_y = data_y[:100, :, :]

    tensor_x = torch.Tensor(data_x)  # transform to torch tensor
    tensor_s = torch.Tensor(data_s)
    tensor_y = torch.Tensor(data_y)

    label_names = npz['label_names']
    feature_names = npz['feature_names'].item()
    ventids = npz['ventids']

    dataset = TensorDataset(tensor_x, tensor_s, tensor_y)

    return dataset, label_names, feature_names, ventids


def get_data(config):
    train_set, label_names, feature_names, train_ventids = import_data(data_path='{}train/data.npz'.format(config['data_dir']), debug=config['debug'])
    val_set, _, _, val_ventids = import_data(data_path='{}val/data.npz'.format(config['data_dir']))
    test_set, _, _, test_ventids = import_data(data_path='{}test/data.npz'.format(config['data_dir']))

    ventids = [train_ventids] + [val_ventids] + [test_ventids]

    return train_set, val_set, test_set, label_names, feature_names, ventids


def initialise_weights(model):
    for m in model.modules():
        if type(m) in [nn.Linear]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(m.weight_hh_l0)
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.zeros_(m.bias_hh_l0)
            nn.init.zeros_(m.bias_ih_l0)