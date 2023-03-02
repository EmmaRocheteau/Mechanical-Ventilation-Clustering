from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers
from model.embedding import PredictionReconstructionEmbedding
from model.utils import get_checkpoint_path
from args import initialise_arguments, add_configs
from basic_training import get_data, initialise_weights, BasicTraining
from utils import write_json
from utils import record_results
import numpy as np
from torch import nn


class PredictionReconstructionTraining(BasicTraining):
    def __init__(self, config, train_set, val_set, test_set, label_names, feature_names, embedding_model):
        super(PredictionReconstructionTraining, self).__init__(config, train_set, val_set, test_set, label_names, feature_names, embedding_model)
        self.binary_reconstruction_indices, self.continuous_reconstruction_indices, self.last_timestep_reconstruction_indices, self.forecast_reconstruction_indices = self.get_indices_for_reconstruction_tasks()
        self.sigmoid = nn.Sigmoid()

    def get_indices_for_reconstruction_tasks(self):
        binary_indices = []
        continuous_indices = []
        last_timestep_indices = []
        forecast_indices = []
        for task_name in self.binary_reconstruction_tasks:
            binary_indices.append(list(self.feature_names['static']).index(task_name))
        for task_name in self.continuous_reconstruction_tasks:
            continuous_indices.append(list(self.feature_names['static']).index(task_name))
        for task_name in self.last_timestep_reconstruction_tasks:
            last_timestep_indices.append(list(self.feature_names['temporal']).index(task_name))
        for task_name in self.forecast_reconstruction_tasks:
            forecast_indices.append(list(self.feature_names['temporal']).index(task_name))
        return binary_indices, continuous_indices, last_timestep_indices, forecast_indices

    def step(self, phase, batch):

        # model
        ts_data, s_data, y_true = batch
        duration_true, binary_true, categorical_true = self.split_labels(y_true)
        embedding, ts_mask, preds, reconstructions = self.model(ts_data, s_data)
        duration_pred, binary_pred, categorical_pred = preds
        last_timestep_reconstruction, forecast, static_binary_reconstruction, static_continuous_reconstruction = reconstructions

        # calculate losses
        # prediction
        duration_loss = self.duration_loss(duration_pred, duration_true, ts_mask).sum() * self.config['duration_weight']
        binary_loss, binary_mask = self.binary_loss(binary_pred, binary_true, ts_mask)
        binary_loss = binary_loss.sum() * self.config['binary_weight']
        categorical_loss, total_cat_loss = self.categorical_loss(categorical_pred, categorical_true, ts_mask)
        total_cat_loss = total_cat_loss * self.config['categorical_weight']

        # reconstruction
        last_timestep_loss = self.ts_reconstruction_loss(last_timestep_reconstruction, ts_data, ts_mask).sum() * self.config['ts_reconstruction_weight']
        forecast_loss = self.forecast_loss(forecast, ts_data, ts_mask).sum() * self.config['ts_forecasting_weight']
        binary_reconstruction_loss = self.binary_reconstruction_loss(static_binary_reconstruction, s_data, ts_mask).sum() * self.config['binary_reconstruction_weight']
        continuous_reconstruction_loss = self.continuous_reconstruction_loss(static_continuous_reconstruction, s_data, ts_mask).sum() * self.config['continuous_reconstruction_weight']

        loss = duration_loss + binary_loss + total_cat_loss + last_timestep_loss + forecast_loss + binary_reconstruction_loss + continuous_reconstruction_loss
        self.log('{}_loss'.format(phase), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_duration_loss'.format(phase), duration_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_binary_loss'.format(phase), binary_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_categorical_loss'.format(phase), total_cat_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_last_timestep_loss'.format(phase), last_timestep_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_forecast_loss'.format(phase), forecast_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_binary_reconstruction_loss'.format(phase), binary_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_continuous_reconstruction_loss'.format(phase), continuous_reconstruction_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss,
                'duration_pred': duration_pred,
                'binary_pred': binary_pred,
                'categorical_pred': categorical_pred,
                'duration_true': duration_true,
                'binary_true': binary_true,
                'categorical_true': categorical_true,
                'binary_mask': binary_mask,
                'ts_mask': ts_mask,
                'embedding': embedding}

    def predict_step(self, batch, batch_idx):
        ts_data, s_data, y_true = batch
        trues = self.split_labels(y_true)
        embedding, ts_mask, preds, reconstructions = self.model(ts_data, s_data)
        # adding a sigmoid activation, as it was trained with BCEWithLogitsLoss, which doesn't include the activation
        preds = (preds[0], self.sigmoid(preds[1]), preds[2])
        reconstructions = (reconstructions[0], reconstructions[1], self.sigmoid(reconstructions[2]), reconstructions[3])
        return embedding, ts_mask, trues, preds, reconstructions

    def epoch_end(self, phase, outputs):
        metrics_dict = self.get_prediction_metrics({}, phase, outputs)
        metrics_dict = self.get_reconstruction_metrics(metrics_dict, phase, outputs)
        return metrics_dict


def train_prediction_reconstruction(config):
    train_set, val_set, test_set, label_names, feature_names, ventids = get_data(config)

    # all the timeseries features except time and the mask variables
    if config['last_timestep_reconstruction_tasks'] == ['all']:
        config['last_timestep_reconstruction_tasks'] = [feature for feature in feature_names['temporal'] if not (feature.endswith('_mask') or feature == 'time')]
    if config['forecast_reconstruction_tasks'] == ['all']:
        config['forecast_reconstruction_tasks'] = [feature for feature in feature_names['temporal'] if not (feature.endswith('_mask') or feature == 'time')]

    # define logger
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], name='PredictionReconstruction', version=config['version'])
    logger.log_hyperparams(params=config)

    # save configuration
    np.save(Path(config['log_path']) / 'PredictionReconstruction' / config['version'] / 'config.npy', config)

    # get the number of categories per categorical prediction task
    num_cats = {}
    _, _, y_true = val_set.tensors
    for task_name in config['categorical_prediction_tasks']:
        ind = list(label_names).index(task_name)
        num_cats[task_name] = len(y_true[:, :, ind].unique())

    # define embedding model
    embedding_model = PredictionReconstructionEmbedding(config, num_cats)
    print(embedding_model)
    initialise_weights(embedding_model)

    # define model
    model = PredictionReconstructionTraining(config=config, train_set=train_set, val_set=val_set, test_set=test_set,
                                             label_names=label_names, feature_names=feature_names, embedding_model=embedding_model)
    chkpt = None if config['load_encoder'] is None else get_checkpoint_path(config['log_path'] + config['load_encoder'])

    trainer = pl.Trainer(
        gpus=config['gpus'],
        logger=logger,
        num_sanity_val_steps=0,
        max_epochs=config['epochs'],
        precision=32,
        default_root_dir=config['log_path'],
        deterministic=True,
        resume_from_checkpoint=chkpt,
        auto_lr_find=config['auto_lr'],
        auto_scale_batch_size=config['auto_bsz']
    )

    trainer.fit(model)

    for phase in ['val']:  # ['test', 'valid'] for finalised models
        if phase == 'val':
            outputs = trainer.validate(model)
        else:
            outputs = trainer.test(model)
        if isinstance(outputs, list):
            outputs = outputs[0]

        test_results = outputs
        res_dir = Path(config['log_path']) / 'PredictionReconstruction'
        if config['version'] is not None:
            res_dir = res_dir / config['version']
        else:
            res_dir = res_dir / ('results_' + str(config['seed']))
        print(phase, ':', test_results)
        Path(res_dir).mkdir(parents=True, exist_ok=True)
        write_json(test_results, res_dir / f'{phase}_results.json', sort_keys=True, verbose=True)

        path_results = Path(config['log_path']) / f'all_{phase}_results.csv'
        record_results(path_results, config, test_results)


if __name__ == '__main__':
    # define configs
    parser = initialise_arguments()
    config = parser.parse_args()
    config = add_configs(config)

    for key in sorted(config):
        print(f'{key}: ', config[key])

    train_prediction_reconstruction(config)