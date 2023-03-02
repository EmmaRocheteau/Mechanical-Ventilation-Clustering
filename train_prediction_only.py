from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers
from model.embedding import PredictionOnlyEmbedding
from model.utils import get_checkpoint_path
from args import initialise_arguments, add_configs
from basic_training import get_data, initialise_weights, BasicTraining
from utils import write_json
from utils import record_results
import numpy as np
from torch import nn


class PredictionOnlyTraining(BasicTraining):
    def __init__(self, config, train_set, val_set, test_set, label_names, feature_names, embedding_model):
        super(PredictionOnlyTraining, self).__init__(config, train_set, val_set, test_set, label_names, feature_names, embedding_model)
        self.sigmoid = nn.Sigmoid()

    def step(self, phase, batch):

        # model
        ts_data, s_data, y_true = batch
        duration_true, binary_true, categorical_true = self.split_labels(y_true)
        embedding, ts_mask, [duration_pred, binary_pred, categorical_pred] = self.model(ts_data, s_data)

        # calculate losses
        duration_loss = self.duration_loss(duration_pred, duration_true, ts_mask)
        binary_loss, binary_mask = self.binary_loss(binary_pred, binary_true, ts_mask)
        categorical_loss, total_cat_loss = self.categorical_loss(categorical_pred, categorical_true, ts_mask)
        loss = duration_loss.sum() * self.config['duration_weight'] + \
               binary_loss.sum() * self.config['binary_weight'] + \
               total_cat_loss * self.config['categorical_weight']
        self.log('{}_loss'.format(phase), loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_duration_loss'.format(phase), duration_loss.sum() * self.config['duration_weight'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_binary_loss'.format(phase), binary_loss.sum() * self.config['binary_weight'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('{}_categorical_loss'.format(phase), total_cat_loss * self.config['categorical_weight'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
        embedding, ts_mask, preds = self.model(ts_data, s_data)
        preds = (preds[0], self.sigmoid(preds[1]), preds[2])  # adding a sigmoid activation, as it was trained with BCEWithLogitsLoss, which doesn't include the activation
        return embedding, ts_mask, trues, preds

    def epoch_end(self, phase, outputs):
        metrics_dict = self.get_prediction_metrics({}, phase, outputs)
        return metrics_dict


def train_prediction_only(config):
    train_set, val_set, test_set, label_names, feature_names, ventids = get_data(config)

    # define logger
    Path(config['log_path']).mkdir(parents=True, exist_ok=True)
    logger = loggers.TensorBoardLogger(config['log_path'], name='PredictionOnly', version=config['version'])
    logger.log_hyperparams(params=config)

    # save configuration
    np.save(Path(config['log_path']) / 'PredictionOnly' / config['version'] / 'config.npy', config)

    # get the number of categories per categorical prediction task
    num_cats = {}
    _, _, y_true = val_set.tensors
    for task_name in config['categorical_prediction_tasks']:
        ind = list(label_names).index(task_name)
        num_cats[task_name] = len(y_true[:, :, ind].unique())

    # define embedding model
    embedding_model = PredictionOnlyEmbedding(config, num_cats)
    print(embedding_model)
    initialise_weights(embedding_model)

    # define model
    model = PredictionOnlyTraining(config=config, train_set=train_set, val_set=val_set, test_set=test_set,
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
        res_dir = Path(config['log_path']) / 'PredictionOnly'
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

    train_prediction_only(config)