import argparse
from kmeans_pytorch import kmeans, kmeans_predict
from args import read_params_from_file
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers
from model.embedding import PredictionOnlyEmbedding, PredictionReconstructionEmbedding
from model.utils import get_checkpoint_path
from basic_training import get_data, initialise_weights
from train_prediction_only import PredictionOnlyTraining
from torch.utils.data import DataLoader
from train_prediction_reconstruction import PredictionReconstructionTraining
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import write_json
from utils import record_results
import numpy as np
import pandas as pd
import torch
from sklearn_extra.cluster import KMedoids


def get_model_outputs(config):

    train_set, val_set, test_set, label_names, feature_names, ventids = get_data(config)
    save = config['save_outputs']

    try:
        train_outputs = np.load(config['experiment_folder'] + '/train_outputs.npy', allow_pickle=True)
        val_outputs = np.load(config['experiment_folder'] + '/val_outputs.npy', allow_pickle=True)
        test_outputs = np.load(config['experiment_folder'] + '/test_outputs.npy', allow_pickle=True)

    except FileNotFoundError:

        prediction_only = 'PredictionOnly' in config['experiment_folder']

        # all the timeseries features except time and the mask variables
        if config['last_timestep_reconstruction_tasks'] == ['all']:
            config['last_timestep_reconstruction_tasks'] = [feature for feature in feature_names['temporal'] if not (feature.endswith('_mask') or feature == 'time')]
        if config['forecast_reconstruction_tasks'] == ['all']:
            config['forecast_reconstruction_tasks'] = [feature for feature in feature_names['temporal'] if not (feature.endswith('_mask') or feature == 'time')]

        # define logger
        Path(config['log_path']).mkdir(parents=True, exist_ok=True)
        logger = loggers.TensorBoardLogger(config['log_path'], name='Clustering', version=config['version'])
        logger.log_hyperparams(params=config)

        # get the number of categories per categorical prediction task
        num_cats = {}
        _, _, y_true = val_set.tensors
        for task_name in config['categorical_prediction_tasks']:
            ind = list(label_names).index(task_name)
            num_cats[task_name] = len(y_true[:, :, ind].unique())

        # define embedding model
        if prediction_only:
            embedding_model = PredictionOnlyEmbedding(config, num_cats)
        else:
            embedding_model = PredictionReconstructionEmbedding(config, num_cats)
        print(embedding_model)
        initialise_weights(embedding_model)

        # define model
        if prediction_only:
            model = PredictionOnlyTraining(config=config, train_set=train_set, val_set=val_set, test_set=test_set,
                                           label_names=label_names, feature_names=feature_names, embedding_model=embedding_model)
        else:
            model = PredictionReconstructionTraining(config=config, train_set=train_set, val_set=val_set, test_set=test_set,
                                                     label_names=label_names, feature_names=feature_names, embedding_model=embedding_model)
        chkpt = get_checkpoint_path(config['experiment_folder'])

        trainer = pl.Trainer(
            gpus=config['gpus'],
            logger=logger,
            num_sanity_val_steps=0,
            precision=32,
            default_root_dir=config['log_path'],
            deterministic=True,
            auto_lr_find=config['auto_lr'],
            auto_scale_batch_size=config['auto_bsz']
        )

        # load checkpoint
        model = model.load_from_checkpoint(chkpt, config=config, train_set=train_set, val_set=val_set, test_set=test_set, label_names=label_names, feature_names=feature_names, embedding_model=embedding_model)

        if config['no_clustering']:

            outputs = trainer.test(model)
            if isinstance(outputs, list):
                outputs = outputs[0]

            write_json(outputs, Path(config['experiment_folder']) / 'test_results.json', sort_keys=True, verbose=True)
            path_results = Path(config['experiment_folder']) / '../all_test_results.csv'
            record_results(path_results, config, outputs)
            return

        else:
            non_shuffle_dataloader = DataLoader(model.trainset, batch_size=model.config['batch_size'],
                                                num_workers=model.config['num_workers'], shuffle=False)
            train_outputs = trainer.predict(model, dataloaders=non_shuffle_dataloader)
            val_outputs = trainer.predict(model, dataloaders=model.val_dataloader())
            test_outputs = trainer.predict(model, dataloaders=model.test_dataloader())

            if save:
                np.save(config['experiment_folder'] + '/train_outputs.npy', train_outputs)
                np.save(config['experiment_folder'] + '/val_outputs.npy', val_outputs)
                np.save(config['experiment_folder'] + '/test_outputs.npy', test_outputs)

        outputs = trainer.test(model)
        if isinstance(outputs, list):
            outputs = outputs[0]

        write_json(outputs, Path(config['experiment_folder']) / 'test_results.json', sort_keys=True, verbose=True)
        path_results = Path(config['experiment_folder']) / '../all_test_results.csv'
        record_results(path_results, config, outputs)

    return [train_outputs, train_set], [val_outputs, val_set], [test_outputs, test_set], label_names, feature_names, ventids

def combine_output_batches(sizes, outputs, index):
    return np.concatenate([output[index].view(sizes) for output in outputs], axis=0)

def combine_output_batches_pred(sizes, outputs, index1, index2):
    return np.concatenate([output[index1][index2].view(sizes) for output in outputs], axis=0)

def save_kmeans_clusters(config, debug=False, num_samples=50000):
    train, val, test, label_names, feature_names, ventids = get_model_outputs(config)
    np.random.seed(config['seed'])

    # For each patient
    for phase, data, vents in zip(['train', 'val', 'test'], [train, val, test], ventids):
        outputs = data[0]
        data_set = data[1]

        embeddings = combine_output_batches((-1, config['max_seq_len'], config['embedding_dim']), outputs, 0)
        mask = combine_output_batches((-1, config['max_seq_len'], 1), outputs, 1)
        predictions = np.concatenate((combine_output_batches_pred((-1, config['max_seq_len'], 2), outputs, 3, 0),
                                      combine_output_batches_pred((-1, config['max_seq_len'], 2), outputs, 3, 1)), axis=2)
        original_data = combine_output_batches((-1, config['max_seq_len'], config['F'] * 2 + 1), data_set, 0)

        masked_embeddings = torch.tensor(embeddings[np.repeat(mask, config['embedding_dim'], axis=2) > 0].reshape(-1, config['embedding_dim']))
        masked_data = torch.tensor(original_data[np.repeat(mask, config['F'] * 2 + 1, axis=2) > 0].reshape(-1, config['F'] * 2 + 1))
        masked_ventids = torch.tensor(np.repeat(vents, config['max_seq_len'])[((mask - 1) > -1).reshape(-1)])

        if phase == 'train':
            idx = np.random.randint(masked_embeddings.numpy().shape[0], size=num_samples)
            if config['num_clusters'] is None:
                wcss = []
                for i in range(2, 15):
                    clustering = KMedoids(n_clusters=i, init='k-medoids++', random_state=config['seed'])
                    #clustering = KMeans(n_clusters=i, init='k-means++', random_state=config['seed'])
                    clustering.fit(masked_embeddings.numpy()[idx,:])
                    wcss.append(clustering.inertia_)
                    print('Done {}'.format(i))
                ks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
                sns.lineplot(x=ks, y=wcss)
                print(wcss)
                plt.ylabel('Within-Cluster-Sum-of-Squares (WCSS)')
                plt.xlabel('Number of Clusters')
                plt.savefig(config['experiment_folder'] + '/elbow_plot.png')
                config['num_clusters'] = int(input('Where is the elbow? I.e. which number of clusters should be used?...'))

            #cluster_ids_x, cluster_centers = kmeans(X=masked_embeddings, num_clusters=config['num_clusters'], distance='euclidean')
            clustering = KMedoids(n_clusters=config['num_clusters'], init='k-medoids++', random_state=config['seed'])
            clustering = clustering.fit(masked_embeddings.numpy()[idx,:])
            cluster_ids_x = clustering.predict(masked_embeddings.numpy())

            # save cluster centres
            centre_inds = idx[clustering.medoid_indices_]
            centre_data = masked_data[centre_inds, :]
            np.savetxt(config['experiment_folder'] + '/centre_ventids.txt', masked_ventids[centre_inds])
            np.savetxt(config['experiment_folder'] + '/centre_times.txt', centre_data[:, 0])

        else:
            #cluster_ids_x = kmeans_predict(X=masked_embeddings, cluster_centers=cluster_centers, distance='euclidean')
            cluster_ids_x = clustering.predict(masked_embeddings.numpy())

        try:
            all_pd = pd.read_csv(config['log_path'] + '/all_{}.csv'.format(phase))

        # if data is not already saved
        except FileNotFoundError:

            ts_data = data_set.tensors[0].numpy()
            s_data = data_set.tensors[1].numpy()
            labels = data_set.tensors[2].numpy()[:, :, :4]  # only keep the duration and binary labels

            static_pd = pd.DataFrame(s_data.squeeze(), columns=feature_names['static'])
            static_pd['ventid'] = vents

            if debug is True:
                vents = vents[:100]

            for i in range(len(vents)):
                ventid = vents[i]
                temp_mask = mask[i]

                temp_pd = pd.DataFrame(ts_data[i][np.repeat(temp_mask, 63, axis=1) == 1].reshape(-1, 63), columns=feature_names['temporal'])
                temp_pd['ventid'] = ventid

                for j, label_name in enumerate(feature_names['label'][:4]):
                    temp_pd[label_name] = labels[i][np.repeat(temp_mask, 4, axis=1) == 1].reshape(-1, 4)[:, j]
                    temp_pd['pred_' + label_name] = predictions[i][np.repeat(temp_mask, 4, axis=1) == 1].reshape(-1, 4)[:, j]

                if i == 0:
                    all_pd = temp_pd
                else:
                    all_pd = all_pd.append(temp_pd, ignore_index=True)

            # save data
            all_pd = all_pd.merge(static_pd, on='ventid', how='inner')
            all_pd.to_csv(config['log_path'] + '/all_{}.csv'.format(phase), index=None)

        all_pd['cluster_ids'] = cluster_ids_x
        all_pd['cluster_ids'].to_csv(config['experiment_folder'] + '/{}_clusters.csv'.format(phase), index=None)
        np.save(config['experiment_folder'] + '/{}_embeddings.npy'.format(phase), masked_embeddings)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_folder', type=str, help='full path to the experiment folder')
    parser.add_argument('--config_file', type=str, default='paths.json',
                        help='Config file path - command line arguments will override those in the file.')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpus', type=int, default=-1, help='number of available GPUs')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_clusters', type=int, default=None)
    parser.add_argument('--save_outputs', action='store_true')
    parser.add_argument('--no_clustering', action='store_true')

    config = parser.parse_args()
    config = vars(config)

    experiment_config = np.load(config['experiment_folder'] + '/config.npy', allow_pickle=True).item()

    # replace key information for this local machine that is running the clustering
    experiment_config['config_file'] = config['config_file']
    experiment_config['experiment_folder'] = config['experiment_folder']
    experiment_config['cpu'] = config['cpu']
    experiment_config['num_clusters'] = config['num_clusters']
    experiment_config['debug'] = config['debug']
    experiment_config['save_outputs'] = config['save_outputs']
    experiment_config['no_clustering'] = config['no_clustering']
    config = read_params_from_file(experiment_config, overwrite=True)

    # slightly hacky way to make the new version allowing for ablation studies to be compatible with old versions
    try:
        config['ts_forecasting_weight']
    except KeyError:
        config['ts_forecasting_weight'] = config['ts_reconstruction_weight']

    if config['cpu']:
        num_gpus = 0
        config['gpus'] = None
        config['gpus_per_trial'] = 0

    for key in sorted(config):
        print(f'{key}: ', config[key])

    if config['no_clustering']:
        get_model_outputs(config)
    else:
        save_kmeans_clusters(config)