import torch.nn as nn
import torch
from args import initialise_arguments, add_configs
from model.tpc import TempPointConv
from model.transformer import Transformer


class Exp(nn.Module):
    def forward(self, X):
        return torch.exp(X)


def linear_block(in_dim, out_dim, dropout=None, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([['lrelu', nn.LeakyReLU()], ['relu', nn.ReLU()], ['softmax', nn.Softmax()], ['sigmoid', nn.Sigmoid()], ['tanh', nn.Tanh()], ['exp', Exp()]])

    if dropout is None:
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            nn.BatchNorm1d(out_dim),
            activations[activation]
        )
    else:
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=dropout),
            activations[activation]
        )

    return block


def linear_block_no_bn(in_dim, out_dim, dropout=None, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([['lrelu', nn.LeakyReLU()], ['relu', nn.ReLU()], ['softmax', nn.Softmax()], ['sigmoid', nn.Sigmoid()], ['tanh', nn.Tanh()], ['exp', Exp()]])

    if dropout is None:
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            activations[activation]
        )
    else:
        block = nn.Sequential(
            nn.Linear(in_dim, out_dim, *args, **kwargs),
            nn.Dropout(p=dropout),
            activations[activation]
        )

    return block


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.F = config['F']
        self.no_flat_features = config['no_flat_features']
        self.num_layers = config['num_layers_lstm']
        self.h_dim_lstm = config['h_dim_lstm']
        self.lstm_dropout = config['lstm_dropout']
        self.lstm = nn.LSTM(input_size=self.F * 2 + 1 + self.no_flat_features,
                            hidden_size=self.h_dim_lstm,
                            num_layers=self.num_layers,
                            dropout=self.lstm_dropout)

    def forward(self, X, flat):
        flat_rep = flat.repeat(1, X.shape[1], 1)
        X_concat = torch.cat([flat_rep, X], dim=2)
        encoding, (hidden_states, cell_states) = self.lstm(X_concat)
        return encoding


class TPCEncoder(nn.Module):
    def __init__(self, config):
        super(TPCEncoder, self).__init__()
        self.dropout = nn.Dropout(p=config['dropout'])
        self.tpc = TempPointConv(config)
        self.encoding = nn.Linear(in_features=config['last_linear_size'], out_features=config['encoding_dim_tpc'])

    def forward(self, X, flat):
        tpc_output = self.tpc(X, flat)
        dropout_output = self.dropout(tpc_output)
        encoding = self.encoding(dropout_output)
        return encoding


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.transformer = Transformer(config)

    def forward(self, X, flat):
        flat_rep = flat.repeat(1, X.shape[1], 1)
        X_concat = torch.cat([flat_rep, X], dim=2)
        encoding = self.transformer(X_concat)
        return encoding


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        # Time series encoder
        if config['encoder'] == 'lstm':
            self.ts_encoder = LSTMEncoder(config)
            self.z_dim = config['h_dim_lstm']  # latent space size

        elif config['encoder'] == 'tpc':
            self.ts_encoder = TPCEncoder(config)
            self.z_dim = config['encoding_dim_tpc']  # latent space size

        elif config['encoder'] == 'transformer':
            self.ts_encoder = TransformerEncoder(config)
            self.z_dim = config['d_model']

        self.embedding_dim = config['embedding_dim']
        if config['variational']:
            self.embedding_dim *= 2
        self.last_step = linear_block(self.z_dim, self.embedding_dim, config['dropout'], config['embedding_activate_fn'])

    def forward(self, X, flat):
        ts_encoding = self.ts_encoder(X, flat)
        embedding = self.last_step(ts_encoding.view(-1, self.z_dim)).view(X.shape[0], X.shape[1], self.embedding_dim)
        return embedding


class Predictor(nn.Module):
    def __init__(self, config, num_cats):
        super(Predictor, self).__init__()
        self.num_duration_tasks = len(config['duration_prediction_tasks'])
        self.num_binary_tasks = len(config['binary_prediction_tasks'])
        self.max_seq_len = config['max_seq_len']
        self.categorical_tasks = config['categorical_prediction_tasks']
        self.duration_predictor = linear_block_no_bn(config['embedding_dim'], self.num_duration_tasks, activation='exp')
        self.binary_predictor = nn.Linear(config['embedding_dim'], self.num_binary_tasks)  # no activation function because we are using the BCEWithLogitsLoss which adds a sigmoid in there
        self.categorical_predictor = nn.ModuleDict()
        for task_name in self.categorical_tasks:
            num_categorical_outputs = num_cats[task_name]
            self.categorical_predictor[task_name] = nn.Sequential(
                linear_block(config['embedding_dim'], config['h_dim_decoder'], dropout=config['dropout'], activation=config['fc_activate_fn']),
                linear_block_no_bn(config['h_dim_decoder'], num_categorical_outputs, activation='softmax'))

    def forward(self, embedding):
        duration_pred = self.duration_predictor(embedding).view(-1, self.max_seq_len, self.num_duration_tasks)
        binary_pred = self.binary_predictor(embedding).view(-1, self.max_seq_len, self.num_binary_tasks)
        categorical_pred = {}
        for task_name in self.categorical_tasks:
            categorical_output = self.categorical_predictor[task_name](embedding)
            categorical_pred[task_name] = categorical_output.view(-1, self.max_seq_len, categorical_output.shape[1])
        return duration_pred, binary_pred, categorical_pred


class Reconstruction(nn.Module):
    def __init__(self, config):
        super(Reconstruction, self).__init__()
        self.F = config['F']
        self.num_binary_tasks = len(config['binary_reconstruction_tasks'])
        self.num_continuous_tasks = len(config['continuous_reconstruction_tasks'])
        self.max_seq_len = config['max_seq_len']
        self.last_timestep_decoder = nn.Sequential(
            linear_block(config['embedding_dim'], config['h_dim_decoder'], dropout=config['dropout'], activation=config['fc_activate_fn']),
            linear_block_no_bn(config['h_dim_decoder'], config['F'], activation='tanh'))  # tanh is the output layer, because the timeseries are scaled between -4 and 4

        self.forecast = nn.Sequential(
            linear_block(config['embedding_dim'], config['h_dim_decoder'], dropout=config['dropout'], activation=config['fc_activate_fn']),
            linear_block_no_bn(config['h_dim_decoder'], config['F'], activation='tanh'))  # tanh is the output layer, because the timeseries are scaled between -4 and 4

        self.static_binary_decoder = nn.Linear(config['embedding_dim'], self.num_binary_tasks)  # no activation function because we are using the BCEWithLogitsLoss which adds a sigmoid in there
        self.static_continuous_decoder = linear_block_no_bn(config['embedding_dim'], self.num_continuous_tasks, activation='tanh')

    def forward(self, embedding):
        # the multiplication by 4 is because the data is strictly between -4 and 4, and the tanh activation only allows for -1 to 1
        last_timestep_reconstruction = self.last_timestep_decoder(embedding).view(-1, self.max_seq_len, self.F) * 4
        forecast = self.forecast(embedding).view(-1, self.max_seq_len, self.F) * 4
        static_binary_reconstruction = self.static_binary_decoder(embedding).view(-1, self.max_seq_len, self.num_binary_tasks)
        # only multiplying by 2 because the data for these tasks happens to vary within -2 and 2, again the tanh function is applied
        static_continuous_reconstruction = self.static_continuous_decoder(embedding).view(-1, self.max_seq_len, self.num_continuous_tasks) * 2
        return last_timestep_reconstruction, forecast, static_binary_reconstruction, static_continuous_reconstruction


class LossFunctions(nn.Module):
    def __init__(self, binary_weights, classification_weights):
        super(LossFunctions, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_loss_weighted = nn.BCEWithLogitsLoss(pos_weight=binary_weights, reduction='none')

        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.ce_loss_weighted = {}
        for task_name, weights in classification_weights.items():
            self.ce_loss_weighted[task_name] = nn.CrossEntropyLoss(reduction='none', weight=weights)

    def msle_loss(self, pred, label):
        return self.mse_loss(torch.log(pred + 1), torch.log(label + 1))


class PredictionOnlyEmbedding(nn.Module):
    def __init__(self, config, num_cats):
        super(PredictionOnlyEmbedding, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.encoder = Encoder(config)
        self.predictor = Predictor(config, num_cats)
        self.variational = config['variational']

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def get_mask(self, ts_data):
        ts_mask = torch.sign(torch.max(torch.abs(ts_data), dim=2)[0])
        return ts_mask.unsqueeze(dim=2)

    def forward(self, ts_data, s_data):
        ts_mask = self.get_mask(ts_data)
        embedding = self.encoder(ts_data, s_data)
        if self.variational:
            x = embedding.view(-1, 2, self.embedding_dim)
            # get `mu` and `log_var`
            mu = x[:, 0, :] # the first feature values as mean
            log_var = x[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            embedding = self.reparameterize(mu, log_var)
        else:
            embedding = embedding.view(-1, self.embedding_dim)
        return embedding, ts_mask, self.predictor(embedding)


class PredictionReconstructionEmbedding(nn.Module):
    def __init__(self, config, num_cats):
        super(PredictionReconstructionEmbedding, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.encoder = Encoder(config)
        self.predictor = Predictor(config, num_cats)
        self.decoder = Reconstruction(config)
        self.variational = config['variational']

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def get_mask(self, ts_data):
        ts_mask = torch.sign(torch.max(torch.abs(ts_data), dim=2)[0])
        return ts_mask.unsqueeze(dim=2)

    def forward(self, ts_data, s_data):
        ts_mask = self.get_mask(ts_data)
        embedding = self.encoder(ts_data, s_data)
        if self.variational:
            x = embedding.view(-1, 2, self.embedding_dim)
            # get `mu` and `log_var`
            mu = x[:, 0, :] # the first feature values as mean
            log_var = x[:, 1, :] # the other feature values as variance
            # get the latent vector through reparameterization
            embedding = self.reparameterize(mu, log_var)
        else:
            embedding = embedding.view(-1, self.embedding_dim)
        return embedding, ts_mask, self.predictor(embedding), self.decoder(embedding)


if __name__=='__main__':

    from train_prediction_only import get_data

    parser = initialise_arguments()
    config = parser.parse_args()
    config = add_configs(config)

    train_set, val_set, test_set, label_names, feature_names, ventids = get_data(config)
    num_cats = {}
    _, _, y_true = val_set.tensors
    for task_name in config['categorical_prediction_tasks']:
        ind = label_names.index(task_name)
        num_cats[task_name] = len(y_true[:, :, ind].unique())

    PredictionOnlyEmbedding(config, num_cats)