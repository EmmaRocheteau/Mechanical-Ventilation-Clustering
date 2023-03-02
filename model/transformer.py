import math
import torch
import torch.nn as nn


# PositionalEncoding adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html. I made the following
# changes:
    # Took out the dropout
    # Changed the dimensions/shape of pe
# I am using the positional encodings suggested by Vaswani et al. as the Attend and Diagnose authors do not specify in
# detail how they do their positional encodings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # changed from max_len * d_model to 1 * d_model * max_len
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is B * d_model * T
        # self.pe[:, :, :X.size(2)] is 1 * d_model * T but is broadcast to B when added
        X = X + self.pe[:, :, :X.size(2)]  # B * d_model * T
        return X  # B * d_model * T


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.F = config['F']
        self.no_flat_features = config['no_flat_features']
        self.num_layers = config['num_layers_transformer']
        self.h_dim_transformer = config['h_dim_transformer']
        self.transformer_dropout = config['transformer_dropout']
        self.d_model = config['d_model']
        self.num_heads = config['num_heads']
        self.max_seq_len = config['max_seq_len']
        self.gpu = config['gpus']
        self.input_channels = self.F * 2 + 1 + self.no_flat_features

        self.input_embedding = nn.Conv1d(in_channels=self.input_channels, out_channels=self.d_model, kernel_size=1)  # B * C * T
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)
        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads,
                                                              dim_feedforward=self.h_dim_transformer, dropout=self.transformer_dropout,
                                                              activation=config['fc_activate_fn'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.trans_encoder_layer, num_layers=self.num_layers)

    def _causal_mask(self, size=None):
        if self.gpu:
            mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1).cuda()
        else:
            mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask  # T * T

    def forward(self, X):
        # X is B * (2F + 1 + no_flat_features) * T

        # multiplication by root(d_model) as described in Vaswani et al. 2017 section 3.4
        X = self.input_embedding(X.view(-1, self.input_channels, self.max_seq_len)) * math.sqrt(self.d_model)  # B * d_model * T
        X = self.pos_encoder(X)  # B * d_model * T
        X = self.transformer_encoder(src=X.permute(2, 0, 1), mask=self._causal_mask(size=self.max_seq_len))  # T * B * d_model
        return X.permute(1, 0, 2).contiguous()  # B * T * d_model