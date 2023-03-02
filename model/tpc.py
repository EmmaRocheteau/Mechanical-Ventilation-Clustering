import torch
import torch.nn as nn
from torch import cat
from torch.nn.functional import pad


class TempPointConv(nn.Module):
    def __init__(self, config):

        # The timeseries data will be of dimensions B * (2F + 1) * T where:
        #   B is the batch size
        #   F is the number of features for convolution (N.B. we start with 2F because there are corresponding mask features)
        #   T is the number of timepoints
        #   The other 2 features represent the sequence number and the hour in the day

        # The flat data will be of dimensions B * no_flat_features

        super(TempPointConv, self).__init__()
        self.n_layers = config['num_layers_tpc']
        self.dropout_rate = config['dropout']
        self.temp_dropout_rate = config['temp_dropout_rate']
        self.kernel_size = config['kernel_size']
        self.temp_kernels = [config['no_temp_kernels']] * config['num_layers_tpc']
        self.point_sizes = [config['point_size']] * config['num_layers_tpc']
        self.momentum = config['momentum']
        self.last_linear_size = config['last_linear_size']
        self.F = config['F']
        self.no_flat_features = config['no_flat_features']

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.temp_dropout = nn.Dropout(p=self.temp_dropout_rate)
        self.remove_none = lambda x: tuple(xi for xi in x if xi is not None)  # removes None items from a tuple
        self.init_tpc()


    def init_tpc(self):

        # non-module layer attributes
        self.layers = []
        for i in range(self.n_layers):
            dilation = i * (self.kernel_size - 1) if i > 0 else 1  # dilation = 1 for the first layer, after that it captures all the information gathered by previous layers
            temp_k = self.temp_kernels[i]
            point_size = self.point_sizes[i]
            self.update_layer_info(layer=i, temp_k=temp_k, point_size=point_size, dilation=dilation, stride=1)

        # module layer attributes
        self.create_temp_pointwise_layers()

        # input shape: (B * T) * ((F + Zt) * (1 + Y) + no_flat_features)
        # output shape: (B * T) * last_linear_size
        input_size = (self.F + self.Zt) * (1 + self.Y) + self.no_flat_features
        self.final_point = nn.Linear(in_features=input_size, out_features=self.last_linear_size)

        return

    def update_layer_info(self, layer=None, temp_k=None, point_size=None, dilation=None, stride=None):

        self.layers.append({})
        if point_size is not None:
            self.layers[layer]['point_size'] = point_size
        if temp_k is not None:
            padding = [(self.kernel_size - 1) * dilation, 0]  # [padding_left, padding_right]
            self.layers[layer]['temp_kernels'] = temp_k
            self.layers[layer]['dilation'] = dilation
            self.layers[layer]['padding'] = padding
            self.layers[layer]['stride'] = stride

        return


    def create_temp_pointwise_layers(self):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)

        self.layer_modules = nn.ModuleDict()

        self.Y = 0
        self.Z = 0
        self.Zt = 0

        for i in range(self.n_layers):

            temp_in_channels = (self.F + self.Zt) * (1 + self.Y) if i > 0 else 2 * self.F  # (F + Zt) * (Y + 1)
            temp_out_channels = (self.F + self.Zt) * self.layers[i]['temp_kernels']  # (F + Zt) * temp_kernels
            linear_input_dim = (self.F + self.Zt - self.Z) * self.Y + self.Z + 2 * self.F + 1 + self.no_flat_features  # (F + Zt-1) * Y + Z + 2F + 1 + no_flat_features
            linear_output_dim = self.layers[i]['point_size']  # point_size

            temp = nn.Conv1d(in_channels=temp_in_channels,  # (F + Zt) * (Y + 1)
                             out_channels=temp_out_channels,  # (F + Zt) * Y
                             kernel_size=self.kernel_size,
                             stride=self.layers[i]['stride'],
                             dilation=self.layers[i]['dilation'],
                             groups=self.F + self.Zt)

            point = nn.Linear(in_features=linear_input_dim, out_features=linear_output_dim)

            bn_temp = nn.BatchNorm1d(num_features=temp_out_channels, momentum=self.momentum)
            bn_point = nn.BatchNorm1d(num_features=linear_output_dim, momentum=self.momentum)

            self.layer_modules[str(i)] = nn.ModuleDict({
                'temp': temp,
                'bn_temp': bn_temp,
                'point': point,
                'bn_point': bn_point})

            self.Y = self.layers[i]['temp_kernels']
            self.Z = linear_output_dim
            self.Zt += self.Z

        return


    # This is really where the crux of TPC is defined. This function defines one TPC layer, as in Figure 3 in the paper:
    # https://arxiv.org/pdf/2007.09483.pdf
    def temp_pointwise(self, B=None, T=None, X=None, repeat_flat=None, X_orig=None, temp=None, bn_temp=None, point=None,
                       bn_point=None, temp_kernels=None, point_size=None, padding=None, prev_temp=None, prev_point=None,
                       point_skip=None):

        ### Notation used for tracking the tensor shapes ###

        # Z is the number of extra features added by the previous pointwise layer (could be 0 if this is the first layer)
        # Zt is the cumulative number of extra features that have been added by all previous pointwise layers
        # Zt-1 = Zt - Z (cumulative number of extra features minus the most recent pointwise layer)
        # Y is the number of channels in the previous temporal layer (could be 0 if this is the first layer)
        # X shape: B * ((F + Zt) * (Y + 1)) * T; N.B exception in the first layer where there are also mask features, in this case it is B * 2F * T
        # repeat_flat shape: (B * T) * no_flat_features
        # X_orig shape: (B * T) * (2F + 1)
        # prev_temp shape: (B * T) * ((F + Zt-1) * (Y + 1))
        # prev_point shape: (B * T) * Z

        Z = prev_point.shape[1] if prev_point is not None else 0

        X_padded = pad(X, padding, 'constant', 0)  # B * ((F + Zt) * (Y + 1)) * (T + padding)
        X_temp = self.temp_dropout(bn_temp(temp(X_padded)))  # B * ((F + Zt) * temp_kernels) * T

        X_concat = cat(self.remove_none((prev_temp,  # (B * T) * ((F + Zt-1) * Y)
                                         prev_point,  # (B * T) * Z
                                         X_orig,  # (B * T) * (2F + 1)
                                         repeat_flat)),  # (B * T) * no_flat_features
                       dim=1)  # (B * T) * (((F + Zt-1) * Y) + Z + 2F + 1 + no_flat_features)

        point_output = self.dropout(bn_point(point(X_concat)))  # (B * T) * point_size

        # point_skip input: B * (F + Zt-1) * T
        # prev_point: B * Z * T
        # point_skip output: B * (F + Zt) * T
        point_skip = cat((point_skip, prev_point.view(B, T, Z).permute(0, 2, 1)), dim=1) if prev_point is not None else point_skip

        temp_skip = cat((point_skip.unsqueeze(2),  # B * (F + Zt) * 1 * T
                         X_temp.view(B, point_skip.shape[1], temp_kernels, T)),  # B * (F + Zt) * temp_kernels * T
                        dim=2)  # B * (F + Zt) * (1 + temp_kernels) * T

        X_point_rep = point_output.view(B, T, point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + temp_kernels), 1)  # B * point_size * (1 + temp_kernels) * T
        X_combined = self.relu(cat((temp_skip, X_point_rep), dim=1))  # B * (F + Zt) * (1 + temp_kernels) * T
        next_X = X_combined.contiguous().view(B, (point_skip.shape[1] + point_size) * (1 + temp_kernels), T)  # B * ((F + Zt + point_size) * (1 + temp_kernels)) * T

        temp_output = X_temp.permute(0, 2, 1).contiguous().view(B * T, point_skip.shape[1] * temp_kernels)  # (B * T) * ((F + Zt) * temp_kernels)

        return (temp_output,  # (B * T) * ((F + Zt) * temp_kernels)
                point_output,  # (B * T) * point_size
                next_X,  # B * ((F + Zt) * (1 + temp_kernels)) * T
                point_skip)  # for keeping track of the point skip connections; B * (F + Zt) * T


    def forward(self, X, flat, time_before_pred=0):

        # flat is B * no_flat_features
        # X is B * T * (2F + 1)
        # X_mask is B * T
        # (the batch is padded to the longest sequence, the + 2 is the time and the hour which are not for temporal convolution)

        # get rid of the time and hour fields - these shouldn't go through the temporal network
        # and split into features and indicator variables
        X_separated = torch.split(X[:, :, 1:], self.F, dim=2)  # tuple ((B * T * F), (B * T * F))

        # prepare repeat arguments and initialise layer loop
        B, T, _ = X_separated[0].shape
        repeat_flat = flat.repeat_interleave(T, dim=0).squeeze()  # (B * T) * no_flat_features
        X_orig = X.contiguous().view(B * T, 2 * self.F + 1)  # (B * T) * (2F + 1)
        repeat_args = {'repeat_flat': repeat_flat,
                       'X_orig': X_orig,
                       'B': B,
                       'T': T}
        next_X = torch.stack(X_separated, dim=2).reshape(B, 2 * self.F, T)  # B * 2F * T
        point_skip = X_separated[0].permute(0, 2, 1)  # keeps track of skip connections generated from linear layers; B * F * T
        temp_output = None
        point_output = None

        for i in range(self.n_layers):
            kwargs = dict(self.layer_modules[str(i)], **repeat_args)
            temp_output, point_output, next_X, point_skip = self.temp_pointwise(X=next_X, point_skip=point_skip,
                                                                        prev_temp=temp_output, prev_point=point_output,
                                                                        temp_kernels=self.layers[i]['temp_kernels'],
                                                                        padding=self.layers[i]['padding'],
                                                                        point_size=self.layers[i]['point_size'],
                                                                        **kwargs)


        # note that we cut off at time_before_pred hours here because the model is only valid from time_before_pred hours onwards
        combined_features = cat((flat.repeat_interleave(T - time_before_pred, dim=0).squeeze(),  # (B * (T - time_before_pred)) * no_flat_features
                                 next_X[:, :, time_before_pred:].permute(0, 2, 1).contiguous().view(B * (T - time_before_pred), -1)), dim=1)  # (B * (T - time_before_pred)) * (((F + Zt) * (1 + Y)) + no_flat_features) for tpc

        output = self.final_point(combined_features).reshape(B, T - time_before_pred, -1)

        return output