import numpy as np
import pandas as pd
from tqdm import tqdm


def padding(x, max_seq_len):
    """Sequence data padding.

    Args:
        - x: temporal features
        - max_seq_len: maximum sequence_length

    Returns:
        - x_hat: padded temporal features
    """
    # Shape of the temporal features
    seq_len, dim = x.shape
    col_name = x.columns.values

    # Padding (0)
    x_pad_hat = np.zeros([max_seq_len - seq_len, dim])
    x_pad_hat = pd.DataFrame(x_pad_hat, columns=col_name)

    x_hat = pd.concat((x, x_pad_hat), axis=0)
    x_hat['ventid'] = np.unique(x['ventid'])[0]

    x_hat = index_reset(x_hat)

    return x_hat


def index_reset(x):
    """Reset index in the pandas dataframe.

    Args:
        x: original pandas dataframe

    Returns:
        x: data with new indice
    """
    x = x.reset_index()
    if 'index' in x.columns:
        x = x.drop(columns=['index'])

    return x


def pd_list_to_np_array(x, drop_columns):
    """Convert list of pandas dataframes to 3d numpy array.

    Args:
        - x: list of pandas dataframe
        - drop_column: column names to drop before converting to numpy array

    Returns:
        - x_hat: 3d numpy array
    """
    x_hat = list()
    for component in x:
        temp = component.drop(columns=drop_columns)
        temp = np.asarray(temp)
        x_hat = x_hat + [temp]

    x_hat_new = np.asarray(x_hat)
    return x_hat_new


def list_diff(list1, list2):
    """Compute list differences in order.

    Args:
        - list1: first list
        - list2: second list

    Returns:
        - out: list difference
    """
    out = []
    for ele in list1:
        if not ele in list2:
            out.append(ele)

    return out


class FinalPreprocessing():
    """Define temporal, static, time and label features.

    Attributes:
        - prediction_labels: label names in list format
        - max_seq_len: maximum sequence length
    """

    def __init__(self, prediction_label_names, max_seq_len):
        self.prediction_label_names = prediction_label_names
        self.max_seq_len = max_seq_len

    def pad_sequence(self, x):
        """Returns pandas DataFrame with padded sequences.
        Args:
            - x: temporal data in DataFrame
        Returns:
            - x_hat: preprocessed temporal data in DataFrame
        """
        uniq_id = np.unique(x['ventid'])
        x_hat = list()
        # For each patient
        for i in tqdm(range(len(uniq_id))):
            idx_x = x.index[x['ventid'] == uniq_id[i]]
            if len(idx_x) >= self.max_seq_len:
                temp_x = x.loc[idx_x[: self.max_seq_len]]
                temp_x = index_reset(temp_x)
            # Padding
            else:
                temp_x = padding(x.loc[idx_x], self.max_seq_len)

            x_hat = x_hat + [temp_x]
        return pd.concat(x_hat)

    def sliding_window_label(self, y):
        """Set sliding window label.

        Set labels for window ahead prediction.

        Args:
            - y: labels

        Returns:
            - y: sliding window label
        """
        if self.window > 0:
            y[:, :(self.max_seq_len - self.window), :] = y[:, self.window:, :]
            y[:, (self.max_seq_len - self.window):, :] = 0
        return y

    def time_to_event(self, y):
        """
        Makes label descend over time

        :param y:
        :return:
        """
        for col in y.columns[1:]:
            y['temp'] = y[col] - (y.index / 24)
            y.loc[y[col] == 0, 'temp'] = 0
            y[col] = y['temp']
        y.drop(columns=['temp'], inplace=True)
        # make sure we don't have any negative values
        return y.clip(lower=0)

    def fit_transform(self, timeseries, flat_features, prediction_labels, data_path, debug=False):
        """Transform the dataset based on the Pandas Dataframe to numpy array.

        Returned dataset has temporal, static, time and label features

        Args:
            - dataset: original dataset

        Returns:
            - dataset: defined dataset for the certain problem
        """
        if debug:
            flat_features = flat_features[:100]
            prediction_labels = prediction_labels[:100]
            timeseries = timeseries.loc[timeseries['ventid'].isin(flat_features['ventid'])]
        x = timeseries
        s = flat_features

        # Set temporal and static features
        temporal_features = list_diff(x.columns.values.tolist(), ['ventid']) if x is not None else None
        static_features = list_diff(s.columns.values.tolist(),
                                    ['ventid', 'patientid', 'admissionid']) if s is not None else None

        # Remove negative times
        x = x.loc[x['time'] >= 0]

        # Merge dataframes
        other_features = x.merge(prediction_labels, how='inner', on='ventid')
        other_features = self.pad_sequence(other_features)
        y_duration = other_features[['ventid'] + self.prediction_label_names['duration']]
        y = self.time_to_event(y_duration)

        prediction_label_names = self.prediction_label_names['binary'] + list(self.prediction_label_names['categorical'].keys())
        y_other = other_features[['ventid'] + prediction_label_names]
        label_names = self.prediction_label_names['duration'] + prediction_label_names  # in a simple list
        for col in y_other.columns:
            y[col] = y_other[col]
        y = index_reset(y)

        # Set temporal + static features
        x = other_features[['ventid'] + temporal_features]
        x = index_reset(x)

        s = s[['ventid'] + static_features]
        s = index_reset(s)

        # Convert DataFrames to 3d numpy arrays.
        uniq_id = np.unique(x['ventid'])
        x_hat = list()
        s_hat = list()
        y_hat = list()
        ventids = list()
        # For each patient
        for i in tqdm(range(len(uniq_id))):
            temp_x = x[x['ventid'] == uniq_id[i]]
            temp_s = s[s['ventid'] == uniq_id[i]]
            temp_y = y[y['ventid'] == uniq_id[i]]

            x_hat = x_hat + [temp_x]
            s_hat = s_hat + [temp_s]
            y_hat = y_hat + [temp_y]
            ventids.append(uniq_id[i])

        data_x = pd_list_to_np_array(x_hat, drop_columns='ventid')
        data_s = pd_list_to_np_array(s_hat, drop_columns='ventid')
        data_y = pd_list_to_np_array(y_hat, drop_columns='ventid')

        # Feature name for visualization
        feature_names = {
            'temporal': temporal_features,
            'static': static_features,
            'label': label_names,
        }

        np.savez(data_path, data_x=data_x, data_s=data_s, data_y=data_y, ventids=ventids,
                 feature_names=np.array(feature_names),
                 label_names=np.array(label_names))


def data_preprocessing(data_loader, data_path, split):
    flat_features_train = pd.read_csv(data_path + split + '/flat.csv')
    labels_train = pd.read_csv(data_path + split + '/labels.csv')
    timeseries_train = pd.read_csv(data_path + split + '/timeseries.csv')
    data_loader.fit_transform(timeseries_train, flat_features_train, labels_train,
                              data_path + split + '/data.npz')

def final_processing_main(data_path, prediction_label_names):
    # one shot is like mortality
    # online is like tidal volume (predicting next choice of treatment a certain window ahead)
    # duration is the remaining time to an event
    data_loader = FinalPreprocessing(prediction_label_names=prediction_label_names,
                                     max_seq_len=21*24)

    print('==> Pre-processing train data...')
    data_preprocessing(data_loader, data_path, split='train')

    print('==> Pre-processing val data...')
    data_preprocessing(data_loader, data_path, split='val')

    print('==> Pre-processing test data...')
    data_preprocessing(data_loader, data_path, split='test')


if __name__ == '__main__':
    from amsterdamUMCdb_preprocessing import amsterdam_path as data_path

    prediction_label_names = {
        'binary': ['diedinicu', 'tracheostomy'],
        'duration': ['lengthofstay_from_ventstart', 'actualventduration'],
        'categorical': {
            'destination': ['destination_15', 'destination_16', 'destination_19',
                            'destination_2', 'destination_25', 'destination_33',
                            'destination_40', 'destination_41', 'destination_45',
                            'destination_9', 'destination_misc'],
            'diagnosissubgroup': ['diagnosissubgroup_Buikchirurgie',
                                  'diagnosissubgroup_CABG en Klepchirurgie',
                                  'diagnosissubgroup_Cardiovasculair',
                                  'diagnosissubgroup_None',
                                  'diagnosissubgroup_Overige',
                                  'diagnosissubgroup_Pulmonaal',
                                  'diagnosissubgroup_Traumatologie',
                                  'diagnosissubgroup_Tumor chirurgie',
                                  'diagnosissubgroup_Vaatchirurgie',
                                  'diagnosissubgroup_misc']
        }
    }
    
    final_processing_main(data_path, prediction_label_names)

