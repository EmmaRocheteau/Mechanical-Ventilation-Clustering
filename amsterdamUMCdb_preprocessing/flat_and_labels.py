import pandas as pd

destination_features = {'15': 0, '16': 1, '19': 2, '2': 3, '25': 4, '33': 5, '40': 6, '41': 7, '45': 8, '9': 9,
                        'Overleden': 10, 'misc': 11}
diagnosissubgroup_features = {'Buikchirurgie': 0, 'CABG en Klepchirurgie': 1, 'Cardiovasculair': 2, 'None': 3,
                              'Overige': 4, 'Pulmonaal': 5, 'Traumatologie': 6, 'Tumor chirurgie': 7, 'Vaatchirurgie': 8,
                              'misc': 9}
admissionyeargroup_features = {'2003-2009': 0, '2010-2016': 1}


def preprocess_flat(flat, data_path):

    flat.set_index('ventid', inplace=True)

    # it's interesting that this is so unevenly distributed (11,390 men and only 5,220 women)
    flat['gender'].replace({'Man': 1, 'Vrouw': 0}, inplace=True)

    cat_features = ['origin', 'location', 'specialty', 'weightsource', 'heightsource']  # categorical features
    # get rid of any really uncommon values
    for f in cat_features:
        # make sure NaNs are filled in
        flat[f].fillna('None', inplace=True)
        # a category is considered too rare if it appears in less than 2% of patients - these are lumped into a 'misc' category
        too_rare = [value for value, count in flat[f].value_counts().iteritems() if count < len(flat)*0.02]
        flat.loc[flat[f].isin(too_rare), f] = 'misc'

    # convert the categorical features to one-hot
    flat = pd.get_dummies(flat, columns=cat_features)

    # these features are ordinal, so we make them numeric discrete
    flat['agegroup'] = flat['agegroup'].replace({'18-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5, '80+': 6})
    flat['weightgroup'] = flat['weightgroup'].replace({'59-': 1, '60-69': 2, '70-79': 3, '80-89': 4, '90-99': 5, '100-109': 6, '110+': 7})
    flat['heightgroup'] = flat['heightgroup'].replace({'159-': 1, '160-169': 2, '170-179': 3, '180-189': 4, '190+': 5})

    # most features are probably not normally distributed, so we don't use the standard deviation to normalise,
    # instead we use the 5th and 95th percentiles
    features_for_min_max = ['agegroup', 'weightgroup', 'heightgroup', 'ventcount', 'admissioncount']
    quantiles = flat[features_for_min_max].quantile([0.05, 0.95])
    maxs = quantiles.loc[0.95]
    mins = quantiles.loc[0.05]
    flat[features_for_min_max] = 2 * (flat[features_for_min_max] - mins) / (maxs - mins) - 1
    pd.concat([mins, maxs], axis=1).to_csv(data_path + 'standardisation_limits.csv', mode='a', header=False)

    # we then need to make sure that ridiculous outliers are clipped to something sensible especially because there is a division involved
    flat[features_for_min_max] = flat[features_for_min_max].clip(lower=-4, upper=4)

    # fill in the NaNs which exist in columns in columns gender, weightgroup and heightgroup
    flat['nullweight'] = flat['weightgroup'].isnull().astype(int)  # 505 null weights
    flat['nullheight'] = flat['heightgroup'].isnull().astype(int)  # 877 null heights
    flat['weightgroup'].fillna(0, inplace=True)
    flat['heightgroup'].fillna(0, inplace=True)
    # there are 427 missing genders but we might as well set this to 0.5 to tell the model we aren't sure
    flat['gender'].fillna(0.5, inplace=True)

    return flat

def preprocess_labels(labels):

    labels.set_index('ventid', inplace=True)

    cat_features = ['admissionyeargroup', 'destination', 'diagnosissubgroup']  # categorical features
    # get rid of any really uncommon values
    for f in cat_features:
        # make sure NaNs are filled in
        labels[f].fillna('None', inplace=True)
        # a category is considered too rare if it appears in less than 2% of patients - these are lumped into a 'misc' category
        too_rare = [value for value, count in labels[f].value_counts().iteritems() if count < len(labels)*0.02] + ['Overige']  # add overige, which means "other" to the misc category
        labels.loc[labels[f].isin(too_rare), f] = 'misc'

    labels_one_hot = pd.get_dummies(labels, columns=cat_features)
    labels['diedinicu'] = labels_one_hot['destination_Overleden']
    labels['duration_until_tracheostomy'] = labels['trachstart'] - labels['ventstart']  # if this is negative it means the first tracheostomy was before this ventilation episode
    labels['tracheostomy'] = labels['trachstart'].notnull().astype(int)
    labels.drop(columns='trachstart', inplace=True)

    # convert categorical features into numbers
    labels['destination'].replace(destination_features, inplace=True)
    labels['diagnosissubgroup'].replace(diagnosissubgroup_features, inplace=True)
    labels['admissionyeargroup'].replace(admissionyeargroup_features, inplace=True)

    # convert continuous labels to days (better range)
    labels['actualventduration'] = labels['actualventduration'] / (60 * 24)
    labels['duration_until_tracheostomy'] = labels['duration_until_tracheostomy'] / (60 * 24)
    labels['lengthofstay_total'] = (labels['dischargedat'] - labels['admittedat']) / (60 * 24)
    labels['lengthofstay_from_ventstart'] = (labels['dischargedat'] - labels['ventstart']) / (60 * 24)
    labels.drop(columns='lengthofstay', inplace=True)

    # some patients haven't yet died, or had tracheostomies, so we fill in these with -1 just to avoid NaN
    labels.fillna(value=-1, inplace=True)

    return labels

def flat_and_labels_main(data_path):

    print('==> Loading data from labels and flat features files...')
    flat = pd.read_csv(data_path + 'flat_features.csv')
    flat = preprocess_flat(flat, data_path)
    flat.sort_index(inplace=True)
    labels = pd.read_csv(data_path + 'labels.csv')
    labels = preprocess_labels(labels)
    labels.sort_index(inplace=True)

    # filters out any ventilation episodes that don't have timeseries associated
    try:
        with open(data_path + 'vents.txt', 'r') as f:
            ts_vents = [int(vent.rstrip()) for vent in f.readlines()]
    except FileNotFoundError:
        ts_vents = pd.read_csv(data_path + 'preprocessed_timeseries.csv')
        ts_vents = [x for x in ts_vents.ventid.unique()]
        with open(data_path + 'vents.txt', 'w') as f:
            for vent in ts_vents:
                f.write("%s\n" % vent)
    flat = flat.loc[ts_vents].copy()
    labels = labels.loc[ts_vents].copy()

    print('==> Saving finalised preprocessed labels and flat features...')
    flat.to_csv(data_path + 'preprocessed_flat.csv')
    labels.to_csv(data_path + 'preprocessed_labels.csv')
    return

if __name__=='__main__':
    from amsterdamUMCdb_preprocessing import amsterdam_path as data_path
    flat_and_labels_main(data_path)
