import pandas as pd
import numpy as np
import os


settings_map = {'patient_triggered': ['Bi Vente', 'NAVA', 'PRVC ', 'PRVC (trig)', 'PS/CPAP (trig)', 'SIMV(PC)+PS',
                                      'SIMV(VC)+PS', 'VC (trig)', 'VS', 'SIMV_ASB', 'CPAP', 'BIPAP-SIMV/ASB', 'MMV_ASB',
                                      'MMV/ASB', 'ASB', 'IPPV/ASSIST', 'CPPV/ASSIST', 'CPPV_Assist', 'IPPV_Assist',
                                      'SIMV/ASB', 'CPAP_ASB', 'PS/CPAP', 'BIPAP/ASB', 'CPAP/ASB'],
                'mandatory_ventilation': ['MMV', 'VC ', 'PC ', 'Pressure Controled', 'PC (No trig)', 'PRVC (No trig)',
                                          'VC (No trig)', 'CPPV', 'IPPV', 'SIMV', 'BIPAP'],
                }

def reconfigure_timeseries(timeseries, offset_column, feature_column=None, test=False):
    """
    Pivots the time series.

    :param timeseries: pandas DataFrame, which initially looks something like this:

                ventid             item  value  offset
    0            0      Leuco's (bloed)    6.5    -840
    1            0          CRP (bloed)    7.0    -840
    2            0       Hartfrequentie   83.0       0
    3            0            Ademfreq.   13.0       0

    :param offset_column: string, which indicates the column of the dataframe which should give the number of minutes
    relative to the time that the ventilated episode starts.
    :param feature_column: string, which indicates the column containing the features which will need to form the columns
    of the pivoted table.
    :param test: bool, for debugging purposes.

    :return: pandas DataFrame, which looks something like this:

    item                     ABP gemiddeld Ademfreq.  ... pCO2 (bloed)  pH (bloed)
    ventid time                                       ...
    0      -1 days +10:00:00           NaN       NaN  ...          NaN        NaN
           00:00:00                   68.0      13.0  ...         39.0       7.40
           00:24:00                    NaN       NaN  ...          NaN        NaN
           00:30:00                   83.0      13.0  ...          NaN        NaN
           01:30:00                   75.0      13.0  ...         37.0       7.40

    """
    if test:
        timeseries = timeseries.iloc[:5000000]
    # set a timedelta index which is a specialised index for keeping track of the time before or since an event
    # (the onset of ventilation in our case). it will be important for using forward filling of the data later
    timeseries.set_index(['ventid', pd.to_timedelta(timeseries[offset_column], unit='T')], inplace=True)  # T unit is minutes
    timeseries.drop(columns=offset_column, inplace=True)
    if feature_column is not None:
        # pivot the feature column so that it makes a new column for each feature type in the pivoted table.
        timeseries = timeseries.pivot_table(columns=feature_column, index=timeseries.index)
    # convert index to multi-index with both ventilation episode and timedelta stamp
    timeseries.index = pd.MultiIndex.from_tuples(timeseries.index, names=['ventid', 'time'])
    return timeseries

def resample_and_mask(timeseries, data_path, header, mask_decay=True, decay_rate=0.8, test=False,
                       verbose=False, vent_cols=[]):
    """
    This is called at the end of timeseries_main (defined below).

    :param timeseries: pandas Dataframe; initially looks like:
                              ABP gemiddeld  ...  patient_triggered
    ventid time                              ...
    0      -1 days +10:00:00            NaN  ...                NaN
           0 days 00:00:00        -0.750000  ...                NaN
           0 days 00:24:00              NaN  ...                NaN
           0 days 00:30:00        -0.125000  ...                NaN
           0 days 01:30:00        -0.458333  ...                NaN
           0 days 02:30:00        -0.666667  ...                NaN
           0 days 03:30:00        -0.541667  ...                NaN
    :param data_path:
    :param header:
    :param mask_decay:
    :param decay_rate:
    :param test:
    :param verbose:
    :param max_length:
    :param min_length:
    :param vent_cols:
    :return:
    """
    if test:
        verbose = True
        #mask_decay = False  # the binary mask is quicker to run, so we use this in debug mode
    if verbose:
        print('Resampling to 1 hour intervals...')
    # take the mean of any duplicate index entries for unstacking. note that for the ventilator settings, there are only
    # 1s to indicate the presence of that setting, the absence will be NaN and not 0, so this step is intended for the other features
    timeseries = timeseries.groupby(level=[0, 1]).mean()

    timeseries.reset_index(level=1, inplace=True)
    timeseries.time = timeseries.time.dt.ceil(freq='H')
    timeseries.set_index('time', append=True, inplace=True)
    timeseries.reset_index(level=0, inplace=True)

    # resampled (below) looks like:
    #                           ABP gemiddeld  ...  patient_triggered
    # ventid time                              ...
    # 0      -1 days +10:00:00            NaN  ...                NaN
    #        -1 days +11:00:00            NaN  ...                NaN
    #        -1 days +12:00:00            NaN  ...                NaN
    #        -1 days +13:00:00            NaN  ...                NaN
    #        -1 days +14:00:00            NaN  ...                NaN
    #        -1 days +15:00:00            NaN  ...                NaN
    #        -1 days +16:00:00            NaN  ...                NaN
    #        -1 days +17:00:00            NaN  ...                NaN
    #        -1 days +18:00:00            NaN  ...                NaN
    #        -1 days +19:00:00            NaN  ...                NaN
    #        -1 days +20:00:00            NaN  ...                NaN
    #        -1 days +21:00:00            NaN  ...                NaN
    #        -1 days +22:00:00            NaN  ...                NaN
    #        -1 days +23:00:00            NaN  ...                NaN
    #        0 days 00:00:00        -0.750000  ...                NaN
    #        0 days 01:00:00        -0.125000  ...                NaN
    #        0 days 02:00:00        -0.458333  ...                NaN
    #        0 days 03:00:00        -0.666667  ...                NaN
    #        0 days 04:00:00        -0.541667  ...                NaN
    resampled = timeseries.groupby('ventid').resample('H', closed='right', label='right').mean().drop(columns='ventid')
    del (timeseries)

    def apply_mask_decay(mask_bool):
        """
        :param mask_bool: pandas DataFrame; for a particular feature it tells us whether there is a measurement during this time window
        # ventid  time
        # 0       -1 days +10:00:00    False
        #         -1 days +11:00:00    False
        #         -1 days +12:00:00    False
        #         -1 days +13:00:00    False
        #         -1 days +14:00:00    False
        #         -1 days +15:00:00    False
        #         -1 days +16:00:00    False
        #         -1 days +17:00:00    False
        #         -1 days +18:00:00    False
        #         -1 days +19:00:00    False
        #         -1 days +20:00:00    False
        #         -1 days +21:00:00    False
        #         -1 days +22:00:00    False
        #         -1 days +23:00:00    False
        #         00:00:00              True
        #         01:00:00              True
        #         02:00:00              True
        #         03:00:00              True
        #         04:00:00              True
        #         05:00:00              True
        #         06:00:00              True
        #         07:00:00              True
        #         08:00:00              True
        #         09:00:00              True
        #         10:00:00              True
        #         11:00:00              True
        #         12:00:00             False
        #         13:00:00             False
        #         14:00:00             False
        # we can see that there are measurements between 00:00:00 and 11:00:00 but then no update for the next three hours

        :return: pandas DataFrame; tells you the mask decay variable for this feature, it looks like:
        # ventid  time
        # 0       -1 days +10:00:00    0.000
        #         -1 days +11:00:00    0.000
        #         -1 days +12:00:00    0.000
        #         -1 days +13:00:00    0.000
        #         -1 days +14:00:00    0.000
        #         -1 days +15:00:00    0.000
        #         -1 days +16:00:00    0.000
        #         -1 days +17:00:00    0.000
        #         -1 days +18:00:00    0.000
        #         -1 days +19:00:00    0.000
        #         -1 days +20:00:00    0.000
        #         -1 days +21:00:00    0.000
        #         -1 days +22:00:00    0.000
        #         -1 days +23:00:00    0.000
        #         00:00:00             1.000
        #         01:00:00             1.000
        #         02:00:00             1.000
        #         03:00:00             1.000
        #         04:00:00             1.000
        #         05:00:00             1.000
        #         06:00:00             1.000
        #         07:00:00             1.000
        #         08:00:00             1.000
        #         09:00:00             1.000
        #         10:00:00             1.000
        #         11:00:00             1.000
        #         12:00:00             0.800
        #         13:00:00             0.640
        #         14:00:00             0.512
        # you can see the decay between 11:00:00 and 14:00:00
        """
        mask = mask_bool.astype(int)  # replace False = 0 and True = 1
        mask.replace({0: np.nan}, inplace=True)  # so that forward fill works
        inv_mask_bool = ~mask_bool  # take the inverse of the bool mask so True means no sample and False means sample

        # count_non_measurements starts from 0 if there is a measurement and starts counting upwards in 1s per hour that there is no measurment.
        # it's always 0 if there is a measurement
        count_non_measurements = inv_mask_bool.cumsum() - \
                                 inv_mask_bool.cumsum().where(mask_bool).ffill().fillna(0)

        # mask.ffill().fillna(0) is 0 if there hasn't been a measurement yet recorded for this variable, but it is 1 thereafter
        # (it makes sure we don't start counting decay before the first measurement has been taken)

        # (decay_rate**count_non_measurements) applies the exponential to the decay_rate
        decay_mask = mask.ffill().fillna(0) * (decay_rate**count_non_measurements)  # see above for what it looks like
        return decay_mask

    # store which values had to be imputed
    if mask_decay:
        if verbose:
            print('Calculating mask decay features...')
        mask_bool = resampled.notnull()
        mask = mask_bool.groupby('ventid').transform(apply_mask_decay)  # apply mask decay separately to each ventid, don't propagate from the previous patient
        del (mask_bool)
    else:
        if verbose:
            print('Calculating binary mask features...')
        mask = resampled.notnull()
        mask = mask.astype(int)

    # carry forward missing values (note they will still be 0 in the nulls table), except for the ventilator settings
    # which shouldn't be assumed to hold true carrying forwards because it is categorical, and in fact if the ventilator
    # setting doesn't change, then the same setting is logged at a frequence of at least once per hour.
    if verbose:
        print('Filling missing data forwards...')
    all_columns = resampled.columns
    timeseries_columns = [column for column in all_columns if column not in vent_cols]
    resampled[timeseries_columns] = resampled[timeseries_columns].fillna(method='ffill')

    # simplify the indexes of both tables
    mask = mask.rename(index=dict(zip(mask.index.levels[1],
                                      mask.index.levels[1].days*24 + mask.index.levels[1].seconds//3600)))
    resampled = resampled.rename(index=dict(zip(resampled.index.levels[1],
                                                resampled.index.levels[1].days*24 +
                                                resampled.index.levels[1].seconds//3600)))
 
    if verbose:
        print('Filling in remaining values with zeros...')
    resampled.fillna(0, inplace=True)

    # rename the columns in pandas for the mask so it doesn't complain
    mask.columns = [str(col) + '_mask' for col in mask.columns]

    # merge the mask with the features
    final = pd.concat([resampled, mask], axis=1)
    final.reset_index(level=1, inplace=True)
    final = final.loc[final.time > 0]  # only take the variables recorded after the ventilation episode, but old measurements can still propagate with forward filling and the masking

    # save to csv
    if test is False:
        if verbose:
            print('Saving...')
        final.to_csv(data_path + 'preprocessed_timeseries.csv', mode='a', header=header)
    return

def timeseries_main(data_path, test=False):
    """
    Loads timeseries.csv and ventilator_settings.csv and generates preprocessed timeseries data and saves it to preprocessed_timeseries.csv.

    :param data_path: path to the data, usually found in paths.json.
    :param test: bool, for debugging purposes.
    :return: the preprocessed_timeseries.csv will look something like:

    ventid,time,ABP gemiddeld,Ademfreq.,Alb.Chem (bloed),CRP (bloed),EtCO2 (%),Exp. tidal volume,Hartfrequentie,Lactaat (bloed),Leuco's (bloed),O2 concentratie,PC,PEEP (Set),PO2 (bloed),Piek druk,Saturatie (Monitor),mandatory_ventilation,pCO2 (bloed),pH (bloed),patient_triggered,ABP gemiddeld_mask,Ademfreq._mask,Alb.Chem (bloed)_mask,CRP (bloed)_mask,EtCO2 (%)_mask,Exp. tidal volume_mask,Hartfrequentie_mask,Lactaat (bloed)_mask,Leuco's (bloed)_mask,O2 concentratie_mask,PC_mask,PEEP (Set)_mask,PO2 (bloed)_mask,Piek druk_mask,Saturatie (Monitor)_mask,mandatory_ventilation_mask,pCO2 (bloed)_mask,pH (bloed)_mask,patient_triggered_mask
    0,-14,0.0,0.0,0.0,-0.9720062208398134,0.0,0.0,0.0,0.0,-0.8257261486544654,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    0,-13,0.0,0.0,0.0,-0.9720062208398134,0.0,0.0,0.0,0.0,-0.8257261486544654,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.8888888888888888,0.0,0.0,0.0,0.0,0.8888888888888888,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    0,-12,0.0,0.0,0.0,-0.9720062208398134,0.0,0.0,0.0,0.0,-0.8257261486544654,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4444444444444444,0.0,0.0,0.0,0.0,0.4444444444444444,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    0,-11,0.0,0.0,0.0,-0.9720062208398134,0.0,0.0,0.0,0.0,-0.8257261486544654,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2962962962962963,0.0,0.0,0.0,0.0,0.2962962962962963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
    0,-10,0.0,0.0,0.0,-0.9720062208398134,0.0,0.0,0.0,0.0,-0.8257261486544654,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.2222222222222222,0.0,0.0,0.0,0.0,0.2222222222222222,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    Note that the time can start up to 24 hours before the ventilation episode begins because we are tracking non-ventilation features such
    as heart rates and blood pressures before the ventilation episode begins.
    """

    # make sure the preprocessed_timeseries.csv file is removed initially because we want to append to it.
    if test is False:
        print('==> Removing the preprocessed_timeseries.csv file if it exists...')
        try:
            os.remove(data_path + 'preprocessed_timeseries.csv')
        except FileNotFoundError:
            pass

    print('==> Loading data from timeseries files...')
    if test:
        timeseries = pd.read_csv(data_path + 'timeseries.csv', nrows=500000)
        ventilator_settings = pd.read_csv(data_path + 'ventilator_settings.csv', nrows=500000)
    else:
        timeseries = pd.read_csv(data_path + 'timeseries.csv')
        ventilator_settings = pd.read_csv(data_path + 'ventilator_settings.csv')

    # PREPARING THE TIMESERIES DATA

    # the two modes of temperature measurement can be the same variable
    timeseries['item'].replace({'Temp Bloed': 'Temp.', 'Temp Axillair': 'Temp.'}, inplace=True)

    # we need to get an offset column which reflects time since the start of the ventilation episode rather than the
    # time since the ICU admission (which is what the measuredat variable is currently using),
    # this requires the ventstart time for each ventid in labels, which we obtain from labels.csv
    labels = pd.read_csv(data_path + 'labels.csv')
    timeseries = pd.merge(timeseries, labels[['ventid', 'ventstart']], on='ventid', how='inner')
    timeseries['offset'] = timeseries['measuredat'] - timeseries['ventstart']  # create a new 'offset' column relative to the ventilation
    timeseries.drop(columns=['measuredat', 'ventstart'], inplace=True)

    # PREPARING THE VENTILATOR SETTINGS DATA

    # we start by sorting the offset column
    ventilator_settings = pd.merge(ventilator_settings, labels[['ventid', 'ventstart', 'ventstop']], on='ventid', how='inner')
    ventilator_settings['offset'] = ventilator_settings['measuredat'] - ventilator_settings['ventstart']

    # then we want to do some 'filling in' of the ventilator settings so that it's sampled every minute without gaps.
    # this is because we won't be able to do forward filling in the usual way AFTER the table has been pivoted because
    # ventilator settings are categorical, which means that we can't treat each type of setting as independent from one
    # another (i.e. if the patient is on mandatory ventilation then they can't simultaneously be on patient triggered).
    # Therefore, we need to do the forward filling BEFORE the table is pivoted. We use minutes as the unit for forward
    # filling because we want to retain the information if the patient was ventilated with both types of setting within
    # a particular hour long interval.

    # originally the data looks like:
    #         ventid           value  measuredat  ventstart  offset
    # 0            0             PC          342        342       0
    # 1            0             PC          372        342      30
    # 2            0             PC          432        342      90
    # 3            0             PC          492        342     150
    # 4            0             PC          552        342     210
    # 5            0             PC          612        342     270

    # we want it to look like:
    #          ventid value  offset
    # 0             0   PC        0
    # 1             0   PC        1
    # 2             0   PC        2
    # 3             0   PC        3
    # 4             0   PC        4
    # 5             0   PC        5
    # 6             0   PC        6

    # concatenate stop time entries into the table with null for the value column - this will make the ffill work all
    # the way to the end of the ventilation episode
    ventilator_settings['ventstop'] = ventilator_settings['ventstop'] - ventilator_settings['ventstart']  # redefine ventstop as relative to the ventilation start time
    stoptimes = ventilator_settings.drop_duplicates(subset='ventid')[['ventid', 'ventstop']]
    stoptimes.columns = ['ventid', 'offset']
    stoptimes.offset = stoptimes.offset.astype(int)
    ventilator_settings = pd.concat([ventilator_settings, stoptimes], ignore_index=True, sort=False)
    ventilator_settings.sort_values(['ventid', 'offset', 'value'], inplace=True, na_position='first')
    ventilator_settings.reset_index(inplace=True)
    ventilator_settings.drop(columns=['ventstop', 'index'], inplace=True)

    # the 'spacing' is a slightly hacky way to make sure the timedelta index doesn't complain about not being unique, the important
    # thing is to make sure the timedelta faithfully records the relative time gaps between observations for a given patient
    stoptimes['spacing'] = stoptimes['offset'].cumsum().shift(1).fillna(0).astype(int) + stoptimes['ventid']  # the ventid is just to give it slightly more spacing so that the last time isn't the same as the start of the next episode
    ventilator_settings = pd.merge(ventilator_settings, stoptimes[['ventid', 'spacing']], on='ventid', how='inner')
    #ventilator_settings['spacing'] = ventilator_settings['ventid'] + ventilator_settings['offset_shift']
    spacings = ventilator_settings.drop_duplicates(subset='ventid')[['ventid', 'spacing']]  # obtain a record of the spacings for each patient

    ventilator_settings['timedelta_with_spacing'] = pd.to_timedelta(ventilator_settings['offset'] + ventilator_settings['spacing'], unit='T')  # resample to minutes (T means minutes)
    # sometimes there are multiple ventilator settings recorded within the same minute, I just keep the last one as an assumption
    # that this is the correct one for the timestamp (note that I set the nulls to appear first rather than last for the
    # case where there is another ventilator setting recorded in the last minute of the ventilation episode)
    ventilator_settings.drop_duplicates(subset='timedelta_with_spacing', keep='last', inplace=True)

    # forward fill the data while it is still being treated as one variable (rather than separate categorical variables)
    ventilator_settings = ventilator_settings.set_index('timedelta_with_spacing').groupby('ventid').resample('T')['value'].ffill().reset_index()
    # get the number of minutes from the timestamp as an integer
    ventilator_settings['minutes_with_spacing'] = (ventilator_settings.timedelta_with_spacing.dt.days*24*60 + ventilator_settings.timedelta_with_spacing.dt.seconds/60).astype(int)

    # remove the spacings
    ventilator_settings = pd.merge(ventilator_settings, spacings, on='ventid', how='inner')
    ventilator_settings['offset'] = ventilator_settings['minutes_with_spacing'] - ventilator_settings['spacing']

    # get rid of columns that aren't needed
    ventilator_settings.drop(columns=['timedelta_with_spacing', 'minutes_with_spacing', 'spacing'], inplace=True)
    del (labels)

    print('==> Reconfiguring timeseries...')
    timeseries = reconfigure_timeseries(timeseries,
                                        offset_column='offset',
                                        feature_column='item',
                                        test=test)
    timeseries.columns = timeseries.columns.droplevel()  # drop the column level which simply holds the word 'value'

    # ADDING THE P/F RATIO AND LUNG COMPLIANCE FEATURES

    # a lot of the measurements are asynchronous, and it seems a shame to only update when there are both present at the same time
    # i.e. if we use timeseries['P/F ratio'] = timeseries['PO2 (bloed)']/timeseries['O2 concentratie'], then we get this behaviour:
    # item                      PO2 (bloed)  O2 concentratie  P/F ratio
    # ventid time
    # 0      -1 days +10:00:00          NaN              NaN        NaN
    #        00:00:00                  90.0             50.0   1.800000
    #        00:24:00                   NaN              NaN        NaN
    #        00:30:00                   NaN             50.0        NaN
    #        01:30:00                 149.0             51.0   2.921569

    # we do some forward filling to allow the P/F ratio to be calculated when only one value is updated:
    # item                      PO2 (bloed)  O2 concentratie  P/F ratio
    # ventid time
    # 0      -1 days +10:00:00          NaN              NaN        NaN
    #        00:00:00                  90.0             50.0   1.800000
    #        00:24:00                   NaN              NaN   1.800000
    #        00:30:00                   NaN             50.0   1.800000
    #        01:30:00                 149.0             51.0   2.921569
    timeseries['P/F ratio'] = timeseries.groupby('ventid')['PO2 (bloed)'].ffill()/timeseries.groupby('ventid')['O2 concentratie'].ffill()
    timeseries['lung compliance'] = timeseries.groupby('ventid')['Exp. tidal volume'].ffill()/(timeseries.groupby('ventid')['Piek druk'].ffill() - timeseries.groupby('ventid')['PEEP (Set)'].ffill())

    # however, for the masking to work properly we should insert NaN when BOTH PO2 and FiO2 are NaN (i.e. there is no
    # update to P/F we want the mask to say it hasn't been recorded for this time, so we want this behaviour:
    # item                      PO2 (bloed)  O2 concentratie  P/F ratio
    # ventid time
    # 0      -1 days +10:00:00          NaN              NaN        NaN
    #        00:00:00                  90.0             50.0   1.800000
    #        00:24:00                   NaN              NaN        NaN
    #        00:30:00                   NaN             50.0   1.800000
    #        01:30:00                 149.0             51.0   2.921569
    timeseries['P/F ratio'] = np.where(np.isnan(timeseries[['PO2 (bloed)', 'O2 concentratie']]).all(axis=1), np.nan, timeseries['P/F ratio'])
    timeseries['lung compliance'] = np.where(np.isnan(timeseries[['Exp. tidal volume', 'Piek druk', 'PEEP (Set)']]).all(axis=1), np.nan, timeseries['lung compliance'])

    print('==> Reconfiguring ventilator settings...')
    # get a map of the original settings to the simplified settings
    inv_settings_map = {value: key for (key, values) in settings_map.items() for value in values}
    ventilator_settings['value'].replace(inv_settings_map, inplace=True)
    ventilator_settings = pd.get_dummies(ventilator_settings, columns=['value'])

    # NOTE: the masking on the ventilator settings doesn't reveal how long it's been since the ventilator setting was
    # updated, but rather how long it has been since the patient had their ventilator status changed to this specific setting.
    # hence we replace the 0s with NaN so that the "masking" is configured correctly, and so that there is no averaging
    # across 0s in the hourly resampling stage.
    ventilator_settings.columns = ventilator_settings.columns.str.replace('value_', '')  # remove 'value' in column names
    columns = ['mandatory_ventilation', 'patient_triggered']
    ventilator_settings[columns] = ventilator_settings[columns].replace({0: np.nan})  # replace 0s with NaN (this is for the decay masking to work correctly)
    ventilator_settings = reconfigure_timeseries(ventilator_settings,
                                                 offset_column='offset',
                                                 test=test)

    # most of the features are not normally distributed, so we don't divide by the standard deviation, instead we use the mid-90 centile range for normalisation
    # the nice thing about this approach is that outliers won't mess things up, as they would for the mean or standard deviation calculations
    quantiles = timeseries.quantile([0.05, 0.95])
    maxs = quantiles.loc[0.95]
    mins = quantiles.loc[0.05]
    timeseries = 2 * (timeseries - mins) / (maxs - mins) - 1
    pd.concat([mins, maxs], axis=1).to_csv(data_path + 'standardisation_limits.csv')  # save to csv for posthoc pipeline

    # we then need to make sure that ridiculous outliers are clipped to something sensible
    timeseries.clip(lower=-4, upper=4, inplace=True)  # room for +- 3 on each side, as variables are scaled roughly between -1 and 1

    # combine the ventilator settings with the other time series to get a master timeseries file.
    merged = timeseries.append(ventilator_settings, sort=True)

    # this calculates the masking, and does the forward filling before saving to csv (see above)
    resample_and_mask(merged, data_path, header=True, mask_decay=True, decay_rate=0.8, test=test, verbose=False, vent_cols=columns)

    return

if __name__=='__main__':
    from amsterdamUMCdb_preprocessing import amsterdam_path as data_path
    test = True
    timeseries_main(data_path, test)