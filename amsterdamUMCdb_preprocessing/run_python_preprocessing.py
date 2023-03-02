# takes a while to run (> 4 hours)
from amsterdamUMCdb_preprocessing.timeseries import timeseries_main
from amsterdamUMCdb_preprocessing.flat_and_labels import flat_and_labels_main
from amsterdamUMCdb_preprocessing.split_train_test import split_train_test
from amsterdamUMCdb_preprocessing.final_processing import final_processing_main
from amsterdamUMCdb_preprocessing import amsterdam_path
import os


if __name__=='__main__':
    print('==> Removing the vents.txt file if it exists...')
    try:
        os.remove(amsterdam_path + 'vents.txt')
    except FileNotFoundError:
        pass
    timeseries_main(amsterdam_path, test=False)
    flat_and_labels_main(amsterdam_path)
    split_train_test(amsterdam_path, is_test=False, cleanup=False)

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

    final_processing_main(amsterdam_path, prediction_label_names)