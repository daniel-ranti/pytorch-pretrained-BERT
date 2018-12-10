import hashlib
import logging
import os
import pandas as pd

# a random seed to be used for the pigeonhole
SEED = 'y7Xs9K7Eup'
TARGET_PATH = '/docker/home/eko/data/ICAHNC1/meg_curated/NLP_Validation_Data_Set_coded.xlsx'
REPORT_PATH = '/docker/home/eko/data/ICAHNC1/reports_raw/'


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def pigeonhole(number_of_groups, wedge, seed):
    """
    :param number_of_groups: the number of groups to bucket into
    :param wedge: something whose __str__ decides which random bucket to return, e.g. an emd5
    :param seed: something you provide in order to alter the random distribution
                 should be unique for each use case to avoid biases
    :returns: The stable "bucket" number for this input
    """
    int_returned = int(hashlib.md5(",".join((str(wedge), str(seed))).encode('utf8')).hexdigest(), 16)
    return (int_returned % number_of_groups) + 1

def _format_reports(targets, report_path=REPORT_PATH):
    """
    Takes in a dictionary of targets and path to reports keyed on the
    Targets: a dictionary keyed on accession_id with subdictionaries of target values
    Report_path: path to head CT reports
    """
    # Get a list of all reports
    report_files = os.listdir(report_path)
    reports = [f for f in report_files if f[-3:] == 'xls']

    # create the HCT dataframe in pd, filter by accession numbers in the filtered data
    hct_ddfs = [pd.read_excel(os.path.join(report_path, report)) for report in reports]
    hct_combined = pd.concat(hct_ddfs)

    hct_filtered = hct_combined[hct_combined['Accession Number'].isin(targets.keys())]
    hct_final = hct_filtered.loc[:, ['Accession Number', 'Report Text']]

    # reorder the dataframe such that it is a dictionary keyed on accession number
    hct_final.set_index('Accession Number', inplace=True)
    logger.info("  Head CT Dataframe")
    logger.info(hct_final.head(5))
    return hct_final.to_dict('index')

def _format_targets(target_path=TARGET_PATH):
    """
    Takes in a path to head CT targets and returns a properly formatted dictionary
    keyed on accession ID
    target_dict: a dictionary of targets
    target_list: a list of target names
    """
    targets = pd.read_excel(target_path)
    targets.set_index('accession_id', inplace=True)
    targets.drop(columns=['my_first_instrument_complete', 'redcap_data_access_group'], inplace=True)
    logger.info('Target Dataframe')
    logger.info(targets.head(5))
    # Get a list of targets
    target_list = targets.columns.tolist()
    return targets.to_dict('index'), target_list

def split_datasets():
    """
    Returns a training, validation and test dataset according to hardcoded percentages
    """
    target_dict, target_list = _format_targets()
    report_dict = _format_reports(targets=target_dict)
    train_set = {} #60%
    test_set = {} #20%
    validation_set = {} #20%
    for index, entry in report_dict.items():
        #pigeonhole spits out a psudorandom reproducible distribution of buckets from 1 - 5
        bucket = pigeonhole(5, index, SEED)
        #60% in the training set
        if bucket < 4:
            train_set[index] = entry
            train_set[index]['targets'] =  target_dict[index]
        #20% in the test set
        elif bucket == 4:
            test_set[index] = entry
            test_set[index]['targets'] =  target_dict[index]
        #20% in the validation set
        elif bucket ==5:
            validation_set[index] = entry
            validation_set[index]['targets'] =  target_dict[index]
    return train_set, validation_set, test_set, target_list

def main():
    if not os.path.exists(REPORT_PATH) and os.path.exists(TARGET_PATH):
        logger.info('head CT repots and/or targets not in expected path')
        exit(0)

    train_set, validation_set, test_set, target_list = split_datasets()
    logger.info('Training set: {}'.format(len(train_set.keys())))
    logger.info('Validation set: {}'.format(len(validation_set.keys())))
    logger.info('Test set: {}'.format(len(test_set.keys())))
    # DEBUG
    return train_set, validation_set, test_set, target_list

if __name__ == "__main__":

    main()
