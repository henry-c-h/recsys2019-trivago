import pandas as pd
import numpy as np
import os
import time

import items, sessions, preprocess
from config import mode, seed, n_cluster, smoothing, using_reduced_train, experiment_days
from filepath import (dataset_path, train_path, test_path, 
                   samples_path, log_path, exploded_path,
                   prepared_path, item_path, session_path,
                   metadata_path, meta_split_path, ground_truth_path)
from features import cluster_features
from meta import combined_meta_cols
from sklearn import preprocessing
from utils import create_logger


np.random.seed(seed)
logger = create_logger()


def main():
    create_dataset(dataset_path, train_path, test_path, 
                   samples_path, log_path, exploded_path,
                   prepared_path, metadata_path, meta_split_path,
                   item_path, session_path, ground_truth_path)
    

def create_dataset(dataset_path, train_path, test_path, 
                   samples_path, log_path, exploded_path,
                   prepared_path, metadata_path, meta_split_path,
                   item_path, session_path, ground_truth_path):
    
    if os.path.exists(prepared_path):
        logger.info('Dataset already created. Loading...')
        df = pd.read_feather(prepared_path)
    else:
        logger.info('Preparing dataset...')
        start = time.time()
        
        # generate training samples and log data
        preprocess.generate_samples(
            dataset_path,
            train_path,
            test_path,
            samples_path,
            log_path,
            exploded_path,
            metadata_path,
            meta_split_path,
            ground_truth_path,
            mode
        )
        logger.info('******')
        # build session features
        sessions.build_session_features(samples_path, 
                                        log_path,
                                        exploded_path,
                                        session_path,
                                        cluster_features,
                                        seed,
                                        n_cluster)
        logger.info('******')
        # build item features
        items.build_item_features(samples_path,
                                    log_path,
                                    exploded_path,
                                    meta_split_path,
                                    combined_meta_cols,
                                    item_path,
                                    session_path,
                                    smoothing)
        logger.info('******')
        # concatenate all features
        samples = pd.read_feather(samples_path)
        session_df = pd.read_feather(session_path)
        item_df = pd.read_feather(item_path)
        logger.info('Concatenating features...')
        df = pd.concat([samples, session_df, item_df], axis=1)
        
        # label encode categories
        logger.info('Label encoding...')
        platform_le = preprocessing.LabelEncoder()
        city_le = preprocessing.LabelEncoder()
        country_le = preprocessing.LabelEncoder()
        device_le = preprocessing.LabelEncoder()
        
        df['platform'] = platform_le.fit_transform(df['platform'])
        df['city'] = city_le.fit_transform(df['city'])
        df['country'] = country_le.fit_transform(df['country'])
        df['device'] = device_le.fit_transform(df['device'])
        
        mem_usg = df.memory_usage().sum() / 1024**2
        logger.info(f'Dataset memory usage: {int(mem_usg)} MB.')
        
        df.to_feather(prepared_path)
        logger.info(f'Full prepared dataset saved to disk. Done in {(time.time()-start)/60:.2f} minutes.')
    
    # split dataset for training and testing
    logger.info('Splitting train and test set...')
    test_set = df[df.is_train == 0]
    train_set = df[df.is_train == 1]
    
    # if having limited RAM, use a reduced training set
    if using_reduced_train:
        logger.info('Using reduced sessions to train.')
        samples_train = pd.read_feather(samples_path)[['new_session_id', 'timestamp', 'is_train']]
        samples_train = samples_train[samples_train.is_train == 1]
        samples_train['time'] = pd.to_datetime(samples_train['timestamp'], unit='s')
        cut_off = samples_train['time'].max() - pd.to_timedelta(experiment_days, unit='D')
        reduced_sessions = (samples_train.loc[samples_train.time >= cut_off]
                            .new_session_id.unique())
        train_set = train_set[train_set.new_session_id.isin(reduced_sessions)]
        reduced_df = pd.DataFrame(reduced_sessions, columns=['new_session_id'])
        reduced_df.to_feather(os.path.join(dataset_path, 'reduced_sessions.feather'))

    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    train_set.to_feather(os.path.join(dataset_path, 'train.feather'))
    test_set.to_feather(os.path.join(dataset_path, 'test.feather'))
    
    logger.info('Training and test sets created.')
    logger.info('Data preparation done. Ready for training.')
    

if __name__ == '__main__':
    main()
    