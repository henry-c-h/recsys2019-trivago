import os
import time
import pandas as pd
import numpy as np
from utils import create_logger, reduce_mem_usage
from config import mode, seed, experiment_days, validation_days
from filepath import (dataset_path, train_path, test_path, 
                      samples_path, log_path, exploded_path, 
                      metadata_path, meta_split_path, ground_truth_path,
                      val_path, conf_path)
from utils import create_logger


np.random.seed(seed)
logger = create_logger()
  

def main():
    generate_samples(dataset_path, train_path, test_path,
                     samples_path, log_path, exploded_path,
                     metadata_path, meta_split_path, ground_truth_path,
                     mode)


def generate_samples(dataset_path, train_path, test_path, 
                     samples_path, log_path, exploded_path,
                     metadata_path, meta_split_path, ground_truth_path,
                     mode):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
        
    if os.path.isfile(samples_path) and os.path.isfile(log_path):
        logger.info('Raw data already preprocessed. Proceed to create features.')
        return
    else:
        logger.info('Start preprocessing data...')
        start = time.time()
        
        # reading data
        df = read_raw_data(train_path, test_path)
        
        # some sessions have duplicate steps, rename these sessions
        df = remove_duplicate_steps(df)
        
        # split pipe columns - impressions/prices/current filters
        df = split_pipe_columns(df)
        
        # get country name from city column
        df = add_country_column(df)
        
        # split dataset into train and test
        # if using full mode, then use existing validation and confirmation sets as test set
        split_data(df, samples_path, log_path, dataset_path,
                    exploded_path, ground_truth_path, mode=mode)
        
        # split metadata into columns
        preprocess_metadata(metadata_path, meta_split_path)
        
        logger.info(f'Preprocessing done in {(time.time()-start)/60:.2f} minutes. ' +
                f'Proceed to create features.')


def read_raw_data(train_path, test_path):
    logger.info('Reading raw data...')

    start = time.time()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train['is_train'] = 1
    test['is_train'] = 0
    full_df = pd.concat([train, test])

    full_df = full_df.sort_values(by=['session_id', 'timestamp', 'step']).reset_index(drop=True)
    full_df = full_df.drop('user_id', axis=1)
        
    logger.info(f'Raw data loaded. Done in {(time.time()-start)/60:.2f} minutes.')
    return full_df


def remove_duplicate_steps(df):
    """Return a new dataframe with new session ids for sessions with duplicate steps"""

    logger.info('Checking sessions with duplicate steps...')

    start = time.time()
    step_sizes = df.groupby(['session_id', 'step']).size()
    session_idx = step_sizes[step_sizes > 1].index.get_level_values(0).unique()
    logger.info(f'Sessions with duplicate steps before: {session_idx.shape[0]}')
    duplicate_df = df[df.session_id.isin(session_idx)].copy()
    
    session_num = ((duplicate_df['step'] - duplicate_df['step'].shift(1)) != 1).cumsum() 
    duplicate_df['new_session_id'] = duplicate_df['session_id'] + '-' + session_num.astype(str)
    df['new_session_id'] = df['session_id']
    df.loc[duplicate_df.index, 'new_session_id'] = duplicate_df['new_session_id']
    step_sizes = df.groupby(['new_session_id', 'step']).size()
    session_idx = step_sizes[step_sizes > 1].index.get_level_values(0).unique()

    logger.info(f'Sessions with duplicate steps after: {session_idx.shape[0]}. '\
        f'Done in {(time.time()-start)/60:.2f} minutes.')

    return df


def split_pipe(pipe_array, to_numeric=True):
    if pipe_array is not None and type(pipe_array) is str :
        split_array = pipe_array.split('|')
        if to_numeric:
            split_array = [int(i) for i in split_array]
        return split_array
    return None


def split_pipe_columns(df):
    logger.info('Splitting pipe columns...')

    start = time.time()
    df['impressions'] = df['impressions'].map(split_pipe)
    df['prices'] = df['prices'].map(split_pipe)
    df['current_filters'] = df['current_filters'].apply(split_pipe, to_numeric=False)
    
    return df


def add_country_column(df):
    country = df['city'].str.split(', ').map(lambda x: x[-1])
    df['country'] = country
    return df


# def change_category_dtype(df):
#     df['new_session_id'] = df['new_session_id'].astype('category')
#     df['session_id'] = df['session_id'].astype('category')
#     df['city'] = df['city'].astype('category')
#     df['country'] = df['country'].astype('category')
#     df['device'] = df['device'].astype('category')
#     df['platform'] = df['platform'].astype('category')
#     df['action_type'] = df['action_type'].astype('category')
#     return df
    
    
def preprocess_metadata(file_path, meta_path):
    logger.info('Processing item metadata...')

    meta_df = pd.read_csv(file_path)
    prop_df = meta_df.properties.str.get_dummies()
    prop_df = prop_df.astype(np.int8)
    meta_df = pd.concat([meta_df, prop_df], axis=1).drop('properties', axis=1)
    meta_df.to_feather(meta_path)
    return meta_df


def label_actions_after_clickout(log):
    log = log.merge(log[log.is_target==1][['new_session_id', 'step']],
        on='new_session_id',
        how='left',
        suffixes=['', '_clickout']
    )

    log['before_clickout'] = 0
    log.loc[log.step <= log.step_clickout, 'before_clickout'] = 1
    log = log.drop('step_clickout', axis=1)

    return log


def explode_all_clickouts(df):
    df = df.drop(columns=['current_filters',
                            'reference',
                            'action_type'])
    df['len_impressions'] = df['impressions'].map(len)
    df['item_position'] = df['len_impressions'].map(np.arange) + 1
    df['price_rank'] = df['prices'].map(
        lambda x: np.searchsorted(np.unique(x), x) + 1)
    df['len_prices'] = df['prices'].map(lambda x: np.unique(x).shape[0])
    
    log_exploded = df.explode(
        ['impressions', 'prices', 'item_position', 'price_rank'],
        ignore_index=True)
    
    log_exploded['impressions'] = log_exploded['impressions'].astype(int)
    log_exploded['prices'] = log_exploded['prices'].astype(int)
    log_exploded['item_position'] = log_exploded['item_position'].astype(int)
    log_exploded['price_rank'] = log_exploded['price_rank'].astype(int)

    log_exploded = log_exploded.rename(
            columns={
                'impressions': 'item_id',
                'prices': 'item_price',
        })
    log_exploded['is_clicked'] = (log_exploded['item_id'] ==
                                  log_exploded['item_ref']).astype(int)

    log_exploded = reduce_mem_usage(log_exploded, logger)
    
    return log_exploded


def get_competition_ground_truth(target_set):
    validation = pd.read_csv(val_path)[['session_id', 'step', 'reference']]
    confirmation = pd.read_csv(conf_path)[['session_id', 'step', 'reference']]
    ground_truth = pd.concat([validation, confirmation])
    target_set = target_set.merge(ground_truth, on=['session_id', 'step'],
                                 how='left')
    target_set = target_set.drop('step', axis=1)
    target_set['reference'] = pd.to_numeric(target_set['reference'])
    target_set = target_set.rename(columns={'reference': 'item_ref'})
    return target_set  


def split_data(df, samples_path, log_path, dataset_path,
               exploded_path, ground_truth_path, mode='experiment'):
    logger.info('Splitting dataset...')
    start = time.time()
    
    if mode == 'experiment':
        logger.info('Using experiment mode.')
        # compute cut-offs for training and validation period
        df = df[df.is_train == 1].copy()
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        experiment_val_cut_off = df.time.max() - pd.to_timedelta(validation_days, unit='D')
        experiment_cut_off = df.time.max() - pd.to_timedelta(experiment_days, unit='D')
        logger.info(f'Validation cut off is {experiment_val_cut_off}. '\
            f'Experiment cut off is {experiment_cut_off}.')

        # select all sessions that fall within the timeframe
        session_min_time = df.groupby('new_session_id').time.min()
        sessions = session_min_time[session_min_time >= experiment_cut_off].index
        df = df[df.new_session_id.isin(sessions)].copy()
        df.loc[df.time >= experiment_val_cut_off, 'is_train'] = 0
        df = df.drop('time', axis=1)

        # item_ref relates to all item interactions that have item id as reference
        # if an action is not related to item interaction or missing reference, fill with -1
        df['item_ref'] = pd.to_numeric(df['reference'], errors='coerce').fillna(-1).astype(int)
        
        # choose the last clickout in a session as the target to be predicted
        click_df = df[df.action_type == 'clickout item']
        target_df = click_df.groupby('new_session_id',
                                     as_index=False).nth(-1).copy()
        df['is_target'] = 0
        df.loc[target_df.index, 'is_target'] = 1
        
        # save ground truth
        ground_truth = df[['new_session_id', 'item_ref']].reset_index(drop=True)
        ground_truth.to_feather(ground_truth_path)
        
        # label whether an action is before (including clickout) or after a clickout
        df = label_actions_after_clickout(df)
        
        # explode all clickouts based on the item id
        # the rows with a target label are the training examples
        log_exploded = explode_all_clickouts(df[df.action_type=='clickout item'].copy())
        samples_df = log_exploded[log_exploded.is_target == 1]
        num_train = samples_df[samples_df.is_train==1].shape[0]
        num_test = samples_df[samples_df.is_train==0].shape[0]
        logger.info(f'Total number of training samples: {num_train}')
        logger.info(f'Total number of testing samples: {num_test}')
        
        # attach session prices, position and price rank to interacted items
        impression_items = log_exploded[['new_session_id', 'item_id', 'item_position', 
                                      'item_price', 'price_rank']].drop_duplicates(
                                          ['new_session_id', 'item_id'], keep='last')
        impression_items = impression_items.rename(columns={'item_price': 'session_price'})
        df = df.merge(impression_items, 
                        left_on=['new_session_id', 'item_ref'], 
                        right_on=['new_session_id', 'item_id'], 
                        how='left').drop('item_id', axis=1)
        
        # mask the item ref for test set
        df.loc[(df.is_target==1) & (df.is_train==0), 'item_ref'] = -1
        
        # save all prepared datasets to disk
        df = df.reset_index(drop=True)
        df = reduce_mem_usage(df, logger)
        df.to_feather(log_path)

        log_exploded.to_feather(exploded_path)
        samples_df = samples_df.reset_index(drop=True)
        samples_df = samples_df.drop(columns=['step', 'is_target', 'before_clickout'])
        samples_df.to_feather(samples_path)
    
    else:
        logger.info('Using full competition mode. ')
        # same as for experiment mode
        df['item_ref'] = pd.to_numeric(df['reference'], 
                                       errors='coerce').fillna(-1).astype(int)
        
        # label validation portion and confirmation portion according to challenge
        validation = pd.read_csv(val_path)
        confirmation = pd.read_csv(conf_path)
        val_sessions = validation.session_id.unique()
        conf_sessions = confirmation.session_id.unique()
        df['is_val'] = 0
        df['is_conf'] = 0
        df.loc[df.session_id.isin(val_sessions), 'is_val'] = 1
        df.loc[df.session_id.isin(conf_sessions), 'is_conf'] = 1
        
        # identify test targets
        test_target_mask = ((df.action_type == 'clickout item') & 
                            (df.reference.isnull()) & (df.is_train==0))
        df['is_target'] = 0
        df.loc[test_target_mask, 'is_target'] = 1
        logger.info(f'Number of test targets: {(test_target_mask == 1).sum()}')
        
        # save ground truth
        target_set = df[df.is_target==1][['session_id', 'step']].copy()
        target_set = get_competition_ground_truth(target_set)
        target_set.to_feather(ground_truth_path)
        
        # identify training targets
        max_train_ts = df[df.is_train == 1].timestamp.max()
        session_max_time = df[df.is_train == 1].groupby('new_session_id').timestamp.max()
        train_sessions = session_max_time[session_max_time <= max_train_ts].index
        
        train_clicks = df[df.new_session_id.isin(train_sessions) & 
                   (df.action_type == 'clickout item')]
        train_target_idx = train_clicks.groupby(
            'new_session_id', as_index=False).nth(-1).index
        df.loc[train_target_idx, 'is_target'] = 1
        logger.info(f'Number of train targets: {train_target_idx.shape[0]}')
        
        # save train ground truth
        train_gt = df[(df.is_train == 1) & (df.is_target == 1)][['new_session_id', 'item_ref']].copy()
        train_gt = train_gt.reset_index(drop=True)
        train_gt.to_feather(os.path.join(dataset_path, 'ground_truth_train.feather'))

        # label whether an action is before (including clickout) or after a clickout
        df = label_actions_after_clickout(df)
        
        # explode all clickouts based on the item id
        # the rows with a target label are the training examples
        log_exploded = explode_all_clickouts(df[df.action_type=='clickout item'].copy())
        samples_df = log_exploded[log_exploded.is_target == 1]
        num_train = samples_df[samples_df.is_train==1].shape[0]
        num_test = samples_df[samples_df.is_train==0].shape[0]
        logger.info(f'Total number of training samples: {num_train}')
        logger.info(f'Total number of testing samples: {num_test}')
        
        # attach session prices, position and price rank to interacted items
        impression_items = log_exploded[['new_session_id', 'item_id', 'item_position', 
                                      'item_price', 'price_rank']].drop_duplicates(
                                          ['new_session_id', 'item_id'], keep='last')
        impression_items = impression_items.rename(columns={'item_price': 'session_price'})
        df = df.merge(impression_items, 
                        left_on=['new_session_id', 'item_ref'], 
                        right_on=['new_session_id', 'item_id'], 
                        how='left').drop('item_id', axis=1)
              
        # save all prepared datasets to disk
        df = df.reset_index(drop=True)
        df = reduce_mem_usage(df, logger)
        df.to_feather(log_path)

        log_exploded.to_feather(exploded_path)
        samples_df = samples_df.reset_index(drop=True)
        samples_df = samples_df.drop(columns=['step', 'is_target', 'before_clickout'])
        samples_df.to_feather(samples_path)
        
    
    logger.info(f'Finished splitting dataset. Done in {(time.time()-start)/60:.2f} minutes.')
    

if __name__ == '__main__':
    main()
