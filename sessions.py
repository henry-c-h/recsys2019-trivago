import os
import time
import gc
import pandas as pd
import numpy as np
from filepath import (samples_path, 
                      log_path,
                      exploded_path,
                      session_path)
from constant import interactions, interactions_dict
from config import seed, n_cluster
from utils import reduce_mem_usage, safe_divide, create_logger
from features import cluster_features

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


np.random.seed(seed)
logger = create_logger()


def main():
    build_session_features(samples_path,
                           log_path,
                           exploded_path,
                           session_path,
                           cluster_features,
                           seed,
                           n_cluster)
    
    
def build_session_features(samples_path, 
                           log_path,
                           exploded_path,
                           session_path,
                           cluster_features,
                           seed,
                           n_cluster,
                           return_df=False):
    if os.path.isfile(session_path):
        logger.info('Session features already created.')
        if return_df:
            df = pd.read_feather(session_path)
            logger.info('Session features loaded.')
            return df
    
    else:
        logger.info('Start creating session features...')
        start = time.time()
        
        samples = pd.read_feather(samples_path)
        log_df = pd.read_feather(log_path)
        log_exploded = pd.read_feather(exploded_path)
        logger.info('Data loaded.')

        log_df = get_dwell_time(log_df)
        log_df = identify_actions(log_df)

        # dataframe for all actions before the clickout
        pre_clickout = log_df[(log_df.before_clickout==1) &  
                                        (log_df.is_target != 1)]

        # dataframe for all item interactions before the clickout
        item_interactions = log_df[(log_df.before_clickout==1) & 
                                        (log_df.item_ref != -1) & 
                                        (log_df.is_target != 1)]

        session_df = (pd.DataFrame(samples.new_session_id.unique())
                    .rename(columns={0:'new_session_id'}))
        session_item_df = samples[['new_session_id', 'item_id', 'item_price',
                            'item_position', 'price_rank']].copy()
        
        # get basic session stats and key features
        session_df = get_session_stats(session_df, pre_clickout,
                                    item_interactions, log_exploded)
        
        session_df = get_key_action_features_in_sess(session_df, 
                                                    item_interactions, 
                                                    samples,
                                                    log_df)
        
        # get basic session item interaction stats
        session_df, session_item_df = get_session_item_stats(session_df, 
                                                            session_item_df,
                                                            item_interactions)
        
        # cluster sessions
        clusters = cluster_sessions(session_df,
                                    cluster_features,
                                    seed, n_cluster)
        session_df['cluster_sess'] = clusters
        
        # merge session features back to samples
        session_item_df = session_item_df.merge(session_df, 
                                                on='new_session_id',
                                                how='left')
        
        del session_df, samples, log_df, log_exploded
        gc.collect()
        
        session_item_df['action_type_last_int_sess'] = (session_item_df['action_type_last_int_sess']
                                                        .fillna('No int'))
        session_item_df['action_type_last_int_sess'] = (session_item_df['action_type_last_int_sess']
                                                        .map(interactions_dict))
        session_item_df = session_item_df.fillna(0)
        session_item_df = combine_item_session_features(session_item_df)
        session_item_df = session_item_df.drop(columns=['new_session_id', 'item_id',
                                                        'item_price', 'item_position', 'price_rank',
                                                        'timestamp_item_last_int',
                                                        'timestamp_item_last_co',
                                                        'timestamp_item_last_image',
                                                        'interacted_in_sess',
                                                        'clicked_in_sess',
                                                        'item_ref_last_int_sess',
                                                        'timestamp_last_int_sess',
                                                        'item_ref_last_co_sess',
                                                        'timestamp_last_co_sess',
                                                        'timestamp_last_imp_sess',
                                                        'item_id_pos_one_sess',
                                                        ])
        session_item_df = reduce_mem_usage(session_item_df, logger)
        session_item_df.to_feather(session_path)
        
        logger.info(f'Session features created and saved to disk. ' + 
            f'Done in {(time.time()-start)/60:.2f} minutes.')

        if return_df:
            return session_item_df
    
    
def get_dwell_time(log):
    timestamp_next = log['timestamp'].shift(-1)
    log['dwell_time'] = timestamp_next - log['timestamp']
    session_id_next = log['new_session_id'].shift(-1) == log['new_session_id']
    log['dwell_time'] = log['dwell_time'] * session_id_next
    log['dwell_time'] = log['dwell_time'].fillna(0)
    return log


def identify_actions(log):
    log['is_price_action'] = ((log.action_type == 'change of sort order') &
                              (log.reference.isin([
                                'price only',
                                'price and recommended']))) | \
                             ((log.action_type == 'filter selection') &
                              (log.reference.isin([
                                'Sort by Price',
                                'Best Value'])))
    log['is_distance_action'] = ((log.action_type == 'change of sort order') &
                              (log.reference.isin([
                                'distance only',
                                'distance and recommended']))) | \
                             ((log.action_type == 'filter selection') &
                              (log.reference.isin([
                                'Sort By Distance',
                                'Focus on Distance'])))
    log['is_rating_action'] = ((log.action_type == 'change of sort order') &
                              (log.reference.isin([
                                'rating only',
                                'rating and recommended']))) | \
                             ((log.action_type == 'filter selection') &
                              (log.reference.isin([
                                'Sort By Rating',
                                'Focus on Rating',
                                'Excellent Rating',
                                'Good Rating',
                                'Very Good Rating',
                                'Satisfactory Rating',])))
    return log
    

def get_session_stats(session_df, pre_clickout, item_interactions,
                      log_exploded):
    # get session duration
    session_start_ts = pre_clickout.groupby(
        'new_session_id').timestamp.min().rename('session_start_ts')
    pre_click_max_ts = pre_clickout.groupby(
        'new_session_id').timestamp.max().rename('pre_click_max_ts')
    sess_duration = (pre_click_max_ts - session_start_ts).rename('duration_sess')
    session_df = session_df.merge(sess_duration, left_on='new_session_id', 
                                right_index=True, how='left')
    
    # get pre-clickout steps
    pre_clickout_steps = (pre_clickout
                        .groupby('new_session_id')
                        .size()
                        .rename('pre_clickout_steps'))
    session_df = session_df.merge(pre_clickout_steps, 
                              left_on='new_session_id',
                              right_index=True, how='left')
    
    session_df['time_per_step'] = (session_df['duration_sess'] / 
                                   session_df['pre_clickout_steps'])
    
    # get total interactions count
    total_interactions_sess = (item_interactions
                            .groupby('new_session_id')
                            .size()
                            .rename('count_total_int_sess'))
    session_df = session_df.merge(total_interactions_sess, 
                              left_on='new_session_id',
                              right_index=True, how='left')

    # get counts of action types
    for i in interactions:
        counts = (pre_clickout[pre_clickout.action_type == i]
                .groupby('new_session_id').size().rename(i))
        session_df = session_df.merge(counts, 
                                    left_on='new_session_id', 
                                    right_index=True, how='left')

    rename_dict = {
        'clickout item': 'count_clickouts_sess',
        'interaction item rating': 'count_rating_actions_sess',
        'interaction item info': 'count_info_actions_sess',
        'interaction item image': 'count_image_actions_sess',
        'interaction item deals': 'count_deals_actions_sess',
        'search for item': 'count_searches_sess',
    }

    session_df = session_df.rename(columns=rename_dict)
    
    # get counts of price/distance/rating actions
    actions_agg = pre_clickout.groupby('new_session_id').agg(
        count_price_action=('is_price_action', 'sum'),
        count_distance_action=('is_distance_action', 'sum'),
        count_rating_action=('is_rating_action', 'sum'),
    )
    session_df = session_df.merge(actions_agg, 
                                left_on='new_session_id', 
                                right_index=True, how='left')

    # get mean and max position/price/price rank/dwell time of all interactions
    int_stats = (item_interactions
                .groupby('new_session_id')[['item_position', 
                                            'session_price', 
                                            'price_rank',
                                            'dwell_time']]
                .agg(['mean', 'max']))

    int_stats.columns = ['_'.join(i)+'_int_sess' for i in int_stats.columns]

    session_df = session_df.merge(int_stats, left_on='new_session_id', 
                                right_index=True, how='left')
    
    # get mean and max position/price/price rank of all clickouts
    co_stats = (item_interactions[item_interactions.action_type=='clickout item']
            .groupby('new_session_id')[['item_position', 
                                        'session_price', 
                                        'price_rank',
                                        'dwell_time']]
            .agg(['mean', 'max']))

    co_stats.columns = ['_'.join(i)+'_co_sess' for i in co_stats.columns]

    session_df = session_df.merge(co_stats, left_on='new_session_id', 
                                right_index=True, how='left')


    # get number of uniquely interacted items in a session
    session_df = get_unique_counts(session_df,
                                   item_interactions,
                                   'item_ref',
                                   'unique_int_items_sess')    
    

    # get number of uniquely clicked items in a session    
    session_df = get_unique_counts(session_df,
                                   item_interactions[item_interactions.action_type=='clickout item'],
                                   'item_ref',
                                   'unique_co_items_sess')
    
    # a ratio measuring likelihood of clicking/interacting with unique items
    # idea from competition winner team logicai-io
    # fillna with 0.5
    session_df['unique_co_ratio'] = ((session_df['unique_co_items_sess'] + 1) 
                                     / (session_df['count_clickouts_sess'] + 2))
    session_df['unique_int_ratio'] = ((session_df['unique_int_items_sess'] + 1) 
                                      / (session_df['count_total_int_sess'] + 2))
    session_df['unique_co_ratio'] = session_df['unique_co_ratio'].fillna(0.5)
    session_df['unique_int_ratio'] = session_df['unique_int_ratio'].fillna(0.5)
    
    # get number of unique cities
    session_df = get_unique_counts(session_df,
                                   pre_clickout,
                                   'city',
                                   'unique_city_sess')
    session_df['unique_city_sess'] = session_df['unique_city_sess'].fillna(1)
    
    
    # get mean/max impression price in a session
    imp_price_sess = (log_exploded
                    .groupby('new_session_id')
                    .item_price.agg(['mean', 'max']))
    rename_dict = {
        'mean': 'imp_price_mean_sess',
        'max': 'imp_price_max_sess',
    }
    imp_price_sess = imp_price_sess.rename(columns=rename_dict)
    session_df = session_df.merge(imp_price_sess, 
                                left_on='new_session_id',
                                right_index=True, how='left')
    
    # fill_dict = {i:0 for i in session_df.columns 
    #                         if not isinstance(session_df[i].dtype, CategoricalDtype)}
    session_df = session_df.fillna(0)
    
    return session_df


def get_unique_counts(session_df, log, col, col_name):
    s = log.groupby('new_session_id')[col].nunique().rename(col_name)
    session_df = session_df.merge(s, left_on='new_session_id', 
                                  right_index=True,
                                  how='left')
    return session_df


def count_item_interactions(item_interactions, session_item_df,
                            col_name, action_type=None, return_counts=False):
    if action_type:
        item_interactions = (item_interactions
                        [item_interactions.action_type == action_type])
    
    counts = (item_interactions
                        .groupby(['new_session_id', 'item_ref'])
                        .size()
                        .rename(col_name))
    session_item_df = session_item_df.merge(counts, 
                                    left_on=['new_session_id', 'item_id'],
                                    right_index=True, how='left')
    
    session_item_df[col_name] = session_item_df[col_name].fillna(0)   
    
    if return_counts:
        return session_item_df, counts
    return session_item_df
    
    
def count_items_above_n_interactions(session_df, counts, n, col_name):
    above_n = ((counts[counts >= n]
                    .groupby('new_session_id')
                    .count()
                    .rename(col_name)))
    
    session_df = session_df.merge(above_n, 
                              left_on='new_session_id',
                              right_index=True, how='left')
    session_df[col_name] = session_df[col_name].fillna(0)
    
    return session_df
    

def get_session_item_stats(session_df, session_item_df,
                           item_interactions):
    # get total interactions count of an item
    session_item_df, interaction_count = count_item_interactions(item_interactions,
                                              session_item_df,
                                              col_name='item_int_count_sess',
                                              return_counts=True)
    # get image views of an item
    session_item_df = count_item_interactions(item_interactions,
                                              session_item_df,
                                              col_name='item_image_int_count_sess',
                                              action_type='interaction item image')
    # get clickouts of an item
    session_item_df, clickout_count = count_item_interactions(item_interactions,
                                              session_item_df,
                                              col_name='item_co_count_sess',
                                              action_type='clickout item',
                                              return_counts=True)
    # get searches of an item
    session_item_df = count_item_interactions(item_interactions,
                                            session_item_df,
                                            col_name='item_search_count_sess',
                                            action_type='search for item')
    # get rating views of an item
    session_item_df = count_item_interactions(item_interactions,
                                            session_item_df,
                                            col_name='item_rating_int_count_sess',
                                            action_type='interaction item rating')
    # get deals views of an item
    session_item_df = count_item_interactions(item_interactions,
                                            session_item_df,
                                            col_name='item_deals_int_count_sess',
                                            action_type='interaction item deals')
    
    # get number of items with more than 2 clickouts
    session_df = count_items_above_n_interactions(session_df, 
                                                  clickout_count, 
                                                  n=2, 
                                                  col_name='count_items_above_2_co_sess')
    
    # get number of items with more than 2/3/5 interactions
    session_df = count_items_above_n_interactions(session_df, 
                                                  interaction_count,
                                                  n=2,
                                                  col_name='count_items_above_2_int_sess')
    session_df = count_items_above_n_interactions(session_df, 
                                                  interaction_count,
                                                  n=3,
                                                  col_name='count_items_above_3_int_sess')
    session_df = count_items_above_n_interactions(session_df, 
                                                  interaction_count,
                                                  n=5,
                                                  col_name='count_items_above_5_int_sess')
    
    # get item mean/max/sum dwell time for interactions/images/clickouts
    session_item_df = get_session_item_dwell_time(session_item_df,
                                                  item_interactions,
                                                  'sess',
                                                  agg_list=['mean', 'max', 'sum'])
    session_item_df = get_session_item_dwell_time(session_item_df,
                                                  item_interactions,
                                                  'image_sess',
                                                  agg_list=['mean', 'max', 'sum'],
                                                  action_type='interaction item image')    
    session_item_df = get_session_item_dwell_time(session_item_df,
                                                  item_interactions,
                                                  'co_sess',
                                                  agg_list=['mean', 'max', 'sum'],
                                                  action_type='clickout item')        
    
    # get item's order, step and timestamp of last interaction/last clickout/last image
    session_item_df = get_item_last_action(session_item_df,
                                           item_interactions,
                                           'last_int')
    session_item_df = get_item_last_action(session_item_df,
                                           item_interactions,
                                           'last_co',
                                           action_type='clickout item')
    session_item_df = get_item_last_action(session_item_df,
                                           item_interactions,
                                           'last_image',
                                           action_type='interaction item image')  
    
    
    # add a label for item indicating whether it was clicked/interacted with in session or not
    was_ever_interacted = item_interactions[['new_session_id', 'item_ref']].drop_duplicates()
    was_ever_interacted['interacted_in_sess'] = 1
    was_ever_interacted = was_ever_interacted.set_index(['new_session_id', 'item_ref'])
    session_item_df = session_item_df.merge(was_ever_interacted,
                                       left_on=['new_session_id', 'item_id'],
                                       right_index=True,
                                       how='left')
    was_ever_clicked = (item_interactions[item_interactions.action_type=='clickout item']
                   [['new_session_id', 'item_ref']].drop_duplicates())
    was_ever_clicked['clicked_in_sess'] = 1
    was_ever_clicked = was_ever_clicked.set_index(['new_session_id', 'item_ref'])
    session_item_df = session_item_df.merge(was_ever_clicked,
                                       left_on=['new_session_id', 'item_id'],
                                       right_index=True,
                                       how='left')
    session_item_df['interacted_in_sess'] = session_item_df['interacted_in_sess'].fillna(0)
    session_item_df['clicked_in_sess'] = session_item_df['clicked_in_sess'].fillna(0)  
    
    return session_df, session_item_df


def get_item_last_action(session_item_df, item_interactions, suffix,
                         action_type=None, get_action_order=True):
    if action_type:
        item_interactions = (item_interactions
                             [item_interactions.action_type == action_type])
    all_actions = (item_interactions[['new_session_id', 'item_ref', 
                                   'step', 'timestamp', 'dwell_time']].copy())
    if get_action_order:
        all_actions['order'] = all_actions.groupby('new_session_id').transform('cumcount') + 1
     
    item_last = all_actions.drop_duplicates(['new_session_id', 'item_ref'], keep='last')
    item_last = item_last.set_index(['new_session_id', 'item_ref'])   
    item_last.columns = [f'{i}_item_{suffix}' for i in item_last.columns]
    
    session_item_df = session_item_df.merge(item_last,
                     left_on=['new_session_id', 'item_id'],
                     right_index=True,
                     how='left')
    session_item_df = session_item_df.fillna({i: 0 for i in item_last.columns})
    
    return session_item_df
    

def get_session_item_dwell_time(session_item_df,
                                item_interactions, suffix,
                                agg_list, action_type=None):
    if action_type:
        item_interactions = (item_interactions
                             [item_interactions.action_type == action_type])
    item_dwell_time = (item_interactions
                    .groupby(['new_session_id', 'item_ref'])
                    .dwell_time
                    .agg(agg_list))
    item_dwell_time.columns = [f'item_{i}_dwell_time_{suffix}' 
                               for i in item_dwell_time.columns]
    
    session_item_df = session_item_df.merge(item_dwell_time,
                                        left_on=['new_session_id', 'item_id'],
                                        right_index=True,
                                        how='left')
    session_item_df = session_item_df.fillna({i: 0 for i in item_dwell_time.columns})
    
    return session_item_df
    
     
def cluster_sessions(session_df, features,
                     seed, n_clusters=6, max_iter=1500):
    # test_sessions = samples[samples.is_train==0].new_session_id.unique()
    # X = session_df[~session_df.new_session_id.isin(test_sessions)]
    X = session_df[features]
    pipe = Pipeline([('scaler', MinMaxScaler()), 
                     ('kmeans', KMeans(n_clusters=n_clusters, 
                                       random_state=seed,
                                       max_iter=max_iter))])
    labels = pipe.fit_predict(X)
    
    return labels


def get_key_action_features_in_sess(session_df, item_interactions, 
                                    samples, log_df):
    # last interaction
    last_interacted = (item_interactions.groupby('new_session_id')
                    [['item_ref', 'step', 
                        'timestamp', 'action_type',
                        'item_position', 'session_price', 
                        'price_rank', 'dwell_time']]
                        .nth(-1))
    last_interacted.columns = [f'{i}_last_int_sess' for i in 
                               last_interacted.columns]
    
    # last clickout
    last_clickout = (item_interactions[item_interactions.action_type == 'clickout item']
                    .groupby('new_session_id')[['item_ref', 'step', 
                                                'timestamp',
                                                'item_position', 'session_price', 
                                                'price_rank', 'dwell_time']]
                    .nth(-1))
    last_clickout.columns = [f'{i}_last_co_sess' for i in last_clickout.columns]
    
    session_df = session_df.merge(last_interacted, left_on='new_session_id',
                                right_index=True,
                                how='left')
    session_df = session_df.merge(last_clickout, left_on='new_session_id',
                                right_index=True,
                                how='left')
    
    session_df['action_type_last_int_sess'] = session_df['action_type_last_int_sess'].fillna('No int')
    
    # merge total dwell time of last interated item and last clickout item
    sum_dwell_time = (item_interactions
                      .groupby(['new_session_id', 'item_ref'])
                      .dwell_time.sum())
    session_df = session_df.merge(sum_dwell_time.rename('dwell_time_last_int_item'),
                 left_on=['new_session_id', 'item_ref_last_int_sess'],
                 right_index=True,
                 how='left')
    session_df = session_df.merge(sum_dwell_time.rename('dwell_time_last_co_item'),
                    left_on=['new_session_id', 'item_ref_last_co_sess'],
                    right_index=True,
                    how='left')
    
    # last impression
    last_imp = (log_df[log_df.is_target == 1]
                .groupby('new_session_id')[['timestamp', 'step']]
                .nth(-1))
    last_imp.columns = [f'{i}_last_imp_sess' for i in last_imp.columns]
    session_df = session_df.merge(last_imp, left_on='new_session_id',
                             right_index=True,
                             how='left')
    
    # time between last pre-clickout interaction/clickout to last clickout
    session_df['time_last_pre_int_to_co'] = (session_df['timestamp_last_imp_sess'] 
                                             - session_df['timestamp_last_int_sess'])
    session_df['time_last_pre_co_to_co'] = (session_df['timestamp_last_imp_sess'] 
                                            - session_df['timestamp_last_co_sess'])
    
    # position 1
    pos_one = (samples[samples.item_position == 1]
               [['new_session_id', 'item_id', 
                 'item_price', 'price_rank']])
    pos_one = (pos_one.merge(
                    sum_dwell_time,
                    left_on=['new_session_id', 'item_id'],
                    right_index=True,
                    how='left'))
    rename_dict = {i: f'{i}_pos_one_sess' for i in pos_one.columns if i != 'new_session_id'}
    pos_one = pos_one.rename(columns=rename_dict)
    session_df = session_df.merge(pos_one, 
                                  on='new_session_id',
                                  how='left')
    
    # get position difference of last 2 interactions
    above_2_ints = (item_interactions.groupby('new_session_id')
            .size()[item_interactions.groupby('new_session_id').size() >= 2].index)
    last_2_ints = (item_interactions[item_interactions.new_session_id.isin(above_2_ints)]
            .groupby('new_session_id')[['step', 'item_position']].nth([-1, -2])
            .sort_values(by=['new_session_id', 'step']))
    
    last_2_ints_pos_diff = (last_2_ints
                            .groupby('new_session_id')
                            .item_position.diff()
                            .dropna()
                            .rename('last_2_ints_pos_diff'))
    session_df = session_df.merge(last_2_ints_pos_diff, 
                              left_on='new_session_id', 
                              right_index=True, how='left')

    return session_df


def combine_item_session_features(session_item_df):
    # compare item with all interactions
    session_item_df['item_pos_over_sess_pos'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_mean_int_sess']))
    session_item_df['item_pos_over_sess_pos_max'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_max_int_sess']))
    session_item_df['item_price_over_sess_price'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_mean_int_sess']))
    session_item_df['item_price_over_sess_price_max'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_max_int_sess']))
    session_item_df['item_pr_over_sess_pr'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_mean_int_sess']))
    session_item_df['item_pr_over_sess_pr_max'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_max_int_sess']))
    session_item_df['item_dt_over_sess_dt'] = (safe_divide(
        session_item_df['item_mean_dwell_time_sess'], 
        session_item_df['dwell_time_mean_int_sess']))
    session_item_df['item_dt_over_sess_dt_max'] = (safe_divide(
        session_item_df['item_max_dwell_time_sess'], 
        session_item_df['dwell_time_max_int_sess']))
    
    # compare with all clickout items in a session
    session_item_df['item_pos_over_co_pos'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_mean_co_sess']))
    session_item_df['item_pos_over_co_pos_max'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_max_co_sess'])) 
    session_item_df['item_price_over_co_price'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_mean_co_sess']))
    session_item_df['item_price_over_co_price_max'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_max_co_sess']))
    session_item_df['item_pr_over_co_pr'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_mean_co_sess']))
    session_item_df['item_pr_over_co_pr_max'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_max_co_sess']))           
    session_item_df['item_dt_over_co_dt'] = (safe_divide(
        session_item_df['item_mean_dwell_time_co_sess'], 
        session_item_df['dwell_time_mean_co_sess']))
    session_item_df['item_dt_over_co_dt_max'] = (safe_divide(
        session_item_df['item_max_dwell_time_co_sess'], 
        session_item_df['dwell_time_max_co_sess']))
    
    # likelihood to click/interact with new items
    # if clicked/interacted previously, use (1 - prob) instead
    session_item_df['unique_co_ratio'] = (np.where(
        session_item_df.clicked_in_sess == 0,
        session_item_df['unique_co_ratio'],
        1 - session_item_df['unique_co_ratio']
        ))
    session_item_df['unique_int_ratio'] = (np.where(
        session_item_df.interacted_in_sess == 0,
        session_item_df['unique_int_ratio'],
        1 - session_item_df['unique_int_ratio']
        ))
    
    # fillna with 1 for unique city
    session_item_df['unique_city_sess'] = session_item_df['unique_city_sess'].fillna(1)
    
    # compare with all impressions
    session_item_df['item_price_over_imp_price'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['imp_price_mean_sess']))        
    session_item_df['item_price_over_imp_price_max'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['imp_price_max_sess']))
    
    
    session_item_df['is_last_interacted'] = (session_item_df['item_id'] ==
                                             session_item_df['item_ref_last_int_sess']).astype(int)
    session_item_df['is_last_clicked'] = (session_item_df['item_id'] ==
                                             session_item_df['item_ref_last_co_sess']).astype(int)    
    
    # compare item with last interaction
    session_item_df['item_price_over_last_int_sess'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_last_int_sess']))
    session_item_df['item_pos_over_last_int_sess'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_last_int_sess']))
    session_item_df['item_pr_over_last_int_sess'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_last_int_sess']))
    session_item_df['item_last_int_dt_over_last_int_sess'] = (safe_divide(
        session_item_df['dwell_time_item_last_int'], 
        session_item_df['dwell_time_last_int_sess']))
    session_item_df['item_total_dt_over_last_int_sess'] = (safe_divide(
        session_item_df['item_sum_dwell_time_sess'], 
        session_item_df['dwell_time_last_int_item']))
    

    # compare item with last clickout
    session_item_df['item_price_over_last_co_sess'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['session_price_last_co_sess']))
    session_item_df['item_pos_over_last_co_sess'] = (safe_divide(
        session_item_df['item_position'], 
        session_item_df['item_position_last_co_sess']))
    session_item_df['item_pr_over_last_co_sess'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_last_co_sess']))
    session_item_df['item_last_co_dt_over_last_co_sess'] = (safe_divide(
        session_item_df['dwell_time_item_last_co'], 
        session_item_df['dwell_time_last_co_sess']))
    session_item_df['item_total_dt_over_last_co_sess'] = (safe_divide(
        session_item_df['item_sum_dwell_time_co_sess'], 
        session_item_df['dwell_time_last_co_item']))  
    
    # compare item with position one item
    session_item_df['item_price_over_pos_one_sess'] = (safe_divide(
        session_item_df['item_price'], 
        session_item_df['item_price_pos_one_sess']))
    session_item_df['item_pr_over_pos_one_sess'] = (safe_divide(
        session_item_df['price_rank'], 
        session_item_df['price_rank_pos_one_sess']))
    session_item_df['item_dt_over_pos_one_sess'] = (safe_divide(
        session_item_df['item_sum_dwell_time_sess'], 
        session_item_df['dwell_time_pos_one_sess']))
    
    # compare item's last action with the final action
    session_item_df['item_last_int_step_recency'] = (safe_divide(1,
        session_item_df['step_last_imp_sess'] -
        session_item_df['step_item_last_int']))
    session_item_df['item_last_co_step_recency'] = (safe_divide(1, 
        session_item_df['step_last_imp_sess'] - 
        session_item_df['step_item_last_co']))
    session_item_df['item_last_image_step_recency'] = (safe_divide(1, 
        session_item_df['step_last_imp_sess'] - 
        session_item_df['step_item_last_image']))
    
    session_item_df['item_last_int_ts_recency'] = (safe_divide(1, 
        session_item_df['timestamp_last_imp_sess'] - 
        session_item_df['timestamp_item_last_int']))
    session_item_df['item_last_co_ts_recency'] = (safe_divide(1, 
        session_item_df['timestamp_last_imp_sess'] - 
        session_item_df['timestamp_item_last_co']))
    session_item_df['item_last_image_ts_recency'] = (safe_divide(1, 
        session_item_df['timestamp_last_imp_sess'] - 
        session_item_df['timestamp_item_last_image']))
    
    session_item_df['num_co_since_item_last_co'] = ( 
        session_item_df['count_clickouts_sess'] - 
        session_item_df['order_item_last_co'])
    session_item_df['num_int_since_item_last_int'] = ( 
        session_item_df['count_total_int_sess'] - 
        session_item_df['order_item_last_int'])
    session_item_df['num_img_since_item_last_image'] = ( 
        session_item_df['count_image_actions_sess'] - 
        session_item_df['order_item_last_image'])
    
    # compute item ctr in session
    session_item_df['ctr_sess'] = safe_divide(
        session_item_df['item_co_count_sess'],
        session_item_df['count_total_int_sess'],
    )
    
    return session_item_df


if __name__ == '__main__':
    main()
    