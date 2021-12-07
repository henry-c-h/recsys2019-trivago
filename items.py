import os
import time
import pandas as pd
import numpy as np
from filepath import (dataset_path,
                      samples_path, 
                      log_path,
                      exploded_path,
                      meta_split_path,
                      session_path,
                      item_path)
from config import seed, smoothing
from meta import combined_meta_cols
from utils import reduce_mem_usage, safe_divide, create_logger
import sessions


np.random.rand(seed)
logger = create_logger()


def main():
    build_item_features(samples_path,
                        log_path,
                        exploded_path,
                        meta_split_path,
                        combined_meta_cols,
                        item_path,
                        session_path,
                        smoothing)


def build_item_features(samples_path,
                        log_path,
                        exploded_path,
                        meta_split_path,
                        combined_meta_cols,
                        item_path,
                        session_path,
                        smoothing=10,
                        return_df=False):
    if os.path.isfile(item_path):
        logger.info('Item features already created.')
        if return_df:
            df = pd.read_feather(item_path)
            logger.info('Item features loaded.')
            return df
    else:
        logger.info('Starting creating item features...')
        start = time.time()
        
        samples = pd.read_feather(samples_path)
        log_df = pd.read_feather(log_path)
        log_exploded = pd.read_feather(exploded_path)
        session_features = pd.read_feather(session_path)
        logger.info('Data loaded.')
        
        item_interactions = log_df[log_df.item_ref != -1]
        
        samples = (samples[['new_session_id', 'platform', 'city', 'country', 
                            'item_id', 'is_train', 'is_clicked',
                            'item_price', 'item_position', 'len_impressions', 
                            'price_rank', 'len_prices']].copy())

        # create click-through-rate related features
        samples = get_ctr_related_features(samples, item_interactions, 
                                        log_exploded, session_features,
                                        smoothing)
        logger.info('CTR done')
        
        # get features related to item's rank in position and price point
        samples = get_rank_features(samples, log_exploded)
        logger.info('Rank done')
        
        # aggregate original meta features into fewer categories
        samples, meta_features = extract_meta_features(samples, log_exploded,
                                        meta_split_path, combined_meta_cols)
        logger.info('Meta done')
        
        meta_features = meta_features.set_index('item_id')
        item_interactions = item_interactions.merge(meta_features[['star_level', 'rating_level']],
                                            left_on='item_ref',
                                            right_index=True, how='left')
        log_exploded = log_exploded.merge(meta_features[['star_level', 'rating_level']],
                                    left_on='item_ref',
                                    right_index=True, how='left')
        
        # get price aggregated by groups (city, platform, star level, etc.)
        samples = get_co_price_features(samples, log_exploded)
        samples = get_imp_price_features(samples, log_exploded)
        samples = get_int_price_features(samples, item_interactions)
        logger.info('Price done')

        samples = samples.drop(columns=['new_session_id', 'platform', 'city', 'country', 
                            'item_id', 'is_train', 'is_clicked',
                            'item_price', 'item_position', 'len_impressions', 
                            'price_rank', 'len_prices'])
        
        samples = reduce_mem_usage(samples, logger)
        samples.to_feather(item_path)
        
        logger.info(f'Item features created and saved to disk. ' + 
            f'Done in {(time.time()-start)/60:.2f} minutes.')
        
        if return_df:
            return samples
    

def get_global_counts(samples, item_int, col_name, 
                      action_type=None, group_key='item_ref'):
    if action_type:
        item_int = item_int[item_int.action_type == action_type]
    
    counts = item_int.groupby(group_key).size().rename(col_name)
    samples = samples.merge(counts, left_on='item_id', 
                            right_index=True, how='left')
    samples[col_name] = samples[col_name].fillna(0)
    return samples
    
    
def get_ctr_related_features(samples, item_int, log_exploded,
                             session_features, smoothing=10):
    # get global action counts
    samples = get_global_counts(samples, item_int, 
                                col_name='count_global_img_int', 
                                action_type='interaction item image')
    samples['count_global_img_int'] = (samples['count_global_img_int'] -
                                       session_features['item_image_int_count_sess'])
    
    samples = get_global_counts(samples, item_int, 
                                col_name='count_global_rating_int',
                                action_type='interaction item rating')
    samples['count_global_rating_int'] = (samples['count_global_rating_int'] -
                                       session_features['item_rating_int_count_sess'])
    
    samples = get_global_counts(samples, item_int, 
                                col_name='count_global_search',
                                action_type='search for item')
    samples['count_global_search'] = (samples['count_global_search'] -
                                       session_features['item_search_count_sess'])
    
    samples = get_global_counts(samples, item_int, 
                                col_name='count_global_info_int',
                                action_type='interaction item info')
    
    samples = get_global_counts(samples, item_int, 
                                col_name='count_global_deals_int',
                                action_type='interaction item deals')
    samples['count_global_deals_int'] = (samples['count_global_deals_int'] -
                                       session_features['item_deals_int_count_sess'])
    
    samples = get_global_counts(samples, item_int, 'count_global_co',
                                action_type='clickout item')
    # samples.loc[(samples.is_train == 1) & (samples.is_clicked == 1), 
    #         'count_global_co'] = (samples.loc[(samples.is_train == 1) 
    #                                             & (samples.is_clicked == 1), 
    #                                             'count_global_co'] - 1)
    
    samples = sessions.count_item_interactions(item_int, samples, 'all_co_in_sess',
                                                 action_type='clickout item')
    samples['count_global_co'] = (samples['count_global_co'] - 
                                  samples['all_co_in_sess'])
    # samples = samples.drop('all_co_in_sess', axis=1)
    
    
    samples = get_global_counts(samples, item_int, 'count_global_int')
    samples = sessions.count_item_interactions(item_int, samples, 'all_int_in_sess')
    samples['count_global_int'] = (samples['count_global_int'] - 
                                  samples['all_int_in_sess'])
    # samples = samples.drop('all_int_in_sess', axis=1)
        
    # samples = get_global_counts(samples, log_exploded, 
    #                             col_name='count_global_imp',
    #                             group_key='item_id')
    # each appearance in a session count as 1 impression
    train_log = log_exploded[log_exploded.is_train == 1]
    unique_imp = (train_log[['new_session_id', 'item_id']]
                  .drop_duplicates(['new_session_id', 'item_id']))
    imp = unique_imp.groupby('item_id').size().rename('count_global_imp')
    samples = samples.merge(imp, left_on='item_id', 
                            right_index=True, how='left')
    samples['count_global_imp'] = samples['count_global_imp'].fillna(0)
    
    
    # ctr and other click rates variations
    samples['ctr'] = safe_divide(samples['count_global_co'], samples['count_global_imp'])
    samples['co_per_int'] = safe_divide(samples['count_global_co'],
                                            (samples['count_global_int'] + smoothing))
    samples['co_per_img_int'] = safe_divide(samples['count_global_co'],
                                            (samples['count_global_img_int'] + smoothing))
    samples['co_per_search'] = safe_divide(samples['count_global_co'],
                                            (samples['count_global_search'] + smoothing))
    samples['co_per_rating_int'] = safe_divide(samples['count_global_co'],
                                            (samples['count_global_rating_int']+ smoothing))
    samples['co_per_deals_int'] = safe_divide(samples['count_global_co'],
                                            (samples['count_global_deals_int']+ smoothing))
    # click rates weighted by interaction counts and dwell time
    samples['co_prob_int'] = (samples['co_per_int'] * 
                              session_features['item_int_count_sess'])
    samples['co_prob_img'] = (samples['co_per_img_int'] * 
                              session_features['item_image_int_count_sess'])
    samples['co_prob_search'] = (samples['co_per_search'] * 
                              session_features['item_search_count_sess'])
    samples['co_prob_rating'] = (samples['co_per_rating_int'] * 
                              session_features['item_rating_int_count_sess'])
    samples['co_prob_deals'] = (samples['co_per_deals_int'] * 
                              session_features['item_deals_int_count_sess'])
    samples['co_prob_int_by_dt'] = (samples['co_per_int'] * 
                              session_features['item_sum_dwell_time_sess'])
    samples['co_prob_img_by_dt'] = (samples['co_per_img_int'] * 
                              session_features['item_sum_dwell_time_image_sess'])
    
    # compute position-adjusted ctr
    # train_log = log_exploded[(log_exploded.is_train == 1)]
    avg_ctr = ((train_log.groupby('item_position').is_clicked
                .mean()).rename('position_ctr'))
    ctr_df = (log_exploded.groupby(['item_id', 'item_position'])
            .size()
            .reset_index()
            .rename(columns={0:'position_count'}).merge(
            avg_ctr,
            left_on='item_position',
            right_index=True,
            how='left'))
    ctr_df['expected_clicks'] = ctr_df['position_ctr'] * ctr_df['position_count']
    total_expected_clicks = ctr_df.groupby('item_id').expected_clicks.sum()
    samples = samples.merge(total_expected_clicks, left_on='item_id', right_index=True, how='left')
    samples = samples.merge(avg_ctr, left_on='item_position', right_index=True, how='left')
    samples['expected_clicks'] = samples['expected_clicks'] - samples['position_ctr']
    samples['pos_adjusted_ctr'] = safe_divide(samples['count_global_co'], samples['expected_clicks'])
    # samples['co_count_diff_expected'] = samples['count_global_co'] - samples['expected_clicks']
    samples['co_count_over_expected'] = safe_divide(samples['count_global_co'], 
                                                    samples['expected_clicks'])

    # measure the likelihood of an item being clicked in the session
    # based on global likelihood (num sessions clicked / num sessions interacted)
    samples = get_global_session_numbers(samples, item_int,
                                     col_name='sess_num_int')
    samples['sess_num_int'] = (samples['sess_num_int'] -  
                               (samples['all_int_in_sess'] > 0).astype(int))
    
    samples = get_global_session_numbers(samples, item_int,
                                     col_name='sess_num_co',
                                     action_type='clickout item')
    samples['sess_num_co'] = (samples['sess_num_co'] -  
                               (samples['all_co_in_sess'] > 0).astype(int))
    
    samples = get_global_session_numbers(samples, item_int,
                                     col_name='sess_num_img',
                                     action_type='interaction item image')
    samples['sess_num_img'] = (samples['sess_num_img'] -  
                               (session_features['item_image_int_count_sess'] > 0).astype(int))
    
    
    samples['co_prob_int'] = (samples['sess_num_co'] / 
                              (samples['sess_num_int'] + smoothing))
    samples['co_prob_img'] = (samples['sess_num_co'] / 
                              (samples['sess_num_img'] + smoothing))
    # weighted by number of actions
    samples['co_prob_in_sess_int'] = (samples['co_prob_int'] * 
                                      session_features['item_int_count_sess'])
    samples['co_prob_in_sess_img'] = (samples['co_prob_img'] * 
                                      session_features['item_image_int_count_sess'])
    # weighted by dwell time
    samples['co_prob_in_sess_int_by_dt'] = (samples['co_prob_int'] * 
                                      session_features['item_sum_dwell_time_sess'])
    samples['co_prob_in_sess_img_by_dt'] = (samples['co_prob_img'] * 
                                      session_features['item_sum_dwell_time_image_sess'])


    return samples


def get_global_session_numbers(samples, item_int, col_name, 
                           exclude_current_sess=False, action_type=None):
    if action_type:
        item_int = item_int[item_int.action_type == action_type]
        
    sess_num = (item_int.groupby('item_ref')
                .new_session_id.nunique().rename(col_name))
    
    samples = samples.merge(sess_num,
                            left_on='item_id',
                            right_index=True, how='left')
    # if exclude_current_sess:
    #     samples.loc[(samples.is_train==1) & (samples.is_clicked==1), 
    #             col_name] = (samples.loc[(samples.is_train==1) 
    #                                      & (samples.is_clicked==1), col_name] - 1)
                
    samples[col_name] = samples[col_name].fillna(0)
    
    return samples


def get_rank_features(samples, log_exploded):
    # rank related features
    samples['position_rel'] = (samples['item_position'] / samples['len_impressions'])
    samples['price_rank_rel'] = (samples['price_rank'] / samples['len_prices'])

    log_exploded['position_rel'] = (log_exploded['item_position'] / log_exploded['len_impressions'])
    log_exploded['price_rank_rel'] = (log_exploded['price_rank'] / log_exploded['len_prices'])

    global_rank = log_exploded.groupby('item_id')[[
            'item_position', 
            'position_rel',
            'price_rank', 
            'price_rank_rel'
        ]].mean().rename(
            columns={
                'item_position': 'position_global_abs',
                'position_rel': 'position_global_rel',
                'price_rank': 'price_rank_global_abs',
                'price_rank_rel': 'price_rank_global_rel',
            }
        )

    samples = samples.merge(global_rank, left_on='item_id', right_index=True, how='left')
        
    # local and global postion diff and ratio
    samples['position_abs_diff'] = samples['item_position'] - samples['position_global_abs']
    samples['position_abs_ratio'] = samples['item_position'] / samples['position_global_abs']
    samples['position_rel_diff'] = samples['position_rel'] - samples['position_global_rel']
    samples['position_rel_ratio'] = samples['position_rel'] / samples['position_global_rel']

    # local and global price rank diff and ratio
    samples['price_rank_abs_diff'] = samples['price_rank'] - samples['price_rank_global_abs']
    samples['price_rank_abs_ratio'] = samples['price_rank'] / samples['price_rank_global_abs']
    samples['price_rank_rel_diff'] = samples['price_rank_rel'] - samples['price_rank_global_rel']
    samples['price_rank_rel_ratio'] = samples['price_rank_rel'] / samples['price_rank_global_rel']

    return samples


def get_co_price_features(samples, log_exploded):
    city = compare_co_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['city'],
                                     suffix='city')
    city_platform = compare_co_price_by_group(samples,
                                              log_exploded,
                                              group_cols=['city', 'platform'],
                                              suffix='city_platfrom')
    country = compare_co_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['country'],
                                     suffix='country')
    city_star = compare_co_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['city', 'star_level'],
                                     suffix='city_star')
    price_features = pd.concat([city, city_platform, country, city_star], axis=1)
    samples = pd.concat([samples, price_features], axis=1)
    
    return samples


def get_imp_price_features(samples, log_exploded):
    city = compare_imp_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['city'],
                                     suffix='city')
    country = compare_imp_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['country'],
                                     suffix='country')
    star = compare_imp_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['star_level'],
                                     suffix='star_level')
    rating = compare_imp_price_by_group(samples,
                                     log_exploded,
                                     group_cols=['rating_level'],
                                     suffix='rating_level')
    price_features = pd.concat([city, country, star, rating], axis=1)
    samples = pd.concat([samples, price_features], axis=1)
    
    return samples
    

def compare_imp_price_by_group(samples, log_exploded, group_cols, suffix):        
    imp_price = (log_exploded
                   .groupby(group_cols).item_price.agg(['mean', 'max']))
    imp_price.columns = [f'{i}_price_imp_{suffix}' for i in imp_price.columns]
    
    merged = samples[group_cols+['item_price']].merge(imp_price, 
                                                      left_on=group_cols,
                                                      right_index=True,
                                                      how='left')

    # compute price ratios
    merged[f'item_price_over_mean_imp_{suffix}'] = (safe_divide(merged['item_price'],
                                                               merged[f'mean_price_imp_{suffix}']))
    merged[f'item_price_over_max_imp_{suffix}'] = (safe_divide(merged['item_price'],
                                                               merged[f'max_price_imp_{suffix}']))
    
    merged = merged.drop(columns=group_cols+['item_price'])

    return merged


def compare_co_price_by_group(samples, log_exploded, group_cols, suffix):        

    new_group = group_cols + ['new_session_id']
    
    log_exploded = (log_exploded[(log_exploded.is_clicked==1)&
                                 (log_exploded.is_target!=1)][new_group+['item_price']]
                    .drop_duplicates(new_group, keep='last'))
    
    grouped = log_exploded.groupby(group_cols)
    n = grouped.item_price.transform('count')
    total = grouped.item_price.transform('sum')
    log_exploded[f'mean_price_co_{suffix}'] = safe_divide((total - log_exploded['item_price']),
                                             (n-1))
    log_exploded = log_exploded.drop('item_price', axis=1)
    
    merged = samples[new_group+['item_price']].merge(log_exploded, 
                                                      on=new_group,
                                                    #   right_index=True,
                                                      how='left')
    merged[f'mean_price_co_{suffix}'] = merged[f'mean_price_co_{suffix}'].fillna(0)

    # compute price ratios
    merged[f'item_price_over_mean_co_{suffix}'] = (safe_divide(merged['item_price'],
                                                               merged[f'mean_price_co_{suffix}']))
    
    merged = merged.drop(columns=new_group+['item_price'])

    return merged


def get_int_price_features(samples, item_int):
    city = compare_int_price_by_group(samples, item_int,
                                      ['city'], 'city')
    city_platform = compare_int_price_by_group(samples, item_int,
                                      ['city', 'platform'], 'city_platform')
    country = compare_int_price_by_group(samples, item_int,
                                      ['country'], 'country')
    country_star = compare_int_price_by_group(samples, item_int,
                                      ['country', 'star_level'], 'star')
    country_rating = compare_int_price_by_group(samples, item_int,
                                      ['country', 'rating_level'], 'rating')
    price_features = pd.concat([city, city_platform, country, country_star,
                                country_rating], axis=1)
    samples = pd.concat([samples, price_features], axis=1)
    return samples


def compare_int_price_by_group(samples, item_int, group_cols, suffix):        
    item_int = item_int[(item_int.is_target != 1)&(item_int.before_clickout==1)]
    int_price = (item_int.groupby(group_cols)
                 .session_price.mean()
                 .rename(f'mean_price_int_{suffix}'))
    
    merged = samples[group_cols+['item_price']].merge(int_price, 
                                                      left_on=group_cols,
                                                      right_index=True,
                                                      how='left')
    merged[f'mean_price_int_{suffix}'] = merged[f'mean_price_int_{suffix}'].fillna(0)

    # compute price ratios
    merged[f'item_price_over_mean_int_{suffix}'] = (safe_divide(merged['item_price'],
                                                               merged[f'mean_price_int_{suffix}']))
    

    
    merged = merged.drop(columns=group_cols+['item_price'])

    return merged


def extract_meta_features(samples, log_exploded, meta_path, combined_cols):
    """Return a metadata dataframe with combined columns"""
    
    start = time.time()
    meta_df = pd.read_feather(meta_path)
    drop_properties = ['From 2 Stars', 'From 3 Stars', 'From 4 Stars']
    meta_df = meta_df.drop(columns=drop_properties)
    
    meta_df['item_num_prop'] = meta_df.iloc[:, 1:].sum(1)
    
    star_df = meta_df[[col for col in meta_df.columns if 'Star' in col]]
    star_level = star_df.transform(
        lambda x: int(x.name.split()[0])*x).sum(1).rename('star_level')
    
    rating_df = meta_df[[col for col in meta_df.columns if 'Rating' in col]]
    rating_dict = {
        'Excellent Rating': 4,
        'Very Good Rating': 3,
        'Good Rating': 2,
        'Satisfactory Rating': 1,
    }
    rating_level = rating_df.transform(
        lambda x: x * rating_dict[x.name]).max(1).rename('rating_level')

    combined = []
    for i in combined_cols:
        s = meta_df[combined_cols[i]].sum(1).rename(i)
        combined.append(s)
    combined_df = pd.concat(combined, axis=1)
    combined_df = pd.concat(
        [meta_df[['item_id', 'item_num_prop']], combined_df, star_level, rating_level], axis=1)

    all_items = log_exploded.item_id.unique()
    items_without_meta = set(all_items) - set(meta_df.item_id.unique())
    df = pd.DataFrame(columns=combined_df.columns)
    df['item_id'] = pd.Series(list(items_without_meta))
    df = df.fillna(0)
    combined_df = pd.concat([combined_df, df]).reset_index(drop=True)

    combined_df = reduce_mem_usage(combined_df, logger)
    combined_df.to_feather(os.path.join(dataset_path, 'hotel_features.feather'))

    samples = samples.merge(combined_df, on='item_id', how='left')
    logger.info(f'Extracted metadata features. Done in {(time.time()-start) / 60:.2f} minutes.')

    return samples, combined_df


if __name__ == '__main__':
    main()
