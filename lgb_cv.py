import os
import time
import pandas as pd
import numpy as np
import lightgbm as lgb

from config import seed, mode
from filepath import (dataset_path,
                      ground_truth_path, model_path)
from features import lgb_features
from utils import create_logger
import gc


np.random.seed(seed)
logger = create_logger()


lgb_params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 64,
    'max_bin': 128,
    'min_data_in_leaf': 600,
    'learning_rate': 0.2,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6, 
    'bagging_freq': 12,
    'num_threads': 8,
    'force_col_wise': True,
    'verbose': 0,
    'seed': seed,
}

ranker_params = {
    'boosting': 'gbdt', # 'dart'
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': 25,
    'num_leaves': 64,
    'max_bin': 128,
    'min_data_in_leaf': 600,
    'learning_rate': 0.2,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6, 
    'bagging_freq': 12,
    'num_threads': 8,
    'force_col_wise': True,
    'verbose': 0,
    'seed': seed,
}

num_boost = 10000
early_stopping = 50
ensemble_algo = True


def main():
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    
    train_set = pd.read_feather(os.path.join(dataset_path, 'train.feather'))
    logger.info('Training set loaded.')
    
    clf_model = RecommenderModel('classifier', train_set,
                               lgb_params, lgb_features, 
                               num_boost, early_stopping, 
                               mode)
    ltr_model = RecommenderModel('learning_to_rank', train_set,
                               ranker_params, lgb_features, 
                               num_boost, early_stopping, 
                               mode)
    # training classifier 
    clf_model.cross_validate()
    prob = np.mean(clf_model.test_prob, axis=1)
    evaluate(prob, mode=mode)
    
    # training ranker
    ltr_model.cross_validate()
    prob = np.mean(ltr_model.test_prob, axis=1)
    evaluate(prob, mode=mode)
    
    if ensemble_algo:
        clf_prob = clf_model.test_prob
        ltr_prob = ltr_model.test_prob
        prob_concat = np.concatenate([clf_prob, ltr_prob], axis=1)
        mean_prob = np.mean(prob_concat, axis=1)
        evaluate(mean_prob, mode=mode)
        
    logger.info('Challenge done.')
        
        
def generate_cv_folds(session_col, num_split):
    idx = []
    
    sessions = session_col.unique()
    np.random.shuffle(sessions)
    
    fold_size = len(sessions) // num_split
    for i in range(num_split):
        start = i * fold_size 
        
        if i == num_split - 1:
            end = len(sessions)
        else:
            end = (i + 1) * fold_size
            
        fold_sessions = sessions[start : end]
        train_idx = session_col[~session_col.isin(fold_sessions)].index
        val_idx = session_col[session_col.isin(fold_sessions)].index
        idx.append((train_idx, val_idx))
        
    return idx


def get_reciprocal_ranks(df):
    """Calculate reciprocal ranks for recommendations."""
    mask = df.item_ref == df.item_recommendations

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(df.item_recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0
    
    
def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)


def get_average_precision(item_id, rec, pos):
    if item_id in rec[:pos]:
        idx = np.nonzero(item_id == rec[:pos])[0][0]
        return 1 / (idx+1)
    return 0


class RecommenderModel:
    def __init__(self, algo, train, params, features, 
                 num_boost, early_stopping, mode):
        self.algo = algo
        self.train_set = train
        self.params = params
        self.num_boost_round = num_boost
        self.early_stopping_rounds = early_stopping
        self.features = features
        # self.cat_cols = ['action_type_last_int_sess', 
        #                   'platform', 'city', 'country', 'device',
        #                   'is_last_interacted', 'cluster_sess']
        self.models = []
        self.split = 5
        self.mode = mode
        if self.algo == 'classifier':
            logger.info('LightGBM classifier initiated.')
        elif self.algo == 'learning_to_rank':
            logger.info('LightGBM ranker initiated.')
        
    def cross_validate(self):
        start = time.time()
        
        test_set = pd.read_feather(os.path.join(dataset_path,'test.feather'))
        self.test_prob = np.zeros((test_set.shape[0], self.split))
        
        cv_idx = generate_cv_folds(self.train_set['new_session_id'], self.split)
        for fold, (train_idx, val_idx) in enumerate(cv_idx):
            logger.info('**********')
            logger.info(f'Start cross validating for fold {fold+1}...')
            
            train = self.train_set.loc[train_idx]
            val = self.train_set.loc[val_idx]
            
            if self.algo == 'classifier':
                suffix = 'clf'
                lgb_train_ds = lgb.Dataset(
                            train[self.features],
                            label=train['is_clicked'],
                            # categorical_feature=self.cat_cols,
                            feature_name=self.features,
                        )

                lgb_val_ds = lgb.Dataset(
                            val[self.features],
                            label=val['is_clicked'], 
                            reference=lgb_train_ds,
                            # categorical_feature=self.cat_cols,
                            feature_name=self.features,
                        )
            elif self.algo == 'learning_to_rank':
                suffix = 'ltr'
                train_group = train.groupby('new_session_id').size().values
                val_group = val.groupby('new_session_id').size().values
            
                lgb_train_ds = lgb.Dataset(
                            train[self.features],
                            label=train['is_clicked'],
                            group=train_group,
                            # categorical_feature=self.cat_cols,
                            feature_name=self.features,
                        )

                lgb_val_ds = lgb.Dataset(
                            val[self.features],
                            label=val['is_clicked'],
                            group=val_group, 
                            reference=lgb_train_ds,
                            # categorical_feature=self.cat_cols,
                            feature_name=self.features,
                        )                
        
            logger.info('LightGBM Dataset created.')
            
            bst = lgb.train(
                self.params,
                lgb_train_ds,
                valid_sets=[lgb_train_ds, lgb_val_ds],
                valid_names=['training', 'validation'],
                num_boost_round=self.num_boost_round,
                early_stopping_rounds=self.early_stopping_rounds,
                # categorical_feature=self.cat_cols,
                verbose_eval=50,
            )
            
            logger.info(f'Finished training fold {fold+1}.')
            
            bst.save_model(os.path.join(model_path, f'lgb_model_{fold+1}_{suffix}.txt'), 
                           num_iteration=bst.best_iteration)
            self.models.append(bst)
            logger.info('Model saved.')
            
            del lgb_train_ds, lgb_val_ds
            gc.collect()
            
            if self.mode == 'full':
                logger.info('Generating recommendations for challenge set...')
            else:
                logger.info('Generating recommendations for validation set...')
                
            test_preds = bst.predict(test_set[self.features], 
                                            num_iteration=bst.best_iteration)
            self.test_prob[:, fold] = test_preds

            #save model feature importance
            feature_importance = (pd.DataFrame({'feature': self.features, 
                                    'importance': bst.feature_importance('gain')})
                          .sort_values(by='importance', ascending=False)
                          .reset_index(drop=True))
            
            feature_importance.to_feather(os.path.join(model_path, 
                                                       f'model{fold+1}_fi_{suffix}.feather'))
            
            # logger.info(f'Finished training fold {fold+1}.')
        
        test_prob_df = pd.DataFrame(self.test_prob, columns=[f'model_{i+1}_{suffix}' for i in range(self.split)])
        test_prob_df.to_feather(os.path.join(model_path, f'model_prob_{suffix}.feather'))
        logger.info(f'Finished cross-validation. Done in {(time.time()-start)/60:.1f} minutes.')
        
        
def get_scores(res_df, ground_truth, session_id_col):
    res_df = res_df.sort_values(by=[session_id_col, 'prob'], ascending=False)
    res_df = (res_df.groupby(session_id_col).item_id.apply(np.array)
                .to_frame()
                .reset_index()
                .rename(columns={'item_id': 'item_recommendations'}))

    res_df = res_df.merge(ground_truth, on=session_id_col, how='inner')
    
    # reciprocal rank
    res_df['rr'] = res_df.apply(get_reciprocal_ranks, axis=1)

    # average precision
    res_df['ap_1'] = pd.Series([get_average_precision(ref, rec, 1)
                                for rec, ref in zip(res_df['item_recommendations'], res_df['item_ref'])])
    res_df['ap_3'] = pd.Series([get_average_precision(ref, rec, 3)
                                for rec, ref in zip(res_df['item_recommendations'], res_df['item_ref'])])
    res_df['ap_5'] = pd.Series([get_average_precision(ref, rec, 5)
                                for rec, ref in zip(res_df['item_recommendations'], res_df['item_ref'])])
    # recall
    res_df['recall_3'] = pd.Series([1 if ref in rec[:3] else 0
                                for rec, ref in zip(res_df['item_recommendations'], res_df['item_ref'])])
    res_df['recall_5'] = pd.Series([1 if ref in rec[:5] else 0
                                for rec, ref in zip(res_df['item_recommendations'], res_df['item_ref'])])

    return res_df

        
def evaluate(test_prob, mode):
    logger.info('Evaluating...')
    ground_truth = pd.read_feather(ground_truth_path)
    test_set = pd.read_feather(os.path.join(dataset_path,'test.feather'))
    # test_prob = np.mean(test_prob, axis=1)
    
    if mode == 'full':
        test_res_df = test_set[['session_id', 'item_id', 'is_val', 'is_conf']].copy()
        test_res_df['prob'] = test_prob
        
        val_res_df = (test_res_df[test_res_df.is_val == 1].copy()
                        .drop(columns=['is_val', 'is_conf']))
        conf_res_df = (test_res_df[test_res_df.is_conf == 1].copy()
                        .drop(columns=['is_val', 'is_conf']))
        
        val_res_df = get_scores(val_res_df, ground_truth, 'session_id')
        conf_res_df = get_scores(conf_res_df, ground_truth, 'session_id')
        val_res_df.to_feather(os.path.join(model_path, 'val_recs.feather'))
        conf_res_df.to_feather(os.path.join(model_path, 'conf_recs.feather'))
        
        val_mrr = val_res_df['rr'].mean()
        val_map_at_1 = val_res_df['ap_1'].mean()
        val_map_at_3 = val_res_df['ap_3'].mean()
        val_map_at_5 = val_res_df['ap_5'].mean()
        val_recall_at_3 = val_res_df['recall_3'].mean()
        val_recall_at_5 = val_res_df['recall_5'].mean()
        
        conf_mrr = conf_res_df['rr'].mean()
        conf_map_at_1 = conf_res_df['ap_1'].mean()
        conf_map_at_3 = conf_res_df['ap_3'].mean()
        conf_map_at_5 = conf_res_df['ap_5'].mean()
        conf_recall_at_3 = conf_res_df['recall_3'].mean()
        conf_recall_at_5 = conf_res_df['recall_5'].mean()
        
        logger.info(f'Challenge validation set performance - MRR: {val_mrr:.3f} | ' +
                f'MAP@1: {val_map_at_1:.3f} | ' + 
                f'MAP@3: {val_map_at_3:.3f} | ' + 
                f'MAP@5: {val_map_at_5:.3f} | ' +
                f'Recall@3: {val_recall_at_3:.3f} | ' +
                f'Recall@5: {val_recall_at_5:.3f}')
        
        logger.info(f'Challenge confirmation set performance - MRR: {conf_mrr:.3f} | ' +
                f'MAP@1: {conf_map_at_1:.3f} | ' + 
                f'MAP@3: {conf_map_at_3:.3f} | ' + 
                f'MAP@5: {conf_map_at_5:.3f} | ' +
                f'Recall@3: {conf_recall_at_3:.3f} | ' +
                f'Recall@5: {conf_recall_at_5:.3f}')
    else:
        res_df = test_set[['new_session_id', 'item_id']].copy()
        res_df['prob'] = test_prob
        res_df = get_scores(res_df, ground_truth, 'new_session_id')

        mrr = res_df['rr'].mean()
        map_at_1 = res_df['ap_1'].mean()
        map_at_3 = res_df['ap_3'].mean()
        map_at_5 = res_df['ap_5'].mean()
        recall_at_3 = res_df['recall_3'].mean()
        recall_at_5 = res_df['recall_5'].mean()
    
        logger.info(f'Validation set performance - MRR: {mrr:.3f} | ' +
            f'MAP@1: {map_at_1:.3f} | ' + 
            f'MAP@3: {map_at_3:.3f} | ' + 
            f'MAP@5: {map_at_5:.3f} | ' +
            f'Recall@3: {recall_at_3:.3f} | ' +
            f'Recall@5: {recall_at_5:.3f}')
            

if __name__ == '__main__':
    main()
    