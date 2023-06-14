#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy.stats
import sklearn
from sklearn import preprocessing


class OU(object):

    def __init__(self, df1, df2, model_size=None, eval_size=None):

        self.df1 = df1
        self.df2 = df2
        self.final_df=None
        self.m_size = model_size
        self.e_size = eval_size
        self.fts = []
        self.split_idx = []
        self.splits = []
        
        assert(df1.shape == df2.shape)

    
    def split_expand(self, n_splits=5):

        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.split_idx = list(tscv.split(self.df1))
        print("Expanding Split Successful.")
    
    
    def split_slide(self, m_size=30000, e_size=10000):

        splits = []
        end_ind = m_size
        cur_ind = 0
        
        assert(m_size < self.df1.shape[0])
        
        while end_ind < self.df1.shape[0]:
            # Finds train indices 
            train_ind = np.array(np.arange(cur_ind, end_ind))
            
            # If test indices for last test split less than e_size, then just use the rest.
            if end_ind+e_size<self.df1.shape[0]:
                test_ind = np.array(np.arange(end_ind, end_ind+e_size))  
            else:
                test_ind = np.array(np.arange(end_ind, self.df1.shape[0]))
                
            splits.append((train_ind, test_ind))
            end_ind += e_size
            cur_ind += e_size
        
        self.split_idx = splits
        print("Sliding Window Split Successful.")
    
    
    def fit_feature(self, s1, s2, feature):

                
        s1 = s1[feature]
        s2 = s2[feature]
        
        # Estimate linear relationship between p1 and p2 using linear regression
        beta, dx, _, _, _ = scipy.stats.linregress(s2, s1)
        # Retrieve the residuals (dx_t) of our estimate

        
        residuals = s1 - (s2*beta)

        # Integrate dx_t to find x_t
        x_t = np.cumsum(residuals)
        lag_price = x_t.shift(1)
        
        # Perform lag-1 auto regression on the x_t and the lag
        b, a, _, _, _ = scipy.stats.linregress(lag_price.iloc[1:], x_t.iloc[1:])

        # Calculate parameters to create a t-score
        # theta = -1/math.log(b)
        mu = a/(1-b)
        sigma = np.sqrt(np.var(x_t))

        t_score = (x_t - mu)/sigma
        t_score.name = feature
        # Return absolute value of t_score because we only care about the spread
        # t_score = np.abs(t_score)

        # Return transformed vector, residuals, beta, and the index of the dataframe we fit on.
        return {'tscore_fit_'+feature: t_score, 'residuals_fit_'+feature: residuals, 
                'beta_fit_'+feature: beta, 'dx_fit_'+feature: dx, 
                'mu_fit_'+feature: mu, 'sigma_fit_'+feature: sigma, 
                'fit_index_'+feature: np.array(s1.index)}
    
    
    def transform(self, t1, t2, feature, fit_dict):

        beta = fit_dict['beta_fit_'+feature]
        dx = fit_dict['dx_fit_'+feature]
        mu = fit_dict['mu_fit_'+feature]
        sigma = fit_dict['sigma_fit_'+feature]
        
        s1 = t1[feature]
        s2 = t2[feature]
        
        residuals = s1 - (s2*beta)

         # Integrate dx_t to find x_t
        x_t = np.cumsum(residuals)

        # Calculate parameters to create a t-score
        t_score = (x_t - mu)/sigma
        
        # Return absolute value of t_score because we only care about the spread
        # t_score = np.abs(t_score)
        t_score.name = feature
        
        # Return transformed vector, residuals, beta, and the index of the dataframe we fit on.
        return {'tscore_transform_'+feature: t_score, 'residuals_transform_'+feature: residuals, 
                'transform_index_': np.array(t1.index)}
    
    
    def fit_transform(self, d1, d2, t1, t2, ou_features, other_features=None):

        fit_dicts = {}
        t_dicts = {}
        
        for feature in ou_features:
            fit_dict = self.fit_feature(d1, d2, feature)
            fit_dicts.update(fit_dict)
            
            t_dict = self.transform(t1, t2, feature, fit_dict)
            t_dicts.update(t_dict)
            
        train = pd.DataFrame([fit_dicts[f] for f in fit_dicts.keys() if 'tscore' in f]).T
        test = pd.DataFrame([t_dicts[t] for t in t_dicts.keys() if 'tscore' in t]).T
        
        if other_features:
            for feat in other_features:
                train[feat+'1'] = d1[feat]
                train[feat+'2'] = d2[feat]
                test[feat+'1'] = t1[feat]
                test[feat+'2'] = t2[feat]

        return {'train': {'df': train, **fit_dicts}, 'test': {'df': test, **t_dicts}}
        
        
    def get_splits(self, ou_features, other_features=None, label_func=None, scale=False):

        assert(self.split_idx)
        
        fts = []
        # Fit-Transform the train and test datasets for each of the splits. 
        for train, test in self.split_idx:
            df_train1 = self.df1.loc[train]
            df_train2 = self.df2.loc[train]
            df_test1 = self.df1.loc[test]
            df_test2 = self.df2.loc[test]
            ft = self.fit_transform(df_train1, df_train2, df_test1, df_test2, ou_features, other_features)
            ft['train']['index'] = train
            ft['test']['index'] = test

            # Create Labels
            if label_func:
                train_labels = label_func(ft['train']['residuals_fit_price'])
                test_labels = label_func(ft['test']['residuals_transform_price'])
                ft['train']['labels'] = train_labels
                ft['test']['labels'] = test_labels
                
            # Perform Feature Scaling
            if scale:
                min_max_scaler = preprocessing.MinMaxScaler()
                x_scaled = min_max_scaler.fit_transform(ft['train']['df'])
                y_scaled = min_max_scaler.transform(ft['test']['df'])
                df_scaledx = pd.DataFrame(x_scaled)
                df_scaledy = pd.DataFrame(y_scaled)
                ft['train']['df_scale'] = df_scaledx
                ft['test']['df_scale'] = df_scaledy
                
            fts.append(ft) 
        self.fts = fts
        return fts
