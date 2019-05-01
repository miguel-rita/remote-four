import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from utils import plot_aux_visu, save_importances, save_submission

'''
LGBM Class definition
'''

class LgbmModel:

    # Constructor
    def __init__(self, train, test, y_tgt, output_dir, fit_params, sample_weight, postprocess_sub, feat_blacklist, cv_random_seed):

        # dataset
        self.train = train
        self.test = test
        self.y_tgt = y_tgt

        # other params
        self.output_dir = output_dir
        self.fit_params = fit_params
        self.feat_blacklist = feat_blacklist
        self.cv_random_seed = cv_random_seed

        self.postprocess_sub = postprocess_sub

        # Initialize sample weight
        self.sample_weight = np.ones(shape=self.y_tgt.shape)

    def fit_predict(self, iteration_name, predict_test=True, save_preds=True, produce_sub=False, save_imps=True,
                    save_aux_visu=False):

        if produce_sub:
            predict_test = True

        '''
        Setup CV
        '''

        # CV cycle collectors
        y_oof = np.zeros(self.y_tgt.size)
        if predict_test:
            y_test = np.zeros(self.test.shape[0])
        eval_metrics = []
        imps = pd.DataFrame()

        # Setup stratified CV
        num_folds = 5
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=self.cv_random_seed)

        # Extract numpy arrays for use in lgbm fit method
        approved_feats = [feat for feat in list(self.train.columns) if feat not in self.feat_blacklist]

        x_all = self.train[approved_feats].values
        if predict_test:
            x_test = self.test[approved_feats].values

        for i, (_train, _eval) in enumerate(folds.split(x_all)):

            print(f'> lgbm : Computing fold number {i} . . .')

            # Setup fold data
            x_train, y_train = x_all[_train], self.y_tgt[_train]
            sample_weight = self.sample_weight[_train]
            x_eval, y_eval = x_all[_eval], self.y_tgt[_eval]

            # Setup binary LGBM
            bst = lgb.LGBMRegressor(
                boosting_type='gbdt',
                num_leaves=self.fit_params['num_leaves'],
                learning_rate=self.fit_params['learning_rate'],
                n_estimators=self.fit_params['n_estimators'],
                objective='mae',
                reg_lambda=self.fit_params['reg_lambda'],
                min_child_samples=self.fit_params['min_child_samples'],
                silent=self.fit_params['silent'],
                bagging_fraction=self.fit_params['bagging_fraction'],
                bagging_freq=self.fit_params['bagging_freq'],
                bagging_seed=self.fit_params['bagging_seed'],
                verbose=self.fit_params['verbose'],
            )

            # Train bst
            bst.fit(
                X=x_train,
                y=y_train,
                sample_weight=sample_weight,
                eval_set=[(x_eval, y_eval)],
                eval_names=['\neval_set'],
                early_stopping_rounds=10,
                verbose=self.fit_params['verbose'],
            )

            # Compute and store oof predictions and metric, performing custom thresholding
            y_oof[_eval] = bst.predict(x_eval)
            metric = mean_absolute_error(y_eval, y_oof[_eval])
            eval_metrics.append(metric)
            print(f'> lgbm : Fold MAE : {metric:.4f}')

            # Build test predictions
            if predict_test:
                y_test += bst.predict(x_test) / num_folds

            # Store importances
            if save_imps:
                imp_df = pd.DataFrame()
                imp_df['feat'] = approved_feats
                imp_df['gain'] = bst.feature_importances_
                imp_df['fold'] = i
                imps = pd.concat([imps, imp_df], axis=0, sort=False)

        print('> lgbm : CV results : ')
        print(pd.Series(eval_metrics).describe())

        np.save('../other/y_oof_.npy', y_oof)
        np.save('../other/y_pred_.npy', y_test)
        np.save('../other/y_tgt_.npy', self.y_tgt)
        print('> lgbm : Postprocessed CV : ')
        asort = np.argsort(y_oof)
        unsort = np.argsort(asort)
        sorted_tgt = np.sort(self.y_tgt)
        new_oof = self.y_tgt[unsort]
        print(mean_absolute_error(self.y_tgt, new_oof))

        '''
        Output wrap-up : save importances, predictions (oof and test), submission and others
        '''

        # Insert here additional metrics
        final_metric = np.mean(eval_metrics)

        if self.postprocess_sub:
            final_name = f'lgbm_{iteration_name}_{final_metric:.4f}_pp'
        else:
            final_name = f'lgbm_{iteration_name}_{final_metric:.4f}'

        test_preds_df = pd.DataFrame(data=y_test[:, None], columns=[final_name], index=self.test.index)

        if save_imps:
            save_importances(imps, filename_='../importances/imps_' + final_name)

        if save_preds:
            train_preds_df = pd.DataFrame(data=y_oof[:, None], columns=[final_name])
            train_preds_df.to_hdf(self.output_dir + f'{final_name}_oof.h5', key='w')

            # No sense in saving test without train hence indent
            if predict_test:
                test_preds_df.to_hdf(self.output_dir + f'{final_name}_test.h5', key='w')

        if produce_sub:
            save_submission(
                test_preds_df,
                sub_name=f'../submissions/{final_name}.csv',
                postprocess=self.postprocess_sub,
            )

        if save_aux_visu:
            if False:
                plot_aux_visu()
            pass

