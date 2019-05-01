import numpy as np
import pandas as pd
from lgbm import LgbmModel
from utils import prepare_datasets

def main():

    '''
    Load and preprocess data
    '''

    # Select relevant cached features
    train_feats_list = [
        '../features/train_stats_v2.h5',
    ]
    test_feats_list = [
        '../features/test_stats_v2.h5',
    ]

    train, test, y_tgt = prepare_datasets(train_feats_list, test_feats_list)

    # Select models to train
    controls = {
        'lgbm-models'   : bool(1),
    }

    feat_blacklist = [
        # 'skew',
    ]

    '''
    LGBM Models
    '''

    seed = 42
    model_name = f'm0_v2'

    if controls['lgbm-models']:

        lgbm_params = {
            'num_leaves' : 8,
            'learning_rate': 0.5,
            'min_child_samples' : 30,
            'n_estimators': 100,
            'reg_lambda': 1,
            'bagging_fraction' : 0.8,
            'bagging_freq' : 1,
            'bagging_seed' : seed,
            'silent': 1,
            'verbose': 1,
        }

        lgbm_model_0 = LgbmModel(
            train=train,
            test=test,
            y_tgt=y_tgt,
            output_dir='../level_1_preds/',
            fit_params=lgbm_params,
            sample_weight=1.0,
            postprocess_sub=True,
            feat_blacklist=feat_blacklist,
            cv_random_seed=seed,
        )

        lgbm_model_0.fit_predict(
            iteration_name=model_name,
            predict_test=True,
            save_preds=False,
            produce_sub=True,
            save_imps=True,
            save_aux_visu=False,
        )

if __name__ == '__main__':
    main()
