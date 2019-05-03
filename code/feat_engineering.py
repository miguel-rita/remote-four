import numpy as np
import pandas as pd
import multiprocessing as mp
import tqdm, glob, time, pickle, re
from scipy.stats import skew, kurtosis
from scipy.fftpack import rfft

def atomic_worker(args):

    source_df_dir, start_ix, end_ix, compute_feats = args

    # Setup
    feat_names = []
    feat_arrays = []

    # Load df chunk
    df_chunk = pd.read_hdf(source_df_dir).iloc[start_ix:end_ix+1]
    index = df_chunk.index
    arr = df_chunk.values

    '''
    Stats features
    '''
    if compute_feats['stats-feats']:

        # Feature names
        stats_feats_names = [
            'std',
            'kurtosis',
        ]

        stats_feats_arr = np.hstack([
            np.std(arr, axis=1, keepdims=True),
            kurtosis(arr, axis=1)[:, None],
        ])

        feat_names.extend(stats_feats_names)
        feat_arrays.append(stats_feats_arr)

    '''
    Fourier features
    '''
    if compute_feats['fourier-feats']:

        # Num of coeffs
        n = 10

        # Calc coeffs
        fft_coefs = rfft(arr, axis=1)[:, :n]

        feat_names.extend([f'rfft_coef_{c:d}' for c in range(n)])
        feat_arrays.append(fft_coefs)

    '''
    Delta features
    '''
    if compute_feats['delta-feats']:

        # Feature names
        names = [
            'mean_abs_delta',
            'std_abs_delta',
        ]# + [f'delta_perc_{p:.2f}' for p in percentiles]

        deltas = arr[:, 1:] - arr[:, :-1]
        arr_ = np.hstack([
            np.mean(np.abs(deltas), axis=1, keepdims=True),
            np.std(np.abs(deltas), axis=1, keepdims=True),
            # np.percentile(arr, percentiles, axis=1).T
        ])

        feat_names.extend(names)
        feat_arrays.append(arr_)

    '''
    Aggregate all feats and return as df
    '''
    # Build final pandas dataframe
    df = pd.DataFrame(
        data=np.hstack(feat_arrays),
        columns=feat_names,
        index=index
    )

    print(f'Computed chunk {start_ix:d} to {end_ix}')

    return df

def gen_feats(save_rel_dir, save_name, source_df_dir, compute_feats):
    '''
    Generate custom features dataframe from stored preprocessed signal chunks

    :param save_rel_dir (str) Relative dir to save calculated feats
    :param save_name (str) Feat set name
    :param source_df_dir (str) Relative dir to source data dataframe
    :param compute_feats (dict) Dict of bools marking the feats to generate
    :return:
    '''

    np.warnings.filterwarnings('ignore')

    # Read source df
    df = pd.read_hdf(source_df_dir)

    # Split source df into batches
    n_chunks = 16
    atomic_args = [(source_df_dir, ix[0], ix[-1], compute_feats) for ix in np.array_split(np.arange(df.shape[0]), n_chunks)]

    print(f'> feature_engineering : Creating mp pool . . .')

    pool = mp.Pool(processes=3)#mp.cpu_count()-4)
    res = pool.map(atomic_worker, atomic_args)
    pool.close()
    pool.join()

    print(f'> feature_engineering : Concating and saving results . . .')

    # Concat atomic computed feats and save df
    df = pd.concat(res, axis=0).sort_index()
    df.to_hdf(save_rel_dir+'/'+save_name, key='w')

    # Also save feature names
    feat_list = list(df.columns)
    with open(save_rel_dir+'/'+save_name.split('.h5')[0]+'.txt', 'w') as f:
        f.writelines([f'{feat_name}\n' for feat_name in feat_list])
    with open(save_rel_dir+'/'+save_name.split('.h5')[0]+'.pkl', 'wb') as f2:
        pickle.dump(feat_list, f2, protocol=pickle.HIGHEST_PROTOCOL)


dataset = 'train'
st = time.time()

compute_feats_template = {
    'stats-feats': bool(0),
    'fourier-feats': bool(0),
    'delta-feats': bool(0),
}

feats_to_gen = {
    'stats-feats': 'stats_v8_x3',
    # 'fourier-feats': 'fourier_v4',
    'delta-feats': 'delta_v8_x3',
}

for ft_name, file_name in feats_to_gen.items():

    cpt_fts = compute_feats_template.copy()
    cpt_fts[ft_name] = True

    gen_feats(
        save_rel_dir='../features',
        save_name=f'{dataset}_{file_name}.h5',
        source_df_dir=f'../data/{dataset}_df_x3.h5',
        compute_feats=cpt_fts,
    )

print(f'> feature_engineering : Done, wall time : {(time.time()-st):.2f} seconds .')