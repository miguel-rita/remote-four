import pandas as pd
import numpy as np
import glob, time, datetime
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns

def build_interp(train_df):
    '''
    Build a 2D matrix with segment start/end for easy interpolation
    '''

    @jit(nopython=True)
    def earthquake_ends(arr):
        prev = np.inf
        curr = np.inf
        res = []
        for i, v in enumerate(arr):
            curr = v
            if curr > prev:
                res.append(i - 1)
            prev = curr
        return np.array(res)

    # Get segment ends
    t_ends = earthquake_ends(train_df['time_to_failure'].values)

    # Add last point as end (not really an ea end but segment end)
    t_ends = np.hstack([t_ends, train_df.shape[0] - 1])

    # Get segment starts, adding the zero point
    t_starts = np.roll(t_ends + 1, 1)
    t_starts[0] = 0

    # Store segment starts and ends intertwined in first column of output
    out = np.zeros((t_starts.size * 2, 2), dtype=np.float64)
    out[::2, 0] = t_starts
    out[1::2, 0] = t_ends
    out[:, 1] = train_df['time_to_failure'].values[out[:, 0].astype(int)]

    return out

def run_build_interp():
    train_df = pd.read_csv(
        '../data/train.csv.zip',
        dtype={
            'acoustic_data': np.int16,
            'time_to_failure': np.float32
        }
    )

    # Save target info to disk
    rec_ttf = build_interp(train_df)
    np.save('../data/ttf_rec.npy', rec_ttf)

    # Save signal column to disk as numpy array
    np.save('../data/signal.npy', train_df['acoustic_data'].values)

def generate_train(signal, ttf_rec, chunksize):
    '''
    Generate train df with one chunk per row. First column contains target
    '''

    # Split signal into train chunks
    n_chunks = np.round(signal.size / chunksize)
    splits = np.array_split(np.arange(signal.size), n_chunks)

    # Get targets for all chunk start/end points
    start_end_pts = [(s[0], s[-1]) for s in splits]
    start_end_tgts = [np.interp(se_pts, ttf_rec[:,0], ttf_rec[:,1]) for se_pts in start_end_pts]

    final_splits = []
    final_tgts = []

    # Don't consider chunks where earthquake occured ie. end_tgt > start_tgt
    for i, se_tgts in enumerate(start_end_tgts):
        if se_tgts[-1] < se_tgts[0]:
            final_splits.append(splits[i])
            final_tgts.append(se_tgts[-1]) # Target is the last ttf in the interval

    # Make chunks all the same size as smallest chunk by removing elements from beginning
    size = np.min([e.size for e in final_splits])
    final_splits = [fs[-size:] for fs in final_splits]

    # Concat chunks
    vals = np.vstack([signal[s] for s in final_splits])

    # Generate df
    df = pd.DataFrame(data=vals)

    # Return train and target

    return df, np.array(final_tgts).astype(np.float32)

def run_generate_train():
    train_df, tgt = generate_train(
        signal=np.load('../data/signal.npy'),
        ttf_rec=np.load('../data/ttf_rec.npy'),
        chunksize=150010
    )
    train_df.to_hdf('../data/train_df_rem.h5', key='train_df', mode='w')
    np.save('../data/target_rem.npy', tgt)

def generate_test(test_dir):
    '''
    Generate train df with one chunk per row
    '''

    # Grab all test chunk paths
    paths = glob.glob(test_dir + '*.csv')

    # Grab seg id from end of path
    seg_ids = [p.split('/')[-1][:-4] for p in paths]

    # Stack all segments into 2d array
    vals = np.hstack([pd.read_csv(p).values.astype(np.int16) for p in paths])

    # Combine into dataframe with one chunk per line
    return pd.DataFrame(data=vals.T, index=seg_ids)

def run_generate_test():
    df = generate_test('../data/test/')
    df.to_hdf('../data/test_df.h5', key='test_df', mode='w')

# Borrowed code

def prepare_datasets(train_feats_list, test_feats_list):
    '''
    From a list of paths to precomputed features, loads and prepares train, test and target datasets
    for use by the models

    :param train_feats_list: list of relative paths to precomputed train feats
    :param test_feats_list: list of relative paths to precomputed test feats
    :return: tuple of train df, test df, target 1d np array
    '''

    # Concatenate train and test feats
    train_feats_dfs = [pd.read_hdf(path, mode='r') for path in train_feats_list]
    train_feats_df = pd.concat(train_feats_dfs, axis=1)

    test_feats_dfs = [pd.read_hdf(path, mode='r') for path in test_feats_list]
    test_feats_df = pd.concat(test_feats_dfs, axis=1)

    # Read metadata target for train set
    y_target = np.load('../data/target.npy')

    # ttf_rec = np.load('../data/ttf_rec.npy')
    # ttf_rec[:,0] = ttf_rec[:,0] * y_target.size / ttf_rec[-1,0]
    # final_ixs = np.arange(ttf_rec[-1, 0])
    # final_ixs_2 = final_ixs[(final_ixs < ttf_rec[9, 0]) | (final_ixs >= ttf_rec[13, 0])]
    # final_ixs_2 = final_ixs_2.astype(int)

    return train_feats_df, test_feats_df, y_target

def plot_aux_visu():
    # TODO
    pass

def save_importances(imps_, filename_):
    mean_gain = imps_[['gain', 'feat']].groupby('feat').mean().reset_index()
    mean_gain.index.name = 'feat'
    plt.figure(figsize=(6, 1*mean_gain.shape[0]))
    sns.barplot(x='gain', y='feat', data=mean_gain.sort_values('gain', ascending=False))
    plt.title(f'Num. feats = {mean_gain.shape[0]:d}')
    plt.tight_layout()

    # Timestamp
    ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
    plt.savefig(f'{filename_}_{ts}.png')
    plt.clf()

def save_submission(test_preds_df, sub_name, postprocess):

    if postprocess:
        pass

    # Pull index and rename columns
    test_preds_df = test_preds_df.reset_index(drop=False)
    test_preds_df.columns = ['seg_id', 'time_to_failure']

    # Sort sub to match sample sub
    test_preds_df.sort_values(by='seg_id', inplace=True)

    # Save sub
    test_preds_df.to_csv(f'../submissions/{sub_name}', index=False)

def main():
    # run_build_interp()
    # run_generate_test()
    run_generate_train()
    pass

if __name__ == '__main__':
    main()
