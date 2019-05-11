import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


class TimeSeriesDataSet(Dataset):
    def __init__(self, data):
        super().__init__()
        self.content = data

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return self.content[idx]


def replace_nan_with_col_mean(x):
    missing_flag = np.isnan(x)
    missing_indices = np.where(missing_flag)
    col_means = np.nanmean(x, axis=0)
    x[missing_indices] = np.take(col_means, missing_indices[1])
    return x, missing_flag


def construct_delta_matrix(time_series, time_stamps, observed_mask):
    delta = np.zeros(time_series.shape)
    for t in range(1, time_series.shape[0]):
        delta[t, :] = time_stamps[t] - time_stamps[t - 1] + (1 - observed_mask[t, :]) * delta[t - 1, :]
    return delta


def load_raw_data(data_dir, testing):
    def load_ith_pt_training(i):
        pt_with_na = pd.read_csv(data_dir / 'train_with_missing' / f'{i}.csv').values
        pt_ground_truth = pd.read_csv(data_dir / 'train_groundtruth' / f'{i}.csv').values

        time_stamps = pt_with_na[:, 0]
        pt_with_na, pt_ground_truth = pt_with_na[:, 1:], pt_ground_truth[:, 1:]
        pt_with_na, missing_flag = replace_nan_with_col_mean(pt_with_na)
        # Note: NaN in ground truth will not be used. Just to avoid NaN in pytorch which does not support nanmean() etc.
        ptmax = np.nanmax(pt_ground_truth, axis=0).reshape(1, -1)
        ptmin = np.nanmin(pt_ground_truth, axis=0).reshape(1, -1)
        pt_ground_truth, missing_flag_gt = replace_nan_with_col_mean(pt_ground_truth)

        eval_mask = (~missing_flag_gt) & missing_flag  # 1: locs masked for eval
        observed_mask = (~missing_flag).astype(float)  # 1: observed, 0: missing
        eval_mask = eval_mask.astype(float)

        delta = construct_delta_matrix(pt_with_na, time_stamps, observed_mask)

        return {
            'pt_with_na': pt_with_na,
            'pt_ground_truth': pt_ground_truth,
            'time_stamps': time_stamps,
            'observed_mask': observed_mask,
            'eval_mask': eval_mask,
            'pt_max': ptmax,
            'pt_min': ptmin,
            'length': pt_with_na.shape[0],
            'delta': delta,
            'pid': i
        }

    def load_ith_pt_testing(i):
        pt = pd.read_csv(data_dir / f'{i}.csv').values
        time_stamps = pt[:, 0]
        pt = pt[:, 1:]
        ptmax = np.nanmax(pt, axis=0).reshape(1, -1)
        ptmin = np.nanmin(pt, axis=0).reshape(1, -1)
        pt, missing_flag = replace_nan_with_col_mean(pt)
        observed_mask = (~missing_flag).astype(float)
        delta = construct_delta_matrix(pt, time_stamps, observed_mask)
        return {
            'pt_with_na': pt,
            'time_stamps': time_stamps,
            'observed_mask': observed_mask,
            'pt_max': ptmax,
            'pt_min': ptmin,
            'length': pt.shape[0],
            'delta': delta,
            'pid': i

        }

    if not testing:
        num_pts = len(list((data_dir / 'train_with_missing').glob('*.csv')))
        all_pts = [load_ith_pt_training(i) for i in range(1, num_pts+1)]
        return all_pts
    else:
        num_pts = len(list(data_dir.glob('*.csv')))
        all_pts_testing = [load_ith_pt_testing(i) for i in range(1, num_pts+1)]
        return all_pts_testing


def load_data(data_path, dump_pkl=True, reload_raw=False, testing=False,
              valid_size=0.2, shuffle=True, random_state=None):
    if not data_path.is_file():
        assert data_path.is_dir(), 'Specified path does not exist.'
        data_path = data_path / f"all_pts_{'training' if not testing else 'testing'}.pkl"

    if reload_raw or not data_path.is_file():
        data = load_raw_data(data_path.parent, testing)
        if dump_pkl:
            with open(data_path, 'wb') as outfile:
                pickle.dump(data, outfile)
    else:
        with open(data_path, 'rb') as infile:
            data = pickle.load(infile)

    if not testing:
        if random_state is not None:
            random.seed(random_state)
        if shuffle:
            random.shuffle(data)

        valid_size = valid_size if valid_size > 1 else int(valid_size * len(data))
        train_size = len(data) - valid_size
        train_set, valid_set = TimeSeriesDataSet(data[:train_size]), TimeSeriesDataSet(data[train_size:])
        return train_set, valid_set
    else:
        test_set = TimeSeriesDataSet(data)
        return test_set


def build_data_loader(dataset,
                      device=torch.device('cpu'),
                      batch_size=64,
                      shuffle=True,
                      testing=False):
    def pad_time_series_batch(batch_data):
        lengths = [x['length'] for x in batch_data]
        pids = [x['pid'] for x in batch_data]
        lengths, data_idx = torch.sort(torch.LongTensor(lengths),
                                       descending=True)
        batch_data = [batch_data[idx] for idx in data_idx]
        pids = [pids[idx] for idx in data_idx]

        data_dict = {}
        data_dict['values'] = pad_sequence([torch.FloatTensor(x['pt_with_na']) for x in batch_data],
                                           batch_first=True).to(device)
        data_dict['masks'] = pad_sequence([torch.FloatTensor(x['observed_mask']) for x in batch_data],
                                          batch_first=True).to(device)
        data_dict['deltas'] = pad_sequence([torch.FloatTensor(x['delta']) for x in batch_data],
                                           batch_first=True).to(device)
        data_dict['time_stamps'] = pad_sequence([torch.FloatTensor(x['time_stamps']) for x in batch_data],
                                                batch_first=True).to(device)
        data_dict['lengths'] = lengths.to(device)
        data_dict['pids'] = pids
        data_dict['max_vals'] = torch.FloatTensor(np.concatenate([x['pt_max']
                                                                  for x in batch_data])).to(device).unsqueeze(1)
        data_dict['min_vals'] = torch.FloatTensor(np.concatenate([x['pt_min']
                                                                  for x in batch_data])).to(device).unsqueeze(1)

        if not testing:
            data_dict['evals'] = pad_sequence([torch.FloatTensor(x['pt_ground_truth']) for x in batch_data],
                                              batch_first=True).to(device)
            data_dict['eval_masks'] = pad_sequence([torch.FloatTensor(x['eval_mask']) for x in batch_data],
                                                   batch_first=True).to(device)


        return data_dict

    data_iter = DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           collate_fn=pad_time_series_batch)

    return data_iter


def build_data_loaders(data_path, train_size, batch_size, device, shuffle=True):
    train_set, test_set = load_data(data_path, valid_size=1-train_size)
    train_iter = build_data_loader(train_set, device, batch_size, shuffle)
    test_iter = build_data_loader(test_set, device, batch_size, shuffle)
    return train_iter, test_iter
