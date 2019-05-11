"""
Implementation of the CATSI (Context-Aware Time Series Imputation) model.
"""
import os
from pathlib import Path
from socket import gethostname
from datetime import date
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import pandas as pd

import tqdm
from model import CATSI
from utils import AverageMeter, build_data_loader, load_data
import argparse


class ContAwareTimeSeriesImp(object):
    def __init__(self, var_names,
                 train_data_path='./data/training',
                 out_path='../results/',
                 force_reload_raw=False,
                 valid_size=0.2,
                 device=None):
        # set params
        self.out_path = Path(out_path) / f'{date.today():%Y%m%d}-CATSI-{gethostname()}-{os.getpid()}'
        if not self.out_path.is_dir():
            self.out_path.mkdir()
        self.var_names = var_names
        self.var_names_dict = {i: item for i, item in enumerate(var_names)}
        self.num_vars = len(var_names)

        # load data
        self.train_set, self.valid_set = load_data(Path(train_data_path), reload_raw=force_reload_raw,
                                                   valid_size=valid_size)

        # create model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CATSI(len(var_names)).to(self.device)

    def fit(self, epochs=300, batch_size=64, eval_batch_size=64, eval_epoch=1, record_imp_epoch=50):
        # construct optimizer
        context_rnn_params = {
            'params': self.model.context_rnn.parameters(),
            'lr': 1e-3,
            'weight_decay': 5e-3
        }
        imp_rnn_params = {
            'params': [p[1] for p in self.model.named_parameters() if p[0].split('.')[0] != 'context_rnn'],
            'lr': 1e-3,
            'weight_decay': 5e-5
        }

        optimizer = optim.Adam([context_rnn_params, imp_rnn_params])

        train_iter = build_data_loader(self.train_set, self.device, batch_size, shuffle=True)
        valid_iter = build_data_loader(self.valid_set, self.device, eval_batch_size, shuffle=True)
        self.eval_batch_size = eval_batch_size

        imp_dfs_train = None
        imp_dfs_valid = None
        for epoch in range(epochs):
            self.model.train()

            pbar_desc = f'Epoch {epoch+1}: '

            pbar = tqdm.tqdm(total=len(train_iter), desc=pbar_desc)
            total_loss = AverageMeter()
            total_loss_eval = AverageMeter()
            verbose_loss = [AverageMeter() for _ in range(6)]
            for idx, data in enumerate(train_iter):

                optimizer.zero_grad()
                ret = self.model(data)
                clip_grad_norm_(self.model.parameters(), 1)
                ret['loss'].backward()
                optimizer.step()

                total_loss.update(ret['loss'].item(), ret['loss_count'].item())
                total_loss_eval.update(ret['loss_eval'].item(), ret['loss_eval_count'].item())
                for i, (k, v, c) in enumerate(ret['verbose_loss']):
                    verbose_loss[i].update(v.item(), c)

                pbar.set_description(pbar_desc + f'Training loss={total_loss.avg:.3e}')
                pbar.update()
            pbar_desc = f'Epoch {epoch + 1} done, Training loss={total_loss.avg:.3e}'
            pbar.set_description(pbar_desc)
            pbar.close()

            if (epoch+1) % eval_epoch == 0:
                self.evaluate(valid_iter)

            if record_imp_epoch and (epoch + 1) % record_imp_epoch == 0:
                if imp_dfs_train is None:
                    imp_dfs_train = self.retrieve_imputation(train_iter, epoch+1)
                else:
                    imp_dfs_train = pd.concat([imp_dfs_train, self.retrieve_imputation(train_iter, epoch+1)],
                                              axis=0)
                if imp_dfs_valid is None:
                    imp_dfs_valid = self.retrieve_imputation(valid_iter, epoch+1)
                else:
                    imp_dfs_valid = pd.concat([imp_dfs_valid, self.retrieve_imputation(valid_iter, epoch+1)],
                                              axis=0)

                imp_dfs_train.to_excel(self.out_path / 'imp_train.xlsx', merge_cells=False)
                imp_dfs_valid.to_excel(self.out_path / 'imp_valid.xlsx', merge_cells=False)

        print('Training is done, performing final evaluation on validation set...')

        loss_valid, mae, mre, nrmsd = self.evaluate(valid_iter)
        with open(self.out_path / 'final_eval.csv', 'w') as txtfile:
            txtfile.write(f'Metrics, ' + (', '.join(self.var_names)) + '\n')
            txtfile.write(f'MAE, ' + (', '.join([f'{x:.3f}' for x in mae])) + '\n')
            txtfile.write(f'MRE, ' + (', '.join([f'{x:.3f}' for x in mre])) + '\n')
            txtfile.write(f'nRMSD, ' + (', '.join([f'{x:.3f}' for x in nrmsd])) + '\n')

    def evaluate(self, data_iter, print_scores=True):
        self.model.eval()

        mae = [AverageMeter() for _ in range(self.num_vars)]
        mre = [AverageMeter() for _ in range(self.num_vars)]
        nrmsd = [AverageMeter() for _ in range(self.num_vars)]
        loss_valid = AverageMeter()

        for idx, data in enumerate(data_iter):
            eval_masks = data['eval_masks']
            eval_ = data['evals']
            max_vals = data['max_vals']
            min_vals = data['min_vals']

            ret = self.model(data)
            imputation = ret['imputations']
            loss_valid.update(ret['loss'], ret['loss_count'])

            abs_err = (eval_masks * (eval_ - imputation).abs()).sum(dim=[0, 1]) / eval_masks.sum(dim=[0, 1])
            rel_err = (eval_masks * (eval_ - imputation).abs() / eval_.clamp(min=1e-5)).sum(dim=[0, 1]) / eval_masks.sum(dim=[0, 1])
            [mae[i].update(abs_err[i], eval_.shape[0]) for i in range(self.num_vars)]
            [mre[i].update(rel_err[i], eval_.shape[0]) for i in range(self.num_vars)]

            range_norm = max_vals - min_vals
            nsd = eval_masks * (eval_ - imputation).abs() / range_norm
            for i, (nsd_val, nsd_num) in enumerate(zip((nsd.norm(dim=[0, 1])**2).tolist(),
                                                       eval_masks.sum(dim=[0, 1]).tolist())):
                nrmsd[i].update(nsd_val/nsd_num, nsd_num)

        mae = [x.avg for x in mae]
        mre = [x.avg for x in mre]
        nrmsd = [x.avg ** 0.5 for x in nrmsd]

        if print_scores:
            print(f'   MAE = ' + ('\t'.join([f'{x:.3f}' for x in mae])))
            print(f'   MRE = ' + ('\t'.join([f'{x:.3f}' for x in mre])))
            print(f' nRMSD = ' + ('\t'.join([f'{x:.3f}' for x in nrmsd])))
            print()

        return loss_valid.avg, mae, mre, nrmsd

    def retrieve_imputation(self, data_iter, epoch, colname='imp'):
        self.model.eval()

        imp_dfs = []
        for idx, data in enumerate(data_iter):
            eval_masks = data['eval_masks']
            eval_ = data['evals']

            ret = self.model(data)
            imputation = ret['imputations']

            pids = data['pids']
            imp_df = pd.DataFrame(eval_masks.nonzero().data.cpu().numpy(), columns=['pid', 'tid', 'colid'])
            imp_df['pid'] = imp_df['pid'].map({i: pid for i, pid in enumerate(pids)})
            imp_df['epoch'] = epoch
            imp_df['analyte'] = imp_df['colid'].map(self.var_names_dict)
            imp_df[colname] = imputation[eval_masks == 1].data.cpu().numpy()
            imp_df[colname + '_feat'] = ret['feat_imp'][eval_masks == 1].data.cpu().numpy()
            imp_df[colname + '_hist'] = ret['hist_imp'][eval_masks == 1].data.cpu().numpy()
            imp_df['ground_truth'] = eval_[eval_masks == 1].data.cpu().numpy()
            imp_dfs.append(imp_df)
        imp_dfs = pd.concat(imp_dfs, axis=0).set_index(['pid', 'tid', 'analyte', 'ground_truth'])
        return imp_dfs

    def impute_test_set(self, data_set, batch_size=None):
        batch_size = batch_size or self.eval_batch_size
        data_iter = build_data_loader(data_set, self.device, batch_size, False, testing=True)
        self.model.eval()

        out_dir = self.out_path / 'imputations_test_set'
        out_dir.mkdir()

        imp_dfs = []
        pbar = tqdm.tqdm(desc='Generating imputation', total=len(data_iter))
        for idx, data in enumerate(data_iter):
            missing_masks = 1 - data['masks']
            ret = self.model(data)
            imputation = ret['imputations']

            pids = data['pids']
            imp_df = pd.DataFrame(missing_masks.nonzero().data.cpu().numpy(), columns=['pid', 'tid', 'colid'])
            imp_df['pid'] = imp_df['pid'].map({i: pid for i, pid in enumerate(pids)})
            imp_df['analyte'] = imp_df['colid'].map(self.var_names_dict)
            imp_df['imputation'] = imputation[missing_masks == 1].data.cpu().numpy()
            imp_dfs.append(imp_df)

            for p in range(len(pids)):
                seq_len = data['lengths'][p]
                time_stamps = data['time_stamps'][p, :seq_len].unsqueeze(1)
                imp = imputation[p, :seq_len, :]
                df = pd.DataFrame(torch.cat([time_stamps, imp], dim=1).data.cpu().numpy(),
                                  columns=['CHARTTIME'] + self.var_names)
                df['CHARTTIME'] = df['CHARTTIME'].apply(np.int)
                df.to_csv(out_dir / f'{pids[p]}.csv', index=False)
            pbar.update()
        pbar.close()
        print(f'Done, results saved in:\n {out_dir.resolve()}')
        return imp_dfs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='training_data/',
                        help='Path of the training data.')
    parser.add_argument('-o', '--output', type=str, default='results/',
                        help='Folder to save the results.')
    parser.add_argument('-t', '--testing', type=str, default='test_data/',
                        help='Path of the test data.')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    var_names = [
        'PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB',
        'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU'
    ]

    model = ContAwareTimeSeriesImp(var_names,
                                   train_data_path=args.input,
                                   out_path=args.output,
                                   force_reload_raw=args.reload,
                                   device=device)

    model.fit(epochs=args.epochs,
              batch_size=args.batch_size,
              eval_batch_size=args.eval_batch_size)

    if args.testing:
        test_set = load_data(Path(args.testing), reload_raw=args.reload, testing=True)
        model.impute_test_set(test_set)
