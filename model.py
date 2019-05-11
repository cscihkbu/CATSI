import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math


class MLPFeatureImputation(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLPFeatureImputation, self).__init__()

        self.W = Parameter(torch.Tensor(input_size, hidden_size, input_size))
        self.b = Parameter(torch.Tensor(input_size, hidden_size))
        self.nonlinear_regression = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        m = torch.ones(input_size, hidden_size, input_size)
        stdv = 1. / math.sqrt(input_size)
        for i in range(input_size):
            m[i, :, i] = 0
        self.register_buffer('m', m)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        hidden = torch.cat(tuple(F.linear(x, self.W[i] * Variable(self.m[i]), self.b[i]).unsqueeze(2)
                            for i in range(len(self.W))), dim=2)
        z_h = self.nonlinear_regression(hidden)
        return z_h.squeeze(-1)


class InputTemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)


class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, seq_lengths):
        T_max = input.shape[1]  # batch x time x dims

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)

        for t in range(T_max):
            h = self.rnn_cell(input[:, t, :], h)
            padding_mask = ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device)
            hn = padding_mask * h + (1-padding_mask) * hn

        return hn


class CATSI(nn.Module):
    def __init__(self, num_vars, hidden_size=64, context_hidden=32):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.context_mlp = nn.Sequential(
            nn.Linear(3*self.num_vars+1, 2*context_hidden),
            nn.ReLU(),
            nn.Linear(2*context_hidden, context_hidden)
        )
        self.context_rnn = RNNContext(2*self.num_vars, context_hidden)
        
        self.initial_hidden = nn.Linear(2*context_hidden, 2*hidden_size)
        self.initial_cell_state = nn.Tanh()

        self.rnn_cell_forward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)
        self.rnn_cell_backward = nn.LSTMCell(2*num_vars+2*context_hidden, hidden_size)

        self.decay_inputs = InputTemporalDecay(input_size=num_vars)

        self.recurrent_impute = nn.Linear(2*hidden_size, num_vars)
        self.feature_impute = MLPFeatureImputation(num_vars)

        self.fuse_imputations = nn.Linear(2*num_vars, num_vars)

    def forward(self, data):
        seq_lengths = data['lengths']

        values = data['values']  # pts x time_stamps x vars
        masks = data['masks']
        deltas = data['deltas']

        # compute context vector, h0 and c0
        T_max = values.shape[1]
        padding_masks = torch.cat(tuple(((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                                   for t in range(T_max)), dim=1)
        padding_masks = padding_masks.unsqueeze(2).repeat(1, 1, values.shape[2])  # pts x time_stamps x vars

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars
        data_variance = ((values - data_means.unsqueeze(1)) ** 2).sum(dim=1) / (masks.sum(dim=1) - 1)
        data_stdev = data_variance ** 0.5
        data_missing_rate = 1 - masks.sum(dim=1) / padding_masks.sum(dim=1)
        data_stats = torch.cat((seq_lengths.unsqueeze(1).float(), data_means, data_stdev, data_missing_rate), dim=1)

        # normalization
        min_max_norm = data['max_vals'] - data['min_vals']
        normalized_values = (values - data['min_vals']) / min_max_norm
        normalized_means = (data_means - data['min_vals'].squeeze(1)) / min_max_norm.squeeze(1)

        if self.training:
            normalized_evals = (data['evals'] - data['min_vals']) / min_max_norm

        x_prime = torch.zeros_like(normalized_values)
        x_prime[:, 0, :] = normalized_values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = normalized_values[:, t-1, :]

        gamma = self.decay_inputs(deltas)
        x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        x_complement = (masks * normalized_values + (1-masks) * x_decay) * padding_masks

        context_mlp = self.context_mlp(data_stats)
        context_rnn = self.context_rnn(torch.cat((x_complement, deltas), dim=-1), seq_lengths)
        context_vec = torch.cat((context_mlp, context_rnn), dim=1)
        h = self.initial_hidden(context_vec)
        c = self.initial_cell_state(h)

        inputs = torch.cat([x_complement, masks, context_vec.unsqueeze(1).repeat(1, T_max, 1)], dim=-1)

        h_forward, c_forward = h[:, :self.hidden_size], c[:, :self.hidden_size]
        h_backward, c_backward = h[:, self.hidden_size:], c[:, self.hidden_size:]
        hiddens_forward = h[:, :self.hidden_size].unsqueeze(1)
        hiddens_backward = h[:, self.hidden_size:].unsqueeze(1)
        for t in range(T_max-1):
            h_forward, c_forward = self.rnn_cell_forward(inputs[:, t, :],
                                                         (h_forward, c_forward))
            h_backward, c_backward = self.rnn_cell_backward(inputs[:, T_max-1-t, :],
                                                            (h_backward, c_backward))
            hiddens_forward = torch.cat((hiddens_forward, h_forward.unsqueeze(1)), dim=1)
            hiddens_backward = torch.cat((h_backward.unsqueeze(1), hiddens_backward), dim=1)

        rnn_imp = self.recurrent_impute(torch.cat((hiddens_forward, hiddens_backward), dim=2))
        feat_imp = self.feature_impute(x_complement).squeeze(-1)

        # imputation fusion
        beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imp = masks * normalized_values + (1-masks) * imp_fusion

        rnn_loss = F.mse_loss(rnn_imp * masks, normalized_values * masks, reduction='sum')
        feat_loss = F.mse_loss(feat_imp * masks, normalized_values * masks, reduction='sum')
        fusion_loss = F.mse_loss(imp_fusion * masks, normalized_values * masks, reduction='sum')
        total_loss = rnn_loss + feat_loss + fusion_loss


        if self.training:
            rnn_loss_eval = F.mse_loss(rnn_imp * data['eval_masks'], normalized_evals * data['eval_masks'], reduction='sum')
            feat_loss_eval = F.mse_loss(feat_imp * data['eval_masks'], normalized_evals * data['eval_masks'], reduction='sum')
            fusion_loss_eval = F.mse_loss(imp_fusion * data['eval_masks'], normalized_evals * data['eval_masks'], reduction='sum')
            total_loss_eval = rnn_loss_eval + feat_loss_eval + fusion_loss_eval

        def rescale(x):
            return torch.where(padding_masks==1, x * min_max_norm + data['min_vals'], padding_masks)

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imp)

        out_dict = {
            'loss': total_loss / masks.sum(),
            'verbose_loss': [
                ('rnn_loss', rnn_loss / masks.sum(), masks.sum()),
                ('feat_loss', feat_loss / masks.sum(), masks.sum()),
                ('fusion_loss', fusion_loss / masks.sum(), masks.sum())
            ],
            'loss_count': masks.sum(),
            'imputations': final_imp,
            'feat_imp': feat_imp,
            'hist_imp': rnn_imp
        }
        if self.training:
            out_dict['loss_eval'] = total_loss_eval / data['eval_masks'].sum()
            out_dict['loss_eval_count'] = data['eval_masks'].sum()
            out_dict['verbose_loss'] += [
                ('rnn_loss_eval', rnn_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('feat_loss_eval', feat_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('fusion_loss_eval', fusion_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum())
            ]

        return out_dict

