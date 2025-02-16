from .metrics import *

import sys

LOCAL = True

hour = 3600

ACCEL = None

TARGET_STEPS = 20000
MAX_EPOCHS = 50
VALIDATION = False

if VALIDATION: print('using extra time to show validation results');
if ACCEL is not None: print(' ** {}x FASTER RUN **'.format(ACCEL))
else: ACCEL = 1

from types import SimpleNamespace   

import random
import datetime
import math
import numpy as np
import os
import time

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
# print(torch. __version__) 
    
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterSampler

from torch.utils.data import Dataset, DataLoader    
from torch.optim import AdamW
    
# import timm
# print('timm', timm.__version__)
from timm.models.vision_transformer import VisionTransformer

class XformerModel:
    def __init__(self, metadata):
        self.metadata_ = metadata
        
    def train(self, dataset, val_dataset = None, val_metadata = None,
                          remaining_time_budget = 5 * hour, seed = None):

        remaining_time_budget /= ACCEL
        self.remaining_time_budget = remaining_time_budget
        
        self.start_time = time.time()
        self.end_time = self.start_time + 0.8 * self.remaining_time_budget
        
        self.experiments = []
        self.done_trialing = False

        params = self.scope(dataset, val_dataset, val_metadata,
                          remaining_time_budget, seed)
        
        self.trials(params)
        
        while not self.done_trialing:
            self.trial()
            
        self.train_final()
            
    def train_final(self):
        print('training final models');
        self.models = []
        max_t = sorted(self.experiments, key = lambda x: x[-1])[-1][-1]
        n_final = min(3, int(len(self.experiments) ** 0.5))
        print()
        for e in sorted(self.experiments, key = lambda x: x[1])[:n_final]:
            print(e)
        print()
        for i in range(n_final):
            params, score, t = sorted(self.experiments, key = lambda x: x[1])[i]
            if self.end_time - time.time() < t and len(self.models) > 0:
                return # no time to retrain
            else:
                self.models.append(self.fit(params))
                
        print('Total time: {:.1f} sec. ({:.0%} of allotted)'.format(
                time.time() - self.start_time, 
                (time.time() - self.start_time) /  self.remaining_time_budget ))
        
            
    def scope(self, dataset, val_dataset, val_metadata,
                          remaining_time_budget, seed):
        
        self.train_dataset = dataset 
        self.val_dataset = val_dataset
        
        self.metric = get_metric_func(dataset.metadata.get_final_metric(),
                                      dataset.metadata.get_task_type(),)
        
        self.full_train_data = XformerDataset(self.train_dataset, test = False)
        
        if val_dataset:
            self.train_data = self.full_train_data
            self.val_data = XformerDataset(val_dataset, )
            self.idx_train = list(range(len(self.train_data)))
            
        else:
            self.idx_train, self.idx_val, _, _ = train_test_split(
                 *[np.arange(len(self.train_dataset))] * 2,
                            random_state = seed if seed is not None
                                else datetime.datetime.now().microsecond,
                test_size = 0.4,
            )
            self.train_data = XformerDataset(self.train_dataset,
                                             self.idx_train, test = False)
            self.val_data = XformerDataset(self.train_dataset, self.idx_val)
            

        self.full_train_dataloader = get_xformer_dataloader(self.full_train_data, 
                                                       test = False)
        self.train_dataloader = get_xformer_dataloader(self.train_data, 
                                                       test = False)
        self.val_dataloader = get_xformer_dataloader(self.val_data, )
                
            
        # assume 5-10it/s; 100 = ~10s  (may be asked for ~360-720s)
        if True:
            start = time.time()
            params = get_xformer_base_params(self.train_data, self.train_dataloader)
            model = get_xformer_model(params, 
                        task_type = self.train_dataset.metadata.get_task_type(),
                    final_metric = self.train_dataset.metadata.get_final_metric(),
                        dataset = self.train_data)

            # roughly 100s for 1 hour run, (or ~16 of 360s, or 300s for 10h run)
            params['steps'] = int(500 * (remaining_time_budget / hour) ** 0.7)
            start = time.time()
            model = fit_xformer(model, params, 
                                self.train_dataloader, self.val_dataloader, );
            self.step_time = (time.time() - start) / params['steps']
            start = time.time()
            yp, y = predict_xformer(model, self.val_dataloader, limit = 20)
            self.infer_time = ( (time.time() - start) / len(yp) 
                                     * len(self.val_data) )
            print('estimated {:.0f}s per 1k steps'.format(self.step_time * 1000))
            print('estimated {:.0f}s inference'.format(self.infer_time))
            
            
        # set up step count
        time_per_run = int((self.end_time - time.time() - self.infer_time * 1) / 10)
        est_steps = max(0, int((time_per_run - self.infer_time/2) / self.step_time))
        print(' estimate ~dozen runs of {} steps @ {}s ea.'.format(
            est_steps, time_per_run))

        steps = int(est_steps ** 0.6 * (TARGET_STEPS) ** 0.2
                           * (len(self.train_dataloader) * 20) ** 0.2)
        steps = min(steps, len(self.train_dataloader) * MAX_EPOCHS)    
        steps = max(steps, 100) 
                                 
        if steps > 3000: steps = 1000 * int(steps/1000)
        elif steps > 300: steps = 100 * int(steps/100)
        elif steps > 30: steps = 10 * int(steps/10)
        print('targeting {} steps'.format(steps))
        
        self.est_runs =  (self.end_time - time.time()
                         ) / (steps * self.step_time + self.infer_time)  
        print('  with {:.1f} runs @ full-scale'.format(self.est_runs))
        
        params['steps'] = steps
        
        if self.est_runs > 2.5: self.trial(params);
        else: self.done_trialing = True; self.experiments = [(params, -1, -1)]
        return params
    
    def trials(self, params):
        self.param_dict = get_xformer_param_dict(params, self.full_train_data)
        
    def _check_trials(self):
        if len(self.experiments) > 0:
            max_t = sorted(self.experiments, key = lambda x: x[-1])[-1][-1]
            n_final = min(3, int(len(self.experiments) ** 0.5))
            if self.end_time - time.time() < (n_final + 1) * max_t:
                self.done_trialing = True
        if len(self.experiments) >= 15:
            self.done_trialing = True
            
    def evo_update(self, params):
        print('before update', params)
        if len(self.experiments) >= 2:
            best_params = [e[0] for e in sorted(self.experiments, 
                                                    key = lambda x: x[1])][:-1]
            best_wts = np.array([np.exp(-0.7 * i) for i in range(len(best_params))])
            best_wts /= 1
            print(best_wts)
            replace_rate = 1 -  1 / (len(best_params) + 0.5 ) ** 0.4
            print(replace_rate)
            for k in list(params.keys()):
                if random.random() < replace_rate:
                    source_params =  random.choices(best_params, best_wts)[0]
                    print(k, source_params.get(k))
                    if k in source_params:
                        params[k] = source_params[k]
                    else:
                        params.pop(k)
        print('after update', params)
        return params
        
    def trial(self, params = None):
        self._check_trials()
        if self.done_trialing: return
    
        start = time.time()
        if params is None:
            params = list(ParameterSampler(self.param_dict, 1, 
                     random_state = datetime.datetime.now().microsecond))[0]
            
        params = self.evo_update(params)
                      
        model_params = updateParams(params, self.train_data)
        model = get_xformer_model(model_params,
                    task_type = self.train_dataset.metadata.get_task_type(),
                    final_metric = self.train_dataset.metadata.get_final_metric(),
                    dataset = self.train_data)
        model, yp, y = fitAndPredict_xformer(model, params, 
                                      self.train_dataloader,
                                         self.val_dataloader)
        del model
        try:
            score = self.metric(y, yp); print(score); print()
            self.experiments.append((params, score, time.time() - start))
        except Exception as e:
            print(e)
        
    def fit(self, params = None):
        if params is None:
            params = get_xformer_base_params(self.train_data, self.train_dataloader)

        # params['steps'] = int(params['steps'] 
        #                       * len(self.full_train_data) / len(self.train_data) )
        model_params = updateParams(params, self.full_train_data)
        model = get_xformer_model(model_params, 
                    task_type = self.train_dataset.metadata.get_task_type(),
                    final_metric = self.train_dataset.metadata.get_final_metric(),
                    dataset = self.full_train_data)

        model = fit_xformer(model, params, self.full_train_dataloader,)
        return model
        
    def test(self, dataset, remaining_time_budget=None):
        test_data = XformerDataset(dataset, )
        test_dataloader = get_xformer_dataloader(test_data, )
        yps = []
        for i, model in enumerate(self.models):
            yp, _ = predict_xformer(model, test_dataloader)        
            yps.append(yp.numpy())
        wts = np.arange(1., 1. +len(yps))[::-1]; 
        wts = (wts / wts.sum()).astype(np.float32)
        yps = [yps[i] * wts[i] for i in range(len(yps))]
        yp = np.stack(yps).sum(axis = 0)
        return yp

def updateParams(params, train_dataset):
    model_params = params.copy()
    for k in list(set(list(params.keys())) & set(dir(train_dataset))):
        print(k)
        train_dataset[k] = model_params.pop(k)
    return model_params

    
class XformerDataset(Dataset):
    def __init__(self, dataset, idxs = None, 
                 mult_gn = 0.05, drop_rate = 0.05, re_crop = 0.2,
                    test = True):
        self.required_batch_size = dataset.required_batch_size
        self.collate_fn = dataset.collate_fn
        
        self.test = test
        self.dataset = dataset
        self.mult_gn = mult_gn
        self.drop_rate = drop_rate
        self.re_crop = re_crop
        if idxs is not None:
            self.idxs = idxs
        else:
            print(' ** USING ENTIRE DATASET **')
            self.idxs = list(range(len(dataset)))
        self.x_dims = torch.tensor(dataset[0][0].shape)
        self.x_order = torch.argsort(self.x_dims
                        ).flip(dims = [0]).tolist()        
    
    def __setitem__(self, k, v):
        self.__dict__[k] = v
        
    def __len__(self):
        return len(self.idxs);
    
    def __getitem__(self, idx):
        x, y = self.dataset[self.idxs[idx]]
        x = torch.tensor(x) if not torch.is_tensor(x) else x
        y = torch.tensor(y) if not torch.is_tensor(y) else y
        x = x.float(); y = y.float()
        
        x = x.permute(*self.x_order).reshape(
            self.x_dims[self.x_order[0]], -1
                                        ).unsqueeze(0)
        if not self.test:
            is_padding = (x.std(dim = -1) == 0)
            padding_frac = (is_padding * 1.0).mean().item()
            if padding_frac > 0.1:
                keep_rows = int((1 - self.re_crop * random.random()
                                      ) * (~is_padding).sum().item())
                # print('has padding', padding_frac)
                # print('keeping {} rows'.format(keep_rows))
                x = torch.cat((
                    x[:, 0, :].repeat(1, x.shape[1] - keep_rows, 1),
                    x[:, -keep_rows:, :]), dim = 1)
            x *= torch.exp( self.mult_gn 
                       * torch.randn((x.shape[0], x.shape[2]))
                      ).unsqueeze(1)
            keep = torch.rand((x.shape[1])) > self.drop_rate
            x = x[:, keep, :] 
            x = torch.cat((x[:, 0, :].repeat(1, (~keep).sum(), 1), x),
                           dim = 1)
        return x.float(), y.float().flatten()

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

def reverseLogit(x): return -torch.log(1/x - 1)


def get_xformer_param_dict(params, data):
    param_dict =  {}
    for k, v in params.items():
        param_dict[k] = [v]
    param_dict.update({
        'lr': (10 ** np.random.normal(-3.5, 0.2, 20)).round(6),
        'wd': (10 ** np.random.normal(-3., 1., 20)).round(6),
        'depth': [3, 4, 5, 6,],
        'init_values': (10 ** np.random.normal(-1., 0.5, 20)).round(6),
        'drop_path_rate': [0., 0., 0.1, 0.1, 0.2],
        # 'attn_drop_rate': [0., 0., 0., 0.,0.1, ],
        'dropout': [0.1, 0.1, 0.1, 0.15, ],
        'final_dropout': [0.1, 0.2, 0.3, 0.4, ],
        'rnn': [None, 'GRU', 'GRU', 'LSTM',  ],
        'rnn_ch_drop': [True, True, False],
        'pos_embed_wt': [0., 0.01, 0.03, 0.05,],
        'loss': [None, None, 'focal', 'focal'],
        'mult_gn': [0.01, 0.03, 0.05, 0.05, 0.05, 0.1],
        'drop_rate': [0.01, 0.03, 0.05, 0.05, 0.05, 0.1],
        're_crop': [ 0.03, 0.1, 0.2, ],
        'patch': (params['patch'] * np.exp(np.random.normal(0.0, 0.3, 50))
                     ).astype(int).clip(1, None),
        'steps': np.array([params['steps'], 
                          params['steps'], 
                          min(params['steps'],      20 * len(data)),
                          min(params['steps'] // 2, 20 * len(data)),
                          min(params['steps'] // 2, 10 * len(data)),
                          min(params['steps'] // 4, 5 * len(data)),
                 ]).astype(int).clip(100, None),
    })
    return param_dict
        
    
class FullNetwork(pl.LightningModule):
    def __init__(self, 
                lr = 5e-4, wd = 1e-3, 
                 loss = None,#'focal',
                 smooth = 1e-3, test_smooth = 1e-2,
                 # embed_dim = 384, 
                 patch = 3, 
                 depth = 4, n_heads = 16,
                 qkv_bias = True, init_values = 0.1,#1e-1, 
                 mlp_ratio = 4.,
                 rnn = 'GRU',
                 rnn_ch_drop = True,
                 n_targets = 4,
                 img_size = (360, 1),
                 steps = 1000, 
                 base_freq = None,
                 base_norm = None,
                 feature_mean = None,
                 feature_norm = None,
                 attn_drop_rate = 0.,
                 drop_path_rate = 0.,
                 dropout = 0.1,
                 final_dropout = 0.2,
                 pos_embed_wt = 0.03,
                 task_type = 'single-label',
                 final_metric = 'zero_one_error',
                ):        
        super().__init__()
        
        n_tokens = img_size[0]//patch
        embed_dim = 512 if n_tokens < 128 else 384
        print('embed_dim:', embed_dim);
        
        patch = patch if n_tokens < 384 else img_size[0] // 384
        print('patch:', patch)
        
        self.params = SimpleNamespace(**locals())
        
        base_freq = torch.tensor(np.array(base_freq))
        base_norm = torch.tensor(np.array(base_norm))
        feature_mean = torch.tensor(np.array(feature_mean))
        feature_norm = torch.tensor(np.array(feature_norm))
        if abs(feature_mean.mean()) < 3 * feature_norm.mean():
            print('feature abs(z) < 1; not adjusting mean')
            feature_mean = None
        if max(feature_norm) < 10 and min(feature_norm) > 0.1:
            print('feature norm in-range; not adjusting norm')
        
        self.base_freq = nn.Parameter(
            ( ( base_freq.float() + 1e-4) / base_freq.sum()
                          if base_freq is not None 
                        else torch.ones(n_targets) / n_targets ),
                requires_grad = False)
        self.base_norm = nn.Parameter(base_norm if base_norm is not None
                                      else torch.ones(len(n_targets)),
                                         requires_grad = False)
        self.feature_mean = nn.Parameter(feature_mean if feature_mean is not None
                                         else torch.zeros(img_size[1]),
                                         requires_grad = False)
        self.feature_norm = nn.Parameter(feature_norm if feature_norm is not None                                       else torch.ones(img_size[1]),
                                         requires_grad = False)
        
        
        
        if self.params.rnn:
            self.rnn_bn = nn.GroupNorm(8, embed_dim)
            self.rnn_dropout = (nn.Dropout2d if rnn_ch_drop
                                    else nn.Dropout)(dropout)
            self.rnn = getattr(nn, rnn)(embed_dim, embed_dim//2, 1, 
                                 batch_first = True, dropout = dropout,
                                     bidirectional = True,)
        self.vt = VisionTransformer(img_size = img_size, 
                                patch_size = (patch, img_size[1]), 
                                in_chans = 1,
                                num_classes = 0,
                                embed_dim = embed_dim, 
                                depth = depth, 
                                num_heads = n_heads,
                                mlp_ratio = mlp_ratio, 
                                qkv_bias = qkv_bias, 
                                init_values = init_values,
                                drop_rate = dropout,
                                attn_drop_rate = attn_drop_rate, 
                                drop_path_rate = drop_path_rate, 
                               )
        self.bn = nn.GroupNorm(8, embed_dim)
        self.dropout = nn.Dropout(final_dropout)
        self.head = nn.Linear(embed_dim, #+ img_size[1], 
                              n_targets)

        with torch.no_grad():            
            
            if base_freq is not None:
                if self.params.task_type == 'single-label':
                    self.head.bias[:] = torch.log(self.base_freq)
                elif self.params.task_type == 'multi-label':
                    self.head.bias[:] = reverseLogit(self.base_freq)
                # self.head.weight[:] /= 10
            self.vt.pos_embed[:, 1:] /= 2
            self.vt.pos_embed[:, 1:] += pos_embed_wt * torch.tensor(
                getPositionEncoding(seq_len = img_size[0] // patch,
                                    d = embed_dim, 
                                    n =  img_size[0] // patch) )

    def pad(self, x):
        pad =  self.params.img_size[0] - x.shape[2];
        assert pad >= 0
        if pad > 0:
            return torch.cat( 
                ( torch.zeros((x.shape[0], x.shape[1], pad, x.shape[-1]), 
                              device = x.device), 
                 x), dim = 2)
        else: 
            return x
            
    def forward(self, x, N = 1):
        x_raw = x
        x = (self.pad(x) - self.feature_mean)/self.feature_norm
        if self.params.rnn:
            x = self.vt.forward_features(x).permute(0, 2, 1)
            x = self.rnn_bn(x)
            x = self.rnn_dropout(x.unsqueeze(-1))[:, :, :, 0].permute(0, 2, 1)
            x = self.rnn(x)[0][:, -1, :]
        else:
            x = self.vt.forward(x)
        # x = torch.cat((x, x_raw.mean(dim = 2).mean(dim = 1)), dim = -1)
        ys = []
        for i in range(N):
            ys.append(self.head(self.dropout(self.bn(x))))
        yp = torch.cat(ys, dim = 0);

        with torch.autocast(enabled = False, device_type = 'cuda'):
            if self.params.task_type == 'single-label':
                yp = F.softmax(yp.float(), dim = 1)
            elif self.params.task_type == 'multi-label':
                yp = torch.sigmoid(yp.float())
            else: # continous/regression
                yp = yp * self.base_norm
                

            smooth = self.params.test_smooth if N == 1 else self.params.smooth
            if 'label' in self.params.task_type:
                yp = (yp * (1 - smooth) +
                        self.base_freq * smooth)
        return yp


    def training_step(self, batch, batch_idx):
        x, y = batch
        yp = self.forward(x, N = 10)
        y = torch.cat([y] * (len(yp) // len(y)))
        loss = self.loss(yp, y)
        return loss
        
    def loss(self, yp, y):
        # print(yp); print(y)
        with torch.autocast(enabled = False, device_type = 'cuda'):
            if 'continuous' in self.params.task_type:
                if 'relative' in self.params.final_metric:
                    # loss = l2_relative_error(y, yp)
                    loss = ( ((yp - y)**2).sum(dim = 1) 
                                / (y ** 2).sum(dim = 1)
                           ).mean()
                elif 'l2' in self.params.final_metric:
                    loss = nn.MSELoss()(yp, y)
                elif 'l1' in self.params.final_metric:
                    loss = nn.L1Loss()(yp, y)
                else:
                    loss = nn.MSELoss()(yp, y)
            elif self.loss == 'focal':
                loss = self.focal_loss(yp, y)
            else:
                loss = nn.BCELoss()(yp, y)
        return loss
    
    def focal_loss(self, yp, y, alpha = 0.25, gamma = 1.5):
        p_t = (y * yp) + ((1 - y) * (1 - yp))
        alpha_factor = y * alpha + (1 - y) * (1. - alpha)
        modulating_factor = (1. - p_t) ** gamma
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')

        return (alpha_factor * modulating_factor * ce).mean()    

    def on_validation_epoch_start(self):
        self.y_true = []
        self.y_pred = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        yp = self.forward(x)
        self.y_true.append(y)
        self.y_pred.append(yp)
        loss = self.loss(yp, y)
        self.log('test_loss', loss.item(), prog_bar = True)
        return loss
        
    def on_validation_epoch_end(self):
        y = torch.cat(self.y_true, dim = 0)
        yp = torch.cat(self.y_pred, dim = 0)
        # print(y); print(yp)
        try:
            if 'label' in self.params.task_type:
                metric = 1 - inv_auroc_score(y.cpu(), yp.cpu())
        except Exception as e: metric = np.nan;# print(e); pass;
        self.log('test_auc', metric, 
                     prog_bar = True)
        
        metric = eval(self.params.final_metric)(y.cpu().numpy(), yp.cpu().numpy())
        self.log('test__' + self.params.final_metric, metric, 
                     prog_bar = True)
        
        # except Exception as e:
        #     print(e)
            # print(self.current_epoch, auc.round(2).item(), mae1.round(2).item(),
#                          mae8.round(2).item(), mae5.round(2).item())
#         except Exception as e:
#             print(e)
                                
    def configure_optimizers(self):
        nd_params = [v for k, v in self.named_parameters() 
                         if '.bias' in k or '.bn.' in k]
        main_params = [v for k, v in self.named_parameters() 
                       if not ('.bias' in k or '.bn.' in k)]
                         
        optimizer = AdamW([{'params': nd_params, 'weight_decay': 0},
                           {'params': main_params, }],
                              lr = self.params.lr,
                                  weight_decay = self.params.wd,)
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, int(self.params.steps), 
                                 eta_min = 0)
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda x: min(1, x / (0.05 * self.params.steps)))
        return [optimizer], [{'scheduler': scheduler1,
                              'interval': 'step',},
                             {'scheduler': scheduler2,
                              'interval': 'step'}]   
    



    
def get_xformer_base_params(train_dataset, train_dataloader):
    x = train_dataset[0][0]
    patch = max(1, int( round ( (x.shape[1] / 64) ** 0.5
                                  * (20 / x.shape[-1]) ** 0.1                       
                              )))
    if x.shape[1]/patch > 384: 
        print('384-token limit');
        patch = x.shape[1]//384
    
    # target 12 full runs--knowing some run double this, some half/qtr;
    # expect test set is 5x the validation set
   
    # print(x.shape, patch)
    # l = math.ceil(x.shape[1] / patch) * patch
    img_size = (x.shape[1], x.shape[2])#x.shape[1:]
    
    img_size, patch, img_size[0]//patch# len(base_freq)
    params = {'img_size': img_size,#x.shape[1:],
              'patch': patch,
             }
    print(params)
    return params


def get_xformer_stats(train_dataset, idx_train = None):
    
    features = []; targets = []
    for i in range(min(10000, len(train_dataset))):
        f, t = train_dataset[i]
        features.append(f * 1.0)
        targets.append(t * 1.0)
    features = (np.stack(features)
                .reshape(-1, train_dataset[0][0].shape[-1]))
    targets = np.stack(targets)
    base_freq = np.mean(targets, axis = 0)
    base_norm = np.std(targets, axis = 0)        
    ch_mean = np.mean(features, axis = 0)
    ch_norm = np.std(features, axis = 0)
    # print(base_freq, base_norm)
    # print(ch_mean, ch_norm)
    return base_freq, base_norm, ch_mean, ch_norm

    
# def get_xformer_datasets(train_dataset, val_dataset):
    
#     return train_data, val_data

# def get_xformer_test_dataset(test_dataset):    
    # return test_data

def get_xformer_dataloader(dataset, test = True):
    if not test:
        dataloader = DataLoader(dataset, 
                                batch_size = dataset.required_batch_size or 12, 
                                  shuffle = True, 
                                    drop_last = True,
                                    collate_fn = dataset.collate_fn,
                                    # num_workers = os.cpu_count()
                               )
    else:
        dataloader = DataLoader(dataset, 
                                batch_size = dataset.required_batch_size or 32, 
                                    collate_fn = dataset.collate_fn,
                                    # num_workers = os.cpu_count()
                               )

    return dataloader

def is_ranking(final_metric):
    m = final_metric;
    return 'zero_one' in m or 'f1' in m or 'auc' in m

def get_xformer_model(params, task_type, final_metric, dataset):
    base_freq, base_norm, feature_mean, feature_norm = (
        get_xformer_stats(dataset))
    
    if 'loss' not in params and is_ranking(final_metric):
        params['loss'] = 'focal'
    elif not is_ranking(final_metric):
        params['loss'] = None

    model = FullNetwork(**params,
                # img_size = img_size, patch = patch,
                    base_freq = base_freq,
                        base_norm = base_norm,
                        feature_mean = feature_mean,
                        feature_norm = feature_norm,                        
                        n_targets = len(base_freq),
                    # cosine_steps = steps, warmup_steps = 0.05 * steps,
                    # lr = 3e-4,
                    # embed_dim = 384, n_heads = 24,
                    task_type = task_type,#.metadata.get_task_type(),
                    final_metric = final_metric,     )
    return model

def fit_xformer(model, params, train_dataloader, val_dataloader = None, ):
    print(); print(params)
    start = time.time()
    trainer = pl.Trainer(logger = False,
                    enable_checkpointing = False,
                         enable_progress_bar = LOCAL,
                    max_steps = params['steps'],
                     accelerator = 'auto',
                     precision="16-mixed",
                         limit_val_batches = 1.0 if VALIDATION else 0.0,
                    # limit_val_batches = 200,
                    # val_check_interval = 500,
                     # check_val_every_n_epoch = 5
                    )
    trainer.fit(model, train_dataloader, val_dataloader) 
    print('Training Time: {:.1f}s'.format(time.time() - start))
    del trainer
    return model
    
def fitAndPredict_xformer(model, params, train_dataloader, val_dataloader, ):
    model = fit_xformer(model, params, train_dataloader, val_dataloader, );
    yp, y = predict_xformer(model, val_dataloader)
    return model, yp, y
    
def predict_xformer(model, test_dataloader, limit = None):
    start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device);
    model.eval();

    ys = []; yps = []
    for batch in test_dataloader:
        x, y = batch
        with torch.no_grad():
            yp = model(x.to(device))
            yps.append(yp)
            ys.append(y)
        if limit is not None and len(ys) >= limit: break;
            # break;

    y = torch.cat(ys, dim = 0)
    yp = torch.cat(yps, dim = 0)
    print('Inference Time: {:.1f}s'.format(time.time() - start));
    return yp.cpu(), y.cpu()
    