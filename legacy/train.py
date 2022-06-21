import argparse
import json
import os
import time
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

import model
import util

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('save_model_dir_name')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--va_not_imp_limit', '-va', type=int, default=500)
parser.add_argument('--n_fold', type=int, default=5)
args = parser.parse_args()

fea_all = []
ans_all = []
with open('data/train.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))
for idx, line in enumerate(cnt):
    fea, ans = util.fea_ext(line.split(','))
    fea_all.append(fea)
    ans_all.append(ans)
fea_all = np.array(fea_all)
ans_all = np.array(ans_all)[:, np.newaxis]
print('Overall shapes:', fea_all.shape, ans_all.shape)

sort_idx = np.argsort(ans_all[:, 0])
fea_all = fea_all[sort_idx]
ans_all = ans_all[sort_idx]
print('Sorted shapes:', sort_idx.shape, fea_all.shape, ans_all.shape)

# --- Setup network
nn_param = {
    'dim_fea': fea_all.shape[1],
    'batch': 256,
    'optm_params': {
        'lr': 0.001
    },
}

print('Setting network')
networks = {}
optimizers = {}
schedulers = {}
loss_func = nn.MSELoss()
for f in range(args.n_fold):
    networks[f] = model.MLP(nn_param['dim_fea'])
    networks[f].to(device)
    optimizers[f] = optim.Adam(list(networks[f].parameters()), lr=nn_param['optm_params']['lr'])
    schedulers[f] = torch.optim.lr_scheduler.StepLR(optimizers[f], step_size=1, gamma=0.9999)

# --- Write config
if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))
util.write_cfg(os.path.join(args.save_model_dir_name, 'config.yml'), nn_param)

data_num = fea_all.shape[0]
for fold in range(args.n_fold):

    best_va_loss = 9999999

    valid_idx = np.where(np.arange(data_num)%args.n_fold==fold)[0]
    train_idx = np.where(np.arange(data_num)%args.n_fold!=fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    data_loader_train = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_all[train_idx],
            'ans': ans_all[train_idx],
        }),
        shuffle=True,
        batch_size=nn_param['batch'],
    )

    data_loader_valid = torch.utils.data.DataLoader(
        util.Data2Torch({
            'fea': fea_all[valid_idx],
            'ans': ans_all[valid_idx],
        }),
        batch_size=nn_param['batch'],
    )

    va_not_imporved_continue_count = 0
    totalTime = 0
    fout = open(os.path.join(args.save_model_dir_name, 'train_report_{}.txt'.format(fold)), 'w')
    for epoch in range(args.epoch):
        util.print_and_write_file(fout, 'epoch {}/{}...'.format(epoch + 1, args.epoch))
        tic = time.time()
        # --- Batch training
        networks[fold].train()
        training_loss = 0
        n_batch = 0
        optimizers[fold].zero_grad()
        for idx, data in enumerate(data_loader_train):
            pred = networks[fold](Variable(data['fea'].to(device)))
            ans = Variable(data['ans'].to(device))
            loss = torch.sqrt(loss_func(pred, ans))
            optimizers[fold].zero_grad()
            loss.backward()
            optimizers[fold].step()
            training_loss += loss.data
            n_batch += 1
        # --- Training loss
        training_loss_avg = training_loss / n_batch
        util.print_and_write_file(
            fout, '\tTraining loss (avg over batch): {}, {}, {}'.format(
                training_loss_avg, training_loss, n_batch
            )
        )
        # --- Batch validation
        networks[fold].eval()
        va_loss = 0
        n_batch = 0
        for idx, data in enumerate(data_loader_valid):
            ans = Variable(data['ans'].to(device)).float()
            with torch.no_grad():
                pred = networks[fold](Variable(data['fea'].to(device)))
                loss = torch.sqrt(loss_func(pred, ans))
            va_loss += loss.data
            n_batch += 1
        # --- Validation loss
        va_loss_avg = va_loss / n_batch
        util.print_and_write_file(
            fout, '\tValidation loss (avg over batch): {}, {}, {}'.format(
                va_loss_avg, va_loss, n_batch
            )
        )
        # --- Save if needed
        if va_loss_avg < best_va_loss:
            best_va_loss = va_loss_avg
            va_not_imporved_continue_count = 0
            util.print_and_write_file(fout, '\tWill save bestVa model')
            torch.save(
                networks[fold].state_dict(),
                os.path.join(args.save_model_dir_name, 'model_{}_bestVa'.format(fold))
            )
        else:
            va_not_imporved_continue_count += 1
            util.print_and_write_file(fout, '\tva_not_imporved_continue_count: {}'.format(va_not_imporved_continue_count))
            if va_not_imporved_continue_count >= args.va_not_imp_limit:
                break
        util.print_and_write_file(fout, '\tLearning rate used for this epoch: {}'.format(schedulers[fold].get_last_lr()[0]))
        if schedulers[fold].get_last_lr()[0] >= 1e-4:
            schedulers[fold].step()
        # --- Time
        toc = time.time()
        totalTime += toc - tic
        util.print_and_write_file(fout, '\tTime: {:.3f} sec, estimated remaining: {:.3} hr'.format(
            toc - tic,
            1.0 * totalTime / (epoch + 1) * (args.epoch - (epoch + 1)) / 3600
        ))
        fout.flush()
    fout.close()
    # Save model
    torch.save(
        networks[fold].state_dict(),
        os.path.join(args.save_model_dir_name, 'model_{}_final'.format(fold))
    )
    print('Model saved in {}'.format(args.save_model_dir_name))
