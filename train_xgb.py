import argparse
import os
import warnings
from copy import deepcopy

import numpy as np
from sklearn.metrics import mean_squared_error

import util
from xgboost import XGBRegressor

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('save_model_dir_name')
parser.add_argument('--n_fold', type=int, default=7)
args = parser.parse_args()

print('Loading data...')
with open('data/train.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('\tData count:', len(cnt))

print('Loading extra data...')
ext_data_dict_month, ext_header_to_row_idx_month = util.load_external_monthly_report()
ext_data_dict_day, ext_header_to_row_idx_day = util.load_external_daily_report()

lat_lon_mod_to_fea = {}
idx_cap = util.header_to_row_idx['Capacity']
idx_lat = util.header_to_row_idx['Lat']
idx_lon = util.header_to_row_idx['Lon']
idx_mod = util.header_to_row_idx['Module']
for idx, line in enumerate(cnt):
    arr = line.split(',')
    key = '{}-{}-{}-{}'.format(arr[idx_lat], arr[idx_lon], arr[idx_mod], arr[idx_cap])
    fea, ans = util.fea_ext(
        arr, ext_data_dict_month, ext_header_to_row_idx_month, ext_data_dict_day, ext_header_to_row_idx_day
    )
    if key not in lat_lon_mod_to_fea:
        lat_lon_mod_to_fea[key] = {
            'fea': [],
            'ans': [],
        }
    lat_lon_mod_to_fea[key]['fea'].append(fea)
    lat_lon_mod_to_fea[key]['ans'].append(ans)

fea_all = []
ans_all = []
fold_all = []
for key in lat_lon_mod_to_fea:
    lat_lon_mod_to_fea[key]['fea'] = np.array(lat_lon_mod_to_fea[key]['fea'])
    lat_lon_mod_to_fea[key]['ans'] = np.array(lat_lon_mod_to_fea[key]['ans'])[:, np.newaxis]

    data_num = lat_lon_mod_to_fea[key]['ans'].shape[0]
    fold = np.zeros(data_num)
    for f in range(1, args.n_fold):
        fold[int(data_num/args.n_fold)*f:] += 1

    fea_all.append(lat_lon_mod_to_fea[key]['fea'])
    ans_all.append(lat_lon_mod_to_fea[key]['ans'])
    fold_all.append(fold)

    print('{}, shapes: {}, {}, {}'.format(
        key, fea_all[-1].shape, ans_all[-1].shape, fold_all[-1].shape,
    ))

fea_all = np.vstack(fea_all)
ans_all = np.vstack(ans_all)
fold_all = np.hstack(fold_all)
print('Overall shapes:', fea_all.shape, ans_all.shape, fold_all.shape)

param = {
    'n_estimators': 1400,
    'max_depth': 9,
    'learning_rate': 0.01,
    'eval_metric': mean_squared_error,
    'min_child_weight': 4,
    'gamma': 0.5,
    'reg_lambda': 2,
    'reg_alpha': 0.001,
    'max_delta_step': 2000,
    'n_jobs': 7,
    'verbosity': 0,
}

if not os.path.exists(args.save_model_dir_name):
    os.makedirs(args.save_model_dir_name, 0o755)
    print('Model will be saved in {}'.format(args.save_model_dir_name))
else:
    warnings.warn('Dir {} already exist, result files will be overwritten.'.format(args.save_model_dir_name))

models = {}
data_num = fea_all.shape[0]
for fold in range(args.n_fold):
    valid_idx = np.where(fold_all == fold)[0]
    train_idx = np.where(fold_all != fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    # 1st pass
    models[fold] = XGBRegressor(**param)
    models[fold].fit(
        fea_all[train_idx],
        ans_all[train_idx],
        eval_set=[(fea_all[valid_idx], ans_all[valid_idx])],
    )

    # 2nd pass
    loss = models[fold].evals_result()['validation_0']['rmse']
    local_param = deepcopy(param)
    local_param['n_estimators'] = np.argsort(loss)[0] + 1
    models[fold] = XGBRegressor(**local_param)
    models[fold].fit(fea_all[train_idx], ans_all[train_idx])

    util.save_pkl(os.path.join(args.save_model_dir_name, 'model_{}_time_range_fold.pkl'.format(fold)), models[fold])
    np.save(os.path.join(args.save_model_dir_name, 'va_pred_{}_time_range_fold.npy'.format(fold)), models[fold].predict(fea_all[valid_idx]))
    np.save(os.path.join(args.save_model_dir_name, 'va_ans_{}_time_range_fold.npy'.format(fold)), ans_all[valid_idx])

sort_idx = np.argsort(ans_all[:, 0])
fea_all = fea_all[sort_idx]
ans_all = ans_all[sort_idx]
print('Sorted shapes:', sort_idx.shape, fea_all.shape, ans_all.shape)

models = {}
data_num = fea_all.shape[0]
for fold in range(args.n_fold):
    valid_idx = np.where(np.arange(data_num)%args.n_fold == fold)[0]
    train_idx = np.where(np.arange(data_num)%args.n_fold != fold)[0]

    print('Fold {}, train num {}, test num {}'.format(
        fold, len(train_idx), len(valid_idx),
    ))

    # 1st pass
    models[fold] = XGBRegressor(**param)
    models[fold].fit(
        fea_all[train_idx],
        ans_all[train_idx],
        eval_set=[(fea_all[valid_idx], ans_all[valid_idx])],
    )

    # 2nd pass
    loss = models[fold].evals_result()['validation_0']['rmse']
    local_param = deepcopy(param)
    local_param['n_estimators'] = np.argsort(loss)[0] + 1
    models[fold] = XGBRegressor(**local_param)
    models[fold].fit(fea_all[train_idx], ans_all[train_idx])

    util.save_pkl(os.path.join(args.save_model_dir_name, 'model_{}_sort_fold.pkl'.format(fold)), models[fold])
    np.save(os.path.join(args.save_model_dir_name, 'va_pred_{}_sort_fold.npy'.format(fold)), models[fold].predict(fea_all[valid_idx]))
    np.save(os.path.join(args.save_model_dir_name, 'va_ans_{}_sort_fold.npy'.format(fold)), ans_all[valid_idx])
