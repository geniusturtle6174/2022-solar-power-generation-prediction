import os
import argparse

import numpy as np
import weightedstats as ws

import util

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--output_file_name', default='submission.csv', help='Results file name')
parser.add_argument('--n_fold_train', type=int, default=7)
args = parser.parse_args()

with open('data/test.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))

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
    fea, _ = util.fea_ext(
        arr, ext_data_dict_month, ext_header_to_row_idx_month, ext_data_dict_day, ext_header_to_row_idx_day
    )
    if key not in lat_lon_mod_to_fea:
        lat_lon_mod_to_fea[key] = {
            'fea': [],
            'id': [],
        }
    lat_lon_mod_to_fea[key]['fea'].append(fea)
    lat_lon_mod_to_fea[key]['id'].append(int(arr[0]))

fea_all = []
id_all = []
for key in lat_lon_mod_to_fea:
    lat_lon_mod_to_fea[key]['fea'] = np.array(lat_lon_mod_to_fea[key]['fea'])
    fea_all.append(lat_lon_mod_to_fea[key]['fea'])
    id_all.append(lat_lon_mod_to_fea[key]['id'])
    print('{}, raw shape: {}'.format(
        key, fea_all[-1].shape
    ))

fea_all = np.vstack(fea_all)
id_all = np.hstack(id_all)
print('Overall shape:', fea_all.shape, id_all.shape)

pred_all = []
# last_loss_all = []
for fold in range(args.n_fold_train):
    print('Predicting fold', fold)
    model_t = util.load_pkl(os.path.join(args.model_dir, 'model_{}_time_range_fold.pkl'.format(fold)))
    model_s = util.load_pkl(os.path.join(args.model_dir, 'model_{}_sort_fold.pkl'.format(fold)))
    # last_loss_all.append(model.evals_result()['validation_0']['rmse'][-1])
    pred_t = model_t.predict(fea_all)
    pred_s = model_s.predict(fea_all)
    print('Shape:', pred_t.shape, pred_s.shape)
    pred_all.append(pred_t)
    pred_all.append(pred_s)
# last_loss_all = np.array(last_loss_all)
# weight_all = 1 / last_loss_all
# weight_all = weight_all / sum(weight_all)

pred_all = np.vstack(pred_all)
print('Finish prediction, shape:', pred_all.shape)

sort_idx = np.argsort(id_all)
pred_all = pred_all[:, sort_idx]

with open(args.output_file_name, 'w') as fout:
    fout.write('ID,Generation\n')
    for i in range(pred_all.shape[1]):
        fout.write('{},{:.6f}\n'.format(
            i+1, np.median(pred_all[:, i])
        ))
