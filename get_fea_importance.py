import os
import argparse

import numpy as np

import util

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--n_fold_train', type=int, default=5)
args = parser.parse_args()

imp_all = []
for fold in range(args.n_fold_train):
    print('Getting fold', fold)
    model = util.load_pkl(os.path.join(args.model_dir, 'model_{}_time_range_fold.pkl'.format(fold)))
    imp_all.append(model.feature_importances_)

imp_all = np.array(imp_all)
print('Shape:', imp_all.shape, len(util.FEA_NAMES))

for idx, name in enumerate(util.FEA_NAMES):
    print('{}\t{}\t{}\t{}\t{}\t{}'.format(
        name, imp_all[0, idx], imp_all[1, idx], imp_all[2, idx], imp_all[3, idx], imp_all[4, idx]
    ))
