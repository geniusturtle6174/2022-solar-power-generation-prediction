import os
import argparse

import matplotlib.pyplot as plt
import numpy as np

import util

np.set_printoptions(linewidth=150)

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', help='Model directory name')
parser.add_argument('--n_fold_train', type=int, default=7)
args = parser.parse_args()

pred_all = []
ans_all = []
for fold in range(args.n_fold_train):
    pred = np.load(os.path.join(args.model_dir, 'va_pred_{}.npy'.format(fold)))
    ans = np.load(os.path.join(args.model_dir, 'va_ans_{}.npy'.format(fold)))[:, 0]
    pred_all.append(pred)
    ans_all.append(ans)
    print(fold, pred.shape, ans.shape, np.mean((pred-ans)**2)**0.5)
    plt.subplot(2, 4, fold+1)
    plt.plot(ans, pred, '.')
    plt.plot([0, 3000], [0, 3000], 'r-')

pred_all = np.hstack(pred_all)
ans_all = np.hstack(ans_all)
plt.subplot(2, 4, 8)
plt.plot(ans_all, pred_all, '.')
plt.plot([0, 3000], [0, 3000], 'r-')

print('RMSE:', np.mean((pred_all-ans_all)**2)**0.5)

plt.show()
