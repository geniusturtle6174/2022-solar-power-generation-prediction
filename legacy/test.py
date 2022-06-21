import os
import argparse

import torch
import numpy as np
from torch.autograd import Variable

import util, model

np.set_printoptions(linewidth=150)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('modeldir', help='Model directory name')
parser.add_argument('--output_file_name', default='submission.csv', help='Results file name')
parser.add_argument('--model_postfix', default='_bestVa')
parser.add_argument('--n_fold_train', type=int, default=5)
parser.add_argument('--test_batch_size', type=int, default=256)
args = parser.parse_args()

nn_param = util.load_cfg(os.path.join(args.modeldir, 'config.yml'))

fea_all = []
with open('data/test.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))
for line in cnt:
    fea, _ = util.fea_ext(line.split(','))
    fea_all.append(fea)
fea_all = np.array(fea_all)
print('Overall shapes:', fea_all.shape)

print('Loading network...')
networks = {}
for fold in range(args.n_fold_train):
    save_dic = torch.load(os.path.join(args.modeldir, 'model_{}{}'.format(fold, args.model_postfix)))
    networks[fold] = model.MLP(nn_param['dim_fea'])
    networks[fold].load_state_dict(save_dic)
    networks[fold].eval()
    networks[fold].to(device)

data_loader = torch.utils.data.DataLoader(
    util.Data2Torch({
        'fea': fea_all,
    }),
    batch_size=nn_param['batch'],
)

pred_all = []
for idx, data in enumerate(data_loader):
    with torch.no_grad():
        pred = [networks[fold](Variable(data['fea'].to(device))).detach().cpu().numpy() for fold in range(args.n_fold_train)]
        pred_all.append(np.array(pred))
pred_all = np.concatenate(pred_all, axis=1)
print('Finish prediction, shape:', pred_all.shape)

with open(args.output_file_name, 'w') as fout:
    fout.write('ID,Generation\n')
    for i in range(pred_all.shape[1]):
        fout.write('{},{:.6f}\n'.format(
            i+1, np.median(pred_all[:, i])
        ))
