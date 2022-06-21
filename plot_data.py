import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np

import util

np.set_printoptions(linewidth=150)

with open('data/train.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))

# idx_mmm = util.header_to_row_idx['Irradiance_m']
# idx_irr = util.header_to_row_idx['Irradiance']
# data = np.array([
#     [float(line.split(',')[idx_mmm]), float(line.split(',')[idx_irr])] \
#         for line in cnt if line.split(',')[idx_irr] != ''
# ])

# idx_tmp = util.header_to_row_idx['Temp']
# idx_mmm = util.header_to_row_idx['Temp_m']
# data = np.array([
#     [float(line.split(',')[idx_tmp]), float(line.split(',')[idx_mmm])] \
#         for line in cnt if line.split(',')[idx_mmm] != '' and line.split(',')[idx_mmm] != '']
# )

idx_mmm = util.header_to_row_idx['Capacity']
idx_tmp = util.header_to_row_idx['Temp_m']
data = np.array([
    [float(line.split(',')[idx_mmm]), float(line.split(',')[idx_tmp])] \
        for line in cnt if line.split(',')[idx_tmp] != ''
])

print('Shape:', data.shape)

plt.plot(data[:, 0], data[:, 1], '.')
plt.show()
