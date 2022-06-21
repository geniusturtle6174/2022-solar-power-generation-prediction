import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np

import util

np.set_printoptions(linewidth=150)

with open('data/test.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))

idx_lat = util.header_to_row_idx['Lat']
idx_lon = util.header_to_row_idx['Lon']
lat_lon = np.array([[float(line.split(',')[idx_lat]), float(line.split(',')[idx_lon])] for line in cnt])
print('Shape:', lat_lon.shape)
print('Lats:', sorted(list(set([float(line.split(',')[idx_lat]) for line in cnt]))))
print('Lons:', sorted(list(set([float(line.split(',')[idx_lon]) for line in cnt]))))
print('All:', sorted(list(set(['{}-{}'.format(line.split(',')[idx_lat], line.split(',')[idx_lon]) for line in cnt]))))

plt.plot(lat_lon[:, 0], lat_lon[:, 1], '.')
plt.show()
