import matplotlib.pyplot as plt
import numpy as np

import util

np.set_printoptions(linewidth=150)

with open('data/train.csv', 'r') as fin:
    cnt = fin.read().splitlines()[1:]
    print('Data count:', len(cnt))

idx_dat = util.header_to_row_idx['Date']
idx_tpm = util.header_to_row_idx['Temp_m']
idx_gen = util.header_to_row_idx['Generation']
idx_irr = util.header_to_row_idx['Irradiance']
idx_cap = util.header_to_row_idx['Capacity']
idx_lat = util.header_to_row_idx['Lat']
idx_lon = util.header_to_row_idx['Lon']
idx_irm = util.header_to_row_idx['Irradiance_m']
idx_tmp = util.header_to_row_idx['Temp']
idx_mod = util.header_to_row_idx['Module']

# --- 一個經緯度下的模組數量
# lat_lon_to_mod_dic = {}
# for line in cnt:
#     arr = line.split(',')
#     key = '{}-{}'.format(arr[idx_lat], arr[idx_lon])
#     if key not in lat_lon_to_mod_dic:
#         lat_lon_to_mod_dic[key] = set()
#     lat_lon_to_mod_dic[key].add(arr[idx_mod])

# for key, mod in lat_lon_to_mod_dic.items():
#     print(key, mod)

lat_lon_mod_to_dat_dic = {}
for line in cnt:
    arr = line.split(',')
    key = '{}-{}-{}-{}'.format(arr[idx_lat], arr[idx_lon], arr[idx_mod], arr[idx_cap])
    if key not in lat_lon_mod_to_dat_dic:
        lat_lon_mod_to_dat_dic[key] = {
            'dates': [],
            'temp_ms': [],
            'temps': [],
            'irr_ms': [],
            'irrs': [],
            'capacity': [],
            'generation': [],
        }
    lat_lon_mod_to_dat_dic[key]['dates'].append(arr[idx_dat])
    lat_lon_mod_to_dat_dic[key]['temp_ms'].append(arr[idx_tpm])
    lat_lon_mod_to_dat_dic[key]['temps'].append(arr[idx_tmp])
    lat_lon_mod_to_dat_dic[key]['irr_ms'].append(arr[idx_irm])
    lat_lon_mod_to_dat_dic[key]['irrs'].append(arr[idx_irr])
    lat_lon_mod_to_dat_dic[key]['capacity'].append(arr[idx_cap])
    lat_lon_mod_to_dat_dic[key]['generation'].append(arr[idx_gen])

keys = sorted(lat_lon_mod_to_dat_dic.keys())

for idx, key in enumerate(keys):
    item = lat_lon_mod_to_dat_dic[key]
    print(
        key,
        # len(item['dates']),
        # ''.join([
        #     '1' if val != '' else '0' for val in item['temp_ms']
        # ]),
        # item['dates']
    )

# plt.figure()
for idx, key in enumerate(keys):
    item = lat_lon_mod_to_dat_dic[key]
    # data = np.array([
    #     [float(t), float(tm)] \
    #         for t, tm in zip(item['irr_ms'], item['temp_ms']) \
    #             if t != '' and tm != ''
    # ])
    data = np.array([float(val) if val != '' else np.nan for val in item['irr_ms']])
    if data.size == 0:
        continue
    plt.figure()
    # plt.subplot(2, 7, idx+1)
    # plt.plot(data[:, 0], data[:, 1], '.')
    plt.plot(data, '.-')
    plt.title(key)
    plt.savefig('figures/{}_irr_ms.png'.format(idx+1))

# plt.show()
