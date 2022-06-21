import math
import os
import pickle
from copy import deepcopy

import numpy as np

# 觀測資料查詢: https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp
header_to_row_idx = {h: i for i, h in enumerate('ID,Date,Temp_m,Generation,Irradiance,Capacity,Lat,Lon,Angle,Irradiance_m,Temp,Module'.split(','))}
lat_lon_to_site = { # nearest
    '24.04-120.52': ['C0G780'], #  彰化 秀水   2021/5/19  2022/2/16 秀水測站 C0G780
    '24.06-120.47': ['C0G770'], #  彰化 鹿港   2021/6/12  2022/2/16 福興測站 C0G770
    '24.07-120.47': ['C0G640'], #  彰化 鹿港   2021/5/27  2022/2/16 鹿港測站 C0G640
    '24.07-120.48': ['C0G780'], #  彰化 鹿港   2021/5/21  2022/2/16 秀水測站 C0G780
    '24.08-120.5': ['C0G780'], #   彰化 和美   2020/9/27  2022/2/16 秀水測站 C0G780
    '24.08-120.52': ['C0G780'], #  彰化 彰化市 2021/5/22  2022/2/16 秀水測站 C0G780
    '24.09-120.52': ['C0G780'], #  彰化 和美   2021/5/21  2022/2/16 秀水測站 C0G780
    '24.107-120.44': ['C0G640'], # 彰化 鹿港   2020/9/23  2022/2/16 鹿港測站 C0G640
    '24.98-121.03': ['467050'], #  桃園 新屋   2020/12/17 2022/2/17 新屋測站 467050
    '25.03-121.08': ['467050'], #  桃園 觀音   2020/12/17 2022/2/17 新屋測站 467050
    '25.11-121.26': ['C0C620'], #  桃園 蘆竹   2020/6/9   2022/2/17 蘆竹測站 C0C620
}
# lat_lon_to_site = { # nearest 2
#     '24.04-120.52': ['C0G780', 'C0G910'], #  彰化 秀水   2021/5/19  2022/2/16 秀水測站 C0G780 花壇測站 C0G910
#     '24.06-120.47': ['C0G770', 'C0G640'], #  彰化 鹿港   2021/6/12  2022/2/16 福興測站 C0G770 鹿港測站 C0G640
#     '24.07-120.47': ['C0G640', 'C0G770'], #  彰化 鹿港   2021/5/27  2022/2/16 鹿港測站 C0G640 福興測站 C0G770
#     '24.07-120.48': ['C0G780', 'C0G640'], #  彰化 鹿港   2021/5/21  2022/2/16 秀水測站 C0G780 鹿港測站 C0G640
#     '24.08-120.5': ['C0G780', 'C0G910'], #   彰化 和美   2020/9/27  2022/2/16 秀水測站 C0G780 花壇測站 C0G910
#     '24.08-120.52': ['C0G780', 'C0G910'], #  彰化 彰化市 2021/5/22  2022/2/16 秀水測站 C0G780 花壇測站 C0G910
#     '24.09-120.52': ['C0G780', 'C0G910'], #  彰化 和美   2021/5/21  2022/2/16 秀水測站 C0G780 花壇測站 C0G910
#     '24.107-120.44': ['C0G640', 'C0G900'], # 彰化 鹿港   2020/9/23  2022/2/16 鹿港測站 C0G640 線西測站 C0G900
#     '24.98-121.03': ['467050', 'C0D650'], #  桃園 新屋   2020/12/17 2022/2/17 新屋測站 467050 湖口測站 C0D650
#     '25.03-121.08': ['467050', 'C0C590'], #  桃園 觀音   2020/12/17 2022/2/17 新屋測站 467050 觀音測站 C0C590
#     '25.11-121.26': ['C0C620', 'C0C700'], #  桃園 蘆竹   2020/6/9   2022/2/17 蘆竹測站 C0C620 中壢測站 C0C700
# }
# lat_lon_to_site = { # m = (a+b)/2, minimize pa + pb + pm
#     '24.04-120.52': ['C0G780', 'C0G910'], #  彰化 秀水   2021/5/19  2022/2/16 秀水測站 C0G780 花壇測站 C0G910
#     '24.06-120.47': ['C0G640', 'C0G780'], #  彰化 鹿港   2021/6/12  2022/2/16 鹿港測站 C0G640 秀水測站 C0G780
#     '24.07-120.47': ['C0G640', 'C0G780'], #  彰化 鹿港   2021/5/27  2022/2/16 鹿港測站 C0G640 秀水測站 C0G780
#     '24.07-120.48': ['C0G640', 'C0G780'], #  彰化 鹿港   2021/5/21  2022/2/16 鹿港測站 C0G640 秀水測站 C0G780
#     '24.08-120.5': ['C0G890', 'C0G780'], #   彰化 和美   2020/9/27  2022/2/16 伸港測站 C0G890 秀水測站 C0G780
#     '24.08-120.52': ['C0G890', 'C0G910'], #  彰化 彰化市 2021/5/22  2022/2/16 伸港測站 C0G890 花壇測站 C0G910
#     '24.09-120.52': ['C0G890', 'C0G910'], #  彰化 和美   2021/5/21  2022/2/16 伸港測站 C0G890 花壇測站 C0G910
#     '24.107-120.44': ['C0G640', 'C0G640'], # 彰化 鹿港   2020/9/23  2022/2/16 線西測站 C0G900 鹿港測站 C0G640
#     '24.98-121.03': ['467050', 'C0D650'], #  桃園 新屋   2020/12/17 2022/2/17 新屋測站 467050 湖口測站 C0D650
#     '25.03-121.08': ['467050', 'C0C590'], #  桃園 觀音   2020/12/17 2022/2/17 新屋測站 467050 觀音測站 C0C590
#     '25.11-121.26': ['C0C620', 'C0C590'], #  桃園 蘆竹   2020/6/9   2022/2/17 蘆竹測站 C0C620 觀音測站 C0C590
# }
lat_lon_dic = {lat_lon: i for i, lat_lon in enumerate(sorted(list(lat_lon_to_site.keys())))}
site_to_idx = {
    'C0G780': 0,
    'C0G770': 1,
    'C0G640': 2,
    '467050': 3,
    'C0C620': 4,
    # 'C0G910': 5,
    # 'C0G890': 6,
    # 'C0D650': 7,
    # 'C0C590': 8,
    # 'C0G900': 9,
    # 'C0C700': 10,
}


def save_pkl(path, pkl):
    with open(path, 'wb') as handle:   
        pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(path):
    with open(path, 'rb') as handle:
        try:
            return pickle.load(handle)
        except:
            return pickle5.load(handle)


def print_and_write_file(fout, cnt, fout_end='\n'):
    print(cnt)
    if fout is not None:
        fout.write(cnt  + fout_end)
        fout.flush()


def load_external_monthly_report(path='data/external-monthly'):
    ret_dict = {}
    ext_header_to_row_idx = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            site, year, month = file.split('.')[0].split('-')
            month = str(int(month))
            if site not in ret_dict:
                ret_dict[site] = {}
            if year not in ret_dict[site]:
                ret_dict[site][year] = {}
            with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
                cnt = fin.read().splitlines()
            if ext_header_to_row_idx is None:
                ext_header_to_row_idx = {h.replace('"', ''): i for i, h in enumerate(cnt[1].split(','))}
            cnt = cnt[1:]
            ret_dict[site][year][month] = [list(map(lambda x: x.replace('"', ''), line.split(','))) for line in cnt]
    return ret_dict, ext_header_to_row_idx


def load_external_daily_report(path='data/external-daily'):
    ret_dict = {}
    ext_header_to_row_idx = None
    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
        site = file.split('_')[0]
        if site not in ret_dict:
            ret_dict[site] = {}
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
            cnt = fin.read().splitlines()
        if ext_header_to_row_idx is None:
            ext_header_to_row_idx = {h.replace('"', ''): i for i, h in enumerate(cnt[0].split(',')[1:])}
        for line in cnt[1:]:
            arr = line.split(',')[1:]
            year, month, date = arr[0].split('-')
            month = str(int(month))
            date = str(int(date))
            hour = str(int(arr[1]))
            if year not in ret_dict[site]:
                ret_dict[site][year] = {}
            if month not in ret_dict[site][year]:
                ret_dict[site][year][month] = {}
            if date not in ret_dict[site][year][month]:
                ret_dict[site][year][month][date] = {}
            if hour not in ret_dict[site][year][month][date]:
                ret_dict[site][year][month][date][hour] = deepcopy(arr)
    return ret_dict, ext_header_to_row_idx


def get_possible_empty_fea(raw_val):
    return [
        float(raw_val == ''),
        float(raw_val if raw_val != '' else 0),
    ]


def get_module_fea(raw_val):
    if raw_val == 'MM60-6RT-300':
        return [1, 0, 0, 0, 300, 32.61, 9.20, 38.97, 09.68, 18.44]
    elif raw_val == 'SEC-6M-60A-295':
        return [0, 1, 0, 0, 295, 31.60, 9.34, 39.40, 09.85, 17.74]
    elif raw_val == 'AUO PM060MW3 320W':
        return [0, 0, 1, 0, 320, 33.48, 9.56, 40.90, 10.24, 19.20]
    elif raw_val == 'AUO PM060MW3 325W':
        return [0, 0, 0, 1, 325, 33.66, 9.66, 41.10, 10.35, 19.50]
    else:
        raise ValueError('Unknown raw_val: ' + raw_val)


def get_lat_lon_fea(lat_str, lon_str):
    key = '{}-{}'.format(lat_str, lon_str)
    ret = [0 for _ in range(len(lat_lon_dic))]
    ret[lat_lon_dic[key]] = 1
    return ret


def get_div_fea(a, b):
    a = float(a) if a != '' else 0
    b = float(b) if b != '' else 0
    return a / b if a * b > 0 else 0


def get_ext_float(raw_str, val_of_empty=-1):
    if raw_str in ('...', 'X', '/', '&', ' ', ''):
        return val_of_empty
    elif raw_str == 'T':
        return 0.01
    else:
        return float(raw_str)


def get_ext_time_fea(raw_str, zero_base):
    try:
        hh, mm = list(map(float, raw_str.split(' ')[1].split(':')))
    except:
        return -1
    if hh >= zero_base:
        hh = hh - zero_base
    else:
        hh = hh + 24 - zero_base
    return hh + mm / 60


def get_ext_one_site_monthly_report_fea(site, this_arr, ext_header_to_row_idx, angle, capacity):
    site_one_hot = [0 for _ in range(len(site_to_idx))]
    site_one_hot[site_to_idx[site]] = 1
    return site_one_hot + [
        get_ext_float(this_arr[ext_header_to_row_idx['StnPres']]),
        get_ext_float(this_arr[ext_header_to_row_idx['SeaPres']]),
        get_ext_float(this_arr[ext_header_to_row_idx['StnPresMax']]),
        get_ext_float(this_arr[ext_header_to_row_idx['StnPresMin']]),
        get_ext_float(this_arr[ext_header_to_row_idx['Temperature']]),
        get_ext_float(this_arr[ext_header_to_row_idx['T Max']]),
        get_ext_float(this_arr[ext_header_to_row_idx['T Min']]),
        get_ext_float(this_arr[ext_header_to_row_idx['Td dew point']]),
        get_ext_float(this_arr[ext_header_to_row_idx['RH']]),
        get_ext_float(this_arr[ext_header_to_row_idx['RHMin']]),
        get_ext_float(this_arr[ext_header_to_row_idx['WS']]),
        get_ext_float(this_arr[ext_header_to_row_idx['WD']]),
        get_ext_float(this_arr[ext_header_to_row_idx['WSGust']]),
        get_ext_float(this_arr[ext_header_to_row_idx['WDGust']]),
        get_ext_float(this_arr[ext_header_to_row_idx['Precp']]),
        get_ext_float(this_arr[ext_header_to_row_idx['PrecpHour']]),
        get_ext_float(this_arr[ext_header_to_row_idx['PrecpMax10']]),
        get_ext_float(this_arr[ext_header_to_row_idx['PrecpMax60']]),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShine']]),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShineRate']]),
        get_ext_float(this_arr[ext_header_to_row_idx['GloblRad']]),
        get_ext_float(this_arr[ext_header_to_row_idx['VisbMean']]),
        get_ext_float(this_arr[ext_header_to_row_idx['EvapA']], val_of_empty=-999),
        get_ext_float(this_arr[ext_header_to_row_idx['UVI Max']]),
        get_ext_float(this_arr[ext_header_to_row_idx['Cloud Amount']]),
        # Fea Eng
        get_ext_float(this_arr[ext_header_to_row_idx['SunShine']]) * math.sin(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShine']]) * math.tanh(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShine']]) * (1 - angle),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShineRate']]) * math.sin(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShineRate']]) * math.tanh(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['SunShineRate']]) * (1 - angle),
        get_ext_float(this_arr[ext_header_to_row_idx['GloblRad']]) * math.sin(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['GloblRad']]) * math.tanh(math.radians(angle)),
        get_ext_float(this_arr[ext_header_to_row_idx['GloblRad']]) * (1 - angle),
    ]


def merge_ext_list(fea_a, fea_b):
    # return [max(va, vb) for va, vb in zip(fea_a, fea_b)]
    return [max(va, vb) if va < 0 or vb < 0 else (va + vb) / 2 for va, vb in zip(fea_a, fea_b)]


def get_ext_monthly_report_fea(lat_str, lon_str, date_str, ext_data_dict, ext_header_to_row_idx, angle, capacity):
    year, month, date = date_str.split('/')
    date = int(date)
    site = lat_lon_to_site['{}-{}'.format(lat_str, lon_str)][0]
    this_arr = ext_data_dict[site][year][month][date]
    return get_ext_one_site_monthly_report_fea(site, this_arr, ext_header_to_row_idx, angle, capacity)
    # site_a = lat_lon_to_site['{}-{}'.format(lat_str, lon_str)][0]
    # site_b = lat_lon_to_site['{}-{}'.format(lat_str, lon_str)][0]
    # this_arr_a = ext_data_dict[site_a][year][month][date]
    # this_arr_b = ext_data_dict[site_b][year][month][date]
    # return merge_ext_list(
    #     get_ext_one_site_monthly_report_fea(site_a, this_arr_a, ext_header_to_row_idx, angle, capacity),
    #     get_ext_one_site_monthly_report_fea(site_b, this_arr_b, ext_header_to_row_idx, angle, capacity),
    # )


def get_ext_one_hour_fea(this_dict, hour, ext_header_to_row_idx, angle, capacity):
    return [
        get_ext_float(this_dict[hour][ext_header_to_row_idx['StnPres']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['SeaPres']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['Temperature']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['Td dew point']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['RH']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['WS']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['WD']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['WSGust']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['WDGust']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['Precp']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['PrecpHour']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['SunShine']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['GloblRad']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['Visb']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['UVI']]),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['Cloud Amount']]),
        # Fea Eng
        get_ext_float(this_dict[hour][ext_header_to_row_idx['SunShine']]) * math.sin(math.radians(angle)),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['SunShine']]) * math.tanh(math.radians(angle)),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['SunShine']]) * (1 - angle),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['GloblRad']]) * math.sin(math.radians(angle)),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['GloblRad']]) * math.tanh(math.radians(angle)),
        get_ext_float(this_dict[hour][ext_header_to_row_idx['GloblRad']]) * (1 - angle),
    ]


def get_ext_one_site_daily_report_fea(site, this_dict, ext_header_to_row_idx, angle, capacity):
    site_one_hot = [0 for _ in range(len(site_to_idx))]
    site_one_hot[site_to_idx[site]] = 1
    return \
        get_ext_one_hour_fea(this_dict, '11', ext_header_to_row_idx, angle, capacity) + \
        get_ext_one_hour_fea(this_dict, '12', ext_header_to_row_idx, angle, capacity) + \
        get_ext_one_hour_fea(this_dict, '13', ext_header_to_row_idx, angle, capacity)


def get_ext_daily_report_fea(lat_str, lon_str, date_str, ext_data_dict, ext_header_to_row_idx, angle, capacity):
    year, month, date = date_str.split('/')
    date = str(int(date))
    site = lat_lon_to_site['{}-{}'.format(lat_str, lon_str)][0]
    this_dict = ext_data_dict[site][year][month][date]
    return get_ext_one_site_daily_report_fea(site, this_dict, ext_header_to_row_idx, angle, capacity)


def fea_ext(row, ext_data_dict_month, ext_header_to_row_idx_month, ext_data_dict_day, ext_header_to_row_idx_day):
    angle = float(row[header_to_row_idx['Angle']])
    month = int(row[header_to_row_idx['Date']].split('/')[1])
    date = int(row[header_to_row_idx['Date']].split('/')[2])
    irradiance = float(row[header_to_row_idx['Irradiance']]) if row[header_to_row_idx['Irradiance']] != '' else 0
    irradiance_m = float(row[header_to_row_idx['Irradiance_m']]) / 1000
    capacity = float(row[header_to_row_idx['Capacity']])
    lat_str = row[header_to_row_idx['Lat']]
    lon_str = row[header_to_row_idx['Lon']]
    return \
        get_possible_empty_fea(row[header_to_row_idx['Temp_m']]) + \
        get_possible_empty_fea(row[header_to_row_idx['Irradiance']]) + \
        get_possible_empty_fea(row[header_to_row_idx['Temp']]) + \
        get_module_fea(row[header_to_row_idx['Module']]) + \
        get_lat_lon_fea(lat_str, lon_str) + \
        get_ext_monthly_report_fea(lat_str, lon_str, row[header_to_row_idx['Date']], ext_data_dict_month, ext_header_to_row_idx_month, angle, capacity) + \
    [
        # Single
        month,
        month - 2 if month >= 3 else month + 10,
        month - 5 if month >= 6 else month + 7,
        month - 8 if month >= 9 else month + 4,
        1 if month in (3, 4, 5) else 0,
        1 if month in (6, 7, 8) else 0,
        1 if month in (9, 10, 11) else 0,
        1 if month in (12, 1, 2) else 0,
        date,
        capacity,
        float(lat_str),
        float(lon_str),
        angle,
        math.sin(math.radians(angle)),
        math.cos(math.radians(angle)),
        math.tanh(math.radians(angle)),
        -abs(math.tanh(math.radians(angle))) + 1,
        1 if angle >= 0 else -1,
        irradiance_m,
        # Comb
        capacity * irradiance_m / 1000,
        irradiance_m * (1- angle),
        irradiance_m * math.sin(math.radians(angle)),
        irradiance_m * math.cos(math.radians(angle)),
        irradiance_m * (math.tanh(math.radians(angle))),
        irradiance_m * (-abs(math.tanh(math.radians(angle))) + 1),
        capacity * irradiance_m * (1- angle),
        capacity * irradiance_m * math.sin(math.radians(angle)),
        capacity * irradiance_m * math.cos(math.radians(angle)),
        capacity * irradiance_m * (math.tanh(math.radians(angle))),
        capacity * irradiance_m * (-abs(math.tanh(math.radians(angle))) + 1),
        get_div_fea(row[header_to_row_idx['Temp_m']], row[header_to_row_idx['Temp']]),
        get_div_fea(row[header_to_row_idx['Irradiance']], row[header_to_row_idx['Irradiance_m']]),
    ], int(row[header_to_row_idx['Generation']]) if row[header_to_row_idx['Generation']] != '' else None


FEA_NAMES = [
    'is Temp_m empty',
    'Temp_m',
    'is Irradiance empty',
    'Irradiance',
    'is Temp empty',
    'Temp',
    'is MM60-6RT-300',
    'is SEC-6M-60A-295',
    'is AUO PM060MW3 320W',
    'is AUO PM060MW3 325W',
    'Pmax',
    'Vmp',
    'Imp',
    'Voc',
    'Isc',
    '%',
    'is 24.04-120.52',
    'is 24.06-120.47',
    'is 24.07-120.47',
    'is 24.07-120.48',
    'is 24.08-120.5',
    'is 24.08-120.52',
    'is 24.09-120.52',
    'is 24.107-120.44',
    'is 24.98-121.03',
    'is 25.03-121.08',
    'is 25.11-121.26',
    'ext-C0G780',
    'ext-C0G770',
    'ext-C0G640',
    'ext-467050',
    'ext-C0C620',
    'ext-StnPres',
    'ext-SeaPres',
    'ext-StnPresMax',
    'ext-StnPresMin',
    'ext-Temperature',
    'ext-T Max',
    'ext-T Min',
    'ext-Td dew point',
    'ext-RH',
    'ext-RHMin',
    'ext-WS',
    'ext-WD',
    'ext-WSGust',
    'ext-WDGust',
    'ext-Precp',
    'ext-PrecpHour',
    'ext-PrecpMax10',
    'ext-PrecpMax60',
    'ext-SunShine',
    'ext-SunShineRate',
    'ext-GloblRad',
    'ext-VisbMean',
    'ext-EvapA',
    'ext-UVI Max',
    'ext-Cloud Amount',
    'ext-SunShine * sin angle',
    'ext-SunShine * tanh angle',
    'ext-SunShine * (1 - angle)',
    'ext-SunShineRate * sin angle',
    'ext-SunShineRate * tanh angle',
    'ext-SunShineRate * (1 - angle)',
    'ext-GloblRad * sin angle',
    'ext-GloblRad * tanh angle',
    'ext-GloblRad * (1 - angle)',
    'month',
    'month base 3',
    'month base 6',
    'month base 9',
    'is spring',
    'is summar',
    'is fall',
    'is winter',
    'date',
    'capacity',
    'lat',
    'lon',
    'angle',
    'sin angle',
    'cos angle',
    'tanh angel',
    '-abs(tanh angel)+1',
    'sign angle',
    'irradiance_m',
    'capacity * irradiance_m / 1000',
    'irradiance_m * (1- angle)',
    'irradiance_m * sin angle',
    'irradiance_m * cos angle',
    'irradiance_m * tanh angle',
    'irradiance_m * (-abs(tanh angel)+1)',
    'capacity * irradiance_m * (1- angle)',
    'capacity * irradiance_m * sin angle',
    'capacity * irradiance_m * cos angle',
    'capacity * irradiance_m * tanh angle',
    'capacity * irradiance_m * (-abs(tanh angel)+1)',
    'Temp_m / temp',
    'Irradiance / irradiance_m',
]
