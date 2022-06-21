from itertools import permutations

import matplotlib.pyplot as plt

pred = list(map(float, '25.11-121.26'.split('-')))

sites = {
    # 彰化
    'shenggang': [24.1489, 120.4844],
    'xianxi': [24.1433, 120.4435],
    'lukang': [24.0753, 120.4304],
    'fuxing': [24.0412, 120.4376],
    'puyang': [24.0003, 120.4316],
    'xiushui': [24.0340, 120.5038],
    'huatan': [24.0320, 120.5494],
    'fenyuan': [24.0156, 120.6213],
    'xihu': [23.9483, 120.4791],
    'puxin': [23.9476, 120.5254],
    'yuanlin': [23.9465, 120.5855],
    # 一些台中
    'wuri': [24.1070, 120.6241],
    'dadu': [24.1529, 120.5721],
    'longjing': [24.1845, 120.5289],
    # 桃園
    # 'jhonda': [24.9661, 121.0085], # 無資料
    # 'shueiwei': [24.9400, 121.0871], # 無資料
    'yangmei': [24.9123, 121.1430],
    'xinwu': [25.0067, 121.0474],
    # 'guanyin_ins': [25.0647, 121.1148], # 無資料
    'guanyin': [25.0270, 121.1533],
    # 'jhuwei': [25.1126, 121.2398], # 無資料
    'luzhu': [25.0842, 121.2657],
    'zhongli': [24.9776, 121.2563],
    'taoyuan': [24.9924, 121.3231],
    'guishan': [25.0284, 121.3865],
    'pingjhen': [24.8975, 121.2146],
    'bade': [24.9287, 121.2832],
    # 一些台北
    # 'n039k': [25.0643, 121.3838], # 無資料
    # 'linkou': [25.0721, 121.3808], # 無資料
    # 一些新竹
    'waihu': [24.9177, 120.9687],
    'hukou': [24.9047, 121.0436],
    'bade': [24.9287, 121.2832],
}


def get_dist_pa(p, a):
    return ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5


def get_dist_pab(p, a, b, op):
    m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    pa = ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
    pb = ((p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2) ** 0.5
    pm = ((p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2) ** 0.5
    if op == 'add':
        return pa + pb # + pm
    elif op == 'mul':
        return pa * pb # * pm
    else:
        raise Exception('op not supported')


def get_dist_pabc(p, a, b, c, op):
    g = ((a[0] + b[0] + c[0]) / 2, (a[1] + b[1] + c[1]) / 2)
    pa = ((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2) ** 0.5
    pb = ((p[0] - b[0]) ** 2 + (p[1] - b[1]) ** 2) ** 0.5
    pc = ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** 0.5
    pg = ((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) ** 0.5
    if op == 'add':
        return pa + pb + pc + pg
    elif op == 'mul':
        return pa * pb * pc * pg
    else:
        raise Exception('op not supported')


site_and_dist = sorted([(name, get_dist_pa(pred, loc)) for name, loc in sites.items()], key=lambda x: x[1])
# site_and_dist = sorted([((name_a, name_b), get_dist_pab(pred, loc_a, loc_b, 'add')) for (name_a, loc_a), (name_b, loc_b) in permutations(sites.items(), r=2)], key=lambda x: x[1])
# site_and_dist = sorted([((name_a, name_b, name_c), get_dist_pabc(pred, loc_a, loc_b, loc_c, 'add')) for (name_a, loc_a), (name_b, loc_b), (name_c, loc_c) in permutations(sites.items(), r=3)], key=lambda x: x[1])
for name, dist in site_and_dist[:10]:
    print(name, dist)

# plt.plot(pred[0], pred[1], 'd', label='Pred')
# for name, pos in sites.items():
#     if name in site_and_dist[0][0]:
#         plt.plot(pos[0], pos[1], 'rx', label='top-1')
#     else:
#         plt.plot(pos[0], pos[1], 'b.')
# plt.legend()
# plt.show()
