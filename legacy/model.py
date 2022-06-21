import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_fea):
        super(MLP, self).__init__()
        DIM_HIDDEN = 512
        self.SCALE_FACTOR = 100
        self.net_1 = nn.Sequential(
            nn.Linear(in_features=dim_fea, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
        )
        self.net_2 = nn.Sequential(
            nn.Linear(in_features=dim_fea+DIM_HIDDEN, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
        )
        self.net_3 = nn.Sequential(
            nn.Linear(in_features=dim_fea+DIM_HIDDEN, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
        )
        self.net_4 = nn.Sequential(
            nn.Linear(in_features=dim_fea+DIM_HIDDEN, out_features=DIM_HIDDEN),
            nn.BatchNorm1d(DIM_HIDDEN),
            nn.SiLU(),
        )
        self.net_5 = nn.Sequential(
            nn.Linear(in_features=DIM_HIDDEN, out_features=1),
            nn.SiLU(),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(in_features=dim_fea, out_features=DIM_HIDDEN),
        #     nn.BatchNorm1d(DIM_HIDDEN),
        #     nn.SiLU(),
        #     nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_HIDDEN),
        #     nn.BatchNorm1d(DIM_HIDDEN),
        #     nn.SiLU(),
        #     nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_HIDDEN),
        #     nn.BatchNorm1d(DIM_HIDDEN),
        #     nn.SiLU(),
        #     nn.Linear(in_features=DIM_HIDDEN, out_features=DIM_HIDDEN),
        #     nn.BatchNorm1d(DIM_HIDDEN),
        #     nn.SiLU(),
        #     nn.Linear(in_features=DIM_HIDDEN, out_features=1),
        #     nn.SiLU(),
        # )

    def forward(self, x):
        h = self.net_1(x)
        h = self.net_2(torch.cat((h, x), dim=1))
        h = self.net_3(torch.cat((h, x), dim=1))
        h = self.net_4(torch.cat((h, x), dim=1))
        h = self.net_5(h)
        return h * self.SCALE_FACTOR
        # return self.net(x) * self.SCALE_FACTOR
