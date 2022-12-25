import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=(1,), padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.dropout = nn.Dropout(p=0.2)  # 0.2
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, args):
        super(Inception, self).__init__()
        self.units = args.units
        self.branchs = []
        if self.units[0]:
            self.branch0 = BasicConv2d(1, args.hidCNN, kernel_size=1, stride=1)
            self.branchs.append(self.branch0)
        if self.units[1]:
            self.branch1 = nn.Sequential(
                BasicConv2d(1, int(args.hidCNN/2), kernel_size=1, stride=1),
                BasicConv2d(int(args.hidCNN/2), args.hidCNN, kernel_size=3, stride=1, padding=1)
            )
            self.branchs.append(self.branch1)
        if self.units[2]:
            self.branch2 = nn.Sequential(
                BasicConv2d(1, int(args.hidCNN/2), kernel_size=1, stride=1),
                BasicConv2d(int(args.hidCNN/2), args.hidCNN, kernel_size=3, stride=1, padding=1),
                BasicConv2d(args.hidCNN, args.hidCNN, kernel_size=3, stride=1, padding=1)
            )
            self.branchs.append(self.branch2)
        if self.units[3]:
            self.branch3 = nn.Sequential(
                nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
                BasicConv2d(1, args.hidCNN, kernel_size=1, stride=1)
            )
            self.branchs.append(self.branch3)

    def forward(self, x):
        features = []
        for branch in self.branchs:
            features.append(branch(x))
        out = torch.cat(features, 3)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 18)):
        super(TemporalBlock, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size)
        self.gru = nn.GRU(out_planes, out_planes)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.conv.weight)
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)

    def forward(self, x):
        out = self.conv(x)
        out = self.gru(out.squeeze().contiguous().permute(2, 0, 1))
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.f = args.feature
        self.hw = args.highway_window
        self.hidC = args.hidCNN
        self.hidR = args.hidRNN
        self.kernel_l = 3  # PM2.5 5  Rate 2 NASDAQ 3
        self.c_net_l = BasicConv2d(1, self.hidC, (self.kernel_l, self.kernel_l))  # 3 3
        self.t_net_l = TemporalBlock(self.hidC, self.hidR, kernel_size=(1, self.f-self.kernel_l+1))

        self.inception = Inception(args)
        self.t_net_g = TemporalBlock(self.hidC, self.hidR, kernel_size=(1, sum(args.units)*self.f))

        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(2 * self.hidR, self.f)

        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)

        self.attention_l = ScaledDotProductAttention(d_model=self.hidR, d_k=self.hidR, d_v=self.hidR, h=8)
        self.attention_g = ScaledDotProductAttention(d_model=self.hidR, d_k=self.hidR, d_v=self.hidR, h=8)

        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.fc.weight)
        if self.hw > 0:
            init.xavier_normal_(self.highway.weight)

    def forward(self, x):

        l_feature = self.c_net_l(x.unsqueeze(1))
        h_o, g_o = self.t_net_l(l_feature)
        g_o_att = self.attention_l(h_o.permute(1, 0, 2), h_o.permute(1, 0, 2), h_o.permute(1, 0, 2))
        g_o = g_o_att[:, -1, :].squeeze() + g_o.squeeze()

        g_feature = self.inception(x.unsqueeze(1))
        h_e, g_e = self.t_net_g(g_feature)
        g_e_att = self.attention_g(h_e.permute(1, 0, 2), h_e.permute(1, 0, 2), h_e.permute(1, 0, 2))
        g_e = g_e_att[:, -1, :].squeeze() + g_e.squeeze()

        res = torch.cat((g_o, g_e), 1)
        res = self.fc(res)

        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.f)
            res = res + z

        return res


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v=0, h=1,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model, 每个时刻的编码维度
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        if d_v!=0:
            self.fc_v = nn.Linear(d_model, h * d_v)
            self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.relu = F.relu

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model) nq查询的时刻范围， d_model每个时刻的编码维度
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        if values is not None:
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)
        # att = att.squeeze(1)
        if values is not None:
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
            out = self.fc_o(out)  # (b_s, nq, d_model)
            return self.relu(out)
        return att