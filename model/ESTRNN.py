import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from .arches import conv1x1, conv3x3, conv5x5, actFunc


# RDB 模块内部的子模块
# Dense layer
class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate, activation='relu'):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)
        self.act = actFunc(activation)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate, activation))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out

# RDB cell 中中间的部分单独抽出来
# Middle network of residual dense blocks
class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks, activation='relu'):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer, activation))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out

# RDB cell 内部左边部分
# DownSampling module
class RDB_DS(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, activation='relu'):
        super(RDB_DS, self).__init__()
        self.rdb = RDB(in_channels, growthRate, num_layer, activation)
        self.down_sampling = conv5x5(in_channels, 2 * in_channels, stride=2)

    def forward(self, x):
        # x: n,c,h,w
        x = self.rdb(x)
        out = self.down_sampling(x)

        return out


# Global spatio-temporal attention module
class GSA(nn.Module):
    def __init__(self, para):
        super(GSA, self).__init__()
        self.n_feats = para.n_features
        self.center = para.past_frames
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.F_f = nn.Sequential(
            nn.Linear(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            actFunc(para.activation),
            nn.Linear(4 * (5 * self.n_feats), 2 * (5 * self.n_feats)),
            nn.Sigmoid()
        )
        # out channel: 160
        self.F_p = nn.Sequential(
            conv1x1(2 * (5 * self.n_feats), 4 * (5 * self.n_feats)),
            conv1x1(4 * (5 * self.n_feats), 2 * (5 * self.n_feats))
        )
        # condense layer
        self.condense = conv1x1(2 * (5 * self.n_feats), 5 * self.n_feats)
        # fusion layer
        self.fusion = conv1x1(self.related_f * (5 * self.n_feats), self.related_f * (5 * self.n_feats))

    def forward(self, hs):
        # hs: [(n=4,c=80,h=64,w=64), ..., (n,c,h,w)]
        self.nframes = len(hs) # 默认是5吧
        f_ref = hs[self.center]
        cor_l = []
        for i in range(self.nframes):
            if i != self.center:
                cor = torch.cat([f_ref, hs[i]], dim=1)

                # 进入上边的支线部分
                w = F.adaptive_avg_pool2d(cor, (1, 1)).squeeze()  # (n,c) : (4,160)
                if len(w.shape) == 1:
                    w = w.unsqueeze(dim=0)
                w = self.F_f(w)
                w = w.reshape(*w.shape, 1, 1)

                # 回归主线
                cor = self.F_p(cor)
                cor = self.condense(w * cor)
                cor_l.append(cor)

        cor_l.append(f_ref)
        out = self.fusion(torch.cat(cor_l, dim=1))

        return out


# RDB-based RNN cell
class RDBCell(nn.Module):
    def __init__(self, para):
        super(RDBCell, self).__init__()
        self.activation = para.activation
        self.n_feats = para.n_features
        self.n_blocks = para.n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = RDB_DS(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation)
        self.F_B2 = RDB_DS(in_channels=2 * self.n_feats, growthRate=int(self.n_feats * 3 / 2), num_layer=3,
                           activation=self.activation)
        self.F_R = RDNet(in_channels=(1 + 4) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks, activation=self.activation)  # in: 80
        # F_h: hidden state part
        self.F_h = nn.Sequential(
            conv3x3((1 + 4) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3, activation=self.activation),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, x, s_last):
        # ft主线
        out = self.F_B0(x)
        out = self.F_B1(out)
        out = self.F_B2(out)
        out = torch.cat([out, s_last], dim=1)
        out = self.F_R(out)

        # 进入ht支线
        s = self.F_h(out)

        return out, s


# Reconstructor
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.para = para
        self.num_ff = para.future_frames
        self.num_fb = para.past_frames
        self.related_f = self.num_ff + 1 + self.num_fb
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5 * self.n_feats) * (self.related_f), 2 * self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            conv5x5(self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    """
    Efficient saptio-temporal recurrent neural network (ESTRNN, ECCV2020)
    """
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        self.n_feats = para.n_features # RDB cell 输出隐藏层状态通道数
        self.num_ff = para.future_frames 
        self.num_fb = para.past_frames
        self.ds_ratio = 4 # 特征图相较原图缩放系数
        self.device = torch.device('cuda')
        self.cell = RDBCell(para)
        self.recons = Reconstructor(para)
        self.fusion = GSA(para)

    def forward(self, x, profile_flag=False):
        print('x.shape: ', x.shape)
        if profile_flag:
            return self.profile_forward(x)
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        # forward h structure: (batch_size, channel, height, width)
        # 一开始隐藏层状态是s
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs.append(h) # hs:[(b,c,h,w), ..., (b,c,h,w)]
        for i in range(self.num_fb, frames - self.num_ff): # 如果frames=6，则这里循环2次
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))
        return torch.cat(outputs, dim=1) # (b,frames,c,h,w) -> (b,frames-2,c,h,w)

    # For calculating GMACs
    def profile_forward(self, x):
        outputs, hs = [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs.append(h)
        for i in range(self.num_fb + self.num_ff): # 这里在后面多加几人造帧，保证frames输入=输出
            hs.append(torch.randn(*h.shape).to(self.device))
        for i in range(self.num_fb, frames + self.num_fb):
            out = self.fusion(hs[i - self.num_fb:i + self.num_ff + 1])
            out = self.recons(out)
            outputs.append(out.unsqueeze(dim=1))

        return torch.cat(outputs, dim=1)


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs

# 返回每个序列步长的平均计算代价和模型的总参数数量，这是通过将总的浮点操作数 (FLOPs) 除以序列长度得到的
def cost_profile(model, H, W, seq_length):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params
