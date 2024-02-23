from importlib import import_module

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()
        self.para = para
        model_name = para.model
        # 我感觉应该是寻找当前文件夹下model_name这个文件，并从该文件Model类中的forward开始运行
        self.module = import_module('model.{}'.format(model_name))
        self.model = self.module.Model(para)

    def forward(self, iter_samples):
        outputs = self.module.feed(self.model, iter_samples)
        return outputs

    def profile(self):
        H, W = self.para.profile_H, self.para.profile_W
        seq_length = self.para.future_frames + self.para.past_frames + 1
        flops, params = self.module.cost_profile(self.model, H, W, seq_length)
        return flops, params
