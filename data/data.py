"""
动态导入模块
先动态导入dataset，再动态导入Dataloader
"""

from importlib import import_module


class Data:
    def __init__(self, para, device_id):
        dataset = para.dataset
        module = import_module('data.' + dataset)
        self.dataloader_train = module.Dataloader(para, device_id, 'train')
        self.dataloader_valid = module.Dataloader(para, device_id, 'valid')
