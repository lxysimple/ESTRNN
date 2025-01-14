import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .utils import normalize, Crop, Flip, ToTensor


class DeblurDataset(Dataset):
    """
    Structure of self_.records:
        seq:
            frame:
                path of images -> {'Blur': <path>, 'Sharp': <path>}
    """

    def __init__(self, path, frames, future_frames, past_frames, crop_size=(256, 256), data_format='RGB',
                 centralize=True, normalize=True):
        assert frames - future_frames - past_frames >= 1
        self.frames = frames # 子序列总帧数，默认是8
        self.num_ff = future_frames # 默认是2
        self.num_pf = past_frames # 默认是2
        self.data_format = data_format
        self.W = 256 # 640
        self.H = 256 # 480
        self.crop_h, self.crop_w = crop_size
        self.normalize = normalize
        self.centralize = centralize

        # self.transform = transforms.Compose([Crop(crop_size), Flip(), ToTensor()])
        self.transform = transforms.Compose([Flip(), ToTensor()])

        self._seq_length = 100 # 被废弃
        self._samples = self._generate_samples(path, data_format)

    def _generate_samples(self, dataset_path, data_format):
        """
        将一个序列中100组帧变成[[1=(Blur_path,Sharp_path),2,3,4,5], [2,3,4,5,6], ...]子序列形式
        返回多个序列中所有子序列混合列表
        """
        samples = list()
        records = dict() # 储存所有序列、每一个帧，每一个帧绝对地址存储在sample['Blur']中
        seqs = sorted(os.listdir(dataset_path), key=int)
        for seq in seqs:
            records[seq] = list()
            seq_path = join(dataset_path, seq)
            seq_len = len(os.listdir(seq_path))

            # for frame in range(self._seq_length):
            for frame in range(seq_len):
                
                # suffix = 'png' if data_format == 'RGB' else 'tiff'
                suffix = 'jpg' 
                sample = dict()

                # sample['Blur'] = join(dataset_path, seq, 'Blur', data_format, '{:08d}.{}'.format(frame, suffix))
                # sample['Sharp'] = join(dataset_path, seq, 'Sharp', data_format, '{:08d}.{}'.format(frame, suffix))
                sample['Blur'] = join(dataset_path, seq, '{}.{}'.format(frame, suffix))
                
                # records[seq1]=[sample1, sample2, ...]
                records[seq].append(sample)

        for seq_records in records.values(): # seq_records is a list
            temp_length = len(seq_records) - (self.frames - 1)
            if temp_length <= 0:
                raise IndexError('Exceed the maximum length of the video sequence')
            # 有很多重叠的帧
            for idx in range(temp_length):
                samples.append(seq_records[idx:idx + self.frames])
        return samples

    def __getitem__(self, item): 
        top = random.randint(0, self.H - self.crop_h)
        left = random.randint(0, self.W - self.crop_w)
        flip_lr = random.randint(0, 1)
        flip_ud = random.randint(0, 1)
        # 裁剪和翻转的相关参数
        sample = {'top': top, 'left': left, 'flip_lr': flip_lr, 'flip_ud': flip_ud}

        blur_imgs, sharp_imgs = [], [] # 获得内存中数据增强、标准化后的子序列
        for sample_dict in self._samples[item]: # 默认一组是5帧，故迭代5次
            blur_img, sharp_img = self._load_sample(sample_dict, sample)
            # sharp_img = self._load_sample(sample_dict, sample)
            blur_imgs.append(blur_img)
            # sharp_imgs.append(sharp_img)
        # 子序列中可去模糊的帧数
        # sharp_imgs = sharp_imgs[self.num_pf:self.frames - self.num_ff]
            
        # [(frames=8, c, h, w), (frames-some=4, c, h, w)] 
        # return [torch.cat(item, dim=0) for item in [blur_imgs, sharp_imgs]]
        return torch.cat(blur_imgs, dim=0) 

    def _load_sample(self, sample_dict, sample):
        """ 将1帧载入内存，做一点数据增强和标准化 """

        if self.data_format == 'RGB':
            sample['image'] = cv2.imread(sample_dict['Blur'])
            # sample['label'] = cv2.imread(sample_dict['Sharp'])
        elif self.data_format == 'RAW':
            sample['image'] = cv2.imread(sample_dict['Blur'], -1)[..., np.newaxis].astype(np.int32)
            # sample['label'] = cv2.imread(sample_dict['Sharp'], -1)[..., np.newaxis].astype(np.int32)
        sample = self.transform(sample)
        val_range = 2.0 ** 8 - 1 if self.data_format == 'RGB' else 2.0 ** 16 - 1
        blur_img = normalize(sample['image'], centralize=self.centralize, normalize=self.normalize, val_range=val_range)
        # sharp_img = normalize(sample['label'], centralize=self.centralize, normalize=self.normalize, val_range=val_range)

        # return blur_img, sharp_img
        return blur_img

    def __len__(self):
        return len(self._samples)


class Dataloader:
    def __init__(self, para, device_id, ds_type='train'):
        path = join(para.data_root, para.dataset, '{}_{}'.format(para.dataset, para.ds_config), ds_type)
        frames = para.frames
        dataset = DeblurDataset(path, frames, para.future_frames, para.past_frames, para.patch_size, para.data_format,
                                para.centralize, para.normalize)
        gpus = para.num_gpus
        bs = para.batch_size
        ds_len = len(dataset)
        if para.trainer_mode == 'ddp':
            # 它适用于使用多个 GPU 进行训练的情况，可以确保每个 GPU 都能够获取到不同的样本。
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=para.num_gpus,
                rank=device_id
            )
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=False,
                num_workers=para.threads,
                pin_memory=True,
                sampler=sampler,
                drop_last=True
            )
            loader_len = np.ceil(ds_len / gpus)
            # 保证数据集是batchsize的整数倍
            self.loader_len = int(np.ceil(loader_len / bs) * bs)

        elif para.trainer_mode == 'dp':
            self.loader = DataLoader(
                dataset=dataset,
                batch_size=para.batch_size,
                shuffle=True,
                num_workers=para.threads,
                pin_memory=True,
                drop_last=True
            )
            self.loader_len = int(np.ceil(ds_len / bs) * bs)

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return self.loader_len


if __name__ == '__main__':
    from para import Parameter

    para = Parameter().args
    para.data_format = 'RAW'
    para.dataset = 'BSD'
    dataloader = Dataloader(para, 0)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
    print(x.type(), y.type())
    print(np.max(x.numpy()), np.min(x.numpy()))
    print(np.max(y.numpy()), np.min(y.numpy()))
