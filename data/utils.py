import numpy as np
import torch


class Crop(object):
    """
    Crop randomly the image in a sample.
    Args: output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        top, left = sample['top'], sample['left']
        new_h, new_w = self.output_size
        sample['image'] = image[top: top + new_h,
                          left: left + new_w]
        sample['label'] = label[top: top + new_h,
                          left: left + new_w]

        return sample


class Flip(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag_lr = sample['flip_lr']
        flag_ud = sample['flip_ud']
        if flag_lr == 1:
            sample['image'] = np.fliplr(sample['image'])
            sample['label'] = np.fliplr(sample['label'])
        if flag_ud == 1:
            sample['image'] = np.flipud(sample['image'])
            sample['label'] = np.flipud(sample['label'])

        return sample


class Rotate(object):
    """
    shape is (h,w,c)
    """

    def __call__(self, sample):
        flag = sample['rotate']
        if flag == 1:
            sample['image'] = sample['image'].transpose(1, 0, 2)
            sample['label'] = sample['label'].transpose(1, 0, 2)

        return sample


class Sharp2Sharp(object):
    def __call__(self, sample):
        flag = sample['s2s']
        if flag < 1:
            sample['image'] = sample['label'].copy()
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.ascontiguousarray(image.transpose((2, 0, 1))[np.newaxis, :])
        label = np.ascontiguousarray(label.transpose((2, 0, 1))[np.newaxis, :])
        sample['image'] = torch.from_numpy(image).float()
        sample['label'] = torch.from_numpy(label).float()
        return sample


def normalize(x, centralize=False, normalize=False, val_range=255.0):
    if centralize:
        x = x - val_range / 2
    if normalize:
        x = x / val_range

    return x



def normalize_reverse(x, centralize=False, normalize=False, val_range=255.0):
    if normalize:
        x = x * val_range
    if centralize:
        x = x + val_range / 2

    return x

def min_max_normalization(x: torch.Tensor): 
    """最小-最大归一化函数

    参数:
    x (tc.Tensor): 输入张量，形状为(batch, f1, ...)

    返回:
    tc.Tensor: 归一化后的张量，保持原始形状
    """
    # 获取输入张量的形状
    shape = x.shape

    # 如果输入张量的维度大于2，将其展平成二维张量
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    # 计算每行的最小值和最大值,只取第1行的值拿来用
    min_ = x.min(dim=-1, keepdim=True)[0]
    max_ = x.max(dim=-1, keepdim=True)[0]

    # 如果最小值的平均值为0，最大值的平均值为1，说明已经是归一化状态，直接返回
    if min_.mean() == 0 and max_.mean() == 1:
        return x.reshape(shape)

    # 进行最小-最大归一化处理
    x = (x - min_) / (max_ - min_ + 1e-9)

    return x.reshape(shape), min_, max_-min_

def min_max_normalization_reverse(x, min_, max_min_): 

    # 获取输入张量的形状
    shape = x.shape

    # 如果输入张量的维度大于2，将其展平成二维张量
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    # 进行最小-最大归一化处理
    x = x*max_min_ + min_

    return x.reshape(shape)