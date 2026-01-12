import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import cv2
from skimage import filters
import pdb

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None, fold_num=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if fold_num == None:
            if self.split == 'train':
                with open(self._base_dir + '/train.txt', 'r') as f1:
                    self.sample_list = f1.readlines()
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]

            elif self.split == 'val':
                with open(self._base_dir + '/test.txt', 'r') as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        else:
            if self.split == 'train':
                with open(self._base_dir + '/train_fold_{}.txt'.format(fold_num), 'r') as f1:
                    self.sample_list = f1.readlines()
                with open(self._base_dir + '/unlab.txt', 'r') as f2:
                    self.sample_list = self.sample_list + f2.readlines()
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]
            elif self.split == 'val':
                with open(self._base_dir + '/test_fold_{}.txt'.format(fold_num), 'r') as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        if ('unlabeled' not in case) and ('PSFHS' not in case):
            # print(case)
            image = cv2.imread(self._base_dir+"/{}".format(case),1)
            label_case = case.split('/')[0]+'/labels/'+case.split('/')[2]
            label = cv2.imread(self._base_dir+"/{}".format(label_case),0)
        else:
            image = cv2.imread(self._base_dir + "/{}".format(case), 1)
            h,w,z = image.shape
            label = np.zeros((h, w))
        sample = {'image': image, 'label': label, 'case':case}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx

        return sample

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        case = sample['case']
        image1 = image
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image1, label)
        x, y, z = image.shape

        # print(label.shape)
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, self.output_size[2] / z), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        j = random.randint(1,3)
        image_weak = image
        image_strong = image
        if 'unlabeled' not in case:
            random_noisy = random.randint(1, 1)

            if random_noisy == 0:
                noisy = GaussNoise(min_std=3, max_std=20, norm_mode='trunc')
                image_weak = noisy(image)
            elif random_noisy == 1:
                noisy = Cutout('mala')
                image_weak = noisy(image)
            elif random_noisy == 2:
                noisy = DCT_2D(degree=80)
                image_weak = noisy(image)
        else :
            random_noisy = random.randint(0, 2)
            if random_noisy == 0:
                noisy = GaussNoise(min_std=3, max_std=20, norm_mode='trunc')
                image_strong = noisy(image)
            elif random_noisy == 1:
                noisy = Cutout('mala')
                image_strong = noisy(image)
            elif random_noisy == 2:
                noisy = DCT_2D(degree=80)
                image_strong = noisy(image)
        image_weak = image_weak.transpose((2,0,1))
        image_weak = torch.from_numpy(image_weak.astype(np.float32))
        image_strong = image_strong.transpose((2,0,1))
        image_strong = torch.from_numpy(image_strong.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {'image_weak': image_weak, 'label': label,'image_strong':image_strong}
        return sample



class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order = 1, mode = 'constant', cval = 0)
        label = sk_trans.resize(label, self.output_size, order = 0)
        assert(np.max(label) == 1 and np.min(label) == 0)
        assert(np.unique(label).shape[0] == 2)
        
        return {'image': image, 'label': label}
    
    
class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size),
                    grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def gen_mask(imgs,
             #net_crop_size=[0,0,0],
             net_crop_size=[0,0],
             mask_counts=80,
             # mask_size_z=8,
             mask_size_xy=15):

    crop_size = list(imgs.shape)
    mask = np.ones_like(imgs, dtype=np.float32)
    for k in range(mask_counts):
        #mz = random.randint(net_crop_size[0], crop_size[0]-mask_size_z-net_crop_size[0])
        my = random.randint(net_crop_size[0], crop_size[0]-mask_size_xy-net_crop_size[0])
        mx = random.randint(net_crop_size[1], crop_size[1]-mask_size_xy-net_crop_size[1])
        #mask[mz:mz+mask_size_z, my:my+mask_size_xy, mx:mx+mask_size_xy] = 0
        mask[my:my + mask_size_xy, mx:mx + mask_size_xy] = 0
    return mask

class Cutout(object):#add some black blocks into image
    def __init__(self, model_type='superhuman'):
        super(Cutout, self).__init__()
        self.model_type = model_type
        # mask size
        if self.model_type == 'mala':
            self.min_mask_size = [5, 6, 6]
            self.max_mask_size = [8, 10, 12]
            self.min_mask_counts = 50
            self.max_mask_counts = 80
            #self.net_crop_size = [14, 106, 106]
            self.net_crop_size = [50, 50]
        else:
            self.min_mask_size = [5, 10, 10]
            self.max_mask_size = [10, 20, 20]
            self.min_mask_counts = 60
            self.max_mask_counts = 100
            self.net_crop_size = [0, 0, 0]

    def __call__(self, data):
        mask_counts = random.randint(self.min_mask_counts, self.max_mask_counts)
        # mask_size_z = random.randint(self.min_mask_size[0], self.max_mask_size[0])
        mask_size_xy = random.randint(self.min_mask_size[1], self.max_mask_size[1])
        # mask = gen_mask(data, net_crop_size=self.net_crop_size, \
        #                 mask_counts=mask_counts, \
        #                 mask_size_z=mask_size_z, \
        #                 mask_size_xy=mask_size_xy)
        mask = gen_mask(data, net_crop_size=self.net_crop_size, \
                        mask_counts=mask_counts, \
                        mask_size_xy=mask_size_xy)
        # mask = np.repeat(np.expand_dims(mask, axis=-1), 3, axis=-1)
        data = data * mask
        return data

def add_gauss_noise(imgs, std=5, norm_mode='norm'):
    gaussian = np.random.normal(0, std, (336, 544))
    gaussian = np.repeat(np.expand_dims(gaussian, axis=-1), 3, axis=-1)
    imgs = imgs + gaussian
    if norm_mode == 'norm':
        imgs = (imgs-np.min(imgs)) / (np.max(imgs)-np.min(imgs))
    elif norm_mode == 'trunc':
        imgs[imgs<0] = 0
        imgs[imgs>255] = 255
    else:
        raise NotImplementedError
    return imgs

class GaussNoise(object):
    def __init__(self, min_std=0.01, max_std=0.2, norm_mode='trunc'):
        super(GaussNoise, self).__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.norm_mode = norm_mode

    def __call__(self, data):
        std = random.uniform(self.min_std, self.max_std)
        data = add_gauss_noise(data, std=std, norm_mode=self.norm_mode)
        return data

class DCT_2D(object):
    def __init__(self, degree=50): #The lower the degree, the greater the noise
        super(DCT_2D, self).__init__()
        self.degree = degree

    def __call__(self, data):
        data = data.astype(np.float32)
        processed_data = np.zeros_like(data)

        # Apply 2D DCT and IDCT to each channel separately
        for channel in range(data.shape[2]):
            # Apply 2D DCT to the current channel
            data_dct = cv2.dct(data[:, :, channel])

            # Set high-frequency components to 1
            data_dct[self.degree:, self.degree:] = 1

            # Apply 2D IDCT to the modified DCT coefficients
            data_idct = cv2.idct(data_dct)

            # Store the result in the corresponding channel of the output array
            processed_data[:, :, channel] = data_idct

        # Convert the result back to uint8
        now_data = processed_data.astype(np.uint8)


        return now_data

def add_sobel(imgs, if_mean=False):
    outs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        # sobelx = cv2.Sobel(temp, cv2.CV_32F, 1, 0)
        # sobely = cv2.Sobel(temp, cv2.CV_32F, 0, 1)
        # sobelx = filters.sobel_h(temp)
        # sobely = filters.sobel_v(temp)
        # dst = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # dst = sobelx * 0.5 + sobely * 0.5
        # dst = cv2.Sobel(temp, cv2.CV_32F, 1, 1)
        if if_mean:
            mean = np.mean(temp)
        else:
            mean = 0
        dst = filters.sobel(temp) + mean
        outs.append(dst)
    outs = np.asarray(outs, dtype=np.float32)
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return outs

class SobelFilter(object):
    def __init__(self, if_mean=False):
        super(SobelFilter, self).__init__()
        self.if_mean = if_mean

    def __call__(self, data):
        data = add_sobel(data, if_mean=self.if_mean)
        return data

def add_gauss_blur(imgs, kernel_size=5, sigma=0):
    outs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        temp = cv2.GaussianBlur(temp, (kernel_size,kernel_size), sigma)
        outs.append(temp)
    outs = np.asarray(outs, dtype=np.float32)
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    outs = outs.squeeze()
    return outs

class GaussBlur(object):
    def __init__(self, min_kernel=3, max_kernel=9, min_sigma=0, max_sigma=2):
        super(GaussBlur, self).__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, data):
        kernel_size = random.randint(self.min_kernel // 2, self.max_kernel // 2)
        kernel_size = kernel_size * 2 + 1
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        data = add_gauss_blur(data, kernel_size=kernel_size, sigma=sigma)
        return data
