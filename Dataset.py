import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import numpy as np
import h5py
import pywt
import cv2
from brisque import brisque


def default_loader(path):
    return Image.open(path).convert('RGB')


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def rgb_dct(im_rgb):
    '''
    DCT for an tensor from the image[C,H,W]
    :param im_rgb: tensor
    :return: ndarray
    '''
    im_dct = []
    im_rgb = np.array(im_rgb)
    for i in range(3):
        im_dct.append(cv2.dct(np.float32(im_rgb[i, :, :])))
    res = np.array(im_dct)
    return res


def NonOverlappingCropPatches(im, im_path, patch_size=32, stride=32 ):
    w, h = im.size
    patches = ()
    features = ()
    fusion_dct_patches =()
    for i in range(2, h - stride, stride):     # Begining with 2 for same patche number in MA
        for j in range(2, w - stride, stride):   # MA:480*320,126 patches    QADS:500*380,165 patches
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            # NSS
            feature = torch.from_numpy(brisque(patch.numpy()))

            # dct
            dct_patch = torch.from_numpy(rgb_dct(patch))

            # patch norm
            patch[0] = LocalNormalization(patch[0].numpy())
            patch[1] = LocalNormalization(patch[1].numpy())
            patch[2] = LocalNormalization(patch[2].numpy())

            # dct & rgb
            fusion_dct_patch = torch.cat((patch, dct_patch), 0)

            fusion_dct_patches = fusion_dct_patches + (fusion_dct_patch,)
            patches = patches + (patch,)
            features = features + (feature,)

    return patches, features, fusion_dct_patches


class IQADataset(Dataset):
    def __init__(self, conf, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['im_dir']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))
        if status == 'all':
            all_index = []
            for i in range(len(ref_ids)):
                if ref_ids[i] in index:
                    all_index.append(i)
            self.index = all_index

        self.mos = Info['subjective_scores'][0, self.index]
        # im_names
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes() \
                        [::2].decode() for i in self.index]

        self.patches = ()
        self.features = ()
        self.fusion_dct_patches = ()
        # self.dct_patches = ()
        self.label = []
        # self.gabor = ()
        # self.label_std = []

        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im_path = os.path.join(im_dir, im_names[idx])
            im = self.loader(im_path)

            patches, features, fusion_dct_patches = NonOverlappingCropPatches(im, self.patch_size, self.stride)

            if status == 'train':
                self.patches = self.patches + patches
                # ==========
                self.fusion_dct_patches = self.fusion_dct_patches + fusion_dct_patches
                # ==========
                self.features = self.features + features

                for i in range(len(patches)):
                    self.label.append(self.mos[idx])

            else:
                self.patches = self.patches + (torch.stack(patches),)
                # ==========
                self.fusion_dct_patches = self.fusion_dct_patches + (torch.stack(fusion_dct_patches),)
                # ==========
                features = torch.stack(features)
                self.features = self.features + (features,)

                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], self.features[idx],self.fusion_dct_patches[idx]), (torch.Tensor([self.label[idx]]))