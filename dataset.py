import torch.utils.data as torch_data
import h5py
import numpy as np
from utils import utils
from glob import glob

class Dataset(torch_data.Dataset):
    def __init__(self, h5_file_path='./datas/chair.h5',
                    skip_rate=1, npoint=2048, use_norm=True, split='train', is_training=True):
        super().__init__()
        
        self.npoint = npoint
        self.use_norm = use_norm
        self.is_training = is_training

        h5_file = h5py.File(h5_file_path)
        self.gtsk = h5_file['gt-sk'][:]
        self.gt = h5_file['gt-pl'][:] 
        self.input = h5_file['partial-pl'][:]
        
        if split in ['train', 'test']:
            with open('./datas/{}_list.txt'.format(split), 'r') as f:
                split_choice = [int(x) for x in f]
            self.gt = self.gt[split_choice, ...]
            self.input = self.input[split_choice, ...]
            self.gtsk = self.gtsk[split_choice, ...]
        elif split != 'all':
            raise NotImplementedError

        assert len(self.input) == len(self.gt) == len(self.gtsk), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid_gt = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        centroid_input = np.mean(self.input[..., :3], axis=1, keepdims=True)
        centroid_skeleton = np.mean(self.gtsk[..., :3], axis=1, keepdims=True)
        furthest_distance_gt = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid_gt) ** 2, axis=-1)), axis=1, keepdims=True)
        furthest_distance_input = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid_input) ** 2, axis=-1)), axis=1, keepdims=True)
        furthest_distance_skeleton = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid_skeleton) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance_gt[:, 0] 

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid_gt
            self.gt[..., :3] /= np.expand_dims(furthest_distance_gt, axis=-1)
            self.input[..., :3] -= centroid_input
            self.input[..., :3] /= np.expand_dims(furthest_distance_input, axis=-1)
            self.gtsk[..., :3] -= centroid_skeleton
            self.gtsk[..., :3] /= np.expand_dims(furthest_distance_skeleton, axis =-1)

        self.input = self.input[::skip_rate]
        self.gt = self.gt[::skip_rate]
        self.gtsk = self.gtsk[::skip_rate]
        self.radius = self.radius[::skip_rate]

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        sk_data = self.gtsk[index]
        radius_data = np.array([self.radius[index]])

        # data augmentation
        # if self.use_norm:
        #     if not self.is_training:
        #         return input_data, gt_data, sk_data, radius_data
            
        #     #for data aug
        #     input_data, sk_data, gt_data, scale = utils.random_scale_point_cloud_and_gt_new(input_data, sk_data, gt_data,
        #                                                                        scale_low=0.9, scale_high=1.1)
        #     input_data, sk_data, gt_data = utils.shift_point_cloud_and_gt_new(input_data, sk_data, gt_data, shift_range=0.1)
        #     radius_data = radius_data * scale

        #     #for input aug
        #     if np.random.rand() > 0.5:
        #         input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
        #     if np.random.rand() > 0.5:
        #         input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        # else:
        #     raise NotImplementedError

        return input_data, gt_data, sk_data, radius_data
