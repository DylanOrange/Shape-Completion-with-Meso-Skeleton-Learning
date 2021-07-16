import torch.nn as nn
from models.model_PFNet import _netG

class PPFNet(nn.Module):
    def __init__(self, num_scales, each_scales_size, point_scales_list, crop_point_num):
        super().__init__()
        self.point_netG_skel = _netG(num_scales, each_scales_size, point_scales_list, crop_point_num)
        self.point_netG_full = _netG(num_scales, each_scales_size, point_scales_list, crop_point_num)
    def forward(self, input):
        skeleton_center1,skeleton_center2,skeleton = self.point_netG_skel(input)
        displacement_center1,displacement_center2,displacement = self.point_netG_full(input)
        return skeleton_center1,skeleton_center2,skeleton, displacement_center1,displacement_center2,displacement