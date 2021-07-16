import torch
import torch.nn as nn
from models.punet import PUNet
def get_model(npoint=2048, up_ratio=1, use_normal=False, use_bn=False, use_res=False):
    return PPUNet(npoint, up_ratio, use_normal, use_bn, use_res)
class PPUNet(nn.Module):
    def __init__(self, npoint = 2048, up_ratio = 1, use_normal =False, use_bn = False, use_res = False):
        super().__init__()
        self.punet1 = PUNet(npoint,up_ratio,use_normal,use_bn,use_res)
        self.punet2 = PUNet(npoint,up_ratio,use_normal,use_bn,use_res)
    def forward(self, input):
        skeleton = self.punet1(input)
        displacement = self.punet2(input)
        pointclouds = skeleton + displacement
        return skeleton,pointclouds

if __name__ == '__main__':
    model = PPUNet(up_ratio = 2, use_normal = False).cuda()
    input = torch.randn([1,1024,3]).float().cuda()
    skeleton, pointclouds = model(input)
    print(skeleton.shape)
    print(pointclouds.shape)