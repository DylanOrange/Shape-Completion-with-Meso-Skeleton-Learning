import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match
import open3d as o3d
from dataset import Dataset
import numpy as np
import importlib

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--up_ratio',  type=int,  default=1, help='Upsampling Ratio [default: 4]')
parser.add_argument("--use_bn", action='store_true', default=True)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument('--resume', default='checkpoint/PUonly/punet_epoch_300.pth', type=str)
parser.add_argument("--alpha", type=float, default=1.0)
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

def save_ply(save_dir,points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_dir,pcd,write_ascii=True)
class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps

    def get_emd_loss(self, pointclouds, gt, pcd_radius):
        idx, _ = auction_match(pointclouds, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pointclouds - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1)
        dist2 = torch.mean(dist2, dim=1, keepdims=True)
        dist2 /= pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, sk, skeleton, pcd_radius):
        cost_for, cost_bac = chamfer_distance(sk, skeleton)
        cost = torch.mean(cost_for) + torch.mean(cost_bac)
        return cost

    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32)
        idx = idx.contiguous()

        pred = pred.transpose(1, 2).contiguous()
        grouped_points = pn2_utils.grouping_operation(pred, idx)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        return uniform_loss

    def forward(self, pred_fullpoint, gt_fullpoint,radius_data):
        return self.get_emd_loss(pred_fullpoint, gt_fullpoint, radius_data)*250,self.get_repulsion_loss(pred_fullpoint)
if __name__ == '__main__':
    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=2048, up_ratio=args.up_ratio,use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state'])
    model.cuda()
    model.eval()
    test_dst = Dataset(npoint=2048, use_norm=True, split='test', is_training=False)

    test_loader = DataLoader(test_dst, batch_size=1,
                        shuffle=False, pin_memory=True, num_workers=0)
    loss_list = []
    emd_loss_list = []
    repu_loss_list = []
    loss_func = UpsampleLoss(alpha=args.alpha)
    for itr, batch in enumerate(test_loader):
        input_data, gt_fullpoint, _, radius_data = batch
        input_data = input_data.float().cuda()
        gt_fullpoint = gt_fullpoint.float().cuda()
        gt_fullpoint = gt_fullpoint[..., :3].contiguous()
        radius_data = radius_data.float().cuda()
        pred_fullpoint = model(input_data)
        emd_loss, repulsion_loss= loss_func(pred_fullpoint, gt_fullpoint,radius_data)
        loss = emd_loss + repulsion_loss
        loss_list.append(loss.item())
        emd_loss_list.append(emd_loss.item())
        repu_loss_list.append(repulsion_loss.item())
        pred_fullpoint = pred_fullpoint.data.cpu().numpy()
        input_data = input_data.data.cpu().numpy()
        gt_fullpoint = gt_fullpoint.data.cpu().numpy()

        save_ply(os.path.join(args.save_dir, '{}_input.ply'.format(itr)), input_data[0, :, :3])
        save_ply(os.path.join(args.save_dir, '{}_input_gt.ply'.format(itr)), gt_fullpoint[0, :, :3])
        save_ply(os.path.join(args.save_dir, '{}_output_pointclouds.ply'.format(itr)), pred_fullpoint[0])
        f=open(os.path.join(args.save_dir,'test_loss.txt'),'a')
        f.write('\n'+' loss {:.4f}, weighted emd loss {:.4f}, repul loss{:.4f}, iter{}'.format(float(loss), float(emd_loss),float(repulsion_loss),int(itr)))
        f.close()
    print(' loss {:.4f}, weighted emd loss {:.4f}, repul loss{:.4f}.'.format(np.mean(loss_list), \
    np.mean(emd_loss_list), np.mean(repu_loss_list)))
