import argparse
import os
from pickle import TRUE
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match
from dataset import Dataset
from PPUnet import PPUNet

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: logs/training_logs]')
parser.add_argument('--npoint', type=int, default=2048,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=1, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=300, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=True)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1.0) # for repulsion loss
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=True)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[100, 300])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument("--results_saved_dir", default='result', help='results_saved_dir [default: results]')
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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

    def forward(self, pred_fullpoint, gt_fullpoint, pred_skeleton, gt_skeleton, radius_data):
        return self.get_emd_loss(pred_fullpoint, gt_fullpoint, radius_data)*250, \
            self.get_cd_loss(pred_skeleton,gt_skeleton,radius_data)*25,\
            self.get_repulsion_loss(pred_fullpoint)

def get_optimizer():
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    else:
        raise NotImplementedError
    
    if args.use_decay:
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in args.decay_step_list:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * args.lr_decay
            return max(cur_decay, args.lr_clip / args.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
        return optimizer, lr_scheduler
    else:
        return optimizer, None


if __name__ == '__main__':
    train_dst = Dataset(npoint=args.npoint, use_norm=True, split='train', is_training=True)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)

    model = PPUNet(npoint=args.npoint, up_ratio=args.up_ratio, use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    model.cuda()

    load_model_dir = 'logs/training_logs_25025_airplane/punet_epoch_336.pth'  # here type in the model you want to continue training
    if os.path.exists(load_model_dir):
        checkpoint = torch.load(load_model_dir)
        model.load_state_dict(checkpoint['model_state'])
        start_epoch = checkpoint['epoch']
        print('load {} checkpoint successfully!'.format(start_epoch))
    else:
        print('No checkpoint detected, retrain')
        start_epoch = 0

    optimizer, lr_scheduler = get_optimizer()
    loss_func = UpsampleLoss(alpha=args.alpha)

    model.train()
    for epoch in range(start_epoch,args.max_epoch):
        loss_list = []
        emd_loss_list = []
        cd_loss_list = []
        repu_loss_list = []
        print('{}th_ epoch training starts!'.format(epoch))

        for iter, batch in enumerate(train_loader):
            input_data, gt_fullpoint, gt_skeleton, radius_data = batch
            optimizer.zero_grad()

            input_data = input_data.float().cuda()
            gt_skeleton = gt_skeleton.float().cuda()
            gt_fullpoint = gt_fullpoint.float().cuda()
            gt_fullpoint = gt_fullpoint[..., :3].contiguous()
            radius_data = radius_data.float().cuda()

            pred_skeleton, pred_fullpoint= model(input_data)
            emd_loss, cd_loss, repulsion_loss= loss_func(pred_fullpoint, gt_fullpoint, pred_skeleton, gt_skeleton, radius_data)
            loss = emd_loss + cd_loss + repulsion_loss

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            cd_loss_list.append(cd_loss.item())
            repu_loss_list.append(repulsion_loss.item())
            
        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, cd loss {:.4f}, repul loss{:.4f},lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(cd_loss_list), np.mean(repu_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if (epoch ) % 2 == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict()}
            save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
            torch.save(state, save_path)
