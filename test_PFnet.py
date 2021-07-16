import os
import numpy as np
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from utils import utils_PFnet
from utils.utils_PFnet import get_cd_loss
from utils.utils_PFnet import get_emd_loss
from utils.utils_PFnet import get_repulsion_loss
from utils.ply_utils import save_ply
from dataset import Dataset
from torch.utils.data import DataLoader
from PPFnet import PPFNet

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=2048,help='0 means do not use else use with this weight')
parser.add_argument('--checkpoint', default='checkpoint/point_netG_skel300.pth', help="path to netG (to continue training)")
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument("--results_saved_dir", default='result', help='results_saved_dir [default: results/training_results]')
opt = parser.parse_args()
print(opt)

test_dset = Dataset(npoint=opt.pnum, use_norm=True, split='test', is_training=True)
test_dataloader = DataLoader(test_dset, batch_size=opt.batchSize, shuffle=True, pin_memory=True, num_workers=opt.workers)

length = len(test_dataloader)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PPFNet(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
model = torch.nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(opt.checkpoint,map_location=lambda storage, location: storage)['state_dict'])   
model.eval()

alpha1 = 0.5
alpha2 = 0.5

loss_list = []
loss_fullpoint_list = []
loss_skeleton_list = []
loss_center_list = []
loss_repulsion_list = []

for i, data in enumerate(test_dataloader, 0):
    
    input_data, gt_fullpoint, gt_skeleton, radius_data = data
            
    real_center = gt_skeleton.float().to(device) 
    input_cropped1 = input_data.float().to(device) 
    full_point  = gt_fullpoint.float().to(device)
    radius_data = radius_data.float().to(device)

    real_center = Variable(real_center,requires_grad=True)

    real_center_key1_idx = utils_PFnet.farthest_point_sample(real_center,64,RAN = False)
    real_center_key1 = utils_PFnet.index_points(real_center,real_center_key1_idx)
    real_center_key1 =Variable(real_center_key1,requires_grad=True) 

    real_center_key2_idx = utils_PFnet.farthest_point_sample(real_center,128,RAN = True)
    real_center_key2 = utils_PFnet.index_points(real_center,real_center_key2_idx)
    real_center_key2 =Variable(real_center_key2,requires_grad=True)

    full_point = Variable(full_point,requires_grad=True)

    full_point_key1_idx = utils_PFnet.farthest_point_sample(full_point,512,RAN = False)
    full_point_key1 = utils_PFnet.index_points(full_point,full_point_key1_idx)
    full_point_key1 =Variable(full_point_key1,requires_grad=True)  

    full_point_key2_idx = utils_PFnet.farthest_point_sample(full_point,1024,RAN = True)
    full_point_key2 = utils_PFnet.index_points(full_point,full_point_key2_idx)
    full_point_key2 =Variable(full_point_key2,requires_grad=True)  

    input_cropped2_idx = utils_PFnet.farthest_point_sample(input_cropped1,opt.point_scales_list[1],RAN = True)
    input_cropped2     = utils_PFnet.index_points(input_cropped1,input_cropped2_idx)
    input_cropped3_idx = utils_PFnet.farthest_point_sample(input_cropped1,opt.point_scales_list[2],RAN = False)
    input_cropped3     = utils_PFnet.index_points(input_cropped1,input_cropped3_idx)

    input_cropped1 = Variable(input_cropped1,requires_grad=True) 
    input_cropped2 = Variable(input_cropped2,requires_grad=True)
    input_cropped3 = Variable(input_cropped3,requires_grad=True)
    input_cropped2 = input_cropped2.to(device)  
    input_cropped3 = input_cropped3.to(device)    
    input_cropped  = [input_cropped1,input_cropped2,input_cropped3] 

    skeleton_center1,skeleton_center2,skeleton, displacement_center1,displacement_center2,displacement  =model(input_cropped)
    
    pred_full = displacement + skeleton
    pred_full_center1 = displacement_center1 + skeleton_center1
    pred_full_center2 = displacement_center2 + skeleton_center2
    
    fullpoint_loss = 1500*get_emd_loss(pred_full, full_point,radius_data)
    skeleton_loss = 1000*get_cd_loss(skeleton,real_center,radius_data)

    center_loss = 10*(alpha1*get_emd_loss(skeleton_center1,real_center_key1, radius_data)\
    + alpha2*get_emd_loss(skeleton_center2,real_center_key2, radius_data)\
    + alpha1*get_cd_loss(pred_full_center1,full_point_key1, radius_data)\
    + alpha2*get_cd_loss(pred_full_center2,full_point_key2, radius_data))

    repulsion_loss = get_repulsion_loss(pred_full)

    error = fullpoint_loss + skeleton_loss + center_loss + repulsion_loss

    loss_list.append(error.item())
    loss_fullpoint_list.append(fullpoint_loss.item())
    loss_skeleton_list.append(skeleton_loss.item())
    loss_center_list.append(center_loss.item())
    loss_repulsion_list.append(repulsion_loss.item())

    print('[%d/%d] Loss_G: %.4f  full point loss: %.4f  skeleton loss: %.4f center loss: %.4f repulsion loss: %.4f'
        % (i, length, error, fullpoint_loss, skeleton_loss, center_loss, repulsion_loss))

    input_data = input_cropped1.data.cpu().numpy() 
    pred_skeleton = skeleton.data.cpu().numpy()
    pred_full = pred_full.data.cpu().numpy()
    gt_point = full_point.data.cpu().numpy()
    gt_skeleton = real_center.data.cpu().numpy()

    for iter in range(opt.batchSize):
        save_ply(os.path.join(opt.results_saved_dir, '{}_th_{}input_partial.ply'.format(i, iter)), input_data[iter])
        save_ply(os.path.join(opt.results_saved_dir, '{}_th_{}gt_skeleton.ply'.format(i,iter)), gt_skeleton[iter])
        save_ply(os.path.join(opt.results_saved_dir, '{}_th_{}gt_fullpoint.ply'.format(i,iter)), gt_point[iter])
        save_ply(os.path.join(opt.results_saved_dir, '{}_th_{}pred_fullpoint.ply'.format(i,iter)), pred_full[iter])
        save_ply(os.path.join(opt.results_saved_dir, '{}_th_{}pred_skeleton.ply'.format(i,iter)), pred_skeleton[iter])

print('average loss: Loss_G: %.4f  full point loss: %.4f  skeleton loss: %.4f center loss: %.4f repulsion loss: %.4f'
    % (np.mean(loss_list), np.mean(loss_fullpoint_list), np.mean(loss_skeleton_list), np.mean(loss_center_list), np.mean(loss_repulsion_list))) 