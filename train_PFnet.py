import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from utils import utils_PFnet
from utils.utils_PFnet import get_cd_loss
from utils.utils_PFnet import get_emd_loss
from utils.utils_PFnet import get_repulsion_loss
import numpy as np
from PPFnet import PPFNet
from dataset import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=2048,help='0 means do not use else use with this weight')
parser.add_argument('--niter', type=int, default=301, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate in training')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--checkpoint', default='', help="path to checkpoint (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048,1024,512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument("--results_saved_dir", default='results', help='results_saved_dir [default: results/training_results]')
opt = parser.parse_args()
print(opt)

USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = PPFNet(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
cudnn.benchmark = True
resume_epoch=0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)
    model.to(device) 
    model.apply(weights_init_normal)
if opt.checkpoint != '' :
    model.load_state_dict(torch.load(opt.checkpoint,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.checkpoint)['epoch']
        
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

dset = Dataset(npoint=opt.pnum, use_norm=True, split='train', is_training=True)
dataloader = DataLoader(dset, batch_size=opt.batchSize, shuffle=True, pin_memory=True, num_workers=opt.workers)
# test_dset = Dataset(npoint=opt.pnum, use_random=True, use_norm=True, split='test', is_training=True)
# test_dataloader = DataLoader(test_dset, batch_size=opt.batchSize, shuffle=True, pin_memory=True, num_workers=opt.workers)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)

def run_epoch(model, optimizer, dataloader, train):
    
    if train:
        model = model.train()
    else:
        model = model.eval()

    alpha1 = 0.5
    alpha2 = 0.5

    loss_list = []
    loss_fullpoint_list = []
    loss_skeleton_list = []
    loss_center_list = []
    loss_repulsion_list = []

    for i, data in enumerate(dataloader, 0):
        
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

        if train:
            model.zero_grad()
        
        with torch.set_grad_enabled(train):

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
            
            if train:
                error.backward()
                optimizer.step()

        loss_list.append(error.item())
        loss_fullpoint_list.append(fullpoint_loss.item())
        loss_skeleton_list.append(skeleton_loss.item())
        loss_center_list.append(center_loss.item())
        loss_repulsion_list.append(repulsion_loss.item())

        print('[%d/%d][%d/%d] Loss_G: %.4f  full point loss: %.4f  skeleton loss: %.4f center loss: %.4f repulsion loss: %.4f'
                % (epoch, opt.niter, i, len(dataloader), 
                error, fullpoint_loss, skeleton_loss, center_loss, repulsion_loss))

    return loss_list, loss_fullpoint_list, loss_skeleton_list, loss_center_list, loss_repulsion_list


for epoch in range(resume_epoch,opt.niter):

    loss_list, loss_fullpoint_list, loss_skeleton_list, loss_center_list, loss_repulsion_list = run_epoch(model, optimizer, dataloader, train=True)
    scheduler.step()

    print(' training loss: -- epoch {}, loss {:.4f}, loss fullPoint  {:.5f}, loss Skeleton{:.5f},loss center{:.5f}, loss_repu{:.5f}'.format(
        epoch, np.mean(loss_list), np.mean(loss_fullpoint_list), np.mean(loss_skeleton_list), \
        np.mean(loss_center_list), np.mean(loss_repulsion_list)))
    
    # loss_list, loss_fullpoint_list, loss_skeleton_list, loss_center_list, loss_repulsion_list = run_epoch(point_netG_skel, None, test_dataloader, train=False)

    # print(' val loss: -- epoch {}, loss {:.4f}, loss fullPoint  {:.5f}, loss Skeleton{:.5f},loss center{:.5f}, loss_repu{:.5f}'.format(
    #     epoch, np.mean(loss_list), np.mean(loss_fullpoint_list), np.mean(loss_skeleton_list), \
    #     np.mean(loss_center_list), np.mean(loss_repulsion_list)))

    if epoch% 30== 0:   
        torch.save({'epoch':epoch+1,
                    'state_dict':model.state_dict()},
                    'Checkpoint/PF_basis'+str(epoch)+'.pth' )
 

    
        
