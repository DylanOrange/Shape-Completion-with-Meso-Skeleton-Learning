#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from pointnet2 import pointnet2_utils as pn2_utils
from chamfer_distance import chamfer_distance
from auction_match import auction_match
from knn_cuda import KNN

def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
    knn_obj = KNN(k=group_size, transpose_mode=transpose_mode)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx

def array2samples_distance(array1, array2):
    """
    arguments: 
        array1: the array, size: (num_point, num_feature)
        array2: the samples, size: (num_point, num_feature)
    returns:
        distances: each entry is the distance from a sample to array1 
    """
    num_point1, num_features1 = array1.shape
    num_point2, num_features2 = array2.shape
    expanded_array1 = array1.repeat(num_point2, 1)
    expanded_array2 = torch.reshape(
            torch.unsqueeze(array2, 1).repeat(1, num_point1, 1),
            (-1, num_features2))
    distances = (expanded_array1-expanded_array2)* (expanded_array1-expanded_array2)
#    distances = torch.sqrt(distances)
    distances = torch.sum(distances,dim=1)
    distances = torch.reshape(distances, (num_point2, num_point1))
    distances = torch.min(distances,dim=1)[0]
    distances = torch.mean(distances)
    return distances

def chamfer_distance_numpy(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist = dist + (0.5*av_dist1+0.5*av_dist2)/batch_size
    return dist*100

def chamfer_distance_numpy_test(array1, array2):
    batch_size, num_point, num_features = array1.shape
    dist_all = 0
    dist1 = 0
    dist2 = 0
    for i in range(batch_size):
        av_dist1 = array2samples_distance(array1[i], array2[i])
        av_dist2 = array2samples_distance(array2[i], array1[i])
        dist_all = dist_all + (av_dist1+av_dist2)/batch_size
        dist1 = dist1+av_dist1/batch_size
        dist2 = dist2+av_dist2/batch_size
    return dist_all, dist1, dist2

class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss,self).__init__()
        
    def forward(self,array1,array2):
        return chamfer_distance_numpy(array1,array2)

class PointLoss_test(nn.Module):
    def __init__(self):
        super(PointLoss_test,self).__init__()
        
    def forward(self,array1,array2):
        return chamfer_distance_numpy_test(array1,array2)

def distance_squre(p1,p2):
    tensor=torch.FloatTensor(p1)-torch.FloatTensor(p2)
    val=tensor.mul(tensor)
    val = val.sum()
    return val

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint,RAN = True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if RAN :
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)
    
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # distance = distance.double()
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def get_cd_loss(sk, skeleton, pcd_radius):
    cost_for, cost_bac = chamfer_distance(sk, skeleton)
    cost = torch.mean(cost_for )+ torch.mean(cost_bac)
    cost /= torch.mean(pcd_radius)
    return cost

def get_emd_loss(pointclouds, gt, pcd_radius):
    idx, _ = auction_match(pointclouds, gt)
    matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous()
    dist2 = (pointclouds - matched_out) ** 2
    dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
    dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
    dist2 /= pcd_radius
    return torch.mean(dist2)

def get_repulsion_loss(pred): #uniform distribution

    nn_size=5
    radius=0.07
    h=0.03
    eps=1e-12

    _, idx = knn_point(nn_size, pred, pred, transpose_mode=True)
    idx = idx[:, :, 1:].to(torch.int32) # remove first one
    idx = idx.contiguous() # B, N, nn

    pred = pred.transpose(1, 2).contiguous() # B, 3, N
    grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    grouped_points = grouped_points - pred.unsqueeze(-1)
    dist2 = torch.sum(grouped_points ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(eps).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / h ** 2)

    uniform_loss = torch.mean((radius - dist) * weight)
    return uniform_loss

if __name__=='__main__':
    a = torch.randn(64,256,3)
    b = torch.randn(64,256,3)
    c = torch.randn(64,64,3)
    d = torch.randn(64,64,3)
    radius = 1
    loss_emd = get_emd_loss(a,b,radius)
    # p = PointLoss()
    # print(p(a,b))
    # print(p(c,d)*0.25)