#!/usr/bin/env python
# coding: utf-8

# In[89]:


import open3d as o3d
import numpy as np
import os
import h5py
import tqdm
import numpy as np


# In[200]:


np.set_printoptions(suppress = True)


# In[227]:


def read_ply(filename):
    points=[]
    with open(filename,'r') as f:
        for line in f.readlines()[8:]:
            value = line.rstrip().split(" ")
            points.append([float(format(float(x),'6f')) for x in value])
        points = np.array(points)
    return points
            
def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4): # Too handle error in .off like OFF492 312 0
            n, m, c = first[3:].split(' ')[:]    
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]    #rstrip() delete char(space)
            n = int(n)
            #m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(format(float(x),'6f')) for x in value])
        #for i in range(m):
            #value = f.readline().rstrip().split(' ')
            #faces.append([int(x) for x in value])
        points = np.array(points)
        #faces = np.array(faces)
    return points
def get_ply(filepath):
    file_list = os.listdir(filepath)
    list.sort(file_list)
    name = []
    B = len(file_list)
    points_ply = np.random.randn(B,2048,3)
    i = 0
    for file in tqdm.tqdm(file_list):
        if file.split('.')[1]=='ply':
            name.append(file)
            path = (os.path.join(filepath,file))
            temp_points = read_ply(path)
            points_ply[i,:,:] = temp_points
            i+=1
    return name, points_ply
def get_off(filepath):
    file_list = os.listdir(filepath)
    list.sort(file_list)
    name = []
    B = len(file_list)
    points_off = np.random.randn(B,100,3)
    i = 0
    for file in tqdm.tqdm(file_list):
        if file.split('.')[1]=='off':
            name.append(file)
            path = (os.path.join(filepath,file))
            temp_points = read_off(path)
            points_off[i,:,:] = temp_points
            i+=1
    return name, points_off
def createh5(name,points_partial,points_gt,points_skeleton):
    f=h5py.File("ppunet.h5","w")
    name = np.array(name)
    d1=f.create_dataset("names", name.shape, dtype = h5py.special_dtype(vlen=str))
    d1[:] = name
    d2=f.create_dataset("partial-pl",data = points_partial)
    d3=f.create_dataset("gt-pl",data = points_gt)
    d4=f.create_dataset("gt-sk",data = points_skeleton)
    for key in f.keys():
        print(f[key].name)
        print(f[key].value)
def show(points):
    plt.figure(figsize=(60, 60))
    x, y, z = points[:,0], points[:,1], points[:,2]
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    ax.scatter(x[:2048], y[:2048], z[:2048], c='r')  # 绘制数据点
    #ax.scatter(x[:2048], y[:2048], z[:2048], c='r')
    #ax.scatter(x[:2048], y[:2048], z[:2048], c='g')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()


# In[228]:


if __name__ == "__main__":
    path = os.getcwd()
    file_path_gt = path + '/gt-pl'
    file_path_partial = path + '/partial-pl'
    file_path_skeleton = path+ '/gt-sk'
    name_partial, points_partial = get_ply(file_path_partial)
    name_gt, points_gt = get_ply(file_path_gt)
    name_skeleton, points_skeleton = get_off(file_path_skeleton)
    createh5(name_gt,points_partial,points_gt,points_skeleton)

