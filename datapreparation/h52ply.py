#!/usr/bin/env python
# coding: utf-8

# In[1]:


import open3d as o3d
import numpy as np
import os
import h5py


# In[22]:


path = os.getcwd()


# In[23]:


dir_list = os.listdir(path)


# In[24]:


for file in dir_list:
   file_path = os.path.join(os.path.abspath(path), file)
   if file_path.endswith(".h5"):
       f = h5py.File(file,'r')
       points=np.array(f['data'])
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(points)
       o3d.io.write_point_cloud(file.split(".")[0]+".ply",pcd,write_ascii=True)
       

