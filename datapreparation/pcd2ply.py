#!/usr/bin/env python
# coding: utf-8

# In[1]:


import open3d as o3d


# In[2]:


import numpy as np


# In[10]:


import os


# In[27]:


path = os.getcwd()


# In[28]:


dir_list = os.listdir(path)


# In[29]:


for file in dir_list:
   file_path = os.path.join(os.path.abspath(path), file)
   if file_path.endswith(".pcd"):
       point = o3d.io.read_point_cloud(file)
       o3d.io.write_point_cloud(file.split(".")[0]+".ply",house,write_ascii=True)
       

