#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os 

file_path = os.getcwd() 
path_list = os.listdir(file_path)

path_name = []

for i in path_list:
    file_path = os.path.join(os.path.abspath(file_path), i)
    if file_path.endswith(".ply"):
        path_name.append(i.split(".")[0]) 

path_name.sort() 

for file_name in path_name:
    with open("test.txt", "a") as file:
        file.write("test/" + file_name + "\n")
    file.close()


# In[ ]:





# In[ ]:




