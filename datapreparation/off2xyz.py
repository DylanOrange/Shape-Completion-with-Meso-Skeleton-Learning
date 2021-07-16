#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np


# In[23]:


def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        print(first)
        if (len(first) > 4): # Too handle error in .off like OFF492 312 0
            n, m, c = first[3:].split(' ')[:]    
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]    #rstrip() delete char(space)
            n = int(n)
            #m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        #for i in range(m):
            #value = f.readline().rstrip().split(' ')
            #faces.append([int(x) for x in value])
        points = np.array(points)
        #faces = np.array(faces)
    return points#, faces


# In[24]:


points= read_off('test.off')


# In[39]:


def write_xyz(points):
    with open("camel.xyz", 'w') as f:
        for i in range(50):
            for each in points:
                f.writelines(str(format(each[0],'6f')) + " " + str(format(each[1],'6f')) + " " + str(format(each[2],'6f')) )
                f.write('\n')


# In[40]:


write_xyz(points)


# In[ ]:




