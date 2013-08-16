import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

wp = np.loadtxt('../../matlab/world_points.txt')
cp = np.loadtxt('../../matlab/camera_points.txt')

fig = plt.figure();
ax = fig.add_subplot(111, projection='3d')

num_planes = wp.size/30
for i in range(0, num_planes):
    k = i*30
    ax.plot(wp[k:k+30,0], wp[k:k+30,1], wp[k:k+30,2])
    
dist = la.norm([cp[0,0]-cp[1,0],cp[0,1]-cp[1,1],cp[0,2]-cp[1,2]])
print(dist)

#ax.plot(camera_points[:,0], camera_points[:,1], camera_points[:,2])

# axes
ax.plot([0,100],[0,0],[0,0])
ax.plot([0,0],[0,100],[0,0])
ax.plot([0,0],[0,0],[0,100])

plt.show()
