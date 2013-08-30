import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la
import cv
import cv2

ref = 1

show_cams = 0
show_pts = 1

x = 6
y = 9
phys = 6.0

if (ref):
    wp = np.loadtxt('../../matlab/world_points.txt')

cp = np.loadtxt('../../matlab/camera_points.txt')
f = np.loadtxt('../../matlab/f.txt')
p = np.loadtxt('../../matlab/plane_params.txt')
print p

fig = plt.figure();
ax = fig.add_subplot(111, projection='3d')

points = x*y

print('Focal Lengths:')
for i in range(0, f.size):
    print(f[i])
print('\n')

zavg = np.array([])
size = np.array([])

if not (ref):
    num_planes = wp[:,0].size/points
else:
    num_planes = p[:,0].size

if not (ref):

    for i in range(0, num_planes):
    
        k = i*points
        
        if (show_pts):
            ax.plot(wp[k:k+points,0], wp[k:k+points,1], wp[k:k+points,2])

            zavg = np.append( zavg, np.mean(wp[k:k+points,2]) )
            
            l = k+x-1
            distx = la.norm([wp[k,0]-wp[l,0], wp[k,1]-wp[l,1], wp[k,2]-wp[l,2]])/(x-1)
            
            l = k+x*(y-1)
            disty = la.norm([wp[k,0]-wp[l,0], wp[k,1]-wp[l,1], wp[k,2]-wp[l,2]])/(y-1)
    
            print('Grid: ' + repr(distx) + ' x ' + repr(disty))

            size = np.append( size, np.mean([distx, disty]) )

if (show_cams):
    if (cp.size>3):
        ax.scatter(cp[:,0],cp[:,1],cp[:,2])
    else:
        ax.scatter(cp[0],cp[1],cp[2])

gridx = np.array([])
gridy = np.array([])
gridz = np.array([])
gridt = np.array([])

for i in range(0,y):
    for j in range(0,x):
        gridx = np.append(gridx, j*phys)
        gridy = np.append(gridy, i*phys)
        gridz = np.append(gridz, 0.0)
        gridt = np.append(gridt, 1.0)

grid = np.array([gridx, gridy, gridz, gridt])

rt = np.zeros((3,4))
for i in range(0, num_planes):

    rot, jac = cv2.Rodrigues(p[i,0:3])
    rt[0:3,0:3] = rot
    for j in range(0,3):
        rt[j,3] = p[i,3+j]

    trans = cv2.gemm(rt, grid, 1.0, grid, 0)
    ax.plot(trans[0,:], trans[1,:], trans[2,:])

# axes
ax.plot([0,100],[0,0],[0,0])
ax.plot([0,0],[0,100],[0,0])
ax.plot([0,0],[0,0],[0,100])

# plotting size vs z
#plt.figure()
#plt.plot(zavg, size)

plt.show()
