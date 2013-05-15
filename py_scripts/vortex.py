import numpy as np
import subprocess
import os
import copy

import sys
sys.path.append("/home/ab9/Blender/modules")

import ptv_math as pm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num = 500
box = 50

fps = 30.0
time = 1.0 # seconds
frames = int(fps*time)

R = 12.5
Z = 0.0
T = 50.0
r = 5.0
dt = 1.0/fps

vz = 20.0

fig = plt.figure();

points = pm.generate_points(num,box,box,box)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[0],points[1],points[2],s=1)
plt.xlabel('x')
plt.ylabel('y')
ax.set_xlim(-25,25)
ax.set_ylim(-25,25)
ax.set_zlim(-25,25)

filename = str('frames/%03d' % 0) + '.png'
plt.savefig(filename, dpi=100)
print filename + " written"

'''
print "x=..."
print(points[0])
print "y=..."
print(points[1])
print "z=..."
print(points[2])
'''

next = pm.advance_vortex_flow

for i in range(0,frames,1):
    
    #Z = Z + vz*dt
    points2 = next(points,R,Z,T,r,dt)
    print(points2)
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points2[0],points2[1],points2[2],s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_xlim(-25,25)
    ax.set_ylim(-25,25)
    ax.set_zlim(-25,25)

    filename = str('frames/%03d' % (i+1)) + '.png'
    plt.savefig(filename, dpi=100)
    print filename + " written"
    
    plt.clf()

    points = np.copy(points2)
    points2 = []

#plt.show()

write_movie = 1
if write_movie:
    
    command = ('mencoder',
               'mf://frames/*.png',
               '-mf',
               'type=png:w=800:h=600:fps=10',
               '-ovc',
               'lavc',
               '-lavcopts',
               'vcodec=mpeg4',
               '-oac',
               'copy',
               '-o',
               'output.avi')
    
    print "\n\nabout to execute:\n%s\n\n" % ' '.join(command)
    subprocess.check_call(command)
    
    print "\n\n The movie was written to 'output.avi'"
    
