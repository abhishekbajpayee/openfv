import numpy as np
import subprocess
import os
import copy
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

frames = 30

data = np.loadtxt('../temp/vortex_particles.txt')

fig = plt.figure();

start = 0

xlim = 80
zlim = 50

for i in range(0,frames):

    num = int(data[start])
    ax = fig.add_subplot(111, projection='3d')

    #print(num)
    for j in range(1,num+1):
        k = start + ((j-1)*3)
        #print(data[k+1],data[k+2],data[k+3])
        ax.scatter(data[k+1],data[k+2],data[k+3],s=1)

    start = start + (num*3) + 1

    #plt.show()
    ax.set_xlim(-xlim,xlim)
    ax.set_ylim(-xlim,xlim)
    ax.set_zlim(-zlim,zlim)

    filename = str('frames/%03d' % i) + '.png'
    plt.savefig(filename, dpi=300)
    print filename + " written"

    plt.clf()

write_movie = 1
if write_movie:
    
    command = ('mencoder',
               'mf://frames/*.png',
               '-mf',
               'type=png:w=800:h=600:fps=30',
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
