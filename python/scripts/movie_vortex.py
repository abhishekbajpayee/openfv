import numpy as np
import subprocess
import os
import copy
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num = 1000
frames = 30
size = 150

points = np.loadtxt('../matlab/vortex_points.txt')

fig = plt.figure();

for i in range(0,frames,1):

    j = i*num
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[j:j+num,0],points[j:j+num,1],points[j:j+num,2],s=1)
    ax.set_xlim(-size*0.5,size*0.5)
    ax.set_ylim(-size*0.5,size*0.5)
    ax.set_zlim(-size*0.5,size*0.5)

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
