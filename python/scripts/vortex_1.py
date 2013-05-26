import numpy as np
import subprocess
import os

import sys
sys.path.append("/home/ab9/Blender/modules")

import ptv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num = 10
box = 50

R = 12.5
Z = 0
T = 50.0
r = 5.0
dt = 1.0/30.0

points = ptv.generate_points(num,box,box,box)
print "x=..."
print(points[0])
print "y=..."
print(points[1])
print "z=..."
print(points[2])

for i in range(0,5,1):

    points2 = ptv.advance_vortex_flow(points,R,Z,T,r,dt)
    points = points2
