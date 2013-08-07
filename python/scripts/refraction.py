import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# all units in mm

p = np.array([0,0,-250])
c = np.array([5,5,500])

t = 5 # assume glass of t thickness lies at z = 0 depth t in positive z direction

n1 = 1.00 # air
n2 = 1.50 # glass
n3 = 1.33 # water

# initial guess for ray of intersection using a straight line

x1 = c[0]+()*(t-cam[2])/()
y1 = c[1]+()*(t-cam[2])/()
z1 = t

'''
x2 = c[0]+a*(-cam[2])/c
y2 = c[1]+b*(-cam[2])/c
z2 = 0
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot([c[0],x1,x2,p[0]],[c[1],y1,y2,p[1]],[c[2],z1,z2,p[2]])

plt.show()

'''
ra = 
rb =
rp = 

da =
db =
dp =

f = 
g = 
'''
