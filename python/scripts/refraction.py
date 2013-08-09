import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy.linalg as la

# all units in mm

p = np.array([0,0,5.0])
c = np.array([5.0,5.0,-10.0])

t = 5 # assume glass of t thickness lies at z = 0 depth t in positive z direction

n1 = 1.00 # air
n2 = 1.50 # glass
n3 = 1.33 # water

# initial guess for ray of intersection using a straight line

d = la.norm(p-c)
pc = p-c
n = (p-c)/d

b = np.array([ c[0] + pc[0]*-c[2]/pc[2], c[1] + pc[1]*-c[2]/pc[2], 0 ])
a = np.array([ c[0] + pc[0]*(-t-c[2])/pc[2], c[1] + pc[1]*(-t-c[2])/pc[2], -t ]) 
print(a)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot([c[0],a[0],b[0],p[0]],[c[1],a[1],b[1],p[1]],[c[2],a[2],b[2],p[2]])
ax.hold(1)

rp = la.norm([ p[0]-c[0], p[1]-c[1] ])
dp = p[2]-b[2]
phi = np.arctan2(pc[1],pc[0])

# refractive stuff
ra = la.norm([ a[0]-c[0], a[1]-c[1] ])
rb = la.norm([ b[0]-c[0], b[1]-c[1] ])

da = a[2]-c[2]
db = b[2]-a[2]

for i in range(0,10):

    f = ( ra/la.norm([ra,da]) ) - ( (n2/n1)*(rb-ra)/la.norm([rb-ra,db]) )
    g = ( (rb-ra)/la.norm([rb-ra,db]) ) - ( (n3/n2)*(rp-rb)/la.norm([rp-rb,dp]) )

    dfdra = ( 1/la.norm([ra,da]) ) \
            - ( ra**2/la.norm([ra,da])**3 ) \
            + ( (n2/n2)/la.norm([ra-rb,db]) ) \
            - ( (n2/n1)*(ra-rb)*(2*ra-2*rb)/(2*la.norm([ra-rb,db])**3) )

    dfdrb = ( (n2/n1)*(ra-rb)*(2*ra-2*rb)/(2*la.norm([ra-rb,db])**3) ) \
            - ( (n2/n2)/la.norm([ra-rb,db]) )

    dgdra = ( (ra-rb)*(2*ra-2*rb)/(2*la.norm([ra-rb,db])**3) ) \
            - ( 1/la.norm([ra-rb,db]) )

    dgdrb = ( 1/la.norm([ra-rb,db]) ) \
            + ( (n3/n2)/la.norm([rb-rp,dp]) ) \
            - ( (ra-rb)*(2*ra-2*rb)/(2*la.norm([ra-rb,db])**3) ) \
            - ( (n3/n2)*(rb-rp)*(2*rb-2*rp)/(2*la.norm([rb-rp,dp])**3) )

    ra_n = ra - ( (f*dgdrb - g*dfdrb)/(dfdra*dgdrb - dfdrb*dgdra) )
    rb_n = rb - ( (g*dfdra - f*dgdra)/(dfdra*dgdrb - dfdrb*dgdra) )

    ra = ra_n
    rb = rb_n

    print(f)

a[0] = ra*np.cos(phi) + c[0]
a[1] = ra*np.sin(phi) + c[1]
b[0] = rb*np.cos(phi) + c[0]
b[1] = rb*np.sin(phi) + c[1]

ax.plot([c[0],a[0],b[0],p[0]],[c[1],a[1],b[1],p[1]],[c[2],a[2],b[2],p[2]])
plt.show()
