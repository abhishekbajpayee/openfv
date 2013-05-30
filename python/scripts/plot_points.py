import numpy as np
import matplotlib.pyplot as plt

'''
data = np.loadtxt('cx_v_z.txt')
data1 = np.loadtxt('cx_v_z_add.txt')

plt.plot(data[:,0],data[:,1],'b--',data1[:,0],data1[:,1],'r--',data[:,0],data[:,3],'b:',data1[:,0],data1[:,3],'r:',)
plt.xlabel('z depth [mm]')
plt.ylabel('Cross Correlation [ ]')
#plt.ylim([0,1.1])
plt.legend(('Multiplicative (T=0)','Additive (T=0)','Multiplicative (T=100)','Additive (T=100)'),loc=0)

'''
'''
data = np.loadtxt('cx_v_dz.txt')
data1 = np.loadtxt('cx_v_dz_add.txt')

plt.plot(data[:,0],data[:,1],'b-o',data1[:,0],data1[:,1],'r-o',)
plt.xlabel('dz [mm]')
plt.ylabel('Cross Correlation [ ]')
plt.xscale('log')
#plt.ylim([0,1.1])
plt.legend(('Multiplicative','Additive'),loc=3)
'''
'''
for i in range(10,20):

    filename = "data_files/i_vs_z/i_vs_z_mult_%d.txt" % i
    data = np.loadtxt(filename)

    data[:,0] = data[:,0]-min(data[:,0])
    data[:,0] = data[:,0]/max(data[:,0])
    data[:,1] = data[:,1]/max(data[:,1])

    plt.scatter(data[:,0],data[:,1],s=10)

x = np.linspace(0,1.0,100)
mu = 0.5
sigma = 0.4
dist = (1/(sigma*np.sqrt(2*np.pi)))*np.exp( -np.square(x-mu) / (2*(sigma**2)) )

plt.plot(x,dist,'r--')

plt.xlim((0,1.0))
plt.ylim((0.4,1.05))
plt.xlabel('Normalized depth [ ]')
plt.ylabel('Normalized Intensity [ ]')
'''
line = np.array([0,1000])

x = np.array([100,250,500,750,1000])
t50 = np.array([116,354,962,1561,3431])
t90 = np.array([100,248,499,748,998])
t150 = np.array([0,8,15,23,72])

mult = np.array([100,248,497,310,987])

plt.plot(line,line,'-', x,t50,'--s', x,t90,'--s', x,t150,'--s', x,mult,'--s', markersize=5)

plt.ylim((0,1000))
plt.xlim((0,1000))
plt.xlabel('Number of particles seeded')
plt.ylabel('Number of particles detected')
plt.legend(('Actual Number','Additive (T=50)','Additive (T=90)','Additive (T=150)','Multiplicative'),loc=0)
plt.show()
