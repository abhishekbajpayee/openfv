import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('../temp/particles.txt')

num = int(data[0])

plt.hold(True)

for i in range(1,num+1):
    j = (i-1)*3
    plt.scatter(data[j+1],data[j+2])

plt.show()
