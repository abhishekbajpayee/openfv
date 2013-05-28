import numpy as np
import matplotlib.pyplot as plt

x = np.array([10, 50, 100, 250, 500, 750, 1000])
t50 =  np.array([19, 97, 202, 515, 1248, 2117, 4036])
t90 =  np.array([9, 49, 99, 237, 494, 619, 1005])
t100 = np.array([9, 47, 90, 224, 449, 460, 917])

plt.plot(x,x,x,t50,x,t90,x,t100)

plt.ylim((0,1000))
plt.xlim((0,1000))

plt.show()
