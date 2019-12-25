from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot as plt


x=np.arange(100, 800, 1)
y=np.arange(56, 1, -1)

X, Y = np.meshgrid(x, y)

Z = -np.log10(Y/X)

fig = plt.figure()
ax = Axes3D(fig)
plt.xlabel('x')
plt.ylabel('y')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
plt.show()