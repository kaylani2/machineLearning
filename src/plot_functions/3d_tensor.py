import numpy as np
xyz = np.array (np.random.random ( (100,3)))
x = xyz[:,0]
y = xyz[:,1]
z = xyz[:,2]*100

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure (figsize = (10, 10))
ax = fig.add_subplot (111, projection = '3d')
ax.set_ylabel ('$F1,\ F2,\ F3\ |\ Label$', fontsize = 20, rotation = 180)
ax.set_zlabel ('$Samples$', fontsize = 20, rotation = 0)

ax.plot ([1.], [1.], [1.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$7$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [1.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$8$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [1.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$9$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [2.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$4$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [2.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$5$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [2.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$6$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$1$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$2$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$3$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [4.], [1.], markerfacecolor = 'r', markeredgecolor = 'r', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [4.], [2.], markerfacecolor = 'g', markeredgecolor = 'g', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [4.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = 'o', markersize = 9, alpha = 0.6)


ax.plot ([1.], [1.], [4.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$70$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [4.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$80$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [4.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$90$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$40$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$50$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$60$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [6.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$10$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [6.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$20$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [6.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$30$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [4.], [4.], markerfacecolor = 'r', markeredgecolor = 'r', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [4.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [4.], [6.], markerfacecolor = 'b', markeredgecolor = 'b', marker = 'o', markersize = 9, alpha = 0.6)

plt.show ()


fig = plt.figure (figsize = (10, 10))
ax = fig.add_subplot (111, projection = '3d')
ax.set_ylabel ('$F1,\ F2,\ F3\ |\ Label$', fontsize = 20, rotation = 180)
ax.set_zlabel ('$Samples$', fontsize = 20, rotation = 0)
ax.set_xlabel ('$Steps$', fontsize = 20, rotation = 0)

ax.plot ([3.], [1.], [5.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$9$', markersize = 9, alpha = 0.6)
ax.plot ([3.], [2.], [5.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$8$', markersize = 9, alpha = 0.6)
ax.plot ([3.], [3.], [5.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$7$', markersize = 9, alpha = 0.6)

ax.plot ([2.], [1.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$6$', markersize = 9, alpha = 0.6)
ax.plot ([2.], [2.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$5$', markersize = 9, alpha = 0.6)
ax.plot ([2.], [3.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$4$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [5.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$3$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [5.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$2$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [5.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$1$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [0.], [1.], markerfacecolor = 'r', markeredgecolor = 'r', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [0.], [2.], markerfacecolor = 'g', markeredgecolor = 'g', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [0.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = 'o', markersize = 9, alpha = 0.6)


ax.plot ([3.], [1.], [3.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$90$', markersize = 9, alpha = 0.6)
ax.plot ([3.], [2.], [3.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$80$', markersize = 9, alpha = 0.6)
ax.plot ([3.], [3.], [3.], markerfacecolor = 'r', markeredgecolor = 'r', marker = '$70$', markersize = 9, alpha = 0.6)

ax.plot ([2.], [1.], [3.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$60$', markersize = 9, alpha = 0.6)
ax.plot ([2.], [2.], [3.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$50$', markersize = 9, alpha = 0.6)
ax.plot ([2.], [3.], [3.], markerfacecolor = 'g', markeredgecolor = 'g', marker = '$30$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [1.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$30$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [2.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$20$', markersize = 9, alpha = 0.6)
ax.plot ([1.], [3.], [3.], markerfacecolor = 'b', markeredgecolor = 'b', marker = '$10$', markersize = 9, alpha = 0.6)

ax.plot ([1.], [0.], [4.], markerfacecolor = 'r', markeredgecolor = 'r', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [0.], [5.], markerfacecolor = 'g', markeredgecolor = 'g', marker = 'X', markersize = 9, alpha = 0.6)
ax.plot ([1.], [0.], [6.], markerfacecolor = 'b', markeredgecolor = 'b', marker = 'o', markersize = 9, alpha = 0.6)

plt.show ()
