import numpy
import matplotlib.pyplot as plt

plt.figure ()
plt.xticks (fontsize = 16)
plt.yticks (fontsize = 16)
plt.xlabel ('x', fontsize = 24)
plt.ylabel ('y', fontsize = 24)

X = numpy.random.rand (100, 2)
for i in range (len (X)):
  #if ((4 * X [i, 0] - 1)/1 < X [i, 1]):
  if (X [i, 0] < X [i, 1]):
    plt.scatter (X [i, 0], X [i, 1], c = 'r')
  else:
    plt.scatter (X [i, 0], X [i, 1], c = 'b')

x = numpy.linspace (0, 1, 2)
plt.plot (x)
plt.legend (['y = x'], loc = 2, fontsize = 16)


plt.tight_layout ()
plt.show ()
plt.savefig ('linear.png')
