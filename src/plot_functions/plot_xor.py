import numpy
import matplotlib.pyplot as plt

plt.figure ()
plt.xticks (fontsize = 16)
plt.yticks (fontsize = 16)
plt.xlabel ('x', fontsize = 24)
plt.ylabel ('y', fontsize = 24)

X = numpy.random.rand (100, 2)
for i in range (len (X)):
  if ( (X [i, 0] > 0.5) and (X [i, 1] > 0.5) ): # 1qd
    plt.scatter (X [i, 0], X [i, 1], c = 'r')
  elif ( (X [i, 0] < 0.5) and (X [i, 1] < 0.5) ): # 3qd
    plt.scatter (X [i, 0], X [i, 1], c = 'r')
  else: # 2qd and 4qd
    plt.scatter (X [i, 0], X [i, 1], c = 'b')



plt.tight_layout ()
#plt.show ()
plt.savefig ('xor.png')
