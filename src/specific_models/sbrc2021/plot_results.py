def found_fit(x):
  return x**3

B_10_E_1 = [[1, 2, 3, 4, 5], [10, 30, 50, 90, 170]]
B_10_E_2 = [[1, 2, 3, 4, 5], [8, 9, 20, 30, 32]]

import numpy as np
from matplotlib import pyplot as plt

x_func = np.linspace(0, 10, 50)
y_func = found_fit(x_func)
plt.plot(x_func, y_func, label='$f(x) = x^3$')

plt.plot (B_10_E_1 [0], B_10_E_1 [1], 'rx', label='B=10, E=1')
plt.plot (B_10_E_2 [0], B_10_E_2 [1], 'b-.', label='B=10, E=2')
plt.legend ()
plt.xlabel ('Número de rodadas')
plt.ylabel ('Acurácia')
plt.xlim (0, 7)
plt.ylim (0, 200)
#plt.show ()
plt.savefig('result.png')


## subplot:
#x_data1 = [0.1, 0.2, 0.3, 0.4]
#y_data1 = [1, 2, 3, 4]
#
#x_data2 = [0.1, 0.2, 0.3, 0.4]
#y_data2 = [1, 4, 9, 16]
#
#fig = plt.figure()
#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)
#ax1.plot(x_data1, y_data1, label='data 1')
#ax2.plot(x_data2, y_data2, label='data 2')
#ax1.set_xlabel('Time (s)')
#ax1.set_ylabel('Scale (Bananas)')
#ax1.set_title('first data set')
#ax1.legend()
#ax2.set_xlabel('Time (s)')
#ax2.set_ylabel('Scale (Bananas)')
#ax2.set_title('second data set')
#ax2.legend()
#
#plt.show()
