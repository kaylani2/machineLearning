import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

line = np.linspace (-3, 3, 100)
plt.grid()
#plt.plot (line, np.heaviside (line, 0.5), label="degrau")
plt.plot (line, np.tanh (line), label="tanh")
#plt.plot (line, np.maximum (line, 0), label="relu")
#plt.plot (line, 1/ (1+np.exp (-line)), label="sigmoid")
#plt.axis([-3, 3, -1.5, 1.5])
plt.legend (loc="best")
plt.xlabel ("x")
plt.ylabel ("tanh (x)")
plt.savefig ('tanh' + '.png')
plt.show ()
