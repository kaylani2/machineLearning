import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

## WAVE
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Atributo")
plt.ylabel("Alvo")
plt.show()

mglearn.plots.plot_linear_regression_wave()
plt.show ()
