B_10_E_1 = [[1, 2, 3, 4, 5], [10, 30, 50, 90, 170]]
B_10_E_2 = [[1, 2, 3, 4, 5], [8, 9, 20, 30, 32]]

from matplotlib import pyplot as plt

plt.plot (B_10_E_1 [0], B_10_E_1 [1], 'rx', label='B=10, E=1')
plt.plot (B_10_E_2 [0], B_10_E_2 [1], 'b-.', label='B=10, E=2')
plt.legend ()
plt.xlabel ('Número de rodadas')
plt.ylabel ('Acurácia')
plt.xlim (0, 7)
plt.ylim (0, 200)
plt.show ()
