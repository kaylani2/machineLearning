#bot-iot
#x = [ 89.94 , 93.84 ] #acc
#x_error = [ 3.26 , 1.21] #acc
#x = [       88.10 , 91.10 ] #f1
#x_error = [ 3.97 , 2.22] #f1

#cicids
#x = [ 85.56 , 86.39 ] #acc
#x_error = [ 4.90 , 3.28]#acc
#x = [       88.03 , 88.34 ]#f1
#x_error = [ 2.37 , 2.08]#f1

#mnist
#x = [ 92.29 , 93.89 ]#acc
#x_error = [ 0.72 , 1.51]#acc
#x = [       93.21 , 94.10 ]#f1
#x_error = [ 0.82 , 1.17]#f1

#labsensing
#x = [ 95.76 , 98.12 ]
#x_error = [ 1.77 , 0.56]
x = [       96.55 , 97.92 ]
x_error = [ 2.56 , 1.85]
bars = ['Divisão tradicional', 'Divisão proposta']#, 'F1-score\nda divisão\ntradicional', 'F1-score\nda divisão\nproposta']


# Libraries
import numpy as np
import matplotlib.pyplot as plt

y_pos = np.arange(len(x))

# Create bars
plt.bar(y_pos, x, yerr = x_error, color = ['#cf5151', '#8c95e6', 'red', 'blue'], capsize = 4)


# Create names on the x-axis
plt.xticks(y_pos, bars)
minimum = 70
maximum = 100
ax = plt.gca()
ax.set_ylim([minimum,maximum])
plt.title('F1-score')
plt.ylabel('Valor percentual')
plt.xlabel('Tipo da divisão')


# Show graphic
plt.savefig('f1_labsensing.png')
