## Adaptado de Nikhil Ketkar - Deep Learning with Python, A Hands-on Introduction
## Mostra o efeito da especializacao de um algoritmo ao incrementar o grau do
## polinomio utilizado no metodo de Minimos Quadrados
## Pega os primeiros 80 pontos pra treinar e os ultimos 20 pra testar

#Generate Toy Dataset
import pylab
import numpy
import matplotlib.pyplot as plt

degrees = [0, 1, 2, 3, 4, 5, 8, 10]
#degrees = []

numpy.random.seed (2)
x = numpy.linspace (-1,1,100)
signal = 2 + x + 2 * x * x
noise = numpy.random.normal (0, 0.1, 100)
y = signal + noise

plt.figure ()
plt.xticks (fontsize = 16)
plt.yticks (fontsize = 16)
plt.xticks (numpy.arange (0, 101, 10))
plt.yticks (numpy.arange (0, 5.1, 1))
plt.xlabel ('x', fontsize = 24)
plt.ylabel ('y', fontsize = 24)
#plt.ylabel ('$2x^2 + x + 2$', fontsize = 24)
plt.tight_layout ()

plt.plot (signal,'b');
plt.plot (y,'g', lw = 3, ls = ':')
plt.plot (noise, 'c')

x_train = x[0:80]
y_train = y[0:80]
plt.axvline (x = 60, linewidth = 2, color = 'black')
plt.legend (["Sinal sem ruido", "Sinal coletado", "Ruido gaussiano"], loc = 2, fontsize = 16)

#plt.show ()
plt.savefig ('ruido.png')


for degree in degrees:
  #pylab.figure ()
  degree += 1
  X_train = numpy.column_stack ([numpy.power (x_train,i) for i in range (0,degree)])
  model = numpy.dot (numpy.dot (numpy.linalg.inv (numpy.dot (X_train.transpose (),X_train)),X_train.transpose ()),y_train)
  predicted = numpy.dot (model, [numpy.power (x,i) for i in range (0,degree)])
  train_rmse1 = numpy.sqrt (numpy.sum (numpy.dot (y[0:80] - predicted[0:80], y_train - predicted[0:80])))
  test_rmse1 = numpy.sqrt (numpy.sum (numpy.dot (y[80:] - predicted[80:], y[80:] - predicted[80:])))

  plt.figure ()
  plt.xlabel ('x', fontsize = 24)
  plt.ylabel ('y', fontsize = 24)
  #plt.ylabel ('$2x^2 + x + 2$', fontsize = 24)
  plt.xticks (numpy.arange (-1, 1.2, 0.5))
  plt.yticks (numpy.arange (0, 5.1, 1))
  plt.xticks (fontsize = 16)
  plt.yticks (fontsize = 16)

  #my_dpi = 50
  #plt.figure (figsize= (800/my_dpi, 800/my_dpi), dpi = my_dpi)
  plt.tight_layout ()
  print ("Train RMSE (Degree = " + str (degree - 1) + ")", train_rmse1)
  print ("Test RMSE (Degree = " + str (degree - 1) + ")", test_rmse1)
  #print ("Test RMSE (Degree = 1)", test_rmse1)
  plt.plot (x,y,'g', lw = 3, ls = ':')
  plt.plot (x, predicted,'r')
  plt.axvline (x = 0.60, linewidth = 2, color = 'black')
  #plt.show ()
  plt.legend (["Sinal coletado", "Predito com grau " + str (degree - 1)], loc = 2, fontsize = 16)
  plt.savefig ('grau_' + str (degree - 1) + '.png')
