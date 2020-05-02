import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import math


#def g(x):
#    i = x
    #return 0.1*x**4 - 0.5*x**3 + 0.05*x**5 + 3*(np.sin (i))
#    return x**2 + (10 - 0.5*x**3) + (9 - 0.00001 * x**8) + 50 + (9 + 0.005 * x**6)

def f(x):
    return x*(x+2)*(x - 3)-(0.1*x+8)
    #return x**3 - 5*x**2 + 5.9*x - 8

def df(x):
    h = 0.000001
    return (f(x + h) - f(x))/h

def tan_plot(a):
    x = np.linspace((a-12), (a+12), 1000)
    y = df(a)*(x - a) + f(a)
    plt.plot(x, f(x))
    plt.plot(a, f(a), 'o', markersize = 10)
    plt.plot(x, y, '--k')
    plt.axhline(color = 'black')
    plt.axvline(color = 'black')
    plt.xlabel ('x')
    plt.ylabel ('f(x)')
    plt.axis([-6, 6, -30, 20])
    plt.savefig ('plot' + str (a) + '.png')
    plt.show ()


#x = np.linspace(-4, 4, 1000)
#plt.plot(x, f(x), label = '$f(x)$')
#plt.plot(x, df(x), label = '$f\'(x)$')
#plt.axhline(color = 'black')
#plt.axvline(color = 'black')
#plt.legend(loc = 'best', frameon = False)
#plt.title("Function and its Derivative")
#plt.show ()

for i in [0, 1, 2.5, 1.6, 1.8]:
  tan_plot (i)


#x = sy.Symbol('x')
#df = sy.diff(g(x), x)
#df = sy.simplify(df)
#a, b = sy.solve(df, x)
#x = np.linspace(-3, 4, 1000)
#plt.plot(x, g(x))
#plt.axhline(color = 'black')
#plt.axvline(color = 'black')
#plt.plot(a, g(a), 'o')
#plt.plot(b, g(b), 'o')
#plt.show ()
