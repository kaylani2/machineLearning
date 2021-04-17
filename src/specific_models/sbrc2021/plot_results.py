import sys
import re
import statistics
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 14})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

#def found_fit (x):
#  return x**3
#
#if __name__ == "__main__":
#  try:
#    #filename = (sys.argv [1])
#    filename = '10e_64b.log'
#  except:
#    print ('Usage: $python parser.py FILENAME.log')
#    exit ()

filename = '10e_64b.log'
my_files = [
  {'filename': '10e_64b_10c.log', 'label': '$E_t=10, B_c=64$', 'line_style': 'r-'},
  {'filename': '20e_64b_10c.log', 'label': '$E_t=20, B_c=64$', 'line_style': 'b-'},
]
for my_file in my_files:
  print (my_file ['filename'])
  results = []
  accuracies = []
  standard_deviations = []
  with open (my_file ['filename']) as f:
    for line in f:
      if ( (re.search ('32/32', line)) and not (re.search ('ETA', line)) ):
        current_line = line.replace ('\n', '')
        current_line = current_line [current_line.find ('accuracy: '):]
        current_line = current_line [len ('accuracy: '):]
        #print (current_line)
        results.append (float (current_line))# * 100) # point notation to percentual notation
      if (re.search ('Number of clients:', line)):
        number_of_clients = int (line.split (' ') [-1])

  ### K: At the end the framework performs a test on each client.
  for _ in range (number_of_clients):
    results.pop ()
  #print ('Results:', len (results))

  for index in range (0, len (results), 2):
    local_accuracies = [float (results [index]), float (results [index + 1])]
    accuracies.append ( (round (statistics.mean (local_accuracies) * 100, 3)))#[:-1]
    standard_deviations.append ( (round (statistics.stdev (local_accuracies) * 100, 3)))#[:-1]



  current_plot = [range (1, len (accuracies) + 1), accuracies, standard_deviations]

  plt.plot (current_plot [0], current_plot [1], my_file ['line_style'], label=my_file ['label'])

plt.axhline (y=67.82, color='k', linestyle='-', label='Centralizado: $67,82\%$')
plt.legend ()
plt.xlabel ('Número de rodadas')
plt.ylabel ('Acurácia')
plt.xlim (0, len (accuracies) + 1)
plt.ylim (0, 100)
#plt.show ()
plt.savefig ('result.png')
print ('saved')


## subplot:
#x_data1 = [0.1, 0.2, 0.3, 0.4]
#y_data1 = [1, 2, 3, 4]
#
#x_data2 = [0.1, 0.2, 0.3, 0.4]
#y_data2 = [1, 4, 9, 16]
#
#fig = plt.figure ()
#ax1 = fig.add_subplot (1, 2, 1)
#ax2 = fig.add_subplot (1, 2, 2)
#ax1.plot (x_data1, y_data1, label='data 1')
#ax2.plot (x_data2, y_data2, label='data 2')
#ax1.set_xlabel ('Time (s)')
#ax1.set_ylabel ('Scale (Bananas)')
#ax1.set_title ('first data set')
#ax1.legend ()
#ax2.set_xlabel ('Time (s)')
#ax2.set_ylabel ('Scale (Bananas)')
#ax2.set_title ('second data set')
#ax2.legend ()
#
#plt.show ()
