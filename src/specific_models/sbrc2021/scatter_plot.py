import sys
import re
import statistics
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 13})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

my_files = [
  {'filename': '10e_64b_10c.log', 'label': '$E_t=10, B_c=64, N_c=10, P_f=0\%$',              'marker': '>', 'regex': '32/32', 'round': 251},
  {'filename': '10e_64b_20c.log', 'label': '$E_t=10, B_c=64, N_c=20, P_f=0\%$',              'marker': '>', 'regex': '16/16', 'round': 251},
  {'filename': '20e_64b_10c.log', 'label': '$E_t=20, B_c=64, N_c=10, P_f=0\%$',              'marker': 'o', 'regex': '32/32', 'round': 190},
  {'filename': '20e_256b_10c.log', 'label': '$E_t=20, B_c=256, N_c=10, P_f=0\%$',            'marker': '>', 'regex': '32/32', 'round': 251},
  {'filename': '10e_64b_5c.log', 'label': '$E_t=10, B_c=64, N_c=5, P_f=0\%$',                'marker': '>', 'regex': '63/63', 'round': 251},
  {'filename': '10e_256b_5c.log', 'label': '$E_t=10, B_c=256, N_c=5, P_f=0\%$',              'marker': '>', 'regex': '63/63', 'round': 251},
  {'filename': '20e_64b_5c.log', 'label': '$E_t=20, B_c=64, N_c=5, P_f=0\%$',                'marker': 'o', 'regex': '63/63', 'round': 0},
  {'filename': '025fraction_20e_64b_10c.log', 'label': '$E_t=20, B_c=64, N_c=10, P_f=25\%$', 'marker': '>', 'regex': '32/32', 'round': 251},
  {'filename': '050fraction_20e_64b_10c.log', 'label': '$E_t=20, B_c=64, N_c=10, P_f=50\%$', 'marker': 'o', 'regex': '32/32', 'round': 190},
  {'filename': '075fraction_20e_64b_10c.log', 'label': '$E_t=20, B_c=64, N_c=10, P_f=75\%$', 'marker': 'o', 'regex': '32/32', 'round': 160},
]

for my_file in my_files:
  print ('file:', my_file ['filename'])
  results = []
  accuracies = []
  standard_deviations = []
  with open (my_file ['filename']) as f:
    for line in f:
      if ( (re.search (my_file ['regex'], line)) and not (re.search ('ETA', line)) ):
        try: # Deadline eh hoje
          current_line = line.replace ('\n', '')
          current_line = current_line [current_line.find ('accuracy: '):]
          current_line = current_line [len ('accuracy: '):]
          results.append (float (current_line))# * 100) # point notation to percentual notation
        except:
          pass
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

  print ('acc:', accuracies [-1])
  plt.plot(my_file ['round'], accuracies [-1], my_file ['marker'], label=my_file ['label'])


  #current_plot = [range (1, len (accuracies) + 1), accuracies, standard_deviations]
  #current_plot = [range (1, len (accuracies) + 1), accuracies, standard_deviations]

  #plt.plot (current_plot [0], current_plot [1], my_file ['line_style'], label=my_file ['label'])


#plt.axhline (y=67.82, color='k', linestyle='-', label='Centralizado: $67,82\%$')
#plt.legend ()
plt.figure(figsize=(1,1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel ('Rodadas para convergir')
plt.ylabel ('Acurácia na convergência')
plt.xlim (0, 300)#len (accuracies) + 1)
plt.ylim (0, 100)
plt.tight_layout()
plt.savefig ('scatter.pdf')
print ('saved')
