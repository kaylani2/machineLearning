import re
import sys
import statistics
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

train_time  = []
accuracy    = []
precision   = []
recall      = []
f1          = []
cohen_kappa = []

with open ('logfile.2020-07-04-11_17_13.log') as f:
  for line in f:
    if (re.search ('to train model', line)):
      train_time.append (float (line.split () [0]))
    if (re.search ('Accuracy', line)):
      accuracy.append (float (line.split () [1]))
    if (re.search ('Precision', line)):
      precision.append (float (line.split () [1]))
    if (re.search ('Recall', line)):
      recall.append (float (line.split () [1]))
    if (re.search ('F1', line)):
      f1.append (float (line.split () [1]))
    if (re.search ('Cohen Kappa', line)):
      cohen_kappa.append (float (line.split () [2]))


print ('Accuracy mean +/- stdv: {} +/- {}'.format (statistics.mean (accuracy), statistics.stdev (accuracy)))
print ('Precision mean +/- stdv: {} +/- {}'.format (statistics.mean (precision), statistics.stdev (precision)))
print ('Recall mean +/- stdv: {} +/- {}'.format (statistics.mean (recall), statistics.stdev (recall)))
print ('F1 mean +/- stdv: {} +/- {}'.format (statistics.mean (f1), statistics.stdev (f1)))
print ('Cohen kappa mean +/- stdv: {} +/- {}'.format (statistics.mean (cohen_kappa), statistics.stdev (cohen_kappa)))
print ('Train time mean +/- stdv: {} +/- {}'.format (statistics.mean (train_time), statistics.stdev (train_time)))


mean_accuracy = str (round (statistics.mean (accuracy) * 100, 3))[:-1]
stdev_accuracy = str (round (statistics.stdev (accuracy) * 100, 3))[:-1]
mean_precision = str (round (statistics.mean (precision) * 100, 3))[:-1]
stdev_precision = str (round (statistics.stdev (precision) * 100, 3))[:-1]
mean_recall = str (round (statistics.mean (recall) * 100, 3))[:-1]
stdev_recall = str (round (statistics.stdev (recall) * 100, 3))[:-1]
mean_f1 = str (round (statistics.mean (f1) * 100, 3))[:-1]
stdev_f1 = str (round (statistics.stdev (f1) * 100, 3))[:-1]
mean_cohen_kappa = str (round (statistics.mean (cohen_kappa) * 100, 3))[:-1]
stdev_cohen_kappa = str (round (statistics.stdev (cohen_kappa) * 100, 3))[:-1]
mean_train_time = str (round (statistics.mean (train_time), 3))[:-1]
stdev_train_time = str (round (statistics.stdev (train_time), 3))[:-1]



output = '\n\n\\thead{Floresta aleatória} & \makecell{$(' + mean_accuracy + '\pm$ \\\\ $' + stdev_accuracy + ')\%$} & \makecell{$(' + mean_precision + '\pm$ \\\\ $' + stdev_precision + ')\%$} & \makecell{$(' + mean_recall + '\pm$ \\\\ $' + stdev_recall + ')\%$} & \makecell{$(' + mean_f1 + '\pm$ \\\\ $' + stdev_f1 + ')\%$} & \makecell{$(' + mean_cohen_kappa + '\pm$ \\\\ $' + stdev_cohen_kappa + ')\%$} & \makecell{$(' + mean_train_time + '\pm$ \\\\ $' + stdev_train_time + ')$ s} \\\\ \hline %'



print (output.replace ('.', ','))
