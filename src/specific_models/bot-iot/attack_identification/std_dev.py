import re
import sys
import statistics

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
