import re
import sys
import statistics


files = [
  {'name': 'kmeans.log', 'model': 'kmeans'},
  #{'name': 'OUT_2DCNN.log', 'model': 'twod_cnn'},
  #{'name': 'OUT_AUTOENCODER.log', 'model': 'autoencoder'},
  #{'name': 'OUT_DECISION_TREES.log', 'model': 'decision_tree'},
  #{'name': 'OUT_LSTM.log', 'model': 'rnn'},
  #{'name': 'OUT_MLP.log', 'model': 'mlp'},
  #{'name': 'OUT_NAIVE_BAYES.log', 'model': 'naive'},
  #{'name': 'OUT_RANDOM_FOREST.log', 'model': 'random_forest'},
  #{'name': 'OUT_SVC.log', 'model': 'svm'},
]

try:
  _ = (sys.argv [1])
except:
  print ('Usage: python std_dev.py {macro,p-attack,n-attack}')
  sys.exit (1)



for my_file in files:
  train_time  = []
  accuracy    = []
  precision   = []
  recall      = []
  f1          = []
  cohen_kappa = []
  tps = [] # true positives
  tns = [] # true negatives
  fps = [] # false positives
  fns = [] # false negatives
  far = [] # false alarm rate -- lower is better


  with open ('results/' + my_file ['name']) as f:
    for line in f:
      if (re.search ('to train model', line)):
        train_time.append (float (line.split () [0]))
      if (re.search ('^TP', line)):
        tps.append (float (line.split () [1]))
      if (re.search ('^TN', line)):
        tns.append (float (line.split () [1]))
      if (re.search ('^FP', line)):
        fps.append (float (line.split () [1]))
      if (re.search ('^FN', line)):
        fns.append (float (line.split () [1]))
  far_value = 0
  far_value = 0
  accuracy_value = 0
  precision_value = 0
  recall_value = 0
  f1_value = 0
  if ((sys.argv [1]) == 'macro'):
    for tp, fp, tn, fn in zip (tps, fps, tns, fns):
      #far_value = (fp / (fp + tn))
      far_value = ((fp / (fp + tn)))
      accuracy_value = ( (tp + tn) / (tp + tn + fp + fn) )
      precision_value = ( tp / (tp + fp) )
      recall_value = ( tp / (tp + fn) )
      f1_value = ( tp / (tp + (fp + fn)/2 ) )

    tps, tns, fps, fns = tns, tps, fns, fps
    for tp, fp, tn, fn in zip (tps, fps, tns, fns):
      #far_value_2 = (fp / (fp + tn))
      far_value_2 = ((fp / (fp + tn)))
      accuracy_value_2 = ( (tp + tn) / (tp + tn + fp + fn) )
      precision_value_2 = ( tp / (tp + fp) )
      recall_value_2 = ( tp / (tp + fn) )
      f1_value_2 = ( tp / (tp + (fp + fn)/2 ) )

      #far.append ((far_value + far_value_2)/2)
      far.append ((far_value + far_value_2)/2)
      accuracy.append ((accuracy_value + accuracy_value_2)/2)
      precision.append ((precision_value + precision_value_2)/2)
      recall.append ((recall_value + recall_value_2)/2)
      f1.append ((f1_value + f1_value_2)/2)

  elif ((sys.argv [1]) == 'p-attack'):
    for tp, fp, tn, fn in zip (tps, fps, tns, fns):
      #far_value = (fp / (fp + tn))
      far_value = ((fp / (fp + tn)))
      accuracy_value = ( (tp + tn) / (tp + tn + fp + fn) )
      precision_value = ( tp / (tp + fp) )
      recall_value = ( tp / (tp + fn) )
      f1_value = ( tp / (tp + (fp + fn)/2 ) )

      #far.append (far_value)
      far.append (far_value)
      accuracy.append (accuracy_value)
      precision.append (precision_value)
      recall.append (recall_value)
      f1.append (f1_value)

  elif ((sys.argv [1]) == 'n-attack'):
    tps, tns, fps, fns = tns, tps, fns, fps
    print (tps)
    for tp, fp, tn, fn in zip (tps, fps, tns, fns):
      #far_value = (fp / (fp + tn))
      far_value = ((fp / (fp + tn)))
      accuracy_value = ( (tp + tn) / (tp + tn + fp + fn) )
      precision_value = ( tp / (tp + fp) )
      recall_value = ( tp / (tp + fn) )
      f1_value = ( tp / (tp + (fp + fn)/2 ) )

      #far.append (far_value)
      far.append (far_value)
      accuracy.append (accuracy_value)
      precision.append (precision_value)
      recall.append (recall_value)
      f1.append (f1_value)

  else:
    print ('Usage: python std_dev.py {macro,p-attack,n-attack}')





  mean_accuracy = str (round (statistics.mean (accuracy) * 100, 3))[:-1]
  stdev_accuracy = str (round (statistics.stdev (accuracy) * 100, 3))[:-1]
  mean_precision = str (round (statistics.mean (precision) * 100, 3))[:-1]
  stdev_precision = str (round (statistics.stdev (precision) * 100, 3))[:-1]
  mean_recall = str (round (statistics.mean (recall) * 100, 3))[:-1]
  stdev_recall = str (round (statistics.stdev (recall) * 100, 3))[:-1]
  mean_f1 = str (round (statistics.mean (f1) * 100, 3))[:-1]
  stdev_f1 = str (round (statistics.stdev (f1) * 100, 3))[:-1]
  mean_train_time = str (round (statistics.mean (train_time), 3))[:-1]
  stdev_train_time = str (round (statistics.stdev (train_time), 3))[:-1]
  mean_far = str (round (statistics.mean (far), 3))[:-1]
  stdev_far = str (round (statistics.stdev (far), 3))[:-1]
  mean_far = str (round (statistics.mean (far) * 100, 3))[:-1]
  stdev_far = str (round (statistics.stdev (far) * 100, 3))[:-1]

  #print ('\nFile:', my_file)
  #print ('{:30s} {:10s} +/- {:10s}'.format ('Accuracy mean +/- stdv:', mean_accuracy, stdev_accuracy))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('Precision mean +/- stdv:', mean_precision, stdev_precision))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('Recall mean +/- stdv:', mean_recall, stdev_recall))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('F1 mean +/- stdv:', mean_f1, stdev_f1))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('(1 - far)  mean +/- stdv:', mean_far, stdev_far))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('Far mean +/- stdv:', mean_far, stdev_far))
  #print ('{:30s} {:10s} +/- {:10s}'.format ('Train time mean +/- stdv:', mean_train_time, stdev_train_time))


  # Formatting for plot_vertical_bars.py
  print ('{} ({}, {}, {}, {}, {}), ({}, {}, {}, {}, {})'.format
        ((my_file ['model'] + ', ' + my_file ['model'] + '_std ='),
         float (mean_accuracy), float (mean_precision),
         float (mean_recall), float (mean_f1),
         float (mean_far), float (stdev_accuracy),
         float (stdev_precision), float (stdev_recall),
         float (stdev_f1), float (stdev_far)
        ))


  # Formatting for Overleaf
  #output = '\n\n\\thead{' + my_file + '} & \makecell{$(' + mean_accuracy + '\pm$ \\\\ $' + stdev_accuracy + ')\%$} & \makecell{$(' + mean_precision + '\pm$ \\\\ $' + stdev_precision + ')\%$} & \makecell{$(' + mean_recall + '\pm$ \\\\ $' + stdev_recall + ')\%$} & \makecell{$(' + mean_f1 + '\pm$ \\\\ $' + stdev_f1 + ')\%$} & \makecell{$(' + mean_train_time + '\pm$ \\\\ $' + stdev_train_time + ')$ s} & \makecell{$(' + mean_far + '\pm$ \\\\ $' + stdev_far + ')\%$} \\\\ \hline %'
  #print (output.replace ('.', ',').replace ('0,', '0,00'))

