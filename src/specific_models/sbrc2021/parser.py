import sys
import re
import statistics

if __name__ == "__main__":
  try:
    #filename = (sys.argv [1])
    filename = '10e_64b.log'
  except:
    print ('Usage: $python parser.py FILENAME.log')
    exit ()

  results = []
  accuracies = []
  standard_deviations = []
  with open (filename) as f:
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
    #print (local_accuracies)
    accuracies.append ((round (statistics.mean (local_accuracies) * 100, 3)))#[:-1]
    standard_deviations.append ((round (statistics.stdev (local_accuracies) * 100, 3)))#[:-1]
    #print (index)
    #print (results [index])

  #mean_accuracy = str (round (statistics.mean (accuracy) * 100, 3))[:-1]
  #stdev_accuracy = str (round (statistics.stdev (accuracy) * 100, 3))[:-1]
