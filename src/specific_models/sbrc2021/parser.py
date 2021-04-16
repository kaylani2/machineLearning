import sys
import re
import statistics

if __name__ == "__main__":
  try:
    #filename = (sys.argv [1])
    filename = './logs/10e_64b.log'
  except:
    print ('Usage: $python parser.py FILENAME.log')
    exit ()

  results = []
  with open (filename) as f:
    for line in f:
      if (re.search ('32/32', line)):
        results.append (line)
        print (line)
