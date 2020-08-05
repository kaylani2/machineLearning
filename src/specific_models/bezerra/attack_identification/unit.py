# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

import pandas as pd
import numpy as np

def load_dataset (verbose = False):
  '''
  Parameters:
  -----------
  verbose: bool, default = True

  Returns:
  --------
  df: pandas.DataFrame

  Examples:
  ---------
  >>> df = load_dataset (verbose = True)
  '''

  DATASET_DIR = '../../../../datasets/Dataset-IoT/'
  NETFLOW_DIRS = ['MC/NetFlow/', 'SC/NetFlow/', 'ST/NetFlow/']

  # MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets'
  # MC_I_SECOND: Has infected data from Mirai botnets
  # MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets
  # MC_L: Has legitimate data, no infection

  path_types = ['MC', 'SC', 'ST']
  data_set_files = [[r'MC_I{}.csv'.format (index) for index in range (1, 4)],
                   [r'SC_I{}.csv'.format (index) for index in range (1, 4)],
                   [r'ST_I{}.csv'.format (index) for index in range (1, 4)] ]

  for path, files in zip (path_types, data_set_files):
    files.append (path + '_L.csv')

  print ("Caminhos construidos")
  ################
  ##reading data##
  ################

  for n, (path, files) in enumerate (zip (NETFLOW_DIRS, data_set_files), start = 1):
    for csvFile in files:
        if n == 1:
            df = pd.read_csv (DATASET_DIR + path + csvFile)
        else:
            aux_df = pd.read_csv (DATASET_DIR + path + csvFile)
            df = pd.concat ( [df, aux_df], ignore_index = True)

  return df


def display_general_information (df, verbose = True):
  '''
  Parameters:
  -----------
  df: pandas.DataFrame

  verbose: bool, default = True

  Returns:
  --------
  NULL

  Examples:
  ---------
  >>> display_general_information (df)
  '''
  print ('Dataframe shape (lines, columns):', df.shape, '\n')
  print ('First 5 entries:\n', df [:5], '\n')
  df.info (verbose = verbose)

  print ('Brief description:')
  df.describe ()

  print ('\nDataframe contains NaN values:', df.isnull ().values.any ())
  nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
  print ('Number of NaN columns:', len (nanColumns))
  print ('NaN columns:', nanColumns, '\n')

  # nUniques = df.nunique () ### K: Takes too long. WHY?
  print ('\nColumn | # of different values')
  nUniques = []
  for column in df.columns:
    nUnique = df [column].nunique ()
    nUniques.append (nUnique)
    print ('{:35s} {:15d}  '.format (column, nUnique))
    #print (column, '|', nUnique)

  print ()
  for column in df.columns:
    nUnique = df [column].nunique ()
  for column, nUnique in zip (df.columns, nUniques):
      if (nUnique < 10):
        print (column, df [column].unique ())
      else:
        print ('{:35s} {:15d}  '.format (column, nUnique))

  my_objects = list (df.select_dtypes ( ['object']).columns)
  print ('\nObjects: (select encoding method)')
  print ('\nCheck for high cardinality.')
  print ('Column | # of different values | values')
  for column in my_objects:
    print (column, '|', df [column].nunique (), '|', df [column].unique ())
  print ('Objects:', list (df.select_dtypes ( ['object']).columns), '\n')

def remove_columns_with_one_value (df, verbose = True):
  '''
  Parameters:
  -----------
  df: pandas.DataFrame

  verbose: bool, default = True

  Returns:
  --------
  df: pandas.DataFrame

  log: str

  Examples:
  ---------
  >>> df, log = remove_nan_columns (df, verbose = False)
  '''
  nColumns = len (df.columns)
  nUniques = []
  if (verbose):
    print ('\nColumn | # of different values (before dropping).')
  for column in df.columns:
    nUnique = df [column].nunique ()
    nUniques.append (nUnique)
    if (verbose):
      print (column, '|', nUnique)

  if (verbose):
    print ('\nRemoving attributes that have only one (or zero) sampled value.')
  for column, nUnique in zip (df.columns, nUniques):
    if (nUnique <= 1):
      df.drop (axis = 'columns', columns = column, inplace = True)

  if (verbose):
    print ('\nColumn | # of different values (after dropping).')
    for column in df.columns:
      nUnique = df [column].nunique ()
      print (column, '|', nUnique)

  log = 'While removing single value columns: '
  if ((len (df.columns)) == nColumns):
    log += 'No columns dropped.'
  else:
    log += str ((nColumns - (len (df.columns)), 'column (s) dropped.'))

  return df, log

def remove_nan_columns (df, threshold, verbose = True):
  '''
  Parameters:
  -----------
  df: pandas.DataFrame

  threshold: float

  verbose: bool, default = True

  Returns:
  --------
  df: pandas.DataFrame

  log: str

  Examples:
  ---------
  >>> df, log = remove_nan_columns (df, 1/2, verbose = False)
  '''
  nColumns = len (df.columns)
  if (verbose):
    print ('Removing attributes with more than half NaN values.')
    print ('\nColumn | NaN values')
    print (df.isnull ().sum ())

  threshold = 1/threshold
  df = df.dropna (axis = 'columns', thresh = df.shape [0] // threshold)

  if (verbose):
    print ('Dataframe contains NaN values:', df.isnull ().values.any ())
    print ('\nColumn | NaN values (after dropping columns)')
    print (df.isnull ().sum ())

  log = 'While removing nan value columns: '
  if ((len (df.columns)) == nColumns):
    log += 'No columns dropped.'
  else:
    log += str ((nColumns - (len (df.columns)), 'column (s) dropped.'))

  return df, log

def display_feature_distribution (df, features):
  '''
  Parameters:
  -----------
  df: pandas.DataFrame

  features: list of str

  --------

  Examples:
  ---------
  >>> display_feature_distribution (df)
  '''
  for feature in features:
    print ('\nFeature:', feature)
    print ('Values:', df [feature].unique ())
    print ('Distribution:')
    print (df [feature].value_counts ())
