# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

import pandas as pd
import numpy as np

def load_dataset (file_schema, file_range, index_column, nan_values,
                  verbose = True):
  '''
  Parameters:
  -----------
  file_schema: str

  file_range: int

  index_column: str

  nan_values: list of str

  Returns:
  --------
  df: pandas.DataFrame

  Examples:
  ---------
  >>> df = load_dataset ('IoT-File_{}.csv', 4, 'package_ID', ['?', 'UNDEFINED'])
  '''
  df = pd.DataFrame ()
  for file_number in range (1, file_range + 1):
    if (verbose):
      print ('Reading', file_schema.format (str (file_number)))
    aux = pd.read_csv (file_schema.format (str (file_number)),
                       index_col = index_column,
                       dtype = {index_column: np.int32},
                       na_values = nan_values,
                       low_memory = False)
    df = pd.concat ([df, aux])
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
    print('{:35s} {:15d}  '.format(column, nUnique))
    #print (column, '|', nUnique)

  print ('\nColumn | different values (up to 10)')
  for column in df.columns:
    nUnique = df [column].nunique ()
  for column, nUnique in zip (df.columns, nUniques):
      if (nUnique < 10):
        print (column, df [column].unique ())
      else:
        print('{:35s} {:15d}  '.format(column, nUnique))

  my_objects = list (df.select_dtypes ( ['object']).columns)
  print ('\nObjects: (select encoding method)')
  print ('\nCheck for high cardinality.')
  print ('Column | # of different values | values')
  for column in my_objects:
    print (column, '|', df [column].nunique (), '|', df [column].unique ())
  print ('Objects:', list (df.select_dtypes (['object']).columns), '\n')

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


def plot_heatmap (df_corr):
    '''
    Parameters:
    -----------
    df_corr: pandas.DataFrame (ideally a return from .corr ())
    
    Examples:
    ---------
    >>> plot_heatmap (df.corr (method = 'pearson'))
    '''
    
#def heatmap(x, y, size):
#    fig, ax = plt.subplots()
#    
#    # Mapping from column names to integer coordinates
#    x_labels = [v for v in sorted(x.unique())]
#    y_labels = [v for v in sorted(y.unique())]
#    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
#    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
#    
#    size_scale = 500
#    ax.scatter(
#        x=x.map(x_to_num), # Use mapping for x
#        y=y.map(y_to_num), # Use mapping for y
#        s=size * size_scale, # Vector of square sizes, proportional to size parameter
#        marker='s' # Use square as scatterplot marker
#    )
#    
#    # Show column labels on the axes
#    ax.set_xticks([x_to_num[v] for v in x_labels])
#    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
#    ax.set_yticks([y_to_num[v] for v in y_labels])
#    ax.set_yticklabels(y_labels)