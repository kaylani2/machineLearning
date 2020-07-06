# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Multilayer Perceptron

import pandas as pd
import numpy as np
import sys
import time


###############################################################################
## Define constants
###############################################################################
# Random state for reproducibility

STATES = [0, 10, 100, 1000, 10000]
STATE = 0
for STATE in STATES:
  np.random.seed (STATE)

  pd.set_option ('display.max_rows', None)
  pd.set_option ('display.max_columns', 5)

  BOT_IOT_DIRECTORY = '../../../../datasets/bot-iot/'
  BOT_IOT_FEATURE_NAMES = 'UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv'
  BOT_IOT_FILE_5_PERCENT_SCHEMA = 'UNSW_2018_IoT_Botnet_Full5pc_{}.csv' # 1 - 4
  FIVE_PERCENT_FILES = 4
  BOT_IOT_FILE_FULL_SCHEMA = 'UNSW_2018_IoT_Botnet_Dataset_{}.csv' # 1 - 74
  FULL_FILES = 74
  FILE_NAME = BOT_IOT_DIRECTORY + BOT_IOT_FILE_5_PERCENT_SCHEMA#FULL_SCHEMA
  FEATURES = BOT_IOT_DIRECTORY + BOT_IOT_FEATURE_NAMES
  NAN_VALUES = ['?', '.']
  TARGET = 'attack'

  ###############################################################################
  ## Load dataset
  ###############################################################################
  featureDf = pd.read_csv (FEATURES)
  featureColumns = featureDf.columns.to_list ()
  featureColumns = [f.strip () for f in featureColumns]

  df = pd.DataFrame ()
  for fileNumber in range (1, FIVE_PERCENT_FILES + 1):#FULL_FILES + 1):
    print ('Reading', FILE_NAME.format (str (fileNumber)))
    aux = pd.read_csv (FILE_NAME.format (str (fileNumber)),
                       #names = featureColumns,
                       index_col = 'pkSeqID',
                       dtype = {'pkSeqID' : np.int32}, na_values = NAN_VALUES,
                       low_memory = False)
    df = pd.concat ([df, aux])


  ###############################################################################
  ## Display generic (dataset independent) information
  ###############################################################################
  #print ('Dataframe shape (lines, columns):', df.shape, '\n')
  #print ('First 5 entries:\n', df [:5], '\n')
  #print ('entries:\n', df [4000000//4 - 5:4000000//4 + 5], '\n')
  df.info (verbose = True)

  print ('\nDataframe contains NaN values:', df.isnull ().values.any ())
  nanColumns = [i for i in df.columns if df [i].isnull ().any ()]
  print ('Number of NaN columns:', len (nanColumns))
  print ('NaN columns:', nanColumns, '\n')


  ###############################################################################
  ## Display specific (dataset dependent) information
  ###############################################################################
  print ('\nAttack types:', df ['attack'].unique ())
  print ('Attack distribution:')
  print (df ['attack'].value_counts ())
  print ('\nCateogry types:', df ['category'].unique ())
  print ('Cateogry distribution:')
  print (df [TARGET].value_counts ())
  print ('\nSubcategory types:', df ['subcategory'].unique ())
  print ('Subcategory distribution:')
  print (df ['subcategory'].value_counts ())


  ###############################################################################
  ## Data pre-processing
  ###############################################################################
  #df.replace ( ['NaN', 'NaT'], np.nan, inplace = True)
  #df.replace ('?', np.nan, inplace = True)
  #df.replace ('Infinity', np.nan, inplace = True)

  ###############################################################################
  ### Remove columns with only one value
  print ('\nColumn | # of different values')
  # nUniques = df.nunique () ### K: Takes too long. WHY?
  nUniques = []
  for column in df.columns:
    nUnique = df [column].nunique ()
    nUniques.append (nUnique)
    print (column, '|', nUnique)

  print ('\nRemoving attributes that have only one (or zero) sampled value.')
  for column, nUnique in zip (df.columns, nUniques):
    if (nUnique <= 1): # Only one value: DROP.
      df.drop (axis = 'columns', columns = column, inplace = True)

  print ('\nColumn | # of different values')
  for column in df.columns:
    nUnique = df [column].nunique ()
    print (column, '|', nUnique)

  ###############################################################################
  ### Remove redundant columns
  ### K: These columns are numerical representations of other existing columns.
  redundantColumns = ['state_number', 'proto_number', 'flgs_number']
  print ('\nRemoving redundant columns:', redundantColumns)
  df.drop (axis = 'columns', columns = redundantColumns, inplace = True)

  ###############################################################################
  ### Remove NaN columns (with a lot of NaN values)
  print ('\nColumn | NaN values')
  print (df.isnull ().sum ())
  print ('Removing attributes with more than half NaN values.')
  df = df.dropna (axis = 'columns', thresh = df.shape [0] // 2)
  print ('Dataframe contains NaN values:', df.isnull ().values.any ())
  print ('\nColumn | NaN values (after dropping columns)')
  print (df.isnull ().sum ())

  ###############################################################################
  ### Input missing values
  ### K: Look into each attribute to define the best inputing strategy.
  ### K: NOTE: This must be done after splitting to dataset to avoid data leakge.
  df ['sport'].replace ('-1', np.nan, inplace = True)
  df ['dport'].replace ('-1', np.nan, inplace = True)
  ### K: Negative port values are invalid.
  columsWithMissingValues = ['sport', 'dport']
  ### K: Examine values.
  for column in df.columns:
    nUnique = df [column].nunique ()
  for column, nUnique in zip (df.columns, nUniques):
      if (nUnique < 5):
        print (column, df [column].unique ())
      else:
        print (column, 'unique values:', nUnique)

  # sport  unique values: 91168     # most_frequent?
  # dport  unique values: 115949    # most_frequent?
  imputingStrategies = ['most_frequent', 'most_frequent']


  ###############################################################################
  ### Handle categorical values
  ### K: Look into each attribute to define the best encoding strategy.
  df.info (verbose = False)
  ### K: dtypes: float64 (11), int64 (8), object (9)
  myObjects = list (df.select_dtypes ( ['object']).columns)
  print ('\nObjects:', myObjects, '\n')
  ### K: Objects:
    # 'flgs',
    # 'proto',
    # 'saddr',
    # 'sport',
    # 'daddr',
    # 'dport',
    # 'state',
  # LABELS:
    # TARGET,
    # 'subcategory'

  print ('\nCheck for high cardinality.')
  print ('Column | # of different values | values')
  for column in myObjects:
    print (column, '|', df [column].nunique (), '|', df [column].unique ())

  ### K: NOTE: saddr and daddr (source address and destination address) may incur
  ### into overfitting for a particular scenario of computer network. Since the
  ### classifier will use these IPs and MACs to aid in classifying the traffic.
  ### We may want to drop these attributes to guarantee IDS generalization.
  df.drop (axis = 'columns', columns = 'saddr', inplace = True)
  df.drop (axis = 'columns', columns = 'daddr', inplace = True)

  print ('\nHandling categorical attributes (label encoding).')
  from sklearn.preprocessing import LabelEncoder
  myLabelEncoder = LabelEncoder ()
  df ['flgs'] = myLabelEncoder.fit_transform (df ['flgs'])
  df ['proto'] = myLabelEncoder.fit_transform (df ['proto'])
  df ['sport'] = myLabelEncoder.fit_transform (df ['sport'].astype (str))
  df ['dport'] = myLabelEncoder.fit_transform (df ['dport'].astype (str))
  df ['state'] = myLabelEncoder.fit_transform (df ['state'])
  print ('Objects:', list (df.select_dtypes ( ['object']).columns))

  ###############################################################################
  ### Drop unused targets
  ### K: NOTE: category and subcategory are labels for different
  ### applications, not attributes. They must not be used to aid classification.
  print ('\nDropping category and subcategory.')
  print ('These are labels for other scenarios.')
  df.drop (axis = 'columns', columns = 'category', inplace = True)
  df.drop (axis = 'columns', columns = 'subcategory', inplace = True)


  ###############################################################################
  ## Encode Label
  ###############################################################################
  #print ('\nEnconding label.')
  #myLabels = df [TARGET].unique ()
  #print ('Label types before conversion:', myLabels)
  #for label, code in zip (myLabels, range (len (myLabels))):
  #  df [TARGET].replace (label, code, inplace = True)
  #print ('Label types after conversion:', df [TARGET].unique ())


  ###############################################################################
  ## Split dataset into train, validation and test sets
  ###############################################################################
  from sklearn.model_selection import train_test_split
  TEST_SIZE = 2/10
  VALIDATION_SIZE = 1/4
  print ('\nSplitting dataset (test/train):', TEST_SIZE)
  X_train_df, X_test_df, y_train_df, y_test_df = train_test_split (
                                                 df.iloc [:, :-1],
                                                 df.iloc [:, -1],
                                                 test_size = TEST_SIZE,
                                                 random_state = STATE,)
                                                 #shuffle = False)
  print ('\nSplitting dataset (validation/train):', VALIDATION_SIZE)
  X_train_df, X_val_df, y_train_df, y_val_df = train_test_split (
                                               X_train_df,
                                               y_train_df,
                                               test_size = VALIDATION_SIZE,
                                               random_state = STATE,)
                                               #shuffle = False)
  X_train_df.sort_index (inplace = True)
  y_train_df.sort_index (inplace = True)
  X_val_df.sort_index (inplace = True)
  y_val_df.sort_index (inplace = True)
  X_test_df.sort_index (inplace = True)
  y_test_df.sort_index (inplace = True)
  #X_train_df.sort_values  (by = 'pkSeqID', inplace = True)
  print ('X_train_df shape:', X_train_df.shape)
  print ('y_train_df shape:', y_train_df.shape)
  print ('X_val_df shape:', X_val_df.shape)
  print ('y_val_df shape:', y_val_df.shape)
  print ('X_test_df shape:', X_test_df.shape)
  print ('y_test_df shape:', y_test_df.shape)


  ###############################################################################
  ## Imput missing data
  ###############################################################################
  ### K: NOTE: Only use derived information from the train set to avoid leakage.

  from sklearn.impute import SimpleImputer
  for myColumn, myStrategy in zip (columsWithMissingValues, imputingStrategies):
    myImputer = SimpleImputer (missing_values = np.nan, strategy = myStrategy)
    myImputer.fit (X_train_df [myColumn].values.reshape (-1, 1))
    X_train_df [myColumn] = myImputer.transform (X_train_df [myColumn].values.reshape (-1, 1))
    X_val_df [myColumn] = myImputer.transform (X_val_df [myColumn].values.reshape (-1, 1))
    X_test_df [myColumn] = myImputer.transform (X_test_df [myColumn].values.reshape (-1, 1))


  ###############################################################################
  ## Convert dataframe to a numpy array
  ###############################################################################
  print ('\nConverting dataframe to numpy array.')
  X_train = X_train_df.values
  y_train = y_train_df.values
  X_val = X_val_df.values
  y_val = y_val_df.values
  X_test = X_test_df.values
  y_test = y_test_df.values
  print ('X_train shape:', X_train.shape)
  print ('y_train shape:', y_train.shape)
  print ('X_val shape:', X_val.shape)
  print ('y_val shape:', y_val.shape)
  print ('X_test shape:', X_test.shape)
  print ('y_test shape:', y_test.shape)


  ###############################################################################
  ## Apply normalization
  ###############################################################################
  ### K: NOTE: Only use derived information from the train set to avoid leakage.
  from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
  print ('\nApplying normalization.')
  startTime = time.time ()
  scaler = StandardScaler ()
  #scaler = MinMaxScaler (feature_range = (0, 1))
  scaler.fit (X_train)
  X_train = scaler.transform (X_train)
  X_val = scaler.transform (X_val)
  X_test = scaler.transform (X_test)
  print (str (time.time () - startTime), 'to normalize data.')


  ###############################################################################
  ## Perform feature selection
  ###############################################################################
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
  NUMBER_OF_FEATURES = 2 #'all'
  print ('\nSelecting top', NUMBER_OF_FEATURES, 'features.')
  startTime = time.time ()
  #fs = SelectKBest (score_func = mutual_info_classif, k = NUMBER_OF_FEATURES)
  ### K: ~30 minutes to FAIL fit mutual_info_classif to 5% bot-iot
  #fs = SelectKBest (score_func = chi2, k = NUMBER_OF_FEATURES) # X must be >= 0
  ### K: ~4 seconds to fit chi2 to 5% bot-iot (MinMaxScaler (0, 1))
  fs = SelectKBest (score_func = f_classif, k = NUMBER_OF_FEATURES)
  ### K: ~4 seconds to fit f_classif to 5% bot-iot
  fs.fit (X_train, y_train)
  X_train = fs.transform (X_train)
  X_val = fs.transform (X_val)
  X_test = fs.transform (X_test)
  print (str (time.time () - startTime), 'to select features.')
  print ('X_train shape:', X_train.shape)
  print ('y_train shape:', y_train.shape)
  print ('X_val shape:', X_val.shape)
  print ('y_val shape:', y_val.shape)
  print ('X_test shape:', X_test.shape)
  print ('y_test shape:', y_test.shape)
  bestFeatures = []
  for feature in range (len (fs.scores_)):
    bestFeatures.append ({'f': feature, 's': fs.scores_ [feature]})
  bestFeatures = sorted (bestFeatures, key = lambda k: k ['s'])
  for feature in bestFeatures:
    print ('Feature %d: %f' % (feature ['f'], feature ['s']))

  #pyplot.bar ( [i for i in range (len (fs.scores_))], fs.scores_)
  #pyplot.show ()


  ###############################################################################
  ## Handle imbalanced data
  ###############################################################################
  """
  ### K: 10,000 samples per attack
  from collections import Counter
  from imblearn.over_sampling import RandomOverSampler
  from imblearn.under_sampling import RandomUnderSampler
  print ('\nHandling imbalanced label distribution.')

  ### Only oversample
  myOversampler = RandomOverSampler (sampling_strategy = 'not majority',
                                     random_state = STATE)
  X_over, y_over = myOversampler.fit_resample (X_train, y_train)

  ### Only undersample
  myUndersampler = RandomUnderSampler (sampling_strategy = 'not minority',
                                       random_state = STATE)
  X_under, y_under = myUndersampler.fit_resample (X_train, y_train)

  ### Balanced
  MAX_SAMPLES = int (1e4)
  labels = dict (Counter (y_train))

  sampleDictOver = {k : max (labels [k], MAX_SAMPLES) for k in labels}
  balancedOverSampler = RandomOverSampler (sampling_strategy = sampleDictOver,
                                           random_state = STATE)

  X_bal, y_bal = balancedOverSampler.fit_resample (X_train, y_train)
  labels = dict (Counter (y_bal))

  sampleDictUnder = {k : min (labels [k], MAX_SAMPLES) for k in labels}
  balancedUnderSampler = RandomUnderSampler (sampling_strategy = sampleDictUnder,
                                             random_state = STATE)

  X_bal, y_bal = balancedUnderSampler.fit_resample (X_bal, y_bal)

  print ('Real:', Counter (y_train))
  print ('Over:', Counter (y_over))
  print ('Under:', Counter (y_under))
  print ('Balanced:', Counter (y_bal))
  """


  ###############################################################################
  ## Create learning model (Multilayer Perceptron) and tune hyperparameters
  ###############################################################################
  ### K: One hot encode the output.
  import keras.utils
  from keras.utils import to_categorical
  numberOfClasses = len (df [TARGET].unique ())
  print ('y_val:')
  print (y_val [:50])
  print (y_val.shape)
  y_train = keras.utils.to_categorical (y_train, numberOfClasses)
  y_val = keras.utils.to_categorical (y_val, numberOfClasses)
  y_test = keras.utils.to_categorical (y_test, numberOfClasses)

  from sklearn.model_selection import PredefinedSplit
  from sklearn.model_selection import GridSearchCV
  from sklearn.tree import DecisionTreeClassifier
  import time
  ### -1 indices -> train
  ### 0  indices -> validation
  test_fold = np.repeat ([-1, 0], [X_train.shape [0], X_val.shape [0]])
  myPreSplit = PredefinedSplit (test_fold)


  print ('y_val:')
  print (y_val [:50])
  print (y_val.shape)
  #y_val = y_val.argmax (axis = 1)
  print ('y_val:')
  print (y_val [:50])
  print (y_val.shape)
  #y_train = y_train.argmax (axis = 1)

  print ('\nCreating learning model.')
  from keras.models import Sequential
  from keras.layers import Dense, Dropout
  from keras.optimizers import RMSprop
  from keras.optimizers import Adam
  from keras.wrappers.scikit_learn import KerasClassifier
  from keras import metrics
  from keras.constraints import maxnorm
  #BATCH_SIZE = 64
  #NUMBER_OF_EPOCHS = 4
  #LEARNING_RATE = 0.001

  """
  def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0):
    model = Sequential ()
    model.add (Dense (units = 64, activation = 'relu',
                      kernel_constraint = maxnorm (weight_constraint),
                      input_shape = (X_train.shape [1], )))
    model.add (Dropout (dropout_rate))
    model.add (Dense (32, activation = 'relu'))
    model.add (Dense (numberOfClasses, activation = 'softmax'))
    model.compile (loss = 'binary_crossentropy',
                   optimizer = Adam (lr = learn_rate),
                   metrics = ['accuracy'])#, metrics.CategoricalAccuracy ()])
    return model

  model = KerasClassifier (build_fn = create_model, verbose = 2)
  batch_size = [30]#10, 30, 50]
  epochs = [3]#, 5, 10]
  learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
  dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  weight_constraint = [1, 2, 3, 4, 5]
  param_grid = dict (batch_size = batch_size, epochs = epochs,
                     dropout_rate = dropout_rate, learn_rate = learn_rate,
                     weight_constraint = weight_constraint)
  grid = GridSearchCV (estimator = model, param_grid = param_grid,
                       scoring = 'f1_weighted', cv = myPreSplit, verbose = 2,
                       n_jobs = -1)

  grid_result = grid.fit (np.concatenate ( (X_train, X_val), axis = 0),
                          np.concatenate ( (y_train, y_val), axis = 0))
  print (grid_result.best_params_)

  print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip (means, stds, params):
    print ("%f (%f) with: %r" % (mean, stdev, param))


  #Best: 0.999957 using {'batch_size': 30, 'dropout_rate': 0.7, 'epochs': 3, 'learn_rate': 0.3, 'weight_constraint': 1}
  """

  print ('\nCreating learning model.')
  from keras.models import Sequential
  from keras.layers import Dense, Dropout
  BATCH_SIZE = 30
  NUMBER_OF_EPOCHS = 3
  LEARNING_RATE = 0.001
  WEIGHT_CONSTRAINT = 1
  #from sklearn.utils import class_weight
  #class_weight = class_weight.compute_class_weight ('balanced',
  #                                                  np.unique (y_train),
  #                                                  y_train)
  bestModel = Sequential ()
  bestModel.add (Dense (units = 64, activation = 'relu',
                        #kernel_constraint = maxnorm (WEIGHT_CONSTRAINT),
                        kernel_constraint = maxnorm (WEIGHT_CONSTRAINT),
                        input_shape = (X_train.shape [1], )))
  bestModel.add (Dense (32, activation = 'relu'))
  bestModel.add (Dense (numberOfClasses, activation = 'sigmoid'))
  print ('Model summary:')
  bestModel.summary ()

  ###############################################################################
  ## Compile the network
  ###############################################################################
  print ('\nCompiling the network.')
  from keras.optimizers import RMSprop
  from keras.optimizers import Adam
  from keras import metrics
  bestModel.compile (loss = 'binary_crossentropy',
                     optimizer = Adam (lr = LEARNING_RATE),
                     metrics = ['binary_accuracy',
                                #metrics.Recall (),
                                metrics.Precision ()])



  ###############################################################################
  ## Fit the network
  ###############################################################################
  print ('\nFitting the network.')
  startTime = time.time ()
  history = bestModel.fit (X_train, y_train,
                           batch_size = BATCH_SIZE,
                           epochs = NUMBER_OF_EPOCHS,
                           verbose = 2, #1 = progress bar, not useful for logging
                           workers = 0,
                           use_multiprocessing = True,
                           #class_weight = 'auto',
                           validation_data = (X_val, y_val))
  print (str (time.time () - startTime), 's to train model.')


  ###############################################################################
  ## Analyze results
  ###############################################################################
  from sklearn.metrics import confusion_matrix, classification_report
  from sklearn.metrics import confusion_matrix, precision_score, recall_score
  from sklearn.metrics import f1_score, classification_report, accuracy_score
  from sklearn.metrics import cohen_kappa_score
  #y_pred = bestModel.predict (X_val)
  #y_pred = y_pred.round ()
  #print ('\nPerformance on VALIDATION set:')
  #print ('Confusion matrix:')
  #print (confusion_matrix (y_val.argmax (axis = 1), y_pred.argmax (axis = 1),
  #                         labels = df [TARGET].unique ()))
  #print ('Accuracy:', accuracy_score (y_val, y_pred))
  #print ('Precision:', precision_score (y_val, y_pred, average = 'macro'))
  #print ('Recall:', recall_score (y_val, y_pred, average = 'macro'))
  #print ('F1:', f1_score (y_val, y_pred, average = 'macro'))
  #print ('Cohen Kappa:', cohen_kappa_score (y_val.argmax (axis = 1),
  #                                          y_pred.argmax (axis = 1),
  #                                          labels = df [TARGET].unique ()))

  ### K: NOTE: Only look at test results when publishing...
  print ('\nPerformance on TEST set:')
  y_pred = bestModel.predict (X_test)
  y_pred = y_pred.round ()
  print ('Confusion matrix:')
  print (confusion_matrix (y_test.argmax (axis = 1), y_pred.argmax (axis = 1),
                           labels = df [TARGET].unique ()))
  print ('Accuracy:', accuracy_score (y_test, y_pred))
  print ('Precision:', precision_score (y_test, y_pred, average = 'macro'))
  print ('Recall:', recall_score (y_test, y_pred, average = 'macro'))
  print ('F1:', f1_score (y_test, y_pred, average = 'macro'))
  print ('Cohen Kappa:', cohen_kappa_score (y_test.argmax (axis = 1),
                                            y_pred.argmax (axis = 1),
                                            labels = df [TARGET].unique ()))
