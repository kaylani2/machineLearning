# Author: Kaylani Bochie
# github.com/kaylani2
# kaylani AT gta DOT ufrj DOT br

### K: Model: Autoencoder

import pandas as pd
import numpy as np
import sys
import time
import keras.utils
from keras.utils import to_categorical
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras import metrics
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


###############################################################################
## Define constants
###############################################################################
# Random state for reproducibility

STATES = [0, 10, 100, 1000, 10000]
STATE = 0
for STATE in STATES:
  print ('STATE:', STATE)
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
  df = pd.DataFrame ()
  for fileNumber in range (1, FIVE_PERCENT_FILES + 1):#FULL_FILES + 1):
    print ('Reading', FILE_NAME.format (str (fileNumber)))
    aux = pd.read_csv (FILE_NAME.format (str (fileNumber)),
                       #names = featureColumns,
                       index_col = 'pkSeqID',
                       dtype = {'pkSeqID' : np.int32}, na_values = NAN_VALUES,
                       low_memory = False)
    df = pd.concat ( [df, aux])


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
  print (df ['category'].value_counts ())
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
  ### K: Binary classification. Already encoded.

  ###############################################################################
  ## Split dataset into train, validation and test sets
  ###############################################################################
  ### Isolate attack and normal samples
  mask = df [TARGET] == 0
  # 0 == normal
  df_normal = df [mask]
  # 1 == attack
  df_attack = df [~mask]

  print ('Attack set:')
  print (df_attack [TARGET].value_counts ())
  print ('Normal set:')
  print (df_normal [TARGET].value_counts ())

  ### Sample and drop random attacks
  df_random_attacks = df_attack.sample (n = df_normal.shape [0], random_state = STATE)
  df_attack = df_attack.drop (df_random_attacks.index)

  ### Assemble test set
  df_test = pd.DataFrame ()
  df_test = pd.concat ( [df_test, df_normal])
  df_test = pd.concat ( [df_test, df_random_attacks])
  print ('Test set:')
  print (df_test [TARGET].value_counts ())
  X_test_df = df_test.iloc [:, :-1]
  y_test_df = df_test.iloc [:, -1]
  ### K: y_test is required to plot the roc curve in the end



  df_train = df_attack
  VALIDATION_SIZE = 1/4
  print ('\nSplitting dataset (validation/train):', VALIDATION_SIZE)
  X_train_df, X_val_df, y_train_df, y_val_df = train_test_split (
                                               df_train.iloc [:, :-1],
                                               df_train.iloc [:, -1],
                                               test_size = VALIDATION_SIZE,
                                               random_state = STATE,)


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
  print ('\nApplying normalization.')
  startTime = time.time ()
  scaler = StandardScaler ()
  scaler.fit (X_train)
  X_train = scaler.transform (X_train)
  X_val = scaler.transform (X_val)
  X_test = scaler.transform (X_test)
  print (str (time.time () - startTime), 'to normalize data.')


  ###############################################################################
  ## Create learning model (Autoencoder) and tune hyperparameters
  ###############################################################################

  ###############################################################################
  # Hyperparameter tuning
  #test_fold = np.repeat ( [-1, 0], [X_train.shape [0], X_val.shape [0]])
  #myPreSplit = PredefinedSplit (test_fold)
  #def create_model (learn_rate = 0.01, dropout_rate = 0.0, weight_constraint = 0):
  #  model = Sequential ()
  #  model.add (Dense (X_train.shape [1], activation = 'relu',
  #                    input_shape = (X_train.shape [1], )))
  #  model.add (Dense (32, activation = 'relu'))
  #  model.add (Dense (8,  activation = 'relu'))
  #  model.add (Dense (32, activation = 'relu'))
  #  model.add (Dense (X_train.shape [1], activation = None))
  #  model.compile (loss = 'mean_squared_error',
  #                 optimizer = 'adam',
  #                 metrics = ['mse'])
  #  return model
  #
  #model = KerasRegressor (build_fn = create_model, verbose = 2)
  #batch_size = [30]#, 50]
  #epochs = [5]#, 5, 10]
  #learn_rate = [0.01, 0.1]#, 0.2, 0.3]
  #dropout_rate = [0.0, 0.2]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  #weight_constraint = [0]#1, 2, 3, 4, 5]
  #param_grid = dict (batch_size = batch_size, epochs = epochs,
  #                   dropout_rate = dropout_rate, learn_rate = learn_rate,
  #                   weight_constraint = weight_constraint)
  #grid = GridSearchCV (estimator = model, param_grid = param_grid,
  #                     scoring = 'neg_mean_squared_error', cv = myPreSplit,
  #                     verbose = 2, n_jobs = 16)
  #
  #grid_result = grid.fit (np.vstack ( (X_train, X_val)),#, axis = 1),
  #                        np.vstack ( (X_train, X_val)))#, axis = 1))
  #print (grid_result.best_params_)
  #
  #print ("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  #means = grid_result.cv_results_ ['mean_test_score']
  #stds = grid_result.cv_results_ ['std_test_score']
  #params = grid_result.cv_results_ ['params']
  #for mean, stdev, param in zip (means, stds, params):
  #  print ("%f (%f) with: %r" % (mean, stdev, param))
  #
  ## Best: -0.129429 using {'batch_size': 30, 'dropout_rate': 0.0, 'epochs': 5, 'learn_rate': 0.1, 'weight_constraint': 0}


  ###############################################################################
  ## Finished model
  NUMBER_OF_EPOCHS = 5
  BATCH_SIZE = 30
  LEARNING_RATE = 0.1

  INPUT_SHAPE = (X_train.shape [1], )

  print ('\nCreating learning model.')
  bestModel = Sequential ()
  bestModel.add (Dense (X_train.shape [1], activation = 'relu',
                        input_shape = (X_train.shape [1], )))
  bestModel.add (Dense (32, activation = 'relu'))
  bestModel.add (Dense (8,  activation = 'relu'))
  bestModel.add (Dense (32, activation = 'relu'))
  bestModel.add (Dense (X_train.shape [1], activation = None))


  ###############################################################################
  ## Compile the network
  ###############################################################################
  print ('\nCompiling the network.')
  bestModel.compile (loss = 'mean_squared_error',
                     optimizer = Adam (lr = LEARNING_RATE),
                     metrics = ['mse'])#,metrics.Precision ()])

  print ('Model summary:')
  bestModel.summary ()


  ###############################################################################
  ## Fit the network
  ###############################################################################
  print ('\nFitting the network.')
  startTime = time.time ()
  history = bestModel.fit (X_train, X_train,
                           batch_size = BATCH_SIZE,
                           epochs = NUMBER_OF_EPOCHS,
                           verbose = 2, #1 = progress bar, not useful for logging
                           workers = 0,
                           use_multiprocessing = True,
                           #class_weight = 'auto',
                           validation_data = (X_val, X_val))
  print (str (time.time () - startTime), 's to train model.')


  ###############################################################################
  ## Analyze results
  ###############################################################################
  X_val_pred   = bestModel.predict (X_val)
  X_train_pred = bestModel.predict (X_train)
  print ('Train error:'     , mean_squared_error (X_train_pred, X_train))
  print ('Validation error:', mean_squared_error (X_val_pred, X_val))

  #SAMPLES = 50
  #print ('Error on first', SAMPLES, 'samples:')
  #print ('MSE (pred, real)')
  #for pred_sample, real_sample in zip (X_val_pred [:SAMPLES], X_val [:SAMPLES]):
  #  print (mean_squared_error (pred_sample, real_sample))

  ### K: This looks like another hyperparameter to be adjusted by using a
  ### separate validation set that contains normal and anomaly samples.
  ### K: I've guessed 1%, this may be a future line of research.
  THRESHOLD_SAMPLE_PERCENTAGE = 1/100

  train_mse_element_wise = np.mean (np.square (X_train_pred - X_train), axis = 1)
  val_mse_element_wise = np.mean (np.square (X_val_pred - X_val), axis = 1)

  max_threshold_val = np.max (val_mse_element_wise)
  print ('max_Thresh val:', max_threshold_val)



  print ('samples:')
  print (int (round (val_mse_element_wise.shape [0] *
             THRESHOLD_SAMPLE_PERCENTAGE)))

  top_n_values_val = np.partition (-val_mse_element_wise,
                                   int (round (val_mse_element_wise.shape [0] *
                                               THRESHOLD_SAMPLE_PERCENTAGE)))

  top_n_values_val = -top_n_values_val [: int (round (val_mse_element_wise.shape [0] *
                                                      THRESHOLD_SAMPLE_PERCENTAGE))]


  ### K: O limiar de classificacao sera a mediana dos N maiores custos obtidos
  ### ao validar a rede no conjunto de validacao. N e um hiperparametro que pode
  ### ser ajustado, mas e necessario um conjunto de validacao com amostras
  ### anomalas em adicao ao conjunto de validacao atual, que so tem amostras nao
  ### anomalas. @TODO: Desenvolver e validar o conjunto com esta nova tecnica.
  threshold = np.median (top_n_values_val)
  print ('Thresh val:', threshold)


  ### K: NOTE: Only look at test results when publishing...
  #sys.exit ()
  X_test_pred = bestModel.predict (X_test)
  print (X_test_pred.shape)
  print ('Test error:', mean_squared_error (X_test_pred, X_test))


  y_pred = np.mean (np.square (X_test_pred - X_test), axis = 1)
  #y_pred = []
  #for pred_sample, real_sample, label in zip (X_test_pred, X_test, y_test):
  #  y_pred.append (mean_squared_error (pred_sample, real_sample))

  #print ('\nLabel | MSE (pred, real)')
  #for label, pred in zip (y_test, y_pred):
  #  print (label, '|', pred)

  y_test, y_pred = zip (*sorted (zip (y_test, y_pred)))
  #print ('\nLabel | MSE (pred, real) (ordered)')
  #for label, pred in zip (y_test, y_pred):
  #  print (label, '|', pred)

  # 0 == normal
  # 1 == attack
  print ('\nMSE (pred, real) | Label (ordered)')
  tp, tn, fp, fn = 0, 0, 0, 0
  for label, pred in zip (y_test, y_pred):
  #  if (pred >= threshold):
  #    print ('Classified as anomaly     (NORMAL):', label)
  #  else:
  #    print ('Classified as not anomaly (ATTACK):', label)

    if ((pred >= threshold) and (label == 0)):
      print ('True negative.')
      tn += 1
    elif ((pred >= threshold) and (label == 1)):
      print ('False negative!')
      fn += 1
    elif ((pred < threshold) and (label == 1)):
      print ('True positive.')
      tp += 1
    elif ((pred < threshold) and (label == 0)):
      print ('False positive!')
      fp += 1

  print ('Confusion matrix:')
  print ('tp | fp')
  print ('fn | tn')
  print (tp, '|', fp)
  print (fn, '|', tn)


  ### K: @TODO: Decide how to find the optimal threshold value for classification whilst avoiding data leakage.

  # https://towardsdatascience.com/anomaly-detection-with-autoencoder-b4cdce4866a6
  ## /\ This approach looks directly at the test results to find a good separation (BUT WITHOUT THE LABELS), so it's not leaking...????

  # https://www.pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/
  ## /\ This one uses "quantiles". Poorly explained and organized...

  # https://keras.io/examples/timeseries/timeseries_anomaly_detection/
  ## /\
  #1. Find MAE loss on training samples.
  #2. Find max MAE loss value. This is the worst our model has performed trying to reconstruct a sample. We will make this the threshold for anomaly detection.
  #3. If the reconstruction loss for a sample is greater than this threshold value then we can infer that the model is seeing a pattern that it isn't familiar with. We will label this sample as an anomaly.
  ### K: Esta parece a mais razoável...



  ### K: Scaling the losses was not a good idea...
  #y_pred = np.array (y_pred)
  #scaler = MinMaxScaler ()
  #y_pred = scaler.fit_transform (y_pred.reshape (-1, 1))
  #print (y_pred)
  #y_pred = y_pred.round ()

  #print ('pred | real')
  #print ('All:')
  #for pred, real in zip (y_pred, y_test):
  #  print (pred, '|', real)
  #
  #print ('pred | real')
  #print ('Wrongs:')
  #for pred, real in zip (y_pred, y_test):
  #  if (pred != real):
  #    print (pred, '|', real)




  ### @TODO: Plot ROC on test set
  ### K: ROC
  #ns_probs = [0 for _ in range (len (y_test))] # no skill predictor
  #ns_auc = roc_auc_score (y_test, ns_probs)
  #lr_auc = roc_auc_score (y_test, y_pred)
  #
  #print ('No Skill: ROC AUC = %.3f' % (ns_auc))
  #print ('Autoencoders: ROC AUC = %.3f' % (lr_auc))
  #
  #ns_fpr, ns_tpr, _ = roc_curve (y_test, ns_probs)
  #lr_fpr, lr_tpr, _ = roc_curve (y_test, lr_probs)
  #
  ## plot the roc curve for the model
  #pyplot.plot (ns_fpr, ns_tpr, linestyle = '--', label = 'No Skill')
  #pyplot.plot (lr_fpr, lr_tpr, marker = '.', label = 'Autoencoder')
  ## axis labels
  #pyplot.xlabel ('False Positive Rate')
  #pyplot.ylabel ('True Positive Rate')
  ## show the legend
  #pyplot.legend ()
  ## show the plot
  #pyplot.savefig ('roc_autoencoder.png')
