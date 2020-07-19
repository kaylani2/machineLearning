{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Author: Ernesto Rodríguez\n",
    "# github.com/ernestorodg\n",
    "\n",
    "###############################################################################\n",
    "## Analyse Bezerra's dataset for intrusion detection using Decision Trees\n",
    "###############################################################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import sys\n",
    "\n",
    "###############################################################################\n",
    "## Define constants \n",
    "###############################################################################\n",
    "\n",
    "\n",
    "# Random state for reproducibility\n",
    "STATE = 0\n",
    "np.random.seed(10)\n",
    "# List of available attacks on the dataset\n",
    "\n",
    "\n",
    "# Especific to the repository \n",
    "DATASET_DIRECTORY = r'../datasets/Dataset-bezerra-IoT-20200528T203526Z-001/Dataset-IoT/'\n",
    "NETFLOW_DIRECTORY = r'NetFlow/'\n",
    "\n",
    "\n",
    "# There are different csv files on the Dataset, with different types of data:\n",
    "\n",
    "# Some meanings:\n",
    "# MC: Media Center\n",
    "# I: One hour of legitimate and malicious NetFlow data from profile.\n",
    "# L: One hour of legitimate NetFlow data from profile.\n",
    "\n",
    "MC = r'MC/'\n",
    "ST = r'ST/'\n",
    "SC = r'SC/'\n",
    "\n",
    "\n",
    "# MC_I_FIRST: Has infected data by Hajime, Aidra and BashLite botnets \n",
    "MC_I_FIRST = r'MC_I1.csv'\n",
    "\n",
    "# MC_I_SECOND: Has infected data from Mirai botnets\n",
    "MC_I_SECOND = r'MC_I2.csv'\n",
    "\n",
    "# MC_I_THIR: Has infected data from Mirai, Doflo, Tsunami and Wroba botnets\n",
    "MC_I_THIRD = r'MC_I3.csv'\n",
    "\n",
    "# MC_L: Has legitimate data, no infection\n",
    "MC_L = r'MC_L.csv'\n",
    "\n",
    "\n",
    "# Constants for ST\n",
    "ST_I_FIRST = r'ST_I1.csv'\n",
    "ST_I_SECOND = r'ST_I2.csv'\n",
    "ST_I_THIRD = r'ST_I3.csv'\n",
    "ST_L = r'ST_L.csv'\n",
    "\n",
    "# Constants for SC\n",
    "SC_I_FIRST = r'SC_I1.csv'\n",
    "SC_I_SECOND = r'SC_I2.csv'\n",
    "SC_I_THIRD = r'SC_I3.csv'\n",
    "SC_L = r'SC_L.csv'\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################################################################\n",
    "## Load dataset\n",
    "###############################################################################\n",
    "\n",
    "# For MC data:\n",
    "df_mc_I_first = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_FIRST)\n",
    "df_mc_I_second = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_SECOND)\n",
    "df_mc_I_third = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_I_THIRD)\n",
    "\n",
    "# Add legitimate rows from MC_L\n",
    "legitimate_frame_mc = pd.read_csv (DATASET_DIRECTORY + MC + NETFLOW_DIRECTORY + MC_L)\n",
    "\n",
    "###################\n",
    "\n",
    "# For ST data:\n",
    "df_st_I_first = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_FIRST)\n",
    "df_st_I_second = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_SECOND)\n",
    "df_st_I_third = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_I_THIRD)\n",
    "\n",
    "# Add legitimate rows from SC_L\n",
    "legitimate_frame_st = pd.read_csv (DATASET_DIRECTORY + ST + NETFLOW_DIRECTORY + ST_L)\n",
    "\n",
    "\n",
    "###################\n",
    "\n",
    "# For SC data:\n",
    "df_sc_I_first = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_FIRST)\n",
    "df_sc_I_second = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_SECOND)\n",
    "df_sc_I_third = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_I_THIRD)\n",
    "\n",
    "# Add legitimate rows from MC_L\n",
    "legitimate_frame_sc = pd.read_csv (DATASET_DIRECTORY + SC + NETFLOW_DIRECTORY + SC_L)\n",
    "\n",
    "dataframes_list = [df_mc_I_first,\n",
    "                df_mc_I_second,\n",
    "                df_mc_I_third,\n",
    "                legitimate_frame_mc,\n",
    "                df_st_I_first,\n",
    "                df_st_I_second,\n",
    "                df_st_I_third,\n",
    "                legitimate_frame_st,\n",
    "                df_sc_I_first,\n",
    "                df_sc_I_second,\n",
    "                df_sc_I_third,\n",
    "                legitimate_frame_sc]\n",
    "\n",
    "# Joining the differents DataFrames\n",
    "prev_df = pd.concat(dataframes_list)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "###############################################################################\n",
    "## Modify the DataFrame\n",
    "###############################################################################\n",
    "\n",
    "\n",
    "# Sample the dataset if necessary\n",
    "df = prev_df.sample (frac = 0.1, replace = True, random_state = 0)\n",
    "\n",
    "# We can see that this dataset has a temporal description.\n",
    "# So it is not a good idea to randomly remove rows if using RNN\n",
    "\n",
    "# In this case we drop the index column, since pandas library creates an index\n",
    "# automatically. \n",
    "df = df.drop(df.columns[0], axis=1)\n",
    "\n",
    "# Also drop columns that has no significant data\n",
    "df = df.drop(df.columns[14:], axis=1)\n",
    "\n",
    "# Initial and end time is not a good feature for svm model\n",
    "df = df.drop(['ts', 'te'], axis=1)\n",
    "\n",
    "# Trying another drops to see relation between features and results\n",
    "df = df.drop(['fwd', 'stos'], axis=1)\n",
    "# 'sp', 'dp', 'sa',  'da',  \n",
    "\n",
    "# Counting number of null data\n",
    "nanColumns = [i for i in df.columns if df [i].isnull ().any ()]\n",
    "\n",
    "# Remove NaN and inf values\n",
    "df.replace ('Infinity', np.nan, inplace = True) ## Or other text values\n",
    "df.replace (np.inf, np.nan, inplace = True) ## Remove infinity\n",
    "df.replace (np.nan, 0, inplace = True)\n",
    "\n",
    "\n",
    "# if (df.Label.value_counts()[1] < df.Label.value_counts()[0]):\n",
    "#     remove_n =  df.Label.value_counts()[0] - df.Label.value_counts()[1]  # Number of rows to be removed   \n",
    "#     print(remove_n)\n",
    "#     df_to_be_dropped = df[df.Label == 0]\n",
    "#     drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)\n",
    "#     df = df.drop(drop_indices)\n",
    "# else: \n",
    "#     remove_n =  df.Label.value_counts()[1] - df.Label.value_counts()[0]  # Number of rows to be removed   \n",
    "#     print(remove_n)\n",
    "#     df_to_be_dropped = df[df.Label == 1]\n",
    "#     drop_indices = np.random.choice(df_to_be_dropped.index, remove_n, replace=False)\n",
    "#     df = df.drop(drop_indices)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of non-attacks:  818\n",
      "Number of attacks:  171623\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Slice the dataframe (usually the last column is the target)\n",
    "###############################################################################\n",
    "\n",
    "X = pd.DataFrame(df.iloc [:, 1:])\n",
    "\n",
    "# Selecting other columns\n",
    "# X = pd.concat([X, df.iloc[:, 2]], axis=1)\n",
    "\n",
    "y = df.iloc [:, 0]\n",
    "print('Number of non-attacks: ', y.value_counts()[0])\n",
    "print('Number of attacks: ', y.value_counts()[1])\n",
    "\n",
    "# See Output, only available on jupyter-notebooks\n",
    "# X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of non-attacks:  171623\n",
      "Number of attacks:  171623\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Create artificial non-attacks samples using Random Oversampling\n",
    "###############################################################################\n",
    "\n",
    "from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE\n",
    "\n",
    "ros = RandomOverSampler(random_state=42)\n",
    "\n",
    "X, y = ros.fit_resample(X, y)\n",
    "\n",
    "print('Number of non-attacks: ', y.value_counts()[0])\n",
    "print('Number of attacks: ', y.value_counts()[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ###############################################################################\n",
    "# ## Create artificial non-attacks samples using Random undersampling\n",
    "# ###############################################################################\n",
    "\n",
    "# from imblearn.under_sampling import RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE\n",
    "\n",
    "# ros = RandomUnderSampler(random_state=42)\n",
    "\n",
    "# X, y = ros.fit_resample(X, y)\n",
    "\n",
    "# print('Number of non-attacks: ', y.value_counts()[0])\n",
    "# print('Number of attacks: ', y.value_counts()[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Splitting dataset (validation/train): 0.2\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Split dataset into train and test sets if not using cross validation\n",
    "###############################################################################\n",
    "from sklearn.model_selection import train_test_split\n",
    "TEST_SIZE = 1/5\n",
    "VALIDATION_SIZE = 1/5\n",
    "\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = TEST_SIZE,\n",
    "                                                     random_state = STATE)\n",
    "\n",
    "\n",
    "print ('\\nSplitting dataset (validation/train):', VALIDATION_SIZE)\n",
    "X_train_val, X_val, y_train_val, y_val = train_test_split (\n",
    "                                             X_train,\n",
    "                                             y_train,\n",
    "                                             test_size = VALIDATION_SIZE,\n",
    "                                             random_state = STATE)\n",
    "\n",
    "\n",
    "X_train = pd.DataFrame(X_train)\n",
    "X_test = pd.DataFrame(X_test)\n",
    "X_train_val = pd.DataFrame(X_train_val)\n",
    "X_val = pd.DataFrame(X_val)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "####################################################################\n",
    "# Treat categorical data on train set\n",
    "####################################################################\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "\n",
    "\n",
    "cat_cols = X_train.columns[X_train.dtypes == 'O'] # Returns array with the columns that has Object types elements\n",
    "\n",
    "categories = [\n",
    "    X_train[column].unique() for column in X_train[cat_cols]]\n",
    "\n",
    "for cat in categories:\n",
    "    cat[cat == None] = 'missing'  # noqa\n",
    "\n",
    "# Replacing missing values\n",
    "categorical_imputer = SimpleImputer(missing_values=None, \n",
    "                                    strategy='constant', \n",
    "                                    fill_value='missing')\n",
    "\n",
    "X_train[cat_cols] = categorical_imputer.fit_transform(X_train[cat_cols])\n",
    "\n",
    "# Encoding the categorical data\n",
    "categorical_encoder = OrdinalEncoder(categories = categories)\n",
    "categorical_encoder.fit(X_train[cat_cols])\n",
    "X_train[cat_cols] = categorical_encoder.transform(X_train[cat_cols])\n",
    "\n",
    "categorical_encoder.fit(X_train_val[cat_cols])\n",
    "X_train_val[cat_cols] = categorical_encoder.transform(X_train_val[cat_cols])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "####################################################################\n",
    "# Treat categorical data on test set\n",
    "####################################################################\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "\n",
    "\n",
    "cat_cols = X_test.columns[X_test.dtypes == 'O'] # Returns array with the columns that has Object types elements\n",
    "\n",
    "categories = [\n",
    "    X_test[column].unique() for column in X_test[cat_cols]]\n",
    "\n",
    "for cat in categories:\n",
    "    cat[cat == None] = 'missing'  # noqa\n",
    "\n",
    "# Replacing missing values\n",
    "categorical_imputer = SimpleImputer(missing_values=None, \n",
    "                                    strategy='constant', \n",
    "                                    fill_value='missing')\n",
    "\n",
    "X_test[cat_cols] = categorical_imputer.fit_transform(X_test[cat_cols])\n",
    "\n",
    "# Encoding the categorical data\n",
    "categorical_encoder = OrdinalEncoder(categories = categories)\n",
    "categorical_encoder.fit(X_test[cat_cols])\n",
    "X_test[cat_cols] = categorical_encoder.transform(X_test[cat_cols])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "####################################################################\n",
    "# Treat categorical data on val set\n",
    "####################################################################\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "\n",
    "\n",
    "cat_cols = X_val.columns[X_val.dtypes == 'O'] # Returns array with the columns that has Object types elements\n",
    "\n",
    "categories = [\n",
    "    X_val[column].unique() for column in X_val[cat_cols]]\n",
    "\n",
    "for cat in categories:\n",
    "    cat[cat == None] = 'missing'  # noqa\n",
    "\n",
    "# Replacing missing values\n",
    "categorical_imputer = SimpleImputer(missing_values=None, \n",
    "                                    strategy='constant', \n",
    "                                    fill_value='missing')\n",
    "\n",
    "X_val[cat_cols] = categorical_imputer.fit_transform(X_val[cat_cols])\n",
    "\n",
    "# Encoding the categorical data\n",
    "categorical_encoder = OrdinalEncoder(categories = categories)\n",
    "categorical_encoder.fit(X_val[cat_cols])\n",
    "X_val[cat_cols] = categorical_encoder.transform(X_val[cat_cols])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "####################################################################\n",
    "# Treat numerical data \n",
    "####################################################################\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "\n",
    "num_cols = X_train.columns[(X_train.dtypes == 'float64') | (X_train.dtypes == 'int64')] # Returns array with the columns that has float types elements\n",
    "\n",
    "# Scaling numerical values\n",
    "\n",
    "numerical_scaler = StandardScaler()\n",
    "numerical_scaler.fit(X_train)\n",
    "X_train = numerical_scaler.transform(X_train)\n",
    "\n",
    "X_test = numerical_scaler.transform(X_test)\n",
    "\n",
    "X_val = numerical_scaler.transform(X_val)\n",
    "\n",
    "# X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    171623\n",
       "0    171623\n",
       "Name: Label, dtype: int64"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 1 folds for each of 288 candidates, totalling 288 fits\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.\n",
      "[Parallel(n_jobs=1)]: Done 288 out of 288 | elapsed: 349.8min finished\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'bootstrap': True, 'criterion': 'entropy', 'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 100}\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Create learning model (Random Forest) and tune hyperparameters\n",
    "###############################################################################\n",
    "from sklearn.model_selection import PredefinedSplit\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import time\n",
    "\n",
    "\n",
    "\n",
    "### -1 indices -> train\n",
    "### 0  indices -> validation\n",
    "test_fold = np.repeat ([-1, 0], [X_train_val.shape [0], X_val.shape [0]])\n",
    "myPreSplit = PredefinedSplit (test_fold)\n",
    "#myPreSplit.get_n_splits ()\n",
    "#myPreSplit.split ()\n",
    "#for train_index, test_index in myPreSplit.split ():\n",
    "#    print (\"TRAIN:\", train_index, \"TEST:\", test_index)\n",
    "\n",
    "\n",
    "parameters = {'n_estimators' : [100, 200], \n",
    "              'criterion' : ['gini', 'entropy'],\n",
    "              'max_depth' : [1, 10, 100, 1000, 10000, 100000, 1000000, None],\n",
    "              'min_samples_split' : [2],\n",
    "             'bootstrap' : [True, False]}\n",
    "\n",
    "clf = RandomForestClassifier ()\n",
    "\n",
    "model = GridSearchCV (estimator = clf,\n",
    "                      param_grid = parameters,\n",
    "                      scoring = 'f1_weighted',\n",
    "                      cv = myPreSplit,\n",
    "                      verbose = 1)\n",
    "\n",
    "model.fit (np.concatenate ((X_train_val, X_val), axis = 0),\n",
    "           np.concatenate ((y_train_val, y_val), axis = 0))\n",
    "\n",
    "print (model.best_params_)\n",
    "\n",
    "#{'bootstrap': True, 'criterion': 'entropy', 'max_depth': 1, 'min_samples_split': 2, 'n_estimators': 100}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy Score : 0.5973925710123816\n",
      "Precision Score : 0.5556306741355997\n",
      "Recall Score : 0.9990448901623686\n",
      "F1 Score : 0.7141039565554693\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[ 6493, 27606],\n",
       "       [   33, 34518]])"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# ###############################################################################\n",
    "# ## Obtain metrics from the validation model \n",
    "# ###############################################################################\n",
    "\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import multilabel_confusion_matrix\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# New Model Evaluation metrics \n",
    "print('Accuracy Score : ' + str(accuracy_score(y_test,y_pred)))\n",
    "print('Precision Score : ' + str(precision_score(y_test,y_pred)))\n",
    "print('Recall Score : ' + str(recall_score(y_test,y_pred)))\n",
    "print('F1 Score : ' + str(f1_score(y_test,y_pred)))\n",
    "\n",
    "#Logistic Regression (Grid Search) Confusion matrix\n",
    "confusion_matrix(y_test,y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Plotting confusion matrix\n",
    "###############################################################################\n",
    "from sklearn.metrics import plot_confusion_matrix\n",
    "from matplotlib import pyplot as plt\n",
    "\n",
    "plot_confusion_matrix(model, X_test, y_test)  # doctest: +SKIP\n",
    "plt.savefig(\"random_forest_confusion_matrix_with_tuning.png\", format=\"png\")\n",
    "plt.show()  # doctest: +SKIP\n",
    "# td  sp  dp  pr  flg  ipkt ibyt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--- 14.468616485595703 seconds ---\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Train the model with other parameters\n",
    "###############################################################################\n",
    "\n",
    "# Measure time of this training\n",
    "start_time = time.time()\n",
    "\n",
    "# Assign the model to be used with adjusted parameters\n",
    "clf = RandomForestClassifier(max_depth = 1000000)\n",
    "\n",
    "# Training the model\n",
    "model = clf.fit(X_train, y_train)\n",
    "print(\"--- %s seconds ---\" % (time.time() - start_time))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Precision Score:  0.9828651125722255\n",
      "Recall Score:  0.9994211455529507\n",
      "Accuracy:  0.9909395484340859\n",
      "F1 Score:  0.9910739911600942\n",
      "[[[34531    20]\n",
      "  [  602 33497]]\n",
      "\n",
      " [[33497   602]\n",
      "  [   20 34531]]]\n"
     ]
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Obtain metrics from the above model \n",
    "###############################################################################\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "from sklearn.metrics import accuracy_score\n",
    "from sklearn.metrics import multilabel_confusion_matrix\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "\n",
    "# Predicting from the test slice\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "# Precision == TP / (TP + FP)\n",
    "print('Precision Score: ', precision_score(y_test, y_pred))\n",
    "\n",
    "# Recall == TP / (TP + FN)\n",
    "print('Recall Score: ', recall_score(y_test, y_pred))\n",
    "\n",
    "# Accuracy \n",
    "train_score = model.score(X_test, y_test)\n",
    "print('Accuracy: ', train_score)\n",
    "\n",
    "# f1 \n",
    "f_one_score = f1_score(y_test, y_pred)\n",
    "print('F1 Score: ', f_one_score)\n",
    "\n",
    "# Multilabel Confusion Matrix: \n",
    "# [tn fp]\n",
    "# [fn tp]\n",
    "print(multilabel_confusion_matrix(y_test, y_pred, labels=[0, 1]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "###############################################################################\n",
    "## Plotting confusion matrix\n",
    "###############################################################################\n",
    "\n",
    "plot_confusion_matrix(model, X_test, y_test)  # doctest: +SKIP\n",
    "plt.savefig(\"random_forest_confusion_matrix_without_tuning.png\", format=\"png\")\n",
    "plt.show()  # doctest: +SKIP\n",
    "# td  sp  dp  pr  flg  ipkt ibyt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
