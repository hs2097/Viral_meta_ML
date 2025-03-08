'''  
@author: harshita

- This script is for hypertuning the classification
parameters by randomly searching and. cross validating 
each parameters until optimal accurary is achieved.

- The command for this script is
    
    python3 Randomized_seach.csv.py *(nameofthefeaturedataset)*.csv

- The code outputs the best parameters in the log file.

- Note: Make sure to change the name of the log file when running for another feature set.

'''
#Import libaries
import os
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

#Check for preexisting log, if yes remove it
if os.path.isfile('rscv_classifier_log.log'):
     os.remove('rscv_classifier_log.log')

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('rscv_classifier_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

#Intialize the seed
seed = 130499

#Load and check the dataset
file = sys.argv[1]
master_dataset = pd.read_csv(file)
infoLogger.info("The dataset has been loaded successfully")
infoLogger.info(f"Dataset file: {file}")
infoLogger.info(f'Master dataset:\n{master_dataset.head()}')
infoLogger.info(f'Number of rows in master dataset: {master_dataset.shape[0]}\n')
infoLogger.info(f'Number of columns in master dataset: {master_dataset.shape[1]}\n')

#Buil a training model
columns = ['accession','length','superkingdom','kingdom','phylum', 'class', 'order', 'family', 'genus','species','subspecies','strain']
dataset = master_dataset.drop(columns=columns)
infoLogger.info(f'Final dataset:\n{dataset.head()}')
infoLogger.info(f'Number of rows in dataset: {dataset.shape[0]}\n')
infoLogger.info(f'Number of columns in dataset: {dataset.shape[1]}\n')
dataset.drop(columns=['taxid'], inplace=True)

# Count the total number of instances of each class (category column) in the train dataset
target_counts = dataset['category'].value_counts()
total_instances = len(dataset)

# Calculate proportions
proportions = target_counts / total_instances

infoLogger.info(f"Total instances: {total_instances}")
infoLogger.info(f"Number of instances per class: {target_counts}")

infoLogger.info("\nClass Proportions:")
infoLogger.info(f"Proportions: {proportions}")
infoLogger.info(f"Number of instances per class: {proportions * 100}")

# Visualize the class distribution using Matplotlib
plt.bar(target_counts.index, target_counts.values, color='skyblue', edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig('rscv_dataset_proportion.png', dpi=1500)

#Plot histrogram to check feature frequency
dataset.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8,)
plt.savefig('rscv_feature_histogram.png', dpi=1000)

# Convert categorical columns into numeric using LabelEncoder(). This is will enable xgboost learning.
dataset= dataset.fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (dataset.dtypes == object)

# Get list of categorical column names
categorical_columns = dataset.columns[categorical_mask].tolist()

# Print the head of the categorical columns
infoLogger.info(f'Categorical column:\n{dataset[categorical_columns].head()}\n')
infoLogger.info(f'Number of categorical columns: {len(categorical_columns)}\n')

# Apply LabelEncoder to categorical columns and store mappings
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    label_encoders[col] = le

# Print the head of the LabelEncoded categorical columns
infoLogger.info(f'LabelEncoded categorical columns:\n{dataset[categorical_columns].head()}\n')
infoLogger.info(f'Number of unique values in categorical columns:\n{dataset[categorical_columns].nunique()}\n')

# Log unique values for each categorical column individually
for col in categorical_columns:
    unique_values = dataset[col].unique().tolist()
    infoLogger.info(f'Unique values in the {col} column: {unique_values}\n')
    
    # Log the mapping of each category to its corresponding integer
    mapping = dict(zip(label_encoders[col].classes_, label_encoders[col].transform(label_encoders[col].classes_)))
    infoLogger.info(f'Mapping for column {col}: {mapping}\n')

'''
Training the xgboost Classifier model 
and running RandomizedSearchCV to
obtain the best parameters for the model
'''
# Split the dataset into training and test sets
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
infoLogger.info(X)
infoLogger.info(y)


#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
infoLogger.info(f'X_train shape: {X_train.shape[0]}')
infoLogger.info(f'X_test shape: {X_test.shape[0]}')
infoLogger.info(f'y_train shape: {y_train.shape[0]}')
infoLogger.info(f'y_test shape: {y_test.shape[0]}\n')

#Parmeter ranges to hypertune the classification parameters
param_dist = {
    'eta': uniform(0.0, 1.0),
    'learning_rate': uniform(0.0, 1.0),
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.0, 1.0),
    'colsample_bytree': uniform(0.0, 1.0),
    'gamma': uniform(0, 1),
    'lambda': uniform(0, 1),
    'alpha': uniform(0, 1),
}

#Train the model
xgb_clf = xgb.XGBClassifier(booster='gbtree', objective='multi:softmax', eval_metric='mlogloss', seed=seed, num_class = len(np.unique(y)))

#Cross validate
random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_dist, n_iter=100, scoring='accuracy', cv=3, verbose=1, random_state=seed)

#Fit the model
random_search.fit(X_train, y_train)

#Log the best parameters
infoLogger.info(f"\nBest parameters found: {random_search.best_params_}\n ")
infoLogger.info(f"Best accuracy found: {random_search.best_score_} \n")
#____________________________________________________________________END________________________________________________________________________#