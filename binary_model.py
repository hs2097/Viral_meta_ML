'''  
@author: harshita

- This script is designed to build and evaluate a 
binary classification model using the XGBoost algorithm. 

- It preprocesses the input data, trains the model, 
and evaluates its performance using several metrics, 
saving the output for further use.
'''
#Import libaries
import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,f1_score, precision_score, recall_score, classification_report, log_loss, roc_curve,roc_auc_score, precision_recall_curve,confusion_matrix,ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

#Set up the Parser
parser = argparse.ArgumentParser(description='Binary classification model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--masterdata', required=True, help='Enter the master dataset that will be used for training the model. File format:.csv')
parser.add_argument('--prefix', required=True,help='Enter the prefix you want the outputs to be saved with')
parser.add_argument('--seed',required=True,type=int, default=42, help='Seed for reproducibility')
args = parser.parse_args()

#Check if log exists, if yes remove old
if os.path.isfile(f'{args.prefix}_binary_classifier_log.log'):
     os.remove(f'{args.prefix}_binary_classifier_log.log')

current_working_directory = os.getcwd()

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler(f'{args.prefix}_binary_classifier_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

#Intialize the seed
seed = args.seed

#Load and check the dataset
file = args.masterdata
master_dataset = pd.read_csv(file)
infoLogger.info("The dataset has been loaded successfully")
infoLogger.info(f"Dataset file: {file}")
infoLogger.info(f'Master dataset:\n{master_dataset.head()}')
infoLogger.info(f'Number of rows in master dataset: {master_dataset.shape[0]}\n')
infoLogger.info(f'Number of columns in master dataset: {master_dataset.shape[1]}\n')

#Build a training model
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

infoLogger.info("Class Counts:")
infoLogger.info(target_counts)
infoLogger.info("\nClass Proportions:")
infoLogger.info(proportions)
infoLogger.info("\n")

# Visualize the class distribution using Matplotlib
plt.bar(target_counts.index, target_counts.values, color='skyblue', edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig(f'{args.prefix}_binary_dataset_proportion.png', dpi=1500)
infoLogger.info(f'Dataset proportion saved to: {args.prefix}_binary_dataset_proportion.png')

#Plot histrogram to check feature frequency
dataset.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8,)
plt.savefig(f'{args.prefix}_binary_feature_histogram.png', dpi=1000)
infoLogger.info(f'Feature histogram saved to: {args.prefix}_binary_feature_histogram.png')

'''
Converting categorical columns into numeric using LabelEncoder(). 
This is will enable xgboost learning.

'''
# Create a boolean mask for categorical columns
dataset= dataset.fillna(0)
categorical_mask = (dataset.dtypes == object)

# Get list of categorical column names
categorical_columns = dataset.columns[categorical_mask].tolist()

# Print the head of the categorical columns
infoLogger.info(f'Categorical column:\n{dataset[categorical_columns].head()}\n')

# Print the head of the categorical columns
infoLogger.info(f'Categorical column:\n{dataset[categorical_columns].head()}\n')

# Ensure all data in categorical columns are strings
dataset[categorical_columns] = dataset[categorical_columns].astype(str)

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
dataset[categorical_columns] = dataset[categorical_columns].apply(lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
infoLogger.info(f'LabelEncoded categorical column:\n{dataset[categorical_columns].head()}\n')

'''
Split the dataset into 
train and test sets and
check the distribution of each set.

'''

# Split the dataset into training and test sets
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
infoLogger.info(f'Features:\n{X.head()}')
infoLogger.info(f'Target:\n{y.head()}\n')

# Create an array of indices corresponding to the data
indices = np.arange(len(y))

#Split the data into training and testing sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=seed, stratify=y)
original_test_data = master_dataset.iloc[test_indices]
infoLogger.info(f'X_train shape: {X_train.shape[0]}')
infoLogger.info(f'X_test shape: {X_test.shape[0]}')
infoLogger.info(f'y_train shape: {y_train.shape[0]}')
infoLogger.info(f'y_test shape: {y_test.shape[0]}\n')
infoLogger.info(f'Train indices: {train_indices}\n')
infoLogger.info(f'Test indices: {test_indices}\n')
infoLogger.info(f'Original test data:\n{original_test_data.head()}\n')


# Check the distribution in the training and test sets
train_distribution = pd.Series(y_train).value_counts(normalize=True)
test_distribution = pd.Series(y_test).value_counts(normalize=True)
infoLogger.info("Training set distribution:")
infoLogger.info(train_distribution)
infoLogger.info("\nTest set distribution:")
infoLogger.info(test_distribution)

'''
Training the xgboost Classifier model

'''

#Train the model based on the parameters in XGBClassifier
model = xgb.XGBClassifier(
    booster='gbtree',
    objective='binary:logistic',
    eval_metric='logloss',
    eta=0.2,
    gamma=0.13,
    alpha=0.17,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.5,
    seed=seed,
    n_estimators=137,
    reg_lambda=0.43,
)

#Model fitting
model.fit(X_train, y_train)

infoLogger.info(f'XGBClassifier model fitted!\n')

#Run predictions for test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

'''
Evaluation metrics for the model:

Metrics Calculated:
- Accuracy
- Precision
- Confusion Matrix
- Classification Report
- F1 Score
- Log Loss
- ROC AUC
- Precision-Recall AUC
- Cross Validation
- Feature Importance
- Decision tree
'''

#Accurracy score
infoLogger.info(f'Accuracy score: {(accuracy_score(y_test, y_pred))*100.0}%')
#Precision score
infoLogger.info(f'Precision score: {precision_score(y_test, y_pred)*100.0}%')
#Recall score
infoLogger.info(f'Recall score: {recall_score(y_test, y_pred)*100.0}%\n')

#Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
infoLogger.info(f'Confusion matrix:\n{cm}')
labels = ['Non-Virus', 'Virus']
# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Greens, ax=ax)
plt.title('Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(np.arange(len(labels)), labels, fontsize=10)
plt.yticks(np.arange(len(labels)), labels, fontsize=10)
plt.savefig(f'{args.prefix}_binary_confusion_matrix.png', dpi=1000)
infoLogger.info(f'Confusion matrix saved to: {args.prefix}_binary_confusion_matrix.png')

#Classification report
infoLogger.info(f'Classification report:\n{classification_report(y_test, y_pred)}\n')

#F1 score
infoLogger.info(f'F1 score: {f1_score(y_test, y_pred)}\n')


#Probability prediction
y_probs = model.predict_proba(X_test)[:, 1]
y_probs = np.where(y_probs > 0.5, 1, 0)
infoLogger.info(f'Probability prediction: {y_probs}\n')

#Log loss score
infoLogger.info(f'Log Loss: {log_loss(y_test, y_pred_proba[:,1])}\n')

#ROC AUC score
infoLogger.info(f'ROC AUC score: {roc_auc_score(y_test, y_pred_proba[:,1])}\n')
#Plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='dashed')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(f'{args.prefix}_binary_roc_curve.png', dpi=1000)
infoLogger.info(f'ROC curve saved to: {args.prefix}_binary_roc_curve.png\n')

#Precision recall calculation
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

#Optimal thresholds
optimal_idx = np.argmax(2 * (precision * recall) / (precision + recall))
optimal_threshold = thresholds[optimal_idx]
y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
infoLogger.info(f'Optimal threshold: {optimal_threshold}')
infoLogger.info(f'Optimal precision: {precision[optimal_idx]}')
infoLogger.info(f'Optimal recall: {recall[optimal_idx]}')
infoLogger.info(f'Optimal F1 score: {f1_score(y_test, y_pred_optimal)}')
infoLogger.info(f'Optimal Log Loss: {log_loss(y_test, y_pred_optimal)}\n')


#Plot precision recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig(f'{args.prefix}_binary_precision_recall_curve.png', dpi = 1000)
infoLogger.info(f'Precision recall curve saved to: {args.prefix}_binary_precision_recall_curve.png\n')


#Plot the feature importance
xgb.plot_importance(model)
plt.savefig(f'{args.prefix}_binary_feature_importance.png', dpi=1000)
infoLogger.info(f'Feature importance plot saved to: {args.prefix}_binary_feature_importance.png\n')

#Plot decision tree
xgb.plot_tree(model,num_trees=26,rankdir="LR")
plt.savefig(f'{args.prefix}_binary_tree.png', dpi=1000)
infoLogger.info(f'Tree plot saved to: {args.prefix}_binary_tree.png\n')

#Perform cross-validation
scores = cross_val_score(model, X, y, cv=8)

#Log the cross-validation scores
infoLogger.info(f"Cross-validation scores: {scores}")
infoLogger.info(f"Mean cross-validation score:{scores.mean()}\n")

'''
- Rearranging the dataset
- Creating and rearranging the prediction table 
'''

def put_Category_predicted(row):
    if row['Predicted value'] == 1:
        return 'Virus'
    else:
        return 'NonVirus'

original_test_data['True value'] = y_test
original_test_data['Predicted value'] = y_pred
original_test_data['Predicted_Category'] = original_test_data.apply(put_Category_predicted, axis=1)
original_test_data.to_csv(f'{args.prefix}_binary_test_pred.csv', index=False)
infoLogger.info(f'Test data with predictions saved to: {args.prefix}_binary_test_pred.csv\n')

dataset.insert(loc=0, column='taxid', value=master_dataset['taxid'])
dataset['category_master']= master_dataset['category']
dataset.to_csv(f'{args.prefix}_binary_training_dataset.csv', index=False)
infoLogger.info(f'\nData added back to dataset!\n')
infoLogger.info(f'Final dataset:\n{dataset.head()}')
infoLogger.info(f'Dataset saved to: {args.prefix}_binary_training_dataset.csv\n')

print(f'Model training and testing completed! Images stored and outputs logged in {args.prefix}_binary_classifier_log.log. Path = {current_working_directory}')
#____________________________________________________________________END________________________________________________________________________