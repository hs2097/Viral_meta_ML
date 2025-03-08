'''  
@author: harshita

- This script builds and evaluates a 
multicategorical classification model using 
the XGBoost algorithm. 

- It processes input data, trains the model, 
and provides various performance metrics, 
saving the results and logs for further analysis.

'''
#Import libaries
import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,f1_score, precision_score, recall_score, classification_report, log_loss, roc_curve, precision_recall_curve,confusion_matrix,ConfusionMatrixDisplay,auc)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

#Set up the Parser
parser = argparse.ArgumentParser(description='Multicategorical classification model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--masterdata', required=True, help='Enter the master dataset that will be used for training the model. File format:.csv')
parser.add_argument('--prefix', required=True,help='Enter the prefix you want the outputs to be saved with')
parser.add_argument('--seed',required=True,type=int, default=42, help='Seed for reproducibility')

args = parser.parse_args()
        
#Check if the log is already present            
if os.path.isfile(f'{args.prefix}_multicategorical_classifier_log.log'):
     os.remove(f'{args.prefix}_multicategorical_classifier_log.log')

#Get working directory
current_working_directory = os.getcwd()

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler(f'{args.prefix}_multicategorical_classifier_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

#Initialize the seed
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

infoLogger.info(f"Total instances: {total_instances}")
infoLogger.info(f"Number of instances per class: {target_counts}")
infoLogger.info("\nClass Proportions:")
infoLogger.info(f"Proportions: {proportions}\n")
infoLogger.info(f"Number of instances per class: {proportions * 100}\n")

# Visualize the class distribution using Matplotlib
fig, ax = plt.subplots(figsize=(15, 11))
plt.bar(target_counts.index, target_counts.values, color='skyblue', edgecolor='black')
plt.title('Class Distribution')
plt.xlabel('Class', fontsize = 10)
plt.ylabel('Frequency', fontsize = 10)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f'{args.prefix}_multicategorical_dataset_proportion.png', dpi=1500)
infoLogger.info(f'Dataset proportion saved to: {args.prefix}_multicategorical_dataset_proportion.png')

#Plot histrogram to check feature frequency
dataset.hist(figsize=(20, 20), bins=50, xlabelsize=8, ylabelsize=8,)
plt.tight_layout()
plt.savefig(f'{args.prefix}_multicategorical_feature_histogram.png', dpi=1000)
infoLogger.info(f'Dataset feature histogram saved to: {args.prefix}_multicategorical_feature_histogram.png')

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

# Ensure all data in categorical columns are strings
dataset[categorical_columns] = dataset[categorical_columns].astype(str)

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
Splitting the dataset into train and test
Make sure to startify.

'''
# Split the dataset into training and test sets
X,y = dataset.iloc[:,:-1], dataset.iloc[:,-1]
infoLogger.info(f'Features:\n{X.head()}\n')
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
Using: XGBClssifier

'''

# Train the model based on the parameters in XGBClassifier
num_classes = len(np.unique(y))

#Train the model based on the parameters in XGBClassifier
model = xgb.XGBClassifier(
    booster='gbtree',
    objective='multi:softmax',
    eval_metric='mlogloss',
    eta=0.2,
    gamma=0.1,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.5,
    seed=seed,
    n_estimators=200,
    alpha = 0.5,
    num_class=num_classes,
    reg_lambda=0.34
)

#Model fitting
model.fit(X_train, y_train)

infoLogger.info(f'\nXGBClassifier model fitted!\n')

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

# Accuracy score
infoLogger.info(f'Accuracy score: {(accuracy_score(y_test, y_pred)) * 100.0}%')
# Precision score
infoLogger.info(f'Precision score: {(precision_score(y_test, y_pred, average="weighted"))* 100.0}%')
# Recall score
infoLogger.info(f'Recall score: {(recall_score(y_test, y_pred, average="weighted"))* 100.0}%\n')


# Calculate confusion matrix with normalization
cm = confusion_matrix(y_test, y_pred, normalize='true')
infoLogger.info(f'Confusion matrix:\n{cm}\n')

# Define the correct order of labels
correct_order = ['Archaea', 'Bacteria', 'Fungi', 'Metazoa', 'Other Eukaryote', 'Plant',  'Virus']

# Ensure the labels are in the correct order
# Note: This assumes target_counts.index contains the correct labels
labels = correct_order  

# Plot the confusion matrix with the corrected label order
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Greens, ax=ax)

plt.title('Confusion Matrix', fontsize=12)
plt.xlabel('Predicted Label', fontsize=10)
plt.ylabel('True Label', fontsize=10)
plt.xticks(np.arange(len(labels)), labels, fontsize=8, rotation=30)
plt.yticks(np.arange(len(labels)), labels, fontsize=8)
plt.tight_layout()
plt.savefig(f'{args.prefix}_multicategorical_confusion_matrix.png', dpi=1000)
infoLogger.info(f'\nConfusion matrix saved to: {args.prefix}_multicategorical_confusion_matrix.png\n')

# Classification report
infoLogger.info(f'Classification report:\n{classification_report(y_test, y_pred)}\n')

# F1 score
infoLogger.info(f'F1 score: {f1_score(y_test, y_pred, average="weighted")}\n')

# Log loss score
infoLogger.info(f'Log Loss: {log_loss(y_test, y_pred_proba)}\n')

# Binarize the output for multiclass precision-recall and ROC AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Precision recall calculation for each class
precision = dict()
recall = dict()
thresholds = dict()
for i in range(n_classes):
    precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
    infoLogger.info(f'Class {i} average precision score: {np.mean(precision[i])*100}%')
    infoLogger.info(f'Class {i} average recall score: {np.mean(recall[i])*100}%\n')  


# Plot precision-recall curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (area = {np.mean(precision[i]):0.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid()
plt.savefig(f'{args.prefix}_multicategorical_precision_recall_curve.png', dpi=1000)
infoLogger.info(f'Precision-Recall Curve saved to: {args.prefix}_multicategorical_precision_recall_curve.png')

# ROC AUC calculation and plotting for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    infoLogger.info(f'Class {i} ROC AUC score: {roc_auc[i]}')

infoLogger.info(f'\n')

# Calculate the average ROC AUC score
average_roc_auc = np.mean(list(roc_auc.values()))
infoLogger.info(f'Average ROC AUC score: {average_roc_auc}\n')

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='dashed')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(f'{args.prefix}_multicategorical_roc_curve.png', dpi=1000)
infoLogger.info(f'ROC Curve saved to: {args.prefix}_multicategorical_roc_curve.png')

#Plot the feature importance
xgb.plot_importance(model)
plt.savefig(f'{args.prefix}_multicategorical_feature_importance.png', dpi=1000)
infoLogger.info(f'Feature importance saved to: {args.prefix}_multicategorical_feature_importance.png')

#Plot decision tree
xgb.plot_tree(model,num_trees=137,rankdir="LR")
plt.savefig(f'{args.prefix}_multicategorical_tree.png', dpi=1000)
infoLogger.info(f'Tree saved to: {args.prefix}_multicategorical_tree.png')

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=12)

# Print the cross-validation scores
infoLogger.info(f"Cross-validation scores: {scores}")
infoLogger.info(f"Mean cross-validation score:{scores.mean()}")


'''
Training dataset rearrangement:

- Rearranging the training dataset
- Creating and rearranging the prediction table 
'''

def put_Category_predicted(row):
    if row['Predicted value'] == 0:
        return 'Archaea'
    elif row['Predicted value'] == 1:
        return 'Bacteria'
    elif row['Predicted value'] == 2:
        return 'Fungi'
    elif row['Predicted value'] == 3:
        return 'Metazoa'
    elif row['Predicted value'] == 4:
        return 'Other Eukaryote'
    elif row['Predicted value'] == 5:
        return 'Plant'
    elif row['Predicted value'] == 6:
        return 'Virus'

original_test_data['True value'] = y_test
original_test_data['Predicted value'] = y_pred
original_test_data['Predicted_Category'] = original_test_data.apply(put_Category_predicted, axis=1)
original_test_data.to_csv(f'{args.prefix}_multicategorical_test_pred.csv', index=False)
infoLogger.info(f'Test data with predictions saved to: {args.prefix}_multicategorical_test_pred.csv\n')    

dataset.insert(loc=0, column='taxid', value=master_dataset['taxid'])
dataset['category_master']= master_dataset['category']
dataset.to_csv(f'{args.prefix}_multicategorical_training_dataset.csv', index=False)
infoLogger.info(f'\nData added back to dataset!\n')
infoLogger.info(f'Final dataset:\n{dataset.head()}')
infoLogger.info(f'Training dataset saved to: {args.prefix}_multicategorical_training_dataset.csv')

print(f'Model training and testing completed! Images stored and outputs logged in {args.prefix}_multicategorical_classifier_log.log. Path = {current_working_directory}')
#____________________________________________________________________END_________________________________________________________________________________#