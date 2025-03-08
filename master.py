'''  
@author: harshita

- This script is used to combine all the information produced for 
each taxid into a master dataset which can be used to map the 
information and also to build a training datatset for further 
machine learning model.

- The script also puts categorical information as Virus/NonVirus (binary) or
Archeaa/Bacteria/Metazoa/Plant/Fungi/Virus (multicategorical) depending on which 
the category has been selected or not

'''

#Import libraries
import os
import logging
import pandas as pd
import argparse

#Remove old log
if os.path.isfile('master_log.log'):
     os.remove('master_log.log')

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('master_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

#Set up the Parser
parser = argparse.ArgumentParser(description='Master training dataset generation: Combine all the information produced for each taxid into a master dataset which can be used to map the information and also to build a training datatset for further machine learning model.  ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--category',required=True, type=str, default='binary',help='Enter binary/multicategorical to add categorical information as Virus/NonVirus(binary) or Archeaa/Bacteria/Metazoa/Plant/Fungi/Virus (multicategorical)')
parser.add_argument('--dataset',required=True, help= 'Enter the subsampled dataset.')
parser.add_argument('--genomic_features',required=False, help='Enter the genomic features file. Please ignore this option if you have used featuretools.')
parser.add_argument('--accession_id',required=True, help='Enter the accession id file.')
parser.add_argument('--output',required=True, help='Enter the output file name.')
parser.add_argument('--featuretools',required=False, action= 'store_true' ,help='Enter this option if featuretools have been used to create genomic features.')
args = parser.parse_args()

# Function to add a category column based 'Virus' or 'Non virus'  
def category_binary(row):
    if 'Viruses' in row['superkingdom']:
        return 'Virus'
    else:
        return 'Non-Virus'

# Function to add a category column based on 'kingdom' and 'superkingdom' values
def category_multi(row):
    if 'Viruses' in row['superkingdom']:
        return 'Virus'
    elif 'Bacteria' in row['superkingdom']:
        return 'Bacteria'
    elif 'Archaea' in row['superkingdom']:
        return 'Archaea'
    elif 'Fungi' in row['kingdom']:
        return 'Fungi'
    elif 'Metazoa' in row['kingdom']:
        return 'Metazoa'
    elif 'Viridiplantae' in row['kingdom']:
        return 'Plant'
    else:
        return 'Other Eukaryote'

infoLogger.info(f'Received category argument: {args.category}')
infoLogger.info(f'Received featuretools argument:{"True" if args.featuretools else "False"}\n')

#If feature tools option is not selected
if args.featuretools == False: 

    # Load the files
    genus_dataset = pd.read_csv(args.dataset, dtype=str)
    genome_features_df = pd.read_csv(args.genomic_features, dtype=str)
    accession_df = pd.read_csv(args.accession_id, dtype=str)

    # Split 'SeqName' into 'accession' and 'organism details'
    split_columns = genome_features_df['SeqName'].str.split(':', n=1, expand=True)
    genome_features_df['accession'] = split_columns[0]
    genome_features_df['organism details'] = split_columns[1]
    genome_features_df.drop(columns=['SeqName'], inplace=True)

    # Reorder columns to make 'accession' and 'organism details' the first two columns
    cols = ['accession', 'organism details'] + [col for col in genome_features_df.columns if col not in ['accession', 'organism details']]
    genome_features_df = genome_features_df[cols]
    infoLogger.info(f'Genome features dataset after splitting:\n{genome_features_df.head()}\n')

    # Merge accession_df with genus_dataset on 'taxid'
    merged_df = pd.merge(accession_df, genus_dataset, on='taxid', how='inner')
    infoLogger.info(f'Merging the dataset and accession df:\n{merged_df.head()}\n')
    infoLogger.info(f'size:\n{merged_df.shape}\n')

    # Merge the updated merged_df with genomic_features_df on the 'accession' column
    final_df = pd.merge(merged_df, genome_features_df, on='accession', how='left')
    final_df.drop(columns=['organism details'], inplace=True) #Organism details not needed. Drop it!
    infoLogger.info(f'Final dataset:\n{final_df.head()}\n')
    infoLogger.info(f'size:\n{final_df.shape}\n')

    # Add the category column and category values 
    if str(args.category) == 'binary':
        final_df['category'] = final_df.apply(category_binary, axis=1)
        infoLogger.info(f'Final dataset with Category:\n{final_df.head()}\n')
        infoLogger.info(f'size:\n{final_df.shape}\n')
    elif str(args.category) == 'multicategorical':
        final_df['category'] = final_df.apply(category_multi, axis=1)
        infoLogger.info(f'Final dataset with Category:\n{final_df.head()}\n')
        infoLogger.info(f'size:\n{final_df.shape}\n')
    else:
        infoLogger.error('Invalid argument. Please enter either "binary" or "multicategorical"')

    # Save the final dataframe to a master .csv file
    final_df.to_csv(args.output, index=False)

#if featuretools is selected
elif args.featuretools == True: 
    # Load the files
    genus_dataset = pd.read_csv(args.dataset, dtype=str)
    accession_df = pd.read_csv(args.accession_id, dtype=str)

    #Create a merged genomic features dataset (filenames are hardcoded so that the user does not have to enter it. However, the script needs to be in the same folder as the mentioned file)
    genome_features_df = pd.read_csv('biases.csv', dtype=str)
    files = ['biases_feature_set_nuc.csv', 'biases_feature_set_dinucbias.csv', 'biases_feature_set_dinuc.csv', 'biases_feature_set_cpb.csv', 'biases_feature_set_codonpair.csv', 'biases_feature_set_codonbias.csv']
    
    # Merge the remaining feature files
    for file_path in files[1:]:
        df_temp = pd.read_csv(file_path)
        genome_features_df = pd.merge(genome_features_df, df_temp, on='SequenceName', how='left')

    # Split 'SequenceName' to only retain 'accession'
    split_columns = genome_features_df['SequenceName'].str.split(':', n=1, expand=True)
    genome_features_df['accession'] = split_columns[0]
   
   #Drop 'SequenceName' column
    genome_features_df.drop(columns=['SequenceName'], inplace=True)

    # Reorder columns to make 'accession' the first column
    cols = ['accession'] + [col for col in genome_features_df.columns if col not in ['accession']] 
    genome_features_df = genome_features_df[cols]

    infoLogger.info(f'Genome features dataset :\n{genome_features_df.head()}\n')
    infoLogger.info(f'Genome features dataset columns :\n{genome_features_df.shape[1]}\n')
    infoLogger.info(f'Genome features dataset rows :\n{genome_features_df.shape[0]}\n')
    infoLogger.info(f'Genome features dataset info :\n{genome_features_df.info()}\n')
    infoLogger.info(f'Genus dataset :\n{genus_dataset.head()}\n')
    infoLogger.info(f'Accession dataset :\n{accession_df.head()}\n')

    # Merge accession_df with genus_dataset on 'taxid'
    merged_df = pd.merge(accession_df, genus_dataset, on='taxid', how='inner')
    infoLogger.info(f'Merging the dataset and accession df:\n{merged_df.head()}\n')
    infoLogger.info(f'size:\n{merged_df.shape}\n')

    # Merge the updated merged_df with genomic_features_df on the 'accession' column
    final_df = pd.merge(merged_df, genome_features_df, on='accession', how='left')
    infoLogger.info(f'Final dataset:\n{final_df.head()}\n')
    infoLogger.info(f'size:\n{final_df.shape}\n')

    # Add the category column and category values 
    if str(args.category) == 'binary':
        final_df['category'] = final_df.apply(category_binary, axis=1)
        infoLogger.info(f'Final dataset with Category:\n{final_df.head()}\n')
        infoLogger.info(f'size:\n{final_df.shape}\n')
    elif str(args.category) == 'multicategorical':
        final_df['category'] = final_df.apply(category_multi, axis=1)
        infoLogger.info(f'Final dataset with Category:\n{final_df.head()}\n')
        infoLogger.info(f'size:\n{final_df.shape}\n')
    else:
        infoLogger.error('Invalid argument. Please enter either "binary" or "multicategorical"')


    # Save the final dataframe to a master .csv file
    final_df.to_csv(args.output, index=False)
    

infoLogger.info(f'Master dataset prepared and saved to {args.output}.')
infoLogger.info('Process complete.\n')
#____________________________________________________________________END________________________________________________________________________#