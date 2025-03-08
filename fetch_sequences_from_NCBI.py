"""
@author: harshita

- This script downloads sequences of random lengths 
from the NCBI database based on provided accession numbers. 

- It saves these sequences in a FASTA file and generates 
an accession file containing the fetched accession IDs, 
allowing for further analysis or model training.

"""
#Import libraries
import os
import logging
import argparse
import sys
import pandas as pd
from Bio import Entrez
import fetch_sequences_from_NCBI_functions as functions

#Set up the Parser
parser = argparse.ArgumentParser(description='Fetch sequences from NCBI: Download the sequence of random lengths from the NCBI database and save it in a fasta file. The script will also provide output an accession file of all the accession ids that were fetched.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', required=True, help='Enter the name of the dataset file for which sequences are to be fetched. File format:.csv')
parser.add_argument('--accessionfile', required=True,help='Enter the file which contained accesssion numbers corresponding to taxids.File format:.csv')
parser.add_argument('--contiginfo', required=True,help='Enter the a sample list of contigs to create a length distribution. This will be used to get the start and stop positions for the the sequence.File format:.csv')
parser.add_argument('--batch', required=False,default=10,type=int,help='Enter the batch size to fetch the sequences from NCBI')
parser.add_argument('--seed',default=10,type=int,required=False, help='Enter a seed value')
parser.add_argument('--outputprefix', required=True, help='Enter the prefix to save the fasta file and fetched accession information.')
parser.add_argument('--api', required=True,type=str, help='Enter the API key for NCBI.')
parser.add_argument('--email', required=True,type=str, help='Enter the email address for NCBI.')
args = parser.parse_args()

#Remove old log
if os.path.isfile('sequence_fetch_log.log'):
     os.remove('sequence_fetch_log.log')

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('sequence_fetch_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)


infoLogger.info('Fetch Sequences form NCBI:')
infoLogger.info('Starting the program')
infoLogger.info('Dataset: {}'.format(args.dataset))
infoLogger.info('Accession file: {}'.format(args.accessionfile))
infoLogger.info('Contig info: {}'.format(args.contiginfo))
infoLogger.info('Batch size: {}'.format(args.batch))
infoLogger.info('Seed: {}'.format(args.seed))
infoLogger.info('Output prefix: {}'.format(args.outputprefix))
infoLogger.info('API key: {}'.format(args.api))
infoLogger.info('Email: {}'.format(args.email))
infoLogger.info('')

#Input intialization
dataset_file = args.dataset
ta_file = args.accessionfile
contig_file = args.contiginfo
batch = args.batch
seed = args.seed
output_fasta = str(args.outputprefix + '_fasta.fasta')
output_accession_csv = str(args.outputprefix + '_accession.csv')
output_genomic_features = str(args.outputprefix + '_genomic_features.csv')
master_output= str(args.outputprefix + '_master_dataset.csv')
api = args.api
email = args.email
category = args.category

try:
    # Read dataset and accession input files
    dataset = pd.read_csv(dataset_file)
    taxid_accession_df = pd.read_csv(ta_file, sep='\t', header=None, names=['taxid', 'organism', 'accession', 'length'], dtype=str)
except Exception as e:
    infoLogger.error('Error in loading the input files')
    infoLogger.error(e)
    sys.exit(1)

# Ensure the taxid columns are strings
dataset['taxid'] = dataset['taxid'].astype(str)
taxid_accession_df['taxid'] = taxid_accession_df['taxid'].astype(str)

# Get the unique taxid values
unique_taxid_list = dataset['taxid'].unique().tolist()

# Ensure the list and DataFrame are sorted (necessary for binary search)
unique_taxid_list = sorted(unique_taxid_list)
taxid_accession_df = taxid_accession_df.sort_values(by=['taxid'])

#Iiterate over the list to select the accession of the largest length and save it in a csv file
rows = [] #Intilialize to store the taxid, accession and length information
for taxid in unique_taxid_list:
    try:
        # Peform a binary search to find get a list of all the rows with the current taxid
        subset_df = functions.taxid_binary_search(taxid_accession_df, taxid) 
        #Subset the list to only retain the genomic accession numbers
        filtered_subset_df = subset_df.loc[subset_df['accession'].str.startswith(('NZ', 'NG', 'NW', 'NC', 'NT', 'AC'))]

        if not filtered_subset_df.empty: #if the filtered list is not empty
            # Find the accession with the largest length for the current taxid
            accession_row = functions.get_max_length_row(filtered_subset_df)
            infoLogger.info(f'{accession_row}')
            accession_number = accession_row['accession']
            length = accession_row['length']
            # Append the current taxid, accession id and length information to the 'row' list
            rows.append([taxid, accession_number, length])
        else:
            infoLogger.info(f"\nNo accession numbers found for taxid: {taxid}\n")
            continue
    except Exception as e:
        infoLogger.error(f'Error in processing taxid: {taxid}')
        infoLogger.error(e)
        continue

# Convert the list to a DataFrame
accession_df = pd.DataFrame(rows, columns=['taxid', 'accession', 'length'])


try:
    # Save the accession information to a csv file
    accession_df.to_csv(output_accession_csv, index=False)
    infoLogger.info(f"\nAccession numbers saved to {output_accession_csv}\n{accession_df.head()}\n")

except Exception as e:
    infoLogger.error('Error in saving the accession file:')
    infoLogger.error(e)
    sys.exit(1)


# Find the missing taxid values
missing_taxids = [taxid for taxid in taxid_accession_df['taxid'].values if taxid not in  accession_df['taxid'].values]
if missing_taxids:
    infoLogger.info(f"Missing taxid values: {missing_taxids}\nTotal taxids missing: {len(missing_taxids)}")
else:
    infoLogger.info(f'All taxids have been founds. Proceeding...\n')


# Set the Entrez API key and email
Entrez.api_key = api
Entrez.email = email

# Get the accession numbers from the DataFrame
accession_ids = accession_df['accession'].tolist() #Convert the accession ids to a list
infoLogger.info(f'Total number of Accession IDs : {len(accession_ids)}\n{accession_ids}\n')
lengths= accession_df['length'].tolist() #Covert the length column to list
infoLogger.info(f'Total number of Lengths : {len(lengths)}\n{lengths}\n')

# Initialize an empty list to store the sequences
sequences = []

# Set the contig lengths according to the dataset size
try:
    n = dataset.shape[0] #Get the number of rows in the accession dataframe
    # Get the list of contigs for according to the length of the dataframe
    contig_lengths = functions.distribution(filename=contig_file, column='lengths', n=n, seed=seed)
    infoLogger.info(f'\nList of contigs:\n{contig_lengths}')
    infoLogger.info(f'\nTotal number of contigs: {len(contig_lengths)}\n')
except Exception as e:
    infoLogger.error('Error in getting the contig lengths:')
    infoLogger.error(e)
    sys.exit(1)

infoLogger.info('Fetching the dataset')
infoLogger.info('')

try:
    # Get EPost search keys: WebEnv and QueryKey
    search_results = Entrez.read(Entrez.epost("nucleotide", id=",".join(accession_ids)))
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]
    infoLogger.info(f'\nWebEnv: {webenv}\nQueryKey: {query_key}\n')
except Exception as e:
    infoLogger.error('Error in EPost search:')
    infoLogger.error(e)
    sys.exit(1)

# Iterate over the accession IDs in batches
for i in range(0, len(accession_ids), batch):
    try:
        # Get the current batch of accession IDs
        nucleotide_ids = accession_ids[i:i + batch]
        if not nucleotide_ids: #Check if the nuleotide_id list empty or not
            infoLogger.error(f'No nucleotide accession number found in batch [{i:i+batch}]')

        # Get the lengths of the sequences in the current batch
        length_list = lengths[i:i+batch]
        if not length_list: #Check if the length list is empty or not
            infoLogger.info(f'No length found in batch [{i:i+batch}]') 
        # Get a list of contigs from the distribution curve for the current batch
        contig = contig_lengths[i:i+batch]
        if not contig: #Check if the contig list is empty
            infoLogger.info(f'No contig found in batch [{i:i+batch}]')
        
        
        # Get the start and stop positions for each sequence in the batch
        start, stop = functions.get_positions(sequence_length=length_list, contig_length=contig, seed=seed)
        infoLogger.info(f'\nstart list for batch [{i:i+batch}] = {start}\nnstop for batch [{i:i+batch}] = {stop}\nstart length for batch [{i:i+batch}]: {len(start)}\nstop length for batch [{i:i+batch}]: {len(stop)}\n')
        if not start or not stop:
            infoLogger.error(f"Failed to get start and stop positions.")
            sys.exit(1)

        # Get the sequences for each accession ID in the batch
        record = functions.fetch_nucleotide_record(nucleotide_ids, start_pos=start, stop_pos=stop, webenv=webenv, query_key=query_key)
        
        # Append the record to the list
        sequences.append(record)
    except Exception as e:
        infoLogger.error(f'Error inprocessing for batch [{i:i+batch}]:')
        infoLogger.error(e)
        sys.exit(1)

infoLogger.info(f"Total sequences retrieved: {len(sequences)}")

try:
    # Save the sequences to a FASTA file
    with open(output_fasta, 'w') as f:
        for seq in sequences:
            f.write(seq)
            f.write('\n')
except Exception as e:
    infoLogger.error(f'Error writing to FASTA file: {output_fasta}')
    infoLogger.error(e)
    sys.exit(1)
#______________________________________________________END________________________________________________________#