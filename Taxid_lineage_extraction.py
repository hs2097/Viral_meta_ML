"""
@author: harshita

- This script creates a taxid Lineage database by extracting 
taxonomical lineage information from a list of taxids. 

- It processes the taxid file in chunks and outputs 
the lineage information in a .csv file for further analysis.

"""
#Import libraries
import os
import subprocess
import logging
import argparse
import pandas as pd
import numpy as np
from ete3 import NCBITaxa

#Set up the Parser
parser = argparse.ArgumentParser(description='Taxid Lineage database creator: Takes in taxids, extracts taxonomical lineage information and returns them in the form of a .csv file .\n', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--taxidfile', required=True, help='Enter your taxidfile file')
parser.add_argument('--chunksize', required=False, default=100,type=int,help='Enter the size of the batch to iterate throug the tazid file')
parser.add_argument('--outputfile', required=True, help='Enter the name of your output file')
args = parser.parse_args()

#Check if a log is available, if yes then remove old log
if os.path.isfile('lineage_log.log'):
     os.remove('lineage_log.log')

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('lineage_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

infoLogger.info(f'Starting Taxid Lineage database creator')
infoLogger.info(f'Database generation started\n')
infoLogger.info('Input file: {}\n'.format(args.taxidfile))
infoLogger.info('Chunk size: {}\n'.format(args.chunksize))
infoLogger.info('Output file: {}\n'.format(args.outputfile))

#Parse the taxid_accession catalog
# Assign the file name to the variable
accession_file = args.taxidfile
#Create a list of all the taxonomic ranks
taxonomic_ranks = ['superkingdom','kingdom','phylum', 'class', 'order', 'family', 'genus', 'species','subspecies','strain']

# Read the file in chunks, get lineage information using ETE Toolkit and write into a csv file
chunk_size = int(args.chunksize)
i=1 #Assign a counter to get the chunk count
for chunk in pd.read_csv(accession_file, delimiter='\t', header=None, names=['taxid','organism' ,'accession'], dtype=str, chunksize=chunk_size):
        try:
            infoLogger.info(f'Chunk {i}:\n{chunk.head()}\n')  # print the first few rows of each chunk
            df=chunk #intialise the dataframe

            #Get lineage information using ETE Toolkit
            ncbi = NCBITaxa()

            # Make sure to get all the unique taxids
            taxids = df['taxid'].unique()
            infoLogger.info(f'Unique taxids in chunnk {i}:\n{taxids}\n') #Print the numver of unique taxids
            i+=1

            # Get lineage for each taxid and store in dictionary
            lineage_dict = {}
            for taxid in taxids:
                lineage = ncbi.get_lineage(taxid)
                lineage_dict[taxid] = lineage

            # Convert lineage to names and store in dictionary
            lineage_names = {}
            for taxid, lineage in lineage_dict.items():
                names = ncbi.get_taxid_translator(lineage)
                lineage_names[taxid] = names

            #  Get rank for each taxid and store in dictionary
            lineage_ranks = {}
            for taxid, lineage in lineage_dict.items():
                ranks = ncbi.get_rank(lineage)
                lineage_ranks[taxid] = ranks
            
            # Prepare the data with the correct order of taxonomic ranks
            data = []

            # Iterate over each taxid and its corresponding lineage hierarchy in lineage_dict
            for key, hierarchy in lineage_dict.items():
                # Initialize a row dictionary with empty strings for each desired taxonomic rank
                row = {rank: '' for rank in taxonomic_ranks}
                # Add the taxid to the row dictionary
                row['taxid'] = key
                
                # Iterate over each taxid (level) in the hierarchy list
                for level in hierarchy:
                    # Retrieve the rank for the current level from the lineage_ranks dictionary
                    rank = lineage_ranks[key].get(level)
                    
                    # Check if the rank is one of the desired taxonomic ranks
                    if rank in taxonomic_ranks:
                        # Retrieve the name for the current level from the lineage_names dictionary
                        name = lineage_names[key].get(level, '')
                        # Add the name to the appropriate rank in the row dictionary
                        row[rank] = name
                
                # Append the completed row dictionary to the data list
                data.append(row)
                
            
            #Covert the data list into a pandas dataframe, shift taxid to the first column and write to a csv file
            lineage_df = pd.DataFrame(data)
            first_column=lineage_df.pop('taxid')
            lineage_df.insert(0,'taxid',first_column)
            lineage_df.sort_index(ascending=True)
            lineage_df.to_csv('lineage_info.csv',mode='a', index=False, header=True)

        except FileNotFoundError:
            infoLogger.error('(Taxid_lineage_extraction.py):File {} cannot be opened for reading. Please check and try again'.format(args.taxidfile))
            raise SystemExit(1)
        except Exception as e:
             infoLogger.error(f'Exception (Taxid_lineage_extraction.py):{e}')
             continue

"""
#Run the taxonomy_utils.R Rscript (collaborated by Dr.Nardus Mollentze) to fill up empty taxonomical rank through here.
"""

try:
                
    current_working_directory = os.getcwd() #Get the current working directory

    # Define the input and output files paths (relative to the current working directory)
    input_file = os.path.join(current_working_directory,"lineage_info.csv")
    output_file = os.path.join(current_working_directory, args.outputfile)
    r_script = "taxonomy_utils.R"

    # Construct the command to run the R script
    r_command = ["Rscript", r_script, current_working_directory, input_file, output_file]

    subprocess.run(r_command)

    os.remove('lineage_info.csv')

    infoLogger.info('Database generation completed! \n')
    infoLogger.info('Output file: {}'.format(output_file))

except Exception as e:
    infoLogger.error(f'Exception (Taxid_lineage_extraction.py):{e}')
    raise SystemExit(1)
#________________________________________________________END_____________________________________________________________#