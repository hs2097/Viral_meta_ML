"""
@author: harshita

-This script generates random subsamples from a dataset 
based on taxonomical lineage information. 

- It allows the user to specify a taxonomic rank 
(e.g., genus) and the number of rows to sample. 

- The script outputs the selected subsample into a .csv file, 
which can be used for further analysis or model training.
"""

#Import libraries
import os
import logging
import argparse
import pandas as pd
import numpy as np

#Set up the Parser
parser = argparse.ArgumentParser(description='Random Sample Generator:\nInput the lineage information and subsample the database according to the taxonomical rank and the number of rows of your choice.\n', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--file', required=True, help='Enter the name of the file containing lineage information. File format:.csv')
parser.add_argument('--rank', required=True, default='genus',help='Enter the rank for which you want to select rows')
parser.add_argument('--n', required=False,default=10,type=int,help='Enter the number of rows to select')
parser.add_argument('--seed', required=False,default=123,type=int,help='Enter the seed for random number generation')
parser.add_argument('--lowest_int',default=10,type=int,required=False, help='Enter the lowest integer value to generate random number within')
parser.add_argument('--output', required=True, help='Enter the name of the file to save the sampled data. File format:.csv')
args = parser.parse_args()

#Check if a log is available, if yes then remove old log
if os.path.isfile('subsample_log.log'):
     os.remove('subsample_log.log')

#Set up the logger
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('subsample_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

infoLogger.info(f'Starting Random Sample Generator\n')

#Function to randomly select rows from a given rank 
def random_row_selector(df, rank, n=1, seed = None, lowest_int = 1):
    '''
        To randomly select rows for a given rank:

        df = file containing lineage information
        rank = level of hierarchy for to select rows for
        n = number of rows to select
        seed = seed for random number generation
        lowest_int = lowest integer value to generate random number within
    '''
    #Check if the rank is valid and subset the rank from taxnonomic_ranks
    try:
        taxonomic_ranks = ['superkingdom','kingdom','phylum', 'class', 'order', 'family', 'genus','species','strain']
        if rank in taxonomic_ranks:
            rank_subset_list = [taxonomic_ranks[i] for i in range (0, taxonomic_ranks.index(rank)+1)]
    except ValueError:
        infoLogger.error("(subsample_generator.py): Invalid rank. Please choose from the following ranks:{}\nExiting!!\n" .format(taxonomic_ranks))
        raise SystemExit(1)
    except Exception as e:
            infoLogger.error('Exception (subsample_generator.py): %s. Exiting!!' % e)
            raise SystemExit(1)
                    
    unique_rank_list = df[rank].unique().tolist() #get a list of all the unique names in the particular rank
    infoLogger.info(f'List of all the unique names in rank {rank}:{unique_rank_list}\n')
    np.random.seed(seed) #set up a seed to ensure reproducibility
    random_numbers = np.random.randint(lowest_int, size = len(unique_rank_list)) # randomly generate a number
    #Iterate through the list
    rows = [] #Intialise to append in list
    i=0 #To iterate through the indexes in random number and get the number at that particular index
    for name in unique_rank_list:
        try:
            rank_subset_df = df[df[rank] == name] #subset each name present in the unique list
            if rank_subset_df.shape[0] >= n: #Check if the number of rows for that particular name greater than or equal to n
                df_grouped = rank_subset_df.groupby(rank_subset_list) #group the rows according to the rank list created
                selected_rows = df_grouped.apply(lambda x: x.sample(n=n, random_state = random_numbers[i])) #Random sample
                selected_rows.reset_index(drop=True, inplace=True) #reset index
                rows.append(selected_rows) #Append the rows
                i+=1 #To increment the random number index
            else: #(If the number of rows < n)
                rows.append(rank_subset_df) #Simply subset the 
                i+=1 #To increment the random number index
        except Exception as e:
            infoLogger.error('Exception (subsample_generator.py): %s. Exiting!!' % e)
            raise SystemExit(1)
    #Concatenate the rows
    try:
        sampled_df = pd.concat(rows, ignore_index=True) #Convert the rows into dataframe by using concat
        return sampled_df
    except Exception as e:
        infoLogger.error('Exception (subsample_generator.py): %s. Exiting!!' % e)
        raise SystemExit(1)


#Input intilization
inputfile = args.file
outputfile = args.output
rank = str(args.rank)
n = int(args.n)
seed = int(args.seed)
lowest_int = int(args.lowest_int)

#Read the lineage file
df = pd.read_csv(inputfile)

infoLogger.info(f'File read! Starting sampling for all levels\n')
infoLogger.info(f'Input file: {inputfile}\n{df.head()}')
infoLogger.info(f'Output file: {outputfile}')
infoLogger.info(f'Rank: {rank}')
infoLogger.info(f'Number of rows to be sampled: {n}')
infoLogger.info(f'Seed: {seed}')
infoLogger.info(f'Lowest integer: {lowest_int}')

#Random sampling at mentioned level
infoLogger.info(f'Sampling at {rank} level. n= {n}\n')
df_sampled = random_row_selector(df, rank=rank, n= n, seed=seed, lowest_int=lowest_int) 
infoLogger.info(f'{rank} level sampling done. Number of rows: {df_sampled.shape[0]}\n')
infoLogger.info(f'{df_sampled.head()}\n')
df_sampled.to_csv(outputfile, index=False)
infoLogger.info(f'Sampling done. Output file: {outputfile}\n')
infoLogger.info('Exiting!!\n')

#___________________________________________________END_____________________________________________________#