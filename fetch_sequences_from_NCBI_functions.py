"""
@author: harshita

-This script contains the funstions that are being 
used in fetch_sequences_from_NCBI.py
"""

# Import libraries
import os
import logging
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
from Bio import Entrez

#The info logger will be used to print information into a log file
infoLogger = logging.getLogger('infoLogger') 
infoLogger.setLevel(logging.INFO)
fh = logging.FileHandler('sequence_fetch_log.log')
fh.setFormatter(logging.Formatter('%(message)s'))
infoLogger.addHandler(fh)

#Define a function to get a list of contig lengths
def distribution(filename, column,n,seed):
    """
    This function generates a list of (n) contig lengths according to Poisson distribution.

    filename = file containing the contig infomation
    column = column name in the file that contains the contig lengths
    n = number of contig lengths to generate
    seed = seed for the random number generator
    """
    try:
        # Read the file
        df = pd.read_csv(filename)
        infoLogger.info(f'Contig info:\n{df.head()}\n')

        # Extract the data
        data = df[column].tolist()
        infoLogger.info(f'Maximum contig length in contig file {max(data)}')
        infoLogger.info(f'Minimum contig length in contig file {min(data)}')

        #Count the number of times each contig length occurs and create a .csv file
        counts = df[column].value_counts()
        infoLogger.info(f'Contig lengths counts:\n{counts}\n')
        counts.to_csv('contig_lengths_counts.csv')

        # Fit the data to a Poisson distribution
        mean_estimate = np.mean(data)
        infoLogger.info(f"Mean: {mean_estimate}")

        np.random.seed(seed=seed)
        # Generate random samples from the fitted Poisson distribution
        random_lengths = st.poisson.rvs(mean_estimate, size=n,)

        # Calculate the Probability Mass Function (PMF) of the fitted Poisson distribution
        x = np.arange(min(random_lengths), max(random_lengths) + 1)
        p = st.poisson.pmf(x, mean_estimate)
        # Plot histogram of the data
        plt.hist(random_lengths, bins=range(int(min(random_lengths)), int(max(random_lengths)) + 1), density=True, alpha=0.6, color='b', align='left')
        plt.plot(x, p, 'k', linewidth=2, marker='o', linestyle='None')
        title = "Fit results: mean = %.2f" % mean_estimate
        plt.title(title)
        plt.xlabel('Lengths')
        plt.ylabel('Estimate')

        # Save the plot
        plt.savefig(f'contig_poisson_distribution_{n}.png', dpi=1000)
        return random_lengths
    
    except Exception as e:
        infoLogger.error(f"Error in distribution function (functions.py): {e}")
        return None


# Define a function to get the row with maximum length
def get_max_length_row(table):
    """
    Get the row with the maximum length from a table.

    table = dataframe that contains the length information.
    Note that the length column should be named as 'length'
    """
    try:
        return table.sort_values('length', ascending=False).iloc[0]
    except Exception as e:
        infoLogger.error(f"Error occured in get_max_length_row function (functions.py): {e}")
        return None

# Define a function to fetch nucleotides
def fetch_nucleotide_record(nucleotide_id, start_pos, stop_pos, webenv, query_key):
    """
    Fetch a nucleotide record from the Entrez database.

    nucleotide_id = list of nucleotides accessions for which sequences are to be fetched
    start_pos = start position of the sequence to be fetched
    stop_pos = stop position of the sequence to be fetched
    webenv = web environment
    query_key = query key
    """
    try:
        # Use Entrez to fetch the nucleotide record
        handle = Entrez.efetch(db="nucleotide", id=nucleotide_id, rettype="fasta", retmode="text", seq_start=start_pos, seq_stop=stop_pos, webenv=webenv, query_key=query_key)
        record = handle.read()
        handle.close()
        return record
    except Exception as e:
        # Handle any errors that occur
        infoLogger.error(f"Error fetch_nucleotide_record function (functions.py): {e}")

# Define a function to get positions
def get_positions(sequence_length, contig_length, seed):
    """
    Get the start and stop positions for a list of sequence lengths.

    sequence_length = list of sequence lengths
    contig_length = list of length of the contigs
    seed = random seed
    """
    try:
        #Initialize empty lists to store start and stop positions
        start_list = []
        stop_list = []

        #Set up random seed
        np.random.seed(seed)

        #Conver contig lengths into int
        contig_length_int = contig_length.astype(int)

        #Iterate over the sequence lengths
        for i in range(len(sequence_length)):
            #Get the current sequence length and contig length
            seq_len = int(sequence_length[i])
            contig_len = contig_length_int[i]

            #Check if the sequence length is less than contig length
            if seq_len < contig_len:
                # Set the start position to 1 and the stop position to the sequence length
                start = 1
                stop = seq_len
            else:
                # If the sequence length is only 1 more than the contig length
                if seq_len - contig_len + 1 <= 1:
                    # Set the start position to 1 and the stop position to the contig length
                    start = 1
                    stop = contig_len
                else:
                    # Otherwise, generate a random start position between 1 and the sequence length minus the contig length plus 1
                    start = np.random.randint(1, seq_len - contig_len + 1)
                    # Set the stop position to the start position plus the contig length
                    stop = start + contig_len
            # Append the start and stop positions to the lists        
            start_list.append(start)
            stop_list.append(stop)
        return start_list, stop_list
    
    except Exception as e:
        infoLogger.error(f"Error in get_positions function (functions.py): {e}")
        return None


#Define a function to binary search
def taxid_binary_search(df, taxid):
    """
    Binary search each taxid and get the rows for the particular taxid as a dataframe.

    df = dataframe for which binary search needs to happen
    taxid = taxid for which rows need to be fetched
    """
    try:
        # Initialize the low and high indices for the binary search
        low, high = 0, len(df) - 1
        # Initialize the result list
        result = []
        
        # Perform the binary search
        while low <= high:
            # Calculate the mid index
            mid = (low + high) // 2
            # Check if the taxid at the mid index matches the target taxid
            if df.iloc[mid]['taxid'] == taxid:
                # If it matches, add the row to the result list
                result.append(df.iloc[mid])
                # Check for duplicates around the middle element
                left, right = mid - 1, mid + 1
                while left >= 0 and df.iloc[left]['taxid'] == taxid:
                    # Add the duplicate rows to the result list
                    result.append(df.iloc[left])
                    left -= 1
                while right < len(df) and df.iloc[right]['taxid'] == taxid:
                    # Add the duplicate rows to the result list
                    result.append(df.iloc[right])
                    right += 1
                break
            elif df.iloc[mid]['taxid'] < taxid:
                # If the taxid at the mid index is less than the target taxid, move the low index to mid + 1
                low = mid + 1
            else:
                # If the taxid at the mid index is greater than the target taxid, move the high index to mid - 1
                high = mid - 1
        
        # Convert the result list to a dataframe and return it
        return pd.DataFrame(result)
    
    except Exception as e:
        infoLogger.error(f"Error taxid_binary_search function (functions.py): {e}")
        return None
#______________________________________________________END________________________________________________________#