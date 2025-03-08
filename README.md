# ReadMe

This study aims to improve the identification and characterization of viral sequences within metagenomic datasets by leveraging machine learning models, specifically Gradient Boosting Machines (GBMs), to analyze genomic compositional features. The study utilizes genomic data from various organisms to create robust models capable of distinguishing viral sequences from non-viral ones and further categorizing them into their respective taxonomic groups. This approach is particularly significant in addressing the challenges posed by "viral dark matter," which refers to the vast array of unidentified viral sequences in metagenomic studies.

This readme file contains information about the data, scripts and their usage. It also includes an overview of the steps and commands used for the study. This will help reproduce the pipeline used to train the models.

__________________________________________________________________________________________________________________________________________________________
  ## 1. Pipeline Overview

  Step 1: Download the RefSeq catalog (Release 244 has been used for this study).
          URL: https://ftp.ncbi.nlm.nih.gov/refseq/release/release-catalog

  Step 2: Extract the taxid, organism, accession number and length information from the RefSeq catalog using the following command:
          
          gunzip -c RefSeq-release**.catalog.gz| cut -f1,2,3,6| sort > taxid_accession_lenght.tsv

  Step 3: Generate Lineage database

  Step 4: Randomly subsample the generated lineage database according to a specified rank and n value.

  Step 5: Fetch the sequences of the subsampled taxonomic identifiers (taxids) according to the start and stop position calculated for each length. 
          *Please note that these lengths are sampled from a Poisson distribution curve created according to the list of sample contig lengths that the user provides.*

  Step 6: Calculate genomic features for the fetched sequences.

  Step 7: Create a master dataset containing taxonomic and genomic information and the class assigned to each instance. 
          The *category* column stores:
           a. **Virus/NonVirus** classes based on the superkingdoms for **binary** models
           b. **Archaea/Bacteria/Fungi/Metazoa/Plant/Virus/Other Eukaryote** based on the superkingdom and kingdom (in case of Fungi/Metazoa/Plant) for **multiclass** models

  Step 8: Train the models and validate.
__________________________________________________________________________________________________________________________________________________________

   ## 2. Scripts

a.	Lineage database generation:

i.	Script: Taxid_Lineage_extraction.py

*NOTE : This script also runs an Rscript taxonomy_utils.R, Mollentze et al.(2021) (https://doi.org/10.1371/journal.pbio.3001390 . Full reference in the report.) to fill up any taxonomic rank values that are empty. Please refer to the mentioned URL for any further information about the script: https://github.com/Nardus/zoonotic_rank/blob/main/Utils/taxonomy_utils.R*
    
ii.	Usage: 
      
      python3 Taxid_lineage_extraction.py --taxidfile TAXIDFILE -- chunksize CHUNKSIZE --outputfile  OUTPUTFILE
    
    Taxid Lineage database creator: Takes in taxids, extracts taxonomical lineage information and returns them in the form of a .csv file .

    options:
      -h, --help            show this help message and exit
      --taxidfile TAXIDFILE
                            Enter your taxidfile file (default: None)
      --chunksize CHUNKSIZE
                            Enter the size of the batch to iterate throug the tazid file (default: 100)
      --outputfile OUTPUTFILE
                            Enter the name of your output file (default: None)

iii.	Example Usage: 

      python3 Taxid_lineage_extraction.py --taxidfile taxids_accession_length.csv -- chunksize 1000000 --outputfile  Lineage_database.csv

iv. Output: A .csv file that comprises taxid, superkingdom, kingdom, phylum, class, order, family, genus, species, subspecies, and strain columns. A *log* file to log information and errors while the script runs.


b.	Subsample generator:

i.	Script: random_subsample_generator.py

ii.	Usage: 

    python3 random_subsample_generator.py [-h] --file FILE --rank RANK [--n N] [--seed SEED] [--lowest_int LOWEST_INT] --output OUTPUT
    
    Random Sample Generator: Input the lineage information and subsample the database according to the taxonomical rank and the number of rows of your choice.

    options:
      -h, --help            show this help message and exit
      --file FILE           Enter the name of the file containing lineage information. File format:.csv (default: None)
      --rank RANK           Enter the rank for which you want to select rows (default: genus)
      --n N                 Enter the number of rows to select (default: 10)
      --seed SEED           Enter the seed for random number generation (default: 123)
      --lowest_int LOWEST_INT
                            Enter the lowest integer value to generate random number within (default: 10)
      --output OUTPUT       Enter the name of the file to save the sampled data. File format:.csv (default: None)

iii. Example Usage: 
      
      python3 random_subsample_generator.py --file Lineage_database.csv --rank genus --n 1 --seed 10 --lowest_int 10 --output dataset_genus.csv

iv. Output - A .csv file after subsampling the data. Contains the following columns: taxid, superkingdom, kingdom, phylum, class, order, family, genus, species, subspecies, and strain. A *.log* file to to log information and errors while the script runs.



c. Fetch sequences from NCBI:

i. Script: fetch_Sequences_from_NCBI.py

ii. Usage: 

    python3 fetch_sequences_from_NCBI.py [-h] --dataset DATASET --accessionfile ACCESSIONFILE --contiginfo CONTIGINFO [--batch BATCH] [--seed SEED] --outputprefix OUTPUTPREFIX --api API --email EMAIL
    
    Fetch sequences from NCBI: Download the sequence of random lengths from the NCBI database and save it in a fasta file. The script will also provide output a .csv file of all the accession ids that were fetched.

    options:
      -h, --help                    show this help message and exit
      --dataset DATASET             Enter the name of the dataset file for which sequences are to be fetched. File format:.csv (default: None)
      --accessionfile ACCESSIONFILE Enter the file which contains accesssion numbers corresponding to taxids.File format:.csv (default: None)
      --contiginfo CONTIGINFO       Enter the a sample list of contigs to create a length distribution. This will be used to get the start and stop positions for the the sequence.File format:.csv (default: None)
      --batch BATCH                 Enter the batch size to fetch the sequences from NCBI (default: 10)
      --seed SEED                   Enter a seed value (default: 10)
      --outputprefix OUTPUTPREFIX   Enter the prefix to save the fasta file and fetched accession information. (default: None)
      --api API                     Enter the API key for NCBI. (default: None)
      --email EMAIL                 Enter the email address for NCBI. (default: None)

iii. Example Usage: 

      python3 fetch_sequences_from_NCBI.py --dataset dataset_genus.csv --accessionfile taxid_length_accession.csv --contiginfo laura_contig_lengths.csv --batch 10000 --seed 201197 --outputprefix genus_subsampled -api e54608dac60c8124ece9e4afc410d507c208 --email 2743102s@student.gla.ac.uk

iv. Output: Two output files
    1. [prefix]_fasta.fasta - Fasta file containing the fetched sequences
    2. [prefix]_accession.csv - Accesion_ids of all the fetched sequences
    3. A *.log* file to to log information and errors while the script runs.


d. Genomic feature calculation:

  1. Reduced dataset:

      i. Script: calc_noncoding_genome_features.py (Mollentze et al., 2021)
      ii. Kindly refer to the mentioned URL for script and output information: https://github.com/Nardus/zoonotic_rank/blob/main/Utils/calc_noncoding_genome_features.py

  2. featuretools:

        This is an unpublished pipeline developed at the School of Biodiversity, One Medicine and Veterinary, University of Glasgow, to calculate genomic feature tools. Please contact the university for access permissions.


e. Master dataset generation:

  i. Script: master.py

  ii. Usage: 
  
      python3 master.py [-h] --category CATEGORY --dataset DATASET [--genomic_features GENOMIC_FEATURES] --accession_id ACCESSION_ID --output OUTPUT [--featuretools]

      Master training dataset generation: Combine all the information produced for each taxid into a master dataset which can be used to map the information and also to build a training datatset for further machine learning model.
      options:
      -h, --help            show this help message and exit
      --category CATEGORY   Enter binary/multicategorical to add categorical information as Virus/NonVirus(binary) or Archeaa/Bacteria/Metazoa/Plant/Fungi/Virus (multicategorical) (default: binary)
      --dataset DATASET       Enter the subsampled dataset. (default: None)
      --genomic_features GENOMIC_FEATURES   Enter the genomic features file. Please ignore this option if you have used featuretools. (default: None)
      --accession_id ACCESSION_ID   Enter the accession id file. (default: None)
      --output OUTPUT         Enter the output file name. (default: None)
      --featuretools          Enter this option if featuretools have been used to create genomic features. (default: False)

  iii. Example Usage: 

        python3 master.py --category binary --dataset dataset_genus.csv --genomic_features genus_genomic_features.csv --accession_id genus_v8_accession.csv --output genus_binary_master_dataset.csv

        python3 master.py --category multicategorical --dataset dataset_genus.csv --genomic_features genus_genomic_features.csv --accession_id genus_v8_accession.csv --output genus_multicategorical_master_dataset.csv

  iv. Output: A .csv file after merging the subsampled dataset, genomic features, and category information. Column names depend on the tool used to produce the genomic features. A *log* file to to log information and errors while the script runs.

f. Gradient boosting ML models:

  1. Binary model:

    i. Script: binary_model.py

    ii. Usage : python3 binary_model.py [-h] --masterdata MASTERDATA --prefix PREFIX --seed SEED

      Binary classification model

      options:
        -h, --help            show this help message and exit
        --masterdata MASTERDATA
                              Enter the master dataset that will be used for training the model. File format:.csv (default: None)
        --prefix PREFIX       Enter the prefix you want the outputs to be saved with (default: None)
        --seed SEED           Seed for reproducibility (default: 42)


    iii. Example Usage: 

      python3 binary_model.py --masterdata genus_binary_master_dataset.csv --prefix genus --seed 130499

    iv. Output - Multiple outputs produced:

        1. Files:

          i. [prefix]_binary_training_dataset.scv
          ii.[prefix]_binary_test_pred.csv
          iii. A *.log file to log information while the script runs.

        2. Images produced:

          i. [prefix]_binary_dataset_proportion.png
          ii. [prefix]_binary_feature_hostogram.png
          iii. [prefix]_binary_confusion_matrix.png
          iv. [prefix]_binary_precision_recall_curve.png
          v. [prefix]_binary_roc_curve.png
          vi. [prefix]_binary_feature_importance.png
          vi. [prefix]_binary_tree.png
          

  2. Multicategorical/multiclass model:

    i. Script : multicategorical_model.py

    ii. Usage : python3 multicategorical_model.py [-h] --masterdata MASTERDATA --prefix PREFIX --seed SEED

      Multicategorical classification model

      options:
        -h, --help            show this help message and exit
        --masterdata MASTERDATA
                              Enter the master dataset that will be used for training the model. File format:.csv (default: None)
        --prefix PREFIX       Enter the prefix you want the outputs to be saved with (default: None)
        --seed SEED           Seed for reproducibility (default: 42)
    
    iii. Example Usage: 

      python3 multicategorical_model.py --masterdata genus_multicategorical_master_dataset.csv --prefix genus --seed 130499

    iv. Output - Multiple outputs produced:

        1. Files:

          i. [prefix]_multicategorical_training_dataset.scv
          ii.[prefix]_multicategorical_test_pred.csv
          iii. A *.log file to log information while the script runs.

        2. Images produced:

          i. [prefix]_multicategorical_dataset_proportion.png
          ii. [prefix]_multicategorical_feature_hostogram.png
          iii. [prefix]_multicategorical_confusion_matrix.png
          iv. [prefix]_multicategorical_precision_recall_curve.png
          v. [prefix]_multicategorical_roc_curve.png
          vi. [prefix]_multicategorical_feature_importance.png
          vi. [prefix]_multicategorical_tree.png

----------------------------------------------------------------------------------------------------------------------------------------------------------
  ## Additional
  a. data_visualization.R : R script containing all the code used to visualize data and create plots for this study.
  b. Randomized_search_cv.py : To hypertune the parameters for both binary and multiclass models.

          Usage : python3 Randomized_search_cv.py *feature_dataset.csv

          Example : python3 Randomized_search_cv.py genus_binary_master_dataset.csv
                    python3 Randomized_search_cv.py genus_multicategorical_master_dataset.csv
          
          Output :  A *.log file which stores the best values for the classification parameters after hypertuning is completed.
