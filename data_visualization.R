# This script has been written to visualize the databases and datasets for the Viral Machine Learning study. 

####Set Working directory####
setwd("/Users/harshitasrivastava/Downloads/Viral_ML")

####Import libraries####
library(dplyr)
library(ggplot2)
library(stringr)
library(tidyverse)
library(tibble)
library(hrbrthemes)
library(data.table)
library(viridis)
library(forcats)
library(reshape2)
library(caret)
library(readxl)
library(rlang)
library(gridExtra)
library(readr)
library(tidyr)

####Functions####

##To create a table for count and category and save each count as a seperate .csv file##

calculate_rank_count = function(dataset, file_prefix, path= getwd()) {

  for (column in names(dataset)) {
    #Check if a column is empty, then print the column name of the console
    if (all(is.na(dataset[[column]])) || length(dataset[[column]]) == 0) {
      print(paste0(file_prefix," dataset: ",column, " column is empty."))
      next
    }
    #Create the name of the file
    filename = paste0(path,'/',file_prefix, "_", column, "_count_table.csv")
    
    # Ensure the column is a character vector
    dataset[[column]] = as.character(dataset[[column]])
    
    # Create a table with the count of each value in the column
    count_table = as.data.frame(table(dataset[[column]]))
    colnames(count_table) = c(column, "count")
    
    # Write the count table to a CSV file
    write.csv(count_table, filename, row.names = FALSE)
  }
  return("Files created")
}


##To create a barplot according to the category and cutoff##

make_barplot = function(table, category_col, cutoff,title = "Bar Plot", xlab = "Category", ylab = "Count", size = 0.5, flip = FALSE, legend=TRUE) {
  
  # Ensure the category column is a character vector
  table[[category_col]] = as.character(table[[category_col]])
  
  # Create a table with the count of each value in the category column
  count_table = as.data.frame(table(table[[category_col]]))
  colnames(count_table) = c("category", "count")
  count_table = tibble::rownames_to_column(count_table, "id")
  
  # Filter data to keep counts greater than or equal to cutoff
  filtered_data = subset(count_table, count >= cutoff)
  
  # Check if filtered_data is empty
  if (nrow(filtered_data) == 0) {
    stop("No categories meet the cutoff criteria.")
  }
  
  # Add an ID column for plotting
  filtered_data$id = seq(1, nrow(filtered_data))
  
  #Sort filtered data
  sorted_order = order(filtered_data$count)
  data_sorted = filtered_data[sorted_order,]
  
  if (flip == TRUE)
  {
    # Create the bar plot using ggplot2
    ggp = ggplot(data_sorted, aes(x = reorder(category, count), y = count, fill = category)) +
      geom_bar(stat = "identity") +
      coord_flip()+
      geom_text(aes(label = count), hjust = -0.5, size.unit = "pt", size = (size-2)) +
      #scale_fill_manual(values = rainbow(length(filtered_data$category))) +
      scale_fill_viridis(discrete = TRUE)+
      theme_minimal() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      theme(
        axis.text.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.y = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman"),
        legend.text = element_text( family = "Times New Roman", face = 'bold'),
        legend.title = element_text(family = "Times New Roman", face = 'bold'),
      )
  }
  else
  {
    # Create the bar plot using ggplot2
    ggp = ggplot(data_sorted, aes(x = reorder(category, count), y = count, fill = category)) +
      geom_bar(stat = "identity") +
      geom_text(aes(label = count), vjust = -0.3, size.unit = "pt", size = (size-2)) +
      #scale_fill_manual(values = rainbow(length(filtered_data$category))) +
      scale_fill_viridis(discrete = TRUE)+
      theme_minimal() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      theme(
        axis.text.x = element_text(angle = 90, hjust = 1, color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.y = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman"),
        legend.text = element_text( family = "Times New Roman", face = 'bold'),
        legend.title = element_text(family = "Times New Roman", face = 'bold'),
      )
  }
  
  #If no legend is needed:

  if (legend == FALSE)
  {
    ggp = ggp+  theme(legend.position = "none")
  }
  return (ggp)
}


# Function to make circular barplot for a particular category and cutoff

make_circular_barplot = function(table, category_col, cutoff = 1, min_y_limit = 100, max_y_limit = 100, alpha = 0.5, size = 2.5) {

  # Ensure the category column is a character vector
  table[[category_col]] = as.character(table[[category_col]])
  
  # Create a table with the count of each value in the category column
  count_table = as.data.frame(table(table[[category_col]]))
  colnames(count_table) = c("category", "count")
  count_table = tibble::rownames_to_column(count_table, "id")
  
  # Filter data to keep counts greater than or equal to cutoff
  filtered_data = subset(count_table, count >= cutoff)
  
  # Add an ID column for plotting
  filtered_data$id = seq(1, nrow(filtered_data))
  
  # Calculate the angle of the labels
  number_of_bar = nrow(filtered_data)
  angle =  90 - 360 * (filtered_data$id - 0.5) / number_of_bar
  
  # Calculate the alignment of labels: right or left
  filtered_data$hjust = ifelse(angle < -90, 1, 0)
  
  # Flip angle according to bar orientation to make them readable
  filtered_data$angle = ifelse(angle < -90, angle + 180, angle)
  
  # Create the circular bar plot
  ggp = ggplot(filtered_data, aes(x = as.factor(id), y = count, fill = as.factor(id))) +
    geom_bar(stat = "identity") +
    scale_fill_manual(values = rainbow(number_of_bar)) +
    ylim(-min_y_limit, max(filtered_data$count) + max_y_limit) +
    theme_minimal() +
    theme(
      axis.text = element_blank(), 
      axis.title = element_blank(), 
      panel.grid = element_blank(),
      plot.margin = unit(rep(-2, 4), "cm")
    ) +
    coord_polar(start = 0) +
    geom_text(aes(x = id, y = count + 10, label = paste(category, "(", count, ")", sep = ""), hjust = hjust), 
              color = "black", fontface = "bold", alpha = alpha, size = size, angle = filtered_data$angle, inherit.aes = FALSE)
  
  return(ggp)
}

#To create bar plots to for the number of unique entries present in each each rank to a database

visualize_database = function(data,title = "Diversity Across Taxonomic Ranks", xlab = "Taxonomic Rank", ylab = "Number of Unique Entries", size = 10, flip = FALSE)
{
  #Convert all the artificial_* rank names to NA
  filtered_data = data %>%
    mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))
  
  filtered_data = filtered_data %>%
    filter(!grepl('"', as.character(superkingdom), ignore.case = TRUE))
  
  # Calculate unique counts for each rank
  superkingdom_count = filtered_data %>% filter(!is.na(superkingdom)) %>% summarise(count = n_distinct(superkingdom))%>%pull(count)
  kingdom_count = filtered_data %>% filter(!is.na(kingdom)) %>% summarise(count = n_distinct(kingdom))%>%pull(count)
  phylum_count = filtered_data %>% filter(!is.na(phylum)) %>% summarise(count = n_distinct(phylum))%>%pull(count)
  class_count = filtered_data %>% filter(!is.na(class)) %>% summarise(count = n_distinct(class))%>%pull(count)
  order_count = filtered_data %>% filter(!is.na(order)) %>% summarise(count = n_distinct(order))%>%pull(count)
  family_count = filtered_data %>% filter(!is.na(family)) %>% summarise(count = n_distinct(family))%>%pull(count)
  genus_count = filtered_data %>% filter(!is.na(genus)) %>% summarise(count = n_distinct(genus))%>%pull(count)
  
  # Create a data frame for plotting
  diversity_data = data.frame(
    Rank = c("Superkingdom", "Kingdom", "Phylum", "Class", "Order", 
             "Family", "Genus"),
    Count = c(superkingdom_count, kingdom_count, phylum_count, class_count, order_count,
              family_count, genus_count)
  )
  
  diversity_data$Rank = as.factor(diversity_data$Rank)
  diversity_data$Rank = fct_reorder(diversity_data$Rank,diversity_data$Count)
  
  # Create the bar plot using ggplot2
  if (flip == TRUE){
    ggp = ggplot(diversity_data, aes(x = reorder(Rank, Count), y = Count, fill = Rank)) +
      geom_bar(stat = "identity") +
      coord_flip()+
      geom_text(aes(label = Count), hjust = -0.5, size.unit = "pt", size = (size-10)) +
      #scale_fill_manual(values = rainbow(length(diversity_data$Rank))) +
      scale_fill_viridis(discrete = TRUE)+
      theme_minimal() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      theme(
        axis.text.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.y = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman")
      )
  }
  else{
    ggp = ggplot(diversity_data, aes(x = reorder(Rank, Count), y = Count, fill = Rank)) +
      geom_bar(stat = "identity") +
      #coord_flip()+
      geom_text(aes(label = Count), vjust = -0.3, size.unit = "pt", size = (size-2)) +
      #scale_fill_manual(values = rainbow(length(diversity_data$Rank))) +
      scale_fill_viridis(discrete = TRUE)+
      theme_minimal() +
      ggtitle(title) +
      xlab(xlab) +
      ylab(ylab) +
      theme(
        axis.text.x = element_text(angle = 90, hjust= 1,color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.title.y = element_text(vjust = 1,color = "black", face = "bold", size = size, family = "Times New Roman"),
        axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman")
      )
  }
  
  return (ggp)
  
}

#To save plots as png

save_png = function(ggp,filename, path, height, width)
{
  png(file.path(path, filename), height = height, width = width)
  print(ggp)
  dev.off()
} 

#To make line plot combining the size of each dataset file given in the list

make_line_plot = function(file_list) {

  # Initialize a vector to store the row counts
  row_counts = sapply(data_list, nrow)
  
  # Create a data frame with the counts
  file_data = data.frame(
    file = factor(seq_along(data_list)), # Dynamically create file labels
    row_count = row_counts
  )
  
  # Create the line plot
  ggp = ggplot(file_data, aes(x = file, y = row_count)) +
    geom_line(group = 1, color = "blue", size = 1) + 
    geom_point(color = "black", size = 3) +          
    ggtitle("Number of Rows in Each File") +         
    xlab("File") +                                   
    ylab("Number of Rows") +                         
    theme_minimal()                                  
  
  # Return the plot
  return(ggp)
}

#To make stacked bar plots according to the data list (number of table) and levels (taxonomic levels) provided

make_stacked_plots = function(data_list, rank, count=0, size, levels) {
  #Intialize the levels and a list to store subsetted rows
  levels = levels
  subsets = list()
  
  for (i in 1:length(data_list)) {

    # Count occurrences of each value
    count_data = as.data.frame(table(data_list[[i]][[rank]]))
    # Sort by frequency in descending order
    sorted_order = order(count_data[, 2], decreasing = TRUE)
    count_data = count_data[sorted_order, ]
    # Add dataset (which table the counts belong to) column
    count_data = count_data %>%
      mutate(dataset = ifelse(length(count_data) > 0, levels[i], NA))
    # Subset the top N (N=count) categories
    subset_data = count_data[1:count, ]
    subsets[[i]] = subset_data
  }
  
  # Merge all subsets
  merged = rbindlist(subsets)
  merged$Var1 = as.factor(merged$Var1)
  merged$Var1 = fct_reorder(merged$Var1,merged$Freq,.desc = TRUE)
  
  #Create stacked plot
  ggp = ggplot(merged, aes(fill = dataset, y = Freq, x = Var1)) + 
    geom_bar(position = "stack", stat = "identity") +
    coord_flip()+
    scale_fill_viridis(discrete = TRUE) +
    #scale_fill_manual(values=c("red", "blue", "green")) + 
    ggtitle(paste0("Top ",rank, " from all the datasets")) +
    theme_ipsum() +
    xlab("") +
    theme(
      axis.text.x = element_text( face = "bold", size = size),
      axis.text.y = element_text(face = "bold", size = size),
      title = element_text(size = size),
      legend.text = element_text(face = "bold", size = size),
      legend.title = element_text(face = "bold")
    )
  
  print(ggp)
  
}

#To make multifaceted bar plots according to the datalist (list of tables) and levels (taxonomic levels) provided

make_stacked_plots_multifaceted = function(data_list, rank, count, size, ncol, levels) {
  #Intialize the levels and a list to store subsetted rows
  levels = levels
  subsets = list()
  
  for (i in 1:length(data_list)) {
    # Count occurrences of each value
    count_data = as.data.frame(table(data_list[[i]][[rank]]))
    # Sort by frequency in descending order
    sorted_order = order(count_data[, 2], decreasing = TRUE)
    count_data = count_data[sorted_order, ]
    # Add dataset (which table the counts belong to) column
    count_data = count_data %>%
      mutate(dataset = ifelse(length(count_data) > 0, levels[i], NA))
    # Subset the top N categories
    subset_data = count_data[1:count, ]
    subsets[[i]] = subset_data
  }
  
  # Merge all subsets
  merged = rbindlist(subsets)
  merged$Var1 = as.factor(merged$Var1)
  merged$Var1 = fct_reorder(merged$Var1,merged$Freq,.desc = TRUE)
  
  # Create mulltifaceted stacked plot
  ggp = ggplot(merged, aes(fill = dataset, y = Freq, x = Var1)) + 
    geom_bar(position = "dodge", stat = "identity") +
    coord_flip()+
    scale_fill_viridis(discrete = TRUE, option = "E") +
    ggtitle(" ") +
    facet_wrap(~dataset, ncol = ncol) +
    theme_ipsum() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(face = "bold", size = (size-10)),
      axis.text.y = element_text(face = "bold", size = size),
      title = element_text(size = size),
      legend.text = element_text(face = "bold", size = size),
      legend.title = element_text(face = "bold"),
      strip.text = element_text(size = size)
    ) +
    xlab("")+
    ylab("")
  
  return(ggp)
}

#To make heatmaps
make_heatmap <- function(data, category = 'Virus', top_n = 128, size = 60) {
  # Filter data for the specified category and select relevant columns
  virus_pred <- data[data$category == category,]
  virus_pred_subset <- virus_pred %>%
    select(family, category, True.value, Predicted.value, Predicted_Category) %>%
    mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), 'No_family', .))
  
  # Determine if predictions are correct
  virus_pred_subset$Correct <- virus_pred_subset$True.value == virus_pred_subset$Predicted.value
  
  # Assign predicted family
  virus_pred_subset <- virus_pred_subset %>%
    mutate(Predicted_family = ifelse(Correct, family, Predicted_Category))
  
  # Count actual and predicted viruses by family
  family_counts <- virus_pred_subset %>%
    group_by(family) %>%
    summarize(count = n()) %>%
    arrange(desc(count))
  
  predicted_family_counts <- virus_pred_subset %>%
    group_by(Predicted_family) %>%
    summarize(count = n()) %>%
    arrange(desc(count))
  
  # Rename the column (for consistency)
  names(predicted_family_counts)[names(predicted_family_counts) == "Predicted_family"] <- "family"
  
  # Merge the actual and predicted counts
  merged_family <- merge(family_counts, predicted_family_counts, by = "family")
  
  # Renaming columns for clarity
  columnnames <- c("family", "Actual_virus", "Predicted_virus")
  colnames(merged_family) <- columnnames
  
  # Sort the data by actual virus counts
  merged_family <- merged_family %>%
    arrange(desc(Actual_virus))
  
  # Normalize and scale the data
  merged_family$Actual_virus <- scale(merged_family$Actual_virus)
  merged_family$Predicted_virus <- scale(merged_family$Predicted_virus)
  merged_family <- merged_family[!merged_family$family %in% c("No_family", "Bacteria", "Plant", "Metazoa", "Archaea", "Other Eukaryote", "Fungi"), ]
  
  # Subset the top N most abundantly predicted families
  top_families <- merged_family %>%
    head(top_n)
  
  # Arrange families in descending order of Actual_virus counts
  top_families$family <- factor(top_families$family, levels = top_families$family[order(top_families$Actual_virus, decreasing = FALSE)])
  
  # Convert data to long format for ggplot
  top_families_long <- top_families %>%
    gather(key = "Type", value = "Count", Actual_virus, Predicted_virus)
  
  # Define colors for the gradient
  colours <- c("blue", "yellow", "red")
  
  # Plotting the heatmap
  ggp <- ggplot(top_families_long, aes(x = family, y = Type, fill = Count)) +
    geom_tile() + coord_flip() +
    scale_fill_gradientn(colours = colorRampPalette(colours)(100)) +
    labs(title = "Actual virus vs Predicted virus by Family",
         x = "Family",
         y = "") +
    theme_minimal() +
    theme(axis.text.x = element_text(color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.title.x = element_text(color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.title.y = element_text(color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.text.y = element_text(hjust = 1, vjust = 0.5, colour = "black", face = "bold", family = "Times New Roman", size = size),
          legend.text = element_text(family = "Times New Roman", face = 'bold', size = size - 10),
          legend.title = element_text(family = "Times New Roman", face = 'bold', size = size - 10),
          legend.spacing.y = unit(6, "lines"),
          legend.key.size = unit(6, "cm"),
          title = element_text(colour = "black", face = "bold", family = "Times New Roman", size = size)
    )
  
  return(ggp)
}


#To make heatmaps for category and predicted category

make_predicted_category_heatmap <- function(data, col, prediction_col, 
                                            normalize = c("none", "row", "column", "both"),
                                            display = c("none", "count", "percentage", "proportion"),
                                            colours = c("white", "purple"), size = 60, n=100,legend_key = 6,
                                            title = "Viral Families vs Predicted Categories", xlab='Predicted_Category',ylab="Family") {
  # Validate arguments
  normalize <- match.arg(normalize)
  display <- match.arg(display)
  
  # Count occurrences of each Predicted_Category for each family
  family_category_counts <- data %>%
    group_by(!!sym(col), !!sym(prediction_col)) %>%
    summarise(count = n(), .groups = 'drop') %>%
    spread(key = !!sym(prediction_col), value = count, fill = 0)
  
  # Convert the family column to a character vector
  family_category_counts[[col]] <- as.character(family_category_counts[[col]])
  
  # Add a Total column and sort by it in descending order
  family_category_counts <- family_category_counts %>%
    mutate(Total = rowSums(across(where(is.numeric)))) %>%
    arrange(desc(Total))
  
  # Reorder families based on total counts
  family_category_counts[[col]] <- factor(family_category_counts[[col]], 
                                          levels = family_category_counts[[col]][order(-family_category_counts$Total, decreasing = TRUE)])
  
  # Remove the 'Total' column
  family_category_counts <- family_category_counts %>%
    select(-Total)
  
  # Function to normalize by row
  normalize_by_row <- function(x) {
    if(max(x) == min(x)) return(rep(0, length(x)))  # handle cases with no variance
    (x - min(x)) / (max(x) - min(x))
  }
  
  # Function to normalize by column
  normalize_by_column <- function(x) {
    if(max(x) == min(x)) return(rep(0, length(x)))  # handle cases with no variance
    (x - min(x)) / (max(x) - min(x))
  }
  
  # Apply normalization based on user choice
  if (normalize == "row" || normalize == "both") {
    family_category_counts <- family_category_counts %>%
      mutate(across(where(is.numeric), normalize_by_row))
  }
  
  if (normalize == "column" || normalize == "both") {
    family_category_counts[, -which(names(family_category_counts) == col)] <- 
      apply(family_category_counts[, -which(names(family_category_counts) == col)], 2, normalize_by_column)
  }
  
  # If no normalization selected, calculate percentage or proportion if needed
  if (normalize == "none") {
    if (display == "percentage") {
      family_category_counts <- family_category_counts %>%
        mutate(across(where(is.numeric), ~ .x / sum(.x) * 100))
    } else if (display == "proportion") {
      family_category_counts <- family_category_counts %>%
        mutate(across(where(is.numeric), ~ .x / sum(.x)))
    }
  }
  
  # Remove any remaining NaN values
  family_category_counts <- family_category_counts %>% na.omit()
  
  # Convert to long format for ggplot
  melted_counts <- melt(family_category_counts, id.vars = col, variable.name = "Predicted_Category", value.name = "Value")
  
  # Plot the heatmap
  ggp_multi <- ggplot(melted_counts, aes_string(x = "Predicted_Category", y = col, fill = "Value")) +
    geom_tile() +
    scale_fill_gradientn(colours = colorRampPalette(colours)(n)) +
    labs(title = title,
         x = ylab,
         y = xlab) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, color = "black", face = "bold", family = "sans", size = size),
          axis.title.x = element_text(color = "black", face = "bold", family = "sans", size = size),
          axis.title.y = element_text(color = "black", face = "bold", family = "sans", size = size),
          axis.text.y = element_text(hjust = 1, vjust = 0.5, colour = "black", face = "bold", family = "sans", size = size),
          legend.text = element_text(family = "sans", face = 'bold', size = size - 10),
          legend.title = element_text(family = "sans", face = 'bold', size = size - 10),
          legend.spacing.y = unit(6, "lines"),
          legend.key.size = unit(legend_key, "cm"),
          title = element_text(colour = "black", face = "bold", family = "sans", size = size))
  if (display != "none") {
    ggp_multi <- ggp_multi + geom_text(aes(label = round(Value, 2)), size =(size-10))
  }
  
  return(ggp_multi)
}

# To assign genomic type (Baltimore Classification) from VMR table to other table.

assign_genomic_composition <- function(table1, table2) {
  # Convert factors to characters in df1 and df2 if necessary
  df1 <- data.frame(lapply(table1, function(x) if(is.factor(x)) as.character(x) else x), stringsAsFactors = FALSE)
  df2 <- data.frame(lapply(table2, function(x) if(is.factor(x)) as.character(x) else x), stringsAsFactors = FALSE)
  
  # Create a new column for genomic composition in df1
  df1$genome_composition <- NA
  
  # Iterate through each row of df1
  for (i in 1:nrow(df1)) {
    # Check the first 6 cells for a non-NA value
    for (j in 1:6) {
      if (!is.na(df1[i, j])) {
        immediate_value <- df1[i, j]
        column_name <- colnames(df1)[j]
        
        # Find the genomic composition in df2 for the immediate_value
        genomic_composition <- df2 %>%
          filter(!!sym(column_name) == immediate_value) %>%
          pull('Genome.composition')
        
        # Add the genomic composition to df1
        if (length(genomic_composition) > 0) {
          df1$genome_composition[i] <- genomic_composition[1]
          break
        }
      }
    }
  }
  
  return(df1)
}

#To convert baltimore genomic classifications to genomic types(dsDNA, ssDNA and ssRNA).
check_genomic_composition <- function(table, column_name) {
  for (i in 1:nrow(table)) {
    # Check if the current value in the specified column is not NA
    if (!is.na(table[[column_name]][i])) {
      # Covert various forms of 'dsDNA' to a single 'dsDNA' value
      if (table[[column_name]][i] == 'dsDNA' || table[[column_name]][i] == 'dsDNA-RT') {
        table[[column_name]][i] <- 'dsDNA'
      }
      # Convert various forms of 'ssDNA' to a single 'ssDNA' value
      if (table[[column_name]][i] == 'ssDNA' || table[[column_name]][i] == 'ssDNA(+/-)' || table[[column_name]][i] == 'ssDNA(+)' || table[[column_name]][i] == 'ssDNA(-)') {
        table[[column_name]][i] <- 'ssDNA'
      }
      # Convert various forms of 'ssRNA' to a single 'ssRNA' value
      if (table[[column_name]][i] == 'ssRNA' || table[[column_name]][i] == 'ssRNA(+)' || table[[column_name]][i] == 'ssRNA(-)'|| table[[column_name]][i] == 'ssRNA(+/-)') {
        table[[column_name]][i] <- 'ssRNA'
      }
    }
  }
  return(table)
}

#To make confusion matrix

make_confusion_matrix <- function(data, actual_col, predicted_col, size = 20, low_colour = "white", high_colour="purple") {
  # Convert the specified columns to factors
  actual <- as.factor(data[[actual_col]])
  predicted <- as.factor(data[[predicted_col]])
  
  # Check levels of both factors
  levels_actual <- levels(actual)
  levels_predicted <- levels(predicted)
  
  # Print levels for debugging
  print(levels_actual)
  print(levels_predicted)
  
  # Align the levels of the factors
  combined_levels <- union(levels_actual, levels_predicted)
  actual <- factor(actual, levels = combined_levels)
  predicted <- factor(predicted, levels = combined_levels)
  
  confusion_matrix = confusionMatrix(predicted, actual)
  conf_data = as.data.frame(confusion_matrix$table)
  
  
  # Plot using ggplot2
  confusion_grid <- ggplot(conf_data, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), colour = "white") + coord_flip()+
    geom_text(aes(label = Freq), vjust = 1, size = size-10, family = "Times New Roman") +
    scale_fill_gradient(low = low_colour, high = high_colour) +
    labs(title = "Confusion Matrix", x = "Actual value", y = "Predicted value") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.title.x = element_text(color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.title.y = element_text(color = "black", face = "bold", family = "Times New Roman", size = size),
          axis.text.y = element_text(colour = "black", face = "bold", family = "Times New Roman", size = size),
          legend.text = element_text(family = "Times New Roman", face = 'bold', size = size - 10),
          legend.title = element_text(family = "Times New Roman", face = 'bold', size = size - 10),
          title = element_text(colour = "black", face = "bold", family = "Times New Roman", size = size))
  
  return(confusion_grid)
}


#####Load lineage database####
lineage_data = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/Lineage_database.csv",header= TRUE, sep = ',')

####Sorting the lineage_database table according to superkingdom####

sorted_order = order(lineage_data[,'superkingdom'])
lineage_data_sorted = lineage_data[sorted_order,]

# Inspect rows where superkingdom contains unexpected values
unexpected_values = c("Salmonella enterica subsp. enterica", "Salmonella enterica subsp. salamae")
unexpected_rows = lineage_data_sorted[lineage_data_sorted$superkingdom %in% unexpected_values, ]
print(unexpected_rows)

# Filter out unexpected values from the lineage_data
lineage_data_cleaned = lineage_data[!lineage_data$superkingdom %in% unexpected_values, ]

# Verify that the unexpected values are removed
unique(lineage_data_cleaned$superkingdom)

#Get the count of all the ranks in the superkingdom rank sorted table
calculate_rank_count(dataset = lineage_data_cleaned, file_prefix = "lineage_database")

####Subset according to superkingdom and calculating the count of each rank for that table####

#Subset table according to Archea superkingdom
archea_subset = lineage_data_cleaned[lineage_data_cleaned$superkingdom == 'Archaea',]
write.csv(archea_subset, "archea_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = archea_subset, file_prefix = "Archaea")

#Subset table according to Bacteria superkingdom
bacteria_subset = lineage_data_cleaned[lineage_data_cleaned$superkingdom == 'Bacteria',]
write.csv(bacteria_subset, "bacteria_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = bacteria_subset, file_prefix = "Bacteria")

#Subset table according to Eukaryota superkingdom
eukaryota_subset = lineage_data_cleaned[lineage_data_cleaned$superkingdom == 'Eukaryota',]
write.csv(eukaryota_subset, "eukaryota_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = eukaryota_subset, file_prefix = "Eukaryota")

#Subset table according to Viruses supekingdom
virus_subset = lineage_data_cleaned[lineage_data_cleaned$superkingdom == 'Viruses',]
write.csv(virus_subset, "viruses_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = virus_subset, file_prefix = "Virus")

####Create Barplots to visualize the lineage subsets created from lineage data (lineage_data_cleaned)####

#Convert all the overall taxonomies with artificial_* to NA
lineage_data_cleaned = lineage_data_cleaned%>% mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .)) %>% mutate_all(~ ifelse(grepl('"', as.character(.), ignore.case = TRUE), NA, .))

ggp_bar = make_barplot(table = lineage_data_cleaned, category_col = 'superkingdom', cutoff = 4,size = 70, flip = FALSE, legend = FALSE)
ggp_bar
save_png(ggp_bar,'superkingdoms.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 2000, width = 1000)

#Convert all the viral taxonomies with NA to other
virus_subset = virus_subset %>% mutate_all(~ ifelse(is.na(.), 'Other', .))%>% mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))

ggp_bar_virus = make_barplot(table = virus_subset, category_col = 'kingdom', cutoff = 0, size = 60, flip = FALSE, legend = FALSE )
ggp_bar_virus
save_png(ggp_bar_virus,'virus_kingdoms.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 2000, width = 1500)


ggp_bar_virus = make_barplot(table = virus_subset, category_col = 'phylum', cutoff = 20, size = 10 )
ggp_bar_virus

#Convert all the overall eukaryotic taxonomies with artificial_* to NA
eukaryota_subset = eukaryota_subset %>% mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))

ggp_bar_eukaryota = make_barplot(table = eukaryota_subset, category_col = 'kingdom', cutoff = 0, size = 50, legend = FALSE)
ggp_bar_eukaryota
save_png(ggp_bar_eukaryota,'eukaryota_kingdoms.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 1500, width = 800)

####To get the proportion of Archaea, Bacteria, Virus, Metazoa, Plants and Fungi in the lineage_database (lineage_data_cleaned)####

#Subset Archaea, Bacteria and Virus from lineage database
database_subset_superkingdom = lineage_data_cleaned[(lineage_data_cleaned$superkingdom == 'Archaea')|(lineage_data_cleaned$superkingdom == 'Bacteria')|(lineage_data_cleaned$superkingdom == 'Viruses'),]%>%
  mutate(category = superkingdom)

#Subset Fungi, Metazoa and Viridiplantae from lineage database
database_subset_kingdom = lineage_data_cleaned[(lineage_data_cleaned$kingdom == 'Fungi')|(lineage_data_cleaned$kingdom == 'Metazoa')|(lineage_data_cleaned$kingdom == 'Viridiplantae'),]%>%
  mutate(category = kingdom)

#Merge them and convert all taxonomies with artificial_* to NA
database_merged = rbind(database_subset_superkingdom,database_subset_kingdom)%>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .)) %>%
  mutate_all(~ ifelse(grepl('"', as.character(.), ignore.case = TRUE), NA, .))

ggp = make_barplot(database_merged,category_col = 'category', cutoff = 7, size = 10)
ggp


######Subsampled dataset counts and plots####

####Load the data and the count the number of representatives in each rank of the subsampled dataset####
data_superkingdom = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_superkingdom.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
data_superkingdom = data_superkingdom[!grepl('^"', data_superkingdom$superkingdom),, ]

#Calculate and save the count of each taxonomic rank
calculate_rank_count(dataset = data_superkingdom, file_prefix = "superkingdom")

data_kingdom = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_kingdom.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
calculate_rank_count(dataset = data_kingdom, file_prefix = "kingdom")

data_phylum = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_phylum.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
calculate_rank_count(dataset = data_phylum, file_prefix = "phylum")

data_class = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_class.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
calculate_rank_count(dataset = data_class, file_prefix = "class")

data_order = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_order.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
calculate_rank_count(dataset = data_order, file_prefix = "order")

data_family = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_family.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
calculate_rank_count(dataset = data_family, file_prefix = "family")

data_genus = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/dataset_genus.csv",header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
data_genus_count = calculate_rank_count(dataset = data_genus, file_prefix = "genus")
data_genus = data_genus%>% mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .)) %>% mutate_all(~ ifelse(grepl('"', as.character(.), ignore.case = TRUE), NA, .))

####Create a line plot of dataset size for each subsampled set####
# Define file names
file_paths = c("dataset_kingdom.csv", "dataset_phylum.csv", "dataset_class.csv", "dataset_order.csv", "dataset_family.csv", "dataset_genus.csv")

# Initialize a vector to store the dataset count
row_counts = c()

# Loop through each file, read it, and count the dataset size
for (file in file_paths) {
  data = read.csv(file,header= TRUE, sep = ',', fill = TRUE, quote="", na.strings=c(""))
  row_counts = c(row_counts, nrow(data))
}

# Create a data frame with these counts
file_data = data.frame(
  file = factor(seq_along(file_paths)),
  row_count = row_counts
)

#Create a vector with rank names. To be changed upon changes in datset size and whatever ranks you need.
ranks = c('kingdom','phylum','class','order','family','genus')

# Create a line plot
ggp_line = ggplot(file_data, aes(x = ranks, y = row_count)) +
  geom_line(group = 1, color = "blue", size = 1) +
  geom_point(color = "black", size = 1) +            
  geom_text(aes(label = row_count), vjust = -1.5, size.unit = "pt", size = 20, face = 'bold')+
  ggtitle("Size of each subsampled dataset ") +         
  xlab("Rank") +                                  
  ylab("Size") +                         
  theme_minimal()+
  theme(
    axis.text.x = element_text(color = "black", face = "bold", size = 20),
    axis.title.x = element_text(color = "black", face = "bold", size = 20),
    axis.title.y = element_text(color = "black", face = "bold", size = 20),
    axis.text.y = element_text(colour = "black", face = "bold", size = 20),
    title = element_text(colour = "black", face = "bold", size = 20)
  )

ggp_line 

#Call the save function to save the image
save_png(ggp_line, 'subsamples_dataset_size.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', height = 1000 , width =1000 )

####Create bar plots to visualize the count of each rank in all the subsampled datasets####

data_list = list(data_superkingdom, data_kingdom, data_phylum, data_class, data_order, data_family, data_genus)
levels = c("superkingdom", "kingdom", "phylum", "class", "order", "family", "genus")

ggp_stacked_multifaceted = make_stacked_plots_multifaceted(data_list, 'superkingdom', 5, 30, 7, levels)
ggp_stacked_multifaceted

save_png(ggp_stacked_multifaceted, 'superkingdom_top5.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 3000, height = 1000)


ggp_stacked = make_stacked_plots(data_list, 'superkingdom', 5, 20, levels = levels)
ggp_stacked

save_png(ggp_stacked, 'superkingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 3000, height = 2000)

##Create stacked bar plots for lineage and subsampled data
data_list = list(lineage_data_cleaned, data_genus)
levels = c("Lineage database","Training data")


#SUPERKINGDOM#
ggp_stacked = make_stacked_plots(data_list, 'superkingdom', 4, 10, levels = levels)
ggp_stacked

save_png(ggp_stacked, 'superkingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 500, height = 200)


ggp_stacked_multifaceted = make_stacked_plots_multifaceted(data_list, 'superkingdom', 4, 30, 2, levels)
ggp_stacked_multifaceted

save_png(ggp_stacked_multifaceted, 'superkingdom_top5.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 500)

#KINGDOM#
ggp_stacked = make_stacked_plots(data_list = data_list, rank = 'kingdom',count=13 ,size = 20, levels = levels)
ggp_stacked

save_png(ggp_stacked, 'kingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 500)


ggp_stacked_multifaceted = make_stacked_plots_multifaceted(data_list, 'kingdom', 13, 20, 2, levels)
ggp_stacked_multifaceted

save_png(ggp_stacked_multifaceted, 'kingdom_top5.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1200, height = 500)


#####Genomic Feature Analysis#####
genomic_data = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/genus_genomic_features.csv",header= TRUE, sep = ',')

#Normalize the data
genomic_data_normalized = as.data.frame(apply(genomic_data[,-1], 2, function(x) (x - min(x)) / (max(x) - min(x))))

#Reshape the data into long format
genomic_data_normalized$SeqName = genomic_data$SeqName  # Reattach the SeqName column for identification
genomic_data_long = melt(genomic_data_normalized, id.vars = "SeqName")
names(genomic_data_long)=c("SeqName","features","value")

size = 20
n_col = 2

#Create boxplots for all features
ggp = ggplot(genomic_data_long, aes(x = features, y = value, fill = features)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE)+
  labs(title = "Boxplots of Normalized Genomic Features",
       x = "Genomic Features",
       y = "Normalized Value") +
  theme_minimal()+theme(
    axis.text.x = element_text(angle = 0, hjust = 1, color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.title.y = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman"),
    legend.text = element_text( family = "Times New Roman", face = 'bold'),
    legend.title = element_text(family = "Times New Roman", face = 'bold'),
    title = element_text(colour = "black", face = "bold", size = (size-5), family = "Times New Roman")
  )
ggp
save_png(ggp, 'features_boxplot.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1500, height = 500)


genomic_data_long = melt(genomic_data, id.vars = "SeqName")
names(genomic_data_long)=c("SeqName","features","value")

size = 30

#Plotting histograms using ggplot2
ggp = ggplot(genomic_data_long, aes(x = value, fill=features)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  facet_wrap(~ features, scales = "free",) +
  scale_fill_viridis(discrete = TRUE)+
  labs(title = "Histograms of Genomic Features",
       x = "Value",
       y = "Feature Frequency")+
  theme_minimal() +theme(
    axis.text.x = element_text(angle = 0, hjust = 1, color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.title.x = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.title.y = element_text(color = "black", face = "bold", size = size, family = "Times New Roman"),
    axis.text.y = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman"),
    legend.text = element_text( family = "Times New Roman", face = 'bold'),
    legend.title = element_text(family = "Times New Roman", face = 'bold'),
    title = element_text(colour = "black", face = "bold", size = size, family = "Times New Roman"),
    strip.text.x = element_text(colour = "black", face = "bold", size = (size), family = "Times New Roman"),
    panel.spacing = unit(4, "lines"),
    legend.position ="none"
  )

save_png(ggp, 'features_histogram.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 2000, height = 2500)


#####Load training datasets####
##Also, convert all taxonomies with artificial_* to NA

#Binary#
binary_training_data = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/model training/genus_binary_master_dataset.csv",header= TRUE, sep = ',') %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .)) %>%
  mutate_all(~ ifelse(grepl('"', as.character(.), ignore.case = TRUE), NA, .))

#Barplot
ggp_bar_binary = make_barplot(table = binary_training_data, category_col = 'superkingdom', size = 10,cutoff = 7)
ggp_bar_binary

#Multicategorical#
multi_training_data = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/model training/genus_multicategorical_master_dataset.csv",header= TRUE, sep = ',') %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .)) %>%
  mutate_all(~ ifelse(grepl('"', as.character(.), ignore.case = TRUE), NA, .))

#Barplot
ggp_bar_multi = make_barplot(table = multi_training_data, category_col = 'superkingdom', size = 60,cutoff = 7, legend = FALSE, flip = FALSE)
ggp_bar_multi
save_png(ggp_bar_multi, 'training_superkingdom.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 2000, width = 1000)

#Create stacked barplots for lineage database and training data for all levels
data_list = list(lineage_data_cleaned, multi_training_data)
levels = c("Lineage database","Training data")

ggp_stacked_multi = make_stacked_plots(data_list = data_list, levels = levels ,rank = 'superkingdom', count = 7,size = 10)
ggp_stacked_multi
save_png(ggp_stacked_multi, 'superkingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 500, height = 200)

####Barplots for lineage vs training####
data_list = list(lineage_data_cleaned,multi_training_data)
levels = c("Lineage database","Training data")


#SUPERKINGDOM#
ggp_stacked = make_stacked_plots(data_list, 'superkingdom', 4, 10, levels = levels)
ggp_stacked

save_png(ggp_stacked, 'superkingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 500, height = 200)


ggp_stacked_multifaceted = make_stacked_plots_multifaceted(data_list, 'superkingdom', 4, 30, 2, levels)
ggp_stacked_multifaceted

save_png(ggp_stacked_multifaceted, 'superkingdom_top5.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 500)

#KINGDOM#
ggp_stacked = make_stacked_plots(data_list = data_list, rank = 'kingdom',count=13 ,size = 20, levels = levels)
ggp_stacked

save_png(ggp_stacked, 'kingdom_top5_stacked.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 500)


ggp_stacked_multifaceted = make_stacked_plots_multifaceted(data_list, 'kingdom', 13, 20, 2, levels)
ggp_stacked_multifaceted

save_png(ggp_stacked_multifaceted, 'kingdom_top5.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1200, height = 500)

###Subset all superkingdoms from training dataset####

#Subset table according to Archea superkingdom
training_archea_subset = multi_training_data[multi_training_data$superkingdom == 'Archaea',]
write.csv(training_archea_subset, "training_archea_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = training_archea_subset, file_prefix = "training_Archaea")

#Subset table according to Bacteria superkingdom
training_bacteria_subset = multi_training_data[multi_training_data$superkingdom == 'Bacteria',]
write.csv(training_bacteria_subset, "training_bacteria_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = training_bacteria_subset, file_prefix = "training_Bacteria")

#Subset table according to Eukaryota superkingdom
training_eukaryota_subset = multi_training_data[multi_training_data$superkingdom == 'Eukaryota',]
write.csv(training_eukaryota_subset, "training_eukaryota_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = training_eukaryota_subset, file_prefix = "training_Eukaryota")

#Subset table according to Viruses supekingdom
training_virus_subset = multi_training_data[multi_training_data$superkingdom == 'Viruses',]
write.csv(training_virus_subset, "training_viruses_lineage.csv", row.names = FALSE)
calculate_rank_count(dataset = training_virus_subset, file_prefix = "training_Virus")

#Subset multi_training_dataset according to 'Others' category
Others_subset = multi_training_data[multi_training_data$category == 'Other',]
write.csv(Others_subset, "training_Others_category.csv", row.names = FALSE)
calculate_rank_count(dataset = Others_subset, file_prefix = "Others_category")

####Create bar plots for viral subset####

training_virus_subset = training_virus_subset %>%
  mutate_all(~ ifelse(is.na(.), 'Other', .))

ggp_bar_virus = make_barplot(table =training_virus_subset, category_col = 'kingdom', cutoff = 0, size = 60, flip = FALSE, legend = FALSE )
ggp_bar_virus

save_png(ggp_bar_virus,'training_virus_kingdoms.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 2000, width = 1500)

####Create bar plots for eukaryota subset####

training_eukaryota_subset = training_eukaryota_subset %>% mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))

ggp_bar_eukaryota = make_barplot(table = training_eukaryota_subset, category_col = 'kingdom', cutoff = 0, size = 50, legend = FALSE)
ggp_bar_eukaryota

save_png(ggp_bar_eukaryota,'training_eukaryota_kingdoms.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML',height = 1500, width = 800)

####Create bar plots for superkingdom####

####Barplots to access lineage and final database####

lineage_data_ggp = visualize_database(lineage_data_cleaned, size = 30)
lineage_data_ggp

save_png(lineage_data_ggp, 'lineage_database_diversity.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 1000)

subsampled_data_ggp = visualize_database(data_genus, size = 30)
subsampled_data_ggp

save_png(subsampled_data_ggp, 'subsampled_data_diversity.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 1000)

training_data_ggp = visualize_database(multi_training_data, size = 30)
training_data_ggp

save_png(training_data_ggp, 'training_data_diversity.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML', width = 1000, height = 1000)

#####Visualise test-prediction datasets######

####Load VMR data####

vmr = read_excel("/Users/harshitasrivastava/Downloads/Viral_ML/VMR_MSL39_v1.xlsx", sheet = 1)

#####Multicategorical datasets####

multi_pred = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/model training/genus_fullseq_multicategorical_test_pred.csv",header= TRUE, sep = ',')
sorted_order = order(multi_pred[,'category'])
multi_pred_sorted = multi_pred[sorted_order,]
multi_pred_subset = multi_pred_sorted[,c('family','category','True.value','Predicted.value', 'Predicted_Category')]

confusion_grid = make_confusion_matrix(data = multi_pred_subset, actual_col = 'category',predicted_col = 'Predicted_Category', size = 20, high_colour = '#dbff33')
save_png(confusion_grid, 'confusion_matrix_multivariate.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 1000, height = 1000)

####Create heatmap of Actual_Virus vs Predicted_virus####

ggp <- make_heatmap(multi_pred, category = 'Virus', top_n = 128)

save_png(ggp, 'family_heatmap_multivariate.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 2000, height = 6000) 

####Creating heatmap for Actual_Virus vs Predicted_category (includes all categories) by each viral family####

virus_pred <- multi_pred_sorted[multi_pred_sorted$category == 'Virus',] 
write.csv(virus_pred,'/Users/harshitasrivastava/Downloads/Viral_ML/model training/virus_multi_test_.csv',row.names =FALSE)
virus_pred_subset <- virus_pred %>% select(family, category, True.value, Predicted.value, Predicted_Category) %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), 'No_family', .))
virus_pred_subset = virus_pred_subset[!virus_pred_subset$family %in% c("No_family"), ]

multi_virus_pred_distinct = virus_pred_subset %>% distinct(family, .keep_all = TRUE)
write.csv(virus_pred_subset,'/Users/harshitasrivastava/Downloads/Viral_ML/model training/viral_predictions.csv',row.names=FALSE)

ggp_multi <- make_predicted_category_heatmap(virus_pred_subset, title = "Family vs Predicted category (multiclass model)", size = 45, col = 'family',prediction_col = 'Predicted_Category', normalize = 'row', display = 'none', colours = c("white","#900C3F"))

save_png(ggp_multi, 'family_heatmap_multivariate_v4.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 3000, height = 6000) 

####Heatmap for DNA-RNA virus vs Predicted Category####

virus_pred <- multi_pred_sorted[multi_pred_sorted$category == 'Virus',] %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))%>% select(kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
columnnames = c("Kingdom","Phylum","Class","Order","Family","Genus","Species","Category","True.value","Predicted.value","Predicted_Category")
colnames(virus_pred) = columnnames

multi_genomic_composition_df = assign_genomic_composition(virus_pred,vmr)
multi_genomic_composition_df = check_genomic_composition(multi_genomic_composition_df, column_name = 'genome_composition')

multi_genomic_composition_df = multi_genomic_composition_df[!is.na(multi_genomic_composition_df$genome_composition), ]

ggp_dna_rna_multi = make_predicted_category_heatmap(data = multi_genomic_composition_df, col = 'genome_composition', prediction_col = 'Predicted_Category', title = "Genomic Composition (Actual vs Predicted)", size = 20, normalize = 'none', display = 'count', legend_key = 1, colours = c('#d57f6c','#900C3F'))

save_png(ggp_dna_rna_multi, 'family_heatmap_dna_rna_multivariate_v4.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training',width = 1500, height = 1000) 

####Analyse the families that have been predicted as 'Bacteria'####

bacteria_pred = virus_pred[virus_pred$Predicted_Category == "Bacteria",]
bacteria_pred_subset = bacteria_pred[,c('Family','Category','True.value','Predicted.value', 'Predicted_Category')]
bacteria_count = calculate_rank_count(bacteria_pred_subset,'bacteria_pred')

#####Binary datasets####

binary_pred = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/model training/genus_fullseq_binary_test_pred.csv",header= TRUE, sep = ',')
sorted_order = order(binary_pred[,'category'])
binary_pred_sorted = binary_pred[sorted_order,]
binary_pred_subset = binary_pred_sorted[,c('family','category','True.value','Predicted.value', 'Predicted_Category')]

confusion_grid = make_confusion_matrix(data = binary_pred_subset, actual_col = 'category',predicted_col = 'Predicted_Category', size = 20)

save_png(confusion_grid, 'confusion_matrix_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 1000, height = 1000)

####Create heatmap of Actual_Virus vs Predicted_virus####

ggp <- make_heatmap(binary_pred, category = 'Virus', top_n = 128)

save_png(ggp, 'family_heatmap_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 2000, height = 6000)

####Creating heatmap for Actual_Virus vs Predicted_something_else by each viral family####

virus_pred <- binary_pred_sorted[binary_pred_sorted$category == 'Virus',]

virus_pred_subset <- virus_pred %>%
  select(family, category, True.value, Predicted.value, Predicted_Category) %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), 'No_family', .))
virus_pred_subset = virus_pred_subset[!virus_pred_subset$family %in% c("No_family"), ]

binary_virus_pred_distinct = virus_pred_subset %>% distinct(family, .keep_all = TRUE)
write.csv(virus_pred_subset,'/Users/harshitasrivastava/Downloads/Viral_ML/model training/viral_predictions.csv',row.names=FALSE)

ggp_binary <- make_predicted_category_heatmap(virus_pred_subset, col = 'family',prediction_col = 'Predicted_Category',title = "Family vs Predicted category (binary model)", size = 45, normalize = 'row', display = 'none', colours = c("white","darkgreen"))

save_png(ggp_binary, 'family_heatmap_binary_v4.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 2000, height = 6000) 

####Heatmap for DNA-RNA virus vs Predicted Category####

virus_pred <- binary_pred_sorted[binary_pred_sorted$category == 'Virus',] %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))%>% select(kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
columnnames = c("Kingdom","Phylum","Class","Order","Family","Genus","Species","Category","True.value","Predicted.value","Predicted_Category")
colnames(virus_pred) = columnnames

binary_genomic_composition_df = assign_genomic_composition(virus_pred,vmr)
binary_genomic_composition_df = check_genomic_composition(binary_genomic_composition_df, column_name = 'genome_composition')

binary_genomic_composition_df_subset =binary_genomic_composition_df[!is.na(binary_genomic_composition_df$genome_composition), ]

ggp_dna_rna_binary = make_predicted_category_heatmap(data = binary_genomic_composition_df, col = 'genome_composition', prediction_col = 'Predicted_Category', title = "Actual genomic composition vs predictions (binary model)", size = 30, normalize = 'none', colours = c("white","darkgreen"), n=90, display = 'count', legend_key = 1 )

save_png(ggp_dna_rna_binary, 'family_heatmap_dna_rna_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/model training', width = 1500, height = 1000) 

#####Analyse test data with new features#####

####Binary#####
binary_new_test_df = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/genus_binary_test_pred.csv",header= TRUE, sep = ',')%>% select(taxid,superkingdom,kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
sorted_order = order(binary_new_test_df[,'category'])
binary_new_test_df_sorted = binary_new_test_df[sorted_order,]

####Confusion matrix####
binary_ggp_cm = make_confusion_matrix(data = binary_new_test_df_sorted, actual_col = 'category',predicted_col = 'Predicted_Category')

save_png(binary_ggp_cm, 'confusion_matrix_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars', width = 1000, height = 1000)

####Creating heatmap for Actual_Virus vs Predicted_something_else by each viral family####

binary_virus_pred <- binary_new_test_df_sorted[binary_new_test_df_sorted$category == 'Virus',]

binary_virus_pred_subset <- binary_virus_pred %>%
  select(family, category, True.value, Predicted.value, Predicted_Category) %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), 'No_family', .))
binary_virus_pred_subset = binary_virus_pred_subset[!binary_virus_pred_subset$family %in% c("No_family"), ]

binary_new_pred_subset_distinct = binary_virus_pred_subset %>% distinct(family, .keep_all = TRUE)
write.csv(binary_virus_pred_subset,'/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/binary_viral_predictions.csv',row.names=FALSE)

ggp_binary_new_features <- make_predicted_category_heatmap(binary_virus_pred_subset, col = 'family',prediction_col = 'Predicted_Category',title = "Family vs Predicted category (binary model)", size = 45, normalize = 'row',colours = c("white","#bd6cd5"),)

save_png(ggp_binary_new_features, 'family_heatmap_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars', width = 2000, height = 6000)

####Heatmap for DNA-RNA virus vs Predicted Category####

binary_virus_pred <- binary_new_test_df_sorted[binary_new_test_df_sorted$category == 'Virus',] %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))%>% select(kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
columnnames = c("Kingdom","Phylum","Class","Order","Family","Genus","Species","Category","True.value","Predicted.value","Predicted_Category")
colnames(binary_virus_pred) = columnnames

binary_genomic_composition_df = assign_genomic_composition(binary_virus_pred,vmr)
binary_genomic_composition_df = check_genomic_composition(binary_genomic_composition_df, column_name = 'genome_composition')

binary_genomic_composition_df_subset =binary_genomic_composition_df[!is.na(binary_genomic_composition_df$genome_composition), ]

ggp_dna_rna_binary = make_predicted_category_heatmap(data = binary_genomic_composition_df, col = 'genome_composition', prediction_col = 'Predicted_Category', title = "Actual genomic composition vs predictions (binary model)", size = 30, normalize = 'none', colours = c("white","#bd6cd5"), n=90, display = 'count', legend_key = 1 )

save_png(ggp_dna_rna_binary, 'family_heatmap_dna_rna_binary.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars', width = 1500, height = 1000) 

#####Multicategorical####

multi_new_test_df = read.table("/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/genus_multicategorical_test_pred.csv",header= TRUE, sep = ',')%>% select(taxid,superkingdom,kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
sorted_order = order(multi_new_test_df[,'category'])
multi_new_test_df_sorted = multi_new_test_df[sorted_order,]

####Confusion matrix####

multi_ggp_cm = make_confusion_matrix(data = multi_new_test_df_sorted, actual_col = 'category',predicted_col = 'Predicted_Category', size = 30)

save_png(multi_ggp_cm, 'confusion_matrix_multi.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars', width = 2000, height = 2000)

####Creating heatmap for Actual_Virus vs Predicted_something_else by each viral family####

multi_virus_pred <- multi_new_test_df_sorted[multi_new_test_df_sorted$category == 'Virus',]

multi_new_virus_pred_subset <- multi_virus_pred %>%
  select(family, category, True.value, Predicted.value, Predicted_Category) %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), 'No_family', .))
multi_new_virus_pred_subset = multi_new_virus_pred_subset[!multi_new_virus_pred_subset$family %in% c("No_family"), ]

multi_new_pred_subset_distinct = multi_new_virus_pred_subset %>% distinct(family, .keep_all = TRUE)
write.csv(multi_new_virus_pred_subset,'/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/multi_viral_predictions.csv',row.names=FALSE)

ggp_multi_new_features <- make_predicted_category_heatmap(multi_new_virus_pred_subset, col = 'family',prediction_col = 'Predicted_Category',title = "Family vs Predicted category (multivariate model)", size = 45, normalize = 'row', display = 'none', colours = c('white','#2659b9'))

save_png(ggp_multi_new_features, 'family_heatmap_multivariate.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars', width = 3000, height = 6000)

####Heatmap for DNA-RNA virus vs Predicted Category####

multi_new_virus_pred <- multi_new_test_df_sorted[multi_new_test_df_sorted$category == 'Virus',] %>%
  mutate_all(~ ifelse(grepl("artificial", as.character(.), ignore.case = TRUE), NA, .))%>% select(kingdom,phylum,class,order,family,genus,species,category, True.value, Predicted.value, Predicted_Category)
columnnames = c("Kingdom","Phylum","Class","Order","Family","Genus","Species","Category","True.value","Predicted.value","Predicted_Category")
colnames(multi_new_virus_pred) = columnnames

multi_new_genomic_composition_df = assign_genomic_composition(multi_new_virus_pred,vmr)
multi_new_genomic_composition_df = check_genomic_composition(multi_new_genomic_composition_df, column_name = 'genome_composition')

multi_new_genomic_composition_df = multi_new_genomic_composition_df[!is.na(multi_new_genomic_composition_df$genome_composition), ]

ggp_dna_rna_multi = make_predicted_category_heatmap(data = multi_new_genomic_composition_df, col = 'genome_composition', prediction_col = 'Predicted_Category', title = "Genomic Composition (Actual vs Predicted)", size = 20, normalize = 'none', display = 'count', legend_key = 1, colours = c('#d5e4ff','#2659b9'))

save_png(ggp_dna_rna_multi, 'family_heatmap_dna_rna_multivariate.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars',width = 1500, height = 1000) 

#####New feature analysis####
####Load data####

new_feature = read_csv("/Users/harshitasrivastava/Downloads/Viral_ML/master_data/model_training/genus_new_features_binary_master_dataset.csv", show_col_types = FALSE)%>% select(-c(taxid,length,superkingdom,kingdom,phylum,class,order,family,genus,species,subspecies,strain,category))

#####Real metagenomic test features####

test_features = read_csv("/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/case_study/case_master_features_v2.csv", show_col_types = FALSE)%>% select(c('SequenceName','category'))

test_metadata = read_excel("/Users/harshitasrivastava/Downloads/Viral_ML/diamond_all_virus_hits_min200.xlsx", sheet = 1)

####Binary dataset with test features####

binary_new_feature = read_csv("/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/case_study/case_study_binary_test_pred.csv", show_col_types = FALSE)%>% select(c('category','True value','Predicted value','Predicted_Category'))

binary_new_feature = cbind(binary_new_feature, test_features)
binary_new_feature = binary_new_feature[,c('category','True value','Predicted value','Predicted_Category','SequenceName')]
binary_new_feature = binary_new_feature[, c('SequenceName', setdiff(names(binary_new_feature), 'SequenceName'))]
columns = c('Header','category','True value','Predicted value','Predicted_Category')
colnames(binary_new_feature) = columns
binary_new_feature = merge(test_metadata,binary_new_feature,  by='Header', all = TRUE)

ggp_binary_test_heatmap <- make_predicted_category_heatmap(binary_new_feature, col = 'category',prediction_col = 'Predicted_Category',title = "True_value vs Predicted_value (binary model)", xlab = "True category", ylab = 'Predicted Category' ,size = 20, normalize = 'none', display = 'count', colours = c('#d5ffdc','#278666'), legend_key = 1)

save_png(ggp_binary_test_heatmap, 'binary_confusion_matrix.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/case_study',width = 1500, height = 500) 

####Multicategorical dataset with test features####

multi_new_feature = read_csv("/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/case_study/case_study_multicategorical_test_pred.csv", show_col_types = FALSE)%>% select(c('category','True value','Predicted value','Predicted_Category'))

multi_new_feature = cbind(multi_new_feature, test_features)
multi_new_feature = multi_new_feature[,c('category','True value','Predicted value','Predicted_Category','SequenceName')]
multi_new_feature = multi_new_feature[, c('SequenceName', setdiff(names(multi_new_feature), 'SequenceName'))]
columns = c('Header','category','True value','Predicted value','Predicted_Category')
colnames(multi_new_feature) = columns
multi_new_feature = merge(test_metadata,multi_new_feature,  by='Header', all = TRUE)

ggp_multi_test_heatmap <- make_predicted_category_heatmap(multi_new_feature, col = 'category',prediction_col = 'Predicted_Category',title = "True_value vs Predicted_value (multiclass model)", xlab = "True category", ylab = 'Predicted Category' ,size = 20, normalize = 'none', display = 'count', colours = c('#fff8a2','#d3c94b'), legend_key = 1)

save_png(ggp_multi_test_heatmap, 'multicategorical_confusion_matrix.png', path = '/Users/harshitasrivastava/Downloads/Viral_ML/from_mars/case_study',width = 1500, height = 500)
#________________________________________________________________________________________________________________________________________________________#