# Clear the environment
rm(list = ls())

# Load necessary libraries
library(data.table)  # for fread()
library(dplyr)       # for data manipulation
library(ggplot2)     # in case of future plotting needs

# Set the path for the data directory
working_dir <- "./UNDERCANOPY/Climate finance estimation/Raw Data"

# Check if the current working directory is different from the desired one
if (getwd() != working_dir) {
  setwd(working_dir) # Change working directory
  message("Changed working directory to: ", getwd())
} else {
  message("Current working directory is already set to: ", getwd())
}

# Specify the period of interest
years <- 1973:2023
period_label <- paste0(min(years), "-", max(years))

# Years with annual data
annual_years <- 2006:2023



# MODIFICATIONS LF:
# problème de portabilité entre macosx et w11 ?
# Load annual datasets
for (year in annual_years) {
  file_name <- paste0("CRS ", year, " Data") # instruction ok
  # ici je suis obligé d'ajouter à l'instruction le nom du dossier "CRS" car dans le code initial le dossier n'est pas spécifié:
  file_path <- file.path(".", paste0("CRS/CRS ", year, " Data.txt"))
  # CODE INITIAL : file_path <- paste0("./CRS ", year, " Data.txt")
  
  # Vérifier si le fichier existe
  if (file.exists(file_path)) {
    assign(file_name, fread(file_path, encoding = "UTF-8"))
  } else {
    print(paste("Fichier non trouvé :", file_path))
  }
}


gc() # Free up memory

# Create a list of annual datasets
CRS <- lapply(annual_years, function(year) {
  get(paste0("CRS ", year, " Data"))
})

# MODIFICATIONS LF:
# ici les chemins spécifiés ne sont pas les bons ? à moins que ces fichiers ne soient pas à placer dans raw data ?
# je modifie les chemins :

# Load multi-year datasets
CRS_1973_94_data <- fread("CRS/CRS 1973-94 data.txt", encoding = "Latin-1")
CRS_1995_99_data <- fread("CRS/CRS 1995-99 data.txt", encoding = "Latin-1")
CRS_2000_01_data <- fread("CRS/CRS 2000-01 data.txt", encoding = "Latin-1")
CRS_2002_03_data <- fread("CRS/CRS 2002-03 data.txt", encoding = "Latin-1")
CRS_2004_05_data <- fread("CRS/CRS 2004-05 data.txt", encoding = "Latin-1")

# Combine all datasets into a list
all_data <- c(list(CRS_1973_94_data, CRS_1995_99_data, CRS_2000_01_data, CRS_2002_03_data, CRS_2004_05_data), CRS)

# Convert all list elements to data frames
all_data <- lapply(all_data, as.data.frame)

# Function to split datasets by year and combine them
split_and_combine_by_year <- function(data_list) {
  # Initialize an empty list to store results
  result <- list()
  
  # Process each dataset in the list
  for (dataset in data_list) {
    # Get unique years in this dataset
    years <- unique(dataset$Year)
    
    # Split the dataset by year
    for (year in years) {
      year_data <- dataset[dataset$Year == year, ]
      
      # If this year already exists in result, append the data
      if (!is.null(result[[as.character(year)]])) {
        result[[as.character(year)]] <- rbind(result[[as.character(year)]], year_data)
      } else {
        # If this is the first data for this year, create new entry
        result[[as.character(year)]] <- year_data
      }
    }
  }
  
  return(result)
}

# Apply the function to your data
yearly_data <- split_and_combine_by_year(all_data)

# Unlist the data and combine into a single data frame
final_data <- do.call(rbind, unlist(yearly_data, recursive = FALSE))

# Cleanup environment, keeping only the important objects
rm(list = setdiff(ls(), c("final_data", "CRS", "years", "period_label")))
gc() # Free up memory
