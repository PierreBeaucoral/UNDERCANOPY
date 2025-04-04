rm(list = ls())

# Define the desired working directory path
wd <- "/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Travaux CRS"

# Get the current working directory
current_wd <- getwd()

# Check if the current working directory is different from the desired directory
if (current_wd != wd) {
  # If different, set the working directory to the desired directory
  setwd(wd)
  print(paste("Changed working directory to:", getwd()))
} else {
  # If already in the desired directory, print a message indicating the current directory
  print(paste("Current working directory is already set to:", getwd()))
}

#### 0- Import data from .cvs files#### 

source("UploadBase.R")
gc()


#### 1- Get only data of interest #### 
#Comment: Consider only data that get a sector code categorization 


BDD <- lapply (BDD, function(x) drop_na(x, c("Year", "DonorCode", "RecipientCode")))
rm(list=setdiff(ls(), c("BDD", "Bound", "Period")))
gc()

#### 2- Create variable raw_text #### 


processed_df_list <- lapply(BDD, function(df) {
  df <- df %>%
    select(Year, DonorCode, DonorName, CrsID, RecipientCode, RecipientName, FlowCode, Bi_Multi, Category, Finance_t, USD_Commitment_Defl, USD_Disbursement_Defl, ProjectTitle, ShortDescription, LongDescription, PurposeCode,ClimateAdaptation,ClimateMitigation) %>%
    filter(!is.na(ProjectTitle) & !is.na(ShortDescription) & !is.na(LongDescription)) %>%
    unite("raw_text", ProjectTitle:ShortDescription, sep = ' ') %>%
    unite("raw_text", raw_text:LongDescription, sep = ' ')
  gc()
  return(df)
})
gc()


### Lowercasing ###

processed_df_list <- lapply(processed_df_list, function(df) {
  df$raw_text <- tolower(df$raw_text)
  return(df)
})

#### 3- Create  Dataframe #### 


Data <- do.call("rbind", processed_df_list)
gc()

Data <- Data%>%
  distinct()
gc() 

Data <- Data%>%
  filter(!grepl("semi-aggregates", raw_text, fixed = TRUE))%>%
  filter(!grepl("sectors not specified", raw_text, fixed = TRUE))

# Remove underscores from the raw text variable
Data$raw_text <- gsub("_", "", Data$raw_text)

# Create a subsample for beta-testing
rand_df <- Data[sample(nrow(Data), size=10000), ]

write.table(Data, "./Data/DataPB.csv", sep="|")
write.table(rand_df, "./Data/DataPBsample.csv", sep="|")
gc()
# 
# 
# BDD <- lapply(BDD, filter, SectorCode!=998)
# gc()
# 
# #### 2- Select only year from 2000 #### 
# 
# BDD <- BDD[c(28 :49)]
# gc()
# 
# #### 3- Exclude FLows that are not coming from public entities #### 
# #Comment: We exclude flows from private development finance
# 
# # Function to find unique combinations of FlowCode and FlowName
# find_unique_combinations <- function(df) {
#   unique_combinations <- unique(df[, c("FlowCode", "FlowName")])
#   return(unique_combinations)
# }
# 
# # Apply the function to each data frame in the list
# unique_combinations_list <- lapply(BDD, find_unique_combinations)
# 
# # Print or access unique combinations for each data frame
# # print(unique_combinations_list)
# 
# BDD <- lapply(BDD, filter, FlowCode != 30 )
# gc()
# 
# ####4- Creation of variables####
# 
# 
# #####4.2 Type fo financing#####
# 
# BDD <- lapply(BDD, mutate, financetype = case_when(
#   Finance_t %in% c(110,210,310,311) ~ 'Grant',
#   Finance_t %in% c(421,422,423,424,425) ~ 'Debt',
#   Finance_t %in% c(431,432,433) ~ 'Mezzanine',
#   Finance_t %in% c(510,520,530) ~ 'Equity',
#   Finance_t %in% c(610, 611) ~ 'Debt forgiveness',
#   Finance_t %in% c(1100) ~ 'Guarantees',
#   TRUE ~ 'Other'
# ))
# 
# ##### 4.3 Sectors#####
# 
# BDD <- lapply(BDD, mutate, sector = case_when(
#   PurposeCode %in% c(11110, 11120, 11130, 11182, 11220, 11230, 11231, 11232,11240, 11250, 11260, 11320,11330,11420, 11430) ~ 'Education',
#   PurposeCode %in% c(12110,12181, 12182,12191,12220, 12230,12240, 12250, 12261,12262, 12263, 12264, 12281, 12310,12320, 12330,12340,12350, 12382) ~ 'Health',
#   PurposeCode %in% c(14010, 14015,14020,  14021,14022, 14030,14031, 14032,14040,14050, 14081) ~ 'Water Supply & Sanitation',
#   PurposeCode %in% c(15110,15111, 15112,15113,  15114, 15125, 15130, 15142, 15150, 15151, 15152,15153, 15160, 15170,15180, 15190 ) ~ 'Government & Civil Society',
#   PurposeCode %in% c(21010,21020,21030, 21040,21050,21061, 21081 ) ~ 'Transport & Storage',
#   PurposeCode %in% c(22010, 22020, 22030, 22040) ~ 'Communications',
#   PurposeCode %in% c(23110,23181, 23182,23183,23210,23220, 23230,23231,23232,23240,23250,23260,23270,23310,23320,23330, 23340,23350, 23360,23410,23510, 23610,23620, 23630, 23631,23640, 23641,23642 ) ~ 'Energy',
#   PurposeCode %in% c(24010,24020, 24030, 24040,24050,24081 ) ~ 'Banking & Financial Services',
#   PurposeCode %in% c(25010,25020, 25030,25040 ) ~ 'Business & Other Services',
#   PurposeCode %in% c(31110, 31120, 31130,31140,31150, 31161, 31162,31163,31164,31165, 31166,  31181, 31182, 31191, 31192,31193,31194,31195,31210,31220, 31261, 31281, 31282,31291,31310, 31320, 31381,31382,31391 ) ~ 'Agriculture, Forestry, Fishing',
#   PurposeCode %in% c(32110,32120,32130,32140,32161,32162,32163,32164,32165, 32166,32167,  32168,32169,32170, 32171,32172, 32173,32174,32182,32210,32220, 32261, 32262,32263,32264,32265,32266,32267,32268,32310) ~ 'Industry, Mining, Construction',
#   PurposeCode %in% c(41010, 41020,41030,41040,41081,41082) ~ 'General Environment Protection',
#   TRUE ~ 'Other'
# ))
# 
# gc()
# 
# # 4.3.1. more precised Sector
# 
# BDD <- lapply(BDD, mutate, sectorB = case_when(
#   PurposeCode %in% c(11110, 11120, 11130, 11182, 11220, 11230, 11231, 11232,11240, 11250, 11260, 11320,11330,11420, 11430) ~ 'Education',
#   PurposeCode %in% c(12110,12181, 12182,12191,12220, 12230,12240, 12250, 12261,12262, 12263, 12264, 12281, 12310,12320, 12330,12340,12350, 12382) ~ 'Health',
#   PurposeCode %in% c(13010,13020, 13030,13040,13081) ~ 'Population',
#   PurposeCode %in% c(14010, 14015,14020,  14021,14022, 14030,14031, 14032,14040,14050, 14081) ~ 'Water Supply & Sanitation',
#   PurposeCode %in% c(15110,15111, 15112,15113,  15114, 15125, 15130, 15142, 15150, 15151, 15152,15153, 15160, 15170,15180, 15190 ) ~ 'Government & Civil Society',
#   PurposeCode %in% c(21010,21020,21030, 21040,21050,21061, 21081 ) ~ 'Transport & Storage',
#   PurposeCode %in% c(22010, 22020, 22030, 22040) ~ 'Communications',
#   PurposeCode %in% c(23110,23181, 23182,23183,23210,23220, 23230,23231,23232,23240,23250,23260,23270,23310,23320,23330, 23340,23350, 23360,23410,23510, 23610,23620, 23630, 23631,23640, 23641,23642 ) ~ 'Energy',
#   PurposeCode %in% c(24010,24020, 24030, 24040,24050,24081 ) ~ 'Banking & Financial Services',
#   PurposeCode %in% c(25010,25020, 25030,25040 ) ~ 'Business & Other Services',
#   PurposeCode %in% c(31110, 31120, 31130,31140,31150, 31161, 31162,31163,31164,31165, 31166,  31181, 31182, 31191, 31192,31193,31194,31195,31210,31220, 31261, 31281, 31282,31291,31310, 31320, 31381,31382,31391 ) ~ 'Agriculture, Forestry, Fishing',
#   PurposeCode %in% c(32110,32120,32130,32140,32161,32162,32163,32164,32165, 32166,32167,  32168,32169,32170, 32171,32172, 32173,32174,32182,32210,32220, 32261, 32262,32263,32264,32265,32266,32267,32268,32310) ~ 'Industry, Mining, Construction',
#   PurposeCode %in% c(33110,33120,33130,33140,33150,33181,33210) ~ 'Commercial Policy and Regulations',
#   PurposeCode %in% c(41010, 41020,41030,41040,41081,41082) ~ 'General Environment Protection',
#   PurposeCode %in% c(43010, 43030, 43040, 43050, 43060, 43071, 43072, 43073,43081,43082) ~ 'Other Multisectors',
#   PurposeCode %in% c(51010) ~ 'Budgetary Support',
#   PurposeCode %in% c(52010) ~ 'Developmental Food Aid',
#   PurposeCode %in% c(53030, 53040) ~ 'Product Assistance, Other',
#   PurposeCode %in% c(60010,60020, 60030,60040,60061,60062, 60063) ~ 'Actions relying to debt',
#   PurposeCode %in% c(72010,72040,72050 ) ~ 'Emergency Response',
#   PurposeCode %in% c(73010) ~ 'Reconstruction & Rehabilitation',
#   PurposeCode %in% c(74020) ~ 'Disaster Prevention/Preparedness',
#   PurposeCode %in% c(91010) ~ 'Administrative Costs of Donors',
#   PurposeCode %in% c(93010) ~ 'Refugees in Donor Countries',
#   TRUE ~ 'Other'
# ))
# gc()
# 
# #####4.4 Main channels (who receive the funds?)#####
# 
# BDD <-lapply(BDD, mutate, MainChannel= case_when(
#   ChannelCode %in% c(11000, 11001,  11002, 11003, 11004, 12000, 12001,12002,12003,12004, 13000) ~ "PUBLIC SECTOR INSTITUTIONS",
#   ChannelCode %in% c(20000:23504) ~ "NGO",
#   ChannelCode %in% c(30000:32000) ~ "PPP",
#   ChannelCode %in% c(41000:41999) ~ "UN",
#   ChannelCode %in% c(46000:46999) ~ "Regional Development Bank",
#   ChannelCode %in% c(60000:63999) ~ "Private sector institution",
#   TRUE ~ ChannelName ))
# 
# BDD <-lapply(BDD, mutate, Channel= case_when(
#   ChannelCode %in% c(11000, 11001) ~ "Donor Government",
#   ChannelCode %in% c(11003) ~ "Public corporations (donor)",
#   ChannelCode %in% c(12000) ~ "Recipient Government",
#   ChannelCode %in% c(12003) ~ "Public corporations (recipient)",
#   ChannelCode %in% c(13000) ~ "Third Country Government (Delegated co-operation)",
#   ChannelCode %in% c(21000) ~ "International NGO",
#   ChannelCode %in% c(21037) ~ "Women's World Banking",
#   ChannelCode %in% c(22000) ~ "Donor country-based NGO",
#   ChannelCode %in% c(23000) ~ "Developing country-based NGO",
#   ChannelCode %in% c(32000) ~ "Networks",
#   ChannelCode %in% c(42004) ~ "European Investment Bank",
#   ChannelCode %in% c(46000) ~ "Regional Development Banks",
#   ChannelCode %in% c(51000) ~ "University, college or other teaching institution, research institute or think-tank",
#   ChannelCode %in% c(60000) ~ "Private Sector Institutions",
#   ChannelCode %in% c(61000) ~ "Private sector in provider country",
#   ChannelCode %in% c(61001) ~ "Banks (donor)",
#   ChannelCode %in% c(61003) ~ "Investment funds and other collective investment institutions (donor)",
#   ChannelCode %in% c(61004) ~ "Holding companies, trusts and Special Purpose Vehicles (donor)",
#   ChannelCode %in% c(61007) ~ "Other financial corporations (donor)",
#   ChannelCode %in% c(61008) ~ "Exporters (donor)",
#   ChannelCode %in% c(61009) ~ "Other non-financial corporations (donor)",
#   ChannelCode %in% c(60000) ~ "Private Sector Institutions",
#   ChannelCode %in% c(62000) ~ "Private sector in recipient country",
#   ChannelCode %in% c(62001) ~ "Banks (recipient)",
#   ChannelCode %in% c(62002) ~ "Micro Finance Institutions (recipient)",
#   ChannelCode %in% c(62003) ~ "Investment funds and other collective investment institutions (recipient)",
#   ChannelCode %in% c(62004) ~ "Holding companies, trusts and Special Purpose Vehicles (recipient)",
#   ChannelCode %in% c(62007) ~ "Other financial corporations (recipient)",
#   ChannelCode %in% c(62008) ~ "Importers/Exporters (recipient)",
#   ChannelCode %in% c(62009) ~ "Other non-financial corporations (recipient)",
#   ChannelCode %in% c(63000) ~ "Private sector in third party country",
#   ChannelCode %in% c(63001) ~ "Banks (third party)",
#   ChannelCode %in% c(63002) ~ "Micro Finance Institutions (third party)",
#   ChannelCode %in% c(63003) ~ "Investment funds and other collective investment institutions (third party)",
#   ChannelCode %in% c(63004) ~ "Holding companies, trusts and Special Purpose Vehicles (third party)",
#   ChannelCode %in% c(63007) ~ "Other financial corporations (third party)",
#   ChannelCode %in% c(63008) ~ "Importers/Exporters (third party)",
#   ChannelCode %in% c(63009) ~ "Other non-financial corporations (third party)",
#   TRUE ~ ChannelName ))
# 
# gc()
# 
# #####4.5 New flows#####
# 
# BDD <-lapply(BDD, mutate, NewFlow = case_when( 
#   InitialReport %in% c(1) ~ 1,
#   InitialReport %in% c(2,3,4,5,6,7,8,9) ~ 0
# ))
# 
# ####5- Add variables from other sources####
# 
# #####5.1 Iso code#####
# 
# BDD <- lapply(BDD, function(x) cbind(x, Country_Code <- countrycode::countrycode(sourcevar = x$RecipientName, origin = "country.name", destination = "iso3c")))
# BDD <- lapply(BDD, function(df) {
#   df %>% 
#     rename(Country_Code ="Country_Code <- countrycode::countrycode(sourcevar = x$RecipientName, ")
# })
# 
# BDD <- lapply(BDD, function(df) {
#   df[df$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"
#   return(df)
# })
# 
# 
# #####5.2 Region of the recipient countries#####
# 
# BDD <- lapply(BDD, function(x) cbind(x, Region <- countrycode::countrycode(sourcevar = x$Country_Code, origin = "iso3c", destination = "region")))
# BDD <- lapply(BDD, function(df) {
#   df %>% 
#     rename(Region ="Region <- countrycode::countrycode(sourcevar = x$Country_Code, ")
# })
# 
# #Alternative variable of region for some unclassified cases (new variables is created)
# BDD <-lapply(BDD, mutate, RegionB= case_when( 
#   RecipientCode %in% c(55,57,88,89,619, 689) ~ "Europe & Central Asia",
#   RecipientCode %in% c(189,589) ~ "Middle East & North Africa",
#   RecipientCode %in% c(237,289,298, 1027,1028,1029,1030) ~ "Sub-Saharan Africa",
#   RecipientCode %in% c(798,789,860,889,1033,1034,1035) ~ "East Asia & Pacific",
#   RecipientCode %in% c(679,689) ~ "South Asia",
#   RecipientCode %in% c(389,498,489,1031,1032,361) ~ "Latin America & Caribbean",
#   RecipientCode %in% c(998,9998) ~ "Unspecified",
#   TRUE ~ Region ))
# 
# 
# gc()
# 
# #### 6- Compile final database####
# 
# 
# 
# Data <- Data%>%
#   distinct()
# gc() 
# 
# Data <- Data%>%
#   filter(!grepl("Semi-aggregates", raw_text, fixed = TRUE))
# 
# MLData <- Data%>%
# filter(Year %in% c(2000:2017))
# gc()
# 
# predictData <- Data%>%
#   filter(Year %in% c(2018:2021))
# gc()
# 
# 
#  write.table(MLData, "./Data/MLData.csv", sep="|")
#  gc()
# 
#  write.table(predictData, "./Data/predictData.csv", sep="|")
#  gc()
