######### Packages ############
library(readr)
library(png)
library(purrr)
library(gtable)
library(gridExtra)
library(ggplot2)
library(dplyr)
library(knitr)
library(kableExtra)
library(pander)
library(tidyverse)
library(tmap)
library(leaflet)
library(ggforce)
library(treemap)
library(readxl)
library(plotly)
library(ggalluvial)
library(migest)
library(rlist)
library(data.table)
library(hms)
library(flextable)
library(skimr)
library(here)

##### Declare path for datasets #####

# All paths are resolved with here::here() from the repository root (the repo
# ships a `.here` sentinel). Raw CRS .txt files go in
# 'Climate finance estimation/Raw Data/CRS/' (see CRS/Readme.md).

####### Years taken into account #####

##### Begin and end year ####

Period <- c(1973:2022) 
Bound  <- paste0(min(Period), "-", max(Period)) 

### Yearly dataset declaration ###

Yearly <- c(2006: 2021)

######### Datasets charging ########

##### Bases annuelles #####

for (i in  Yearly) {
  filename<-paste0("CRS"," ",i," ", "Data")
  wd<-here::here("Climate finance estimation", "Raw Data", "CRS", paste0("CRS"," ",i," ", "Data", ".txt"))
  assign(filename,fread(wd,  encoding =  "Latin-1"))
}

gc() 

#### Creation of a list dataset ####

CRS <- lapply(Yearly, function(x) {
  dataframeName <- paste0("CRS ", x, " Data")
  return(get(dataframeName))
})

##### For pluri-annual datasets #####


"CRS 1973-94 data" <- fread(here::here("Climate finance estimation", "Raw Data", "CRS", "CRS 1973-94 data.txt"),  encoding =  "Latin-1")
"CRS 1995-99 data" <- fread(here::here("Climate finance estimation", "Raw Data", "CRS", "CRS 1995-99 data.txt"),  encoding =  "Latin-1")
"CRS 2000-01 data" <- fread(here::here("Climate finance estimation", "Raw Data", "CRS", "CRS 2000-01 data.txt"),  encoding =  "Latin-1")
"CRS 2002-03 data" <- fread(here::here("Climate finance estimation", "Raw Data", "CRS", "CRS 2002-03 data.txt"),  encoding =  "Latin-1")
"CRS 2004-05 data" <- fread(here::here("Climate finance estimation", "Raw Data", "CRS", "CRS 2004-05 data.txt"),  encoding =  "Latin-1")


####### Creation of a list with all datasets #######

#### List creation ####
BDD <- list()
gc() 

#### Insertion of datasets ####

BDD <-  c(list(`CRS 1973-94 data`, `CRS 1995-99 data`, `CRS 2000-01 data`, `CRS 2002-03 data`, `CRS 2004-05 data`),CRS)
gc() 

#### Change list element into dataframes #### 

BDD <- lapply(1:length(BDD), function (x){
  as.data.frame(BDD[x])
})
gc()


out <- lapply(1:length(BDD), function (x){
  split( BDD[[x]] , f = BDD[[x]]$Year )
  }) 
gc()



BDD <- unlist(out, recursive = FALSE)
gc()

rm(list=setdiff(ls(), c("BDD", "CRS", "Period", "Bound")))
gc()
