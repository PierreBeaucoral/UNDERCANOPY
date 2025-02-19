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
library(ggalluvial)
library(migest)
library(rlist)
library(data.table)
library(hms)
library(flextable)
library(skimr)
##################################################################################

###### Preparing Data for Climate adaptation ######
setwd("/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Determinant of climate finance")

rm(list=ls())

##### Rio Data ##### 

Rio_Data <- fread('/Users/pierrebeaucoral/Documents/Pro/Thèse CERDI/Recherche/Travaux CRS/Data/DataPB.csv')

Rio_data_adaptation <- Rio_Data%>%
  filter(ClimateAdaptation %in% (1:2))%>%
  group_by(Year, DonorName, RecipientName)%>%
  summarise(Rio_Adapt_Commitment= sum(USD_Commitment_Defl*1000, na.rm = T))

Rio_data_adaptation$ProviderISO <- countrycode::countrycode(sourcevar = Rio_data_adaptation$DonorName, origin = "country.name", destination = "iso3c")
Rio_data_adaptation$RecipientISO <- countrycode::countrycode(sourcevar = Rio_data_adaptation$RecipientName, origin = "country.name", destination = "iso3c")
Rio_data_adaptation[Rio_data_adaptation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"


#### CLimateFinanceBERT Data ####

ClimateBERT <- fread("Data/climate_finance_total.csv")

ClimateBERT_Adaptation <- ClimateBERT%>%
  filter(meta_category == "Adaptation")%>%
  group_by(Year, DonorName, RecipientName, climate_class)%>%
  summarise(ClimateBERT_Adapt_Commitment= sum(USD_Commitment_Defl*1000, na.rm = T))

ClimateBERT_Adaptation$ProviderISO <- countrycode::countrycode(sourcevar = ClimateBERT_Adaptation$DonorName, origin = "country.name", destination = "iso3c")
ClimateBERT_Adaptation$RecipientISO <- countrycode::countrycode(sourcevar = ClimateBERT_Adaptation$RecipientName, origin = "country.name", destination = "iso3c")
ClimateBERT_Adaptation[ClimateBERT_Adaptation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"


#### Huei ####
data_adaptation<-read.csv("Data/Adaptation with gravity vars amended FULL Feb 14 2023.csv")

data_adaptation$NDC15[data_adaptation$NDC15%in%1 & !data_adaptation$Year%in%2015]<-0
data_adaptation$NDC15[data_adaptation$NDC16%in%1 & !data_adaptation$Year%in%2016]<-0
data_adaptation$NDC15[data_adaptation$NDC17%in%1 & !data_adaptation$Year%in%2017]<-0
data_adaptation$NDC15[data_adaptation$NDC18%in%1 & !data_adaptation$Year%in%2018]<-0

## In data_adaptation the amount of Adapt and Mitigation are defined as NA
## after constructing the data_adaptationset, change the adaptation and mitigation to zeros for NA cells

data_adaptation[is.na(data_adaptation[,5]),5]<-0
data_adaptation[is.na(data_adaptation[,6]),6]<-0
data_adaptation[is.na(data_adaptation[,7]),7]<-0
data_adaptation[is.na(data_adaptation[,8]),8]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

data_adaptation[data_adaptation[,5]>0&data_adaptation[,5]<0.1,5]<-0
data_adaptation[data_adaptation[,7]>0&data_adaptation[,7]<0.1,7]<-0


summary(data_adaptation)

## assign NA International Agreement as zero
data_adaptation$InvestAgree[is.na(data_adaptation$InvestAgree)]<-0

regdata_adaptation1<-data_adaptation[,c(1:4,5,7,9:18,21:40)]


regdata_adaptation1$ProviderGDPCur<-log(regdata_adaptation1$ProviderGDPCur)
regdata_adaptation1$RecipientGDPCur<-log(regdata_adaptation1$RecipientGDPCur)
regdata_adaptation1$ProviderPop<-regdata_adaptation1$ProviderPop/10^9
regdata_adaptation1$RecipientPop<-regdata_adaptation1$RecipientPop/10^9

regdata_adaptation1$ProviderfisB<-regdata_adaptation1$ProviderfisB/100
regdata_adaptation1$RecipientfisB<-regdata_adaptation1$RecipientfisB/100

regdata_adaptation1$Providerdebt<-regdata_adaptation1$Providerdebt/100
regdata_adaptation1$Recipientdebt<-regdata_adaptation1$Recipientdebt/100

regdata_adaptation1$distw<-log(regdata_adaptation1$distw)

## change the NA for MDB to zero
regdata_adaptation1$MDBDummy[is.na(regdata_adaptation1$MDBDummy)]<-0

summary(regdata_adaptation1)

regdata_adaptationAdapt<-regdata_adaptation1[complete.cases(regdata_adaptation1),]

names(regdata_adaptationAdapt)[5]<-"AdaptAmount"
names(regdata_adaptationAdapt)[6]<-"MitiAmount"

## remove the zero provider countries
regdata_adaptationAdapt$ProviderISO<-as.character(regdata_adaptationAdapt$ProviderISO)
regdata_adaptationAdapt$RecipientISO<-as.character(regdata_adaptationAdapt$RecipientISO)

library(fastDummies)

regdata_adaptationAdaptD<-dummy_cols(regdata_adaptationAdapt, select_columns = c("Sector","RecipientISO","ProviderISO","Year"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regdata_adaptationAdaptD)[44:111]

reg1<-regdata_adaptationAdaptD[,c(5:149)]

reg1$MitiAmount[reg1$MitiAmount!=0]<-log(reg1$MitiAmount[reg1$MitiAmount!=0]*1000)
reg1$AdaptAmount[reg1$AdaptAmount!=0]<-log(reg1$AdaptAmount[reg1$AdaptAmount!=0]*1000)

reg1$MDBAdapt[reg1$MDBAdapt!=0]<-log(reg1$MDBAdapt[reg1$MDBAdapt!=0]*1000)
reg1$MDBMiti[reg1$MDBMiti!=0]<-log(reg1$MDBMiti[reg1$MDBMiti!=0]*1000)
## choose the benchmark sectors mannually
## drop these variables in the regression

reg1$NDCGHG<-0
reg1$Sector_Others<-0
reg1$Year_2016<-0
reg1$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
reg1$RecipientISO_ARM<-0
#reg1$RecipientISO_ERI<-0
#reg1$RecipientISO_WSM<-0
reg1$IncomeGroup<-as.character(reg1$IncomeGroup)
reg1$IncomeGroup[reg1$IncomeGroup=="UMICs"]<-"BaseUM"

names(reg1)[33:39]<-c("Water","Transport","Agri","EnvProtect", "MultiSec","Others","Disaster")


library(mhurdle)
library(texreg)

## hurdle 1

reg1$WRI<-reg1$RecipientWRIExpo*reg1$RecipientWRIVul
reg1$ProviderGDPtot<-log(exp(reg1$ProviderGDPCur)*reg1$ProviderPop)
reg1$RecipientGDPtot<-log(exp(reg1$RecipientGDPCur)*reg1$RecipientPop)

names(reg1)
## remove ProviderISO_LVA 0.0002 
## LTU 0.002
## introduce provider country fixed effect in hurdle one would bring non-identification challenges
## lapack routine dgesv: system is exactly singular: U[43,43] = 0

Stn <- mhurdle(AdaptAmount ~ WRI+CPIAPublicAdm+CPIAbudget+
                 NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+
                 distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                 ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                 ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                 ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                 ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                 ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+
                 ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                 ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE
               | 
                 WRI+CPIAPublicAdm+CPIAbudget+
                 distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                 ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                 ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+
                 RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+
                 RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+
                 RecipientISO_CIV+RecipientISO_CMR+
                 #RecipientISO_COG+
                 RecipientISO_COM+RecipientISO_CPV+
                 RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+
                 RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+
                 RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+
                 RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+
                 RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+
                 RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+
                 RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+
                 RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+
                 RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+
                 RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+
                 RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE+
                 ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                 ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                 ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                 ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+
                 ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                 ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE+
                 Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+
                 Water+Transport+Agri+EnvProtect+MultiSec+Disaster,
               reg1,
               dist = "n", h2 = TRUE, corr = FALSE, method = "Bhhh", print.level = 0,finalHessian = TRUE)

summary(Stn)



Slnd <- update(Stn, corr = TRUE)
coef(summary(Slnd), "corr")

# # Calculate the heteroscedasticity-robust covariance matrix
# 
# library(sandwich)
# library(lmtest)
# 
# robust_vcov <- sandwich(Slnd)
# 
# # Recompute test statistics with robust standard errors
# robust_tests <- coeftest(Slnd, vcov = robust_vcov)
# 
# # Print results
# summary(robust_tests)

# Extract the summaries of both models
summary_Stn <- summary(Stn)
summary_Slnd <- summary(Slnd)

# Convert the coefficients from both summaries into data frames
coef_df_Stn <- as.data.frame(summary_Stn$coefficients)
coef_df_Slnd <- as.data.frame(summary_Slnd$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn <- coef_df_Stn %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd <- coef_df_Slnd %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn$Variable <- rownames(coef_df_Stn)
coef_df_Slnd$Variable <- rownames(coef_df_Slnd)

# Merge both data frames by the 'Variable' column
combined_df <- merge(coef_df_Stn, coef_df_Slnd, by = "Variable", all = TRUE)

# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn)
logLik_Slnd <- logLik(Slnd)
pseudo_R2_Stn <- summary_Stn$r.squared["lratio"]
coef_det_Stn <- summary_Stn$r.squared["coefdet"]
pseudo_R2_Slnd <- summary_Slnd$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd$r.squared["coefdet"]

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)

# Bind additional parameters to the combined coefficient table
combined_df <- bind_rows(combined_df, additional_params)

# Reorder the columns to have 'Variable' first
combined_df <- combined_df %>%
  select(Variable, everything())

# Export the combined results to a CSV file
write.csv(combined_df, "./regressions/adaptation/combined_regression_results.csv", row.names = FALSE)

result1<-texreg(list(Stn, Slnd),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result1

write.table(result1, "Baseline Result for Adaptation")

# Extract the summary of the models
stn_summary <- summary(Stn)$coefficients
slnd_summary <- summary(Slnd)$coefficients

max_rows <- max(nrow(stn_summary), nrow(slnd_summary))


# Ensure both summaries have the same number of rows by filling missing rows with NA
stn_summary <- rbind(stn_summary, matrix(NA, nrow = max_rows - nrow(stn_summary), ncol = ncol(stn_summary)))
slnd_summary <- rbind(slnd_summary, matrix(NA, nrow = max_rows - nrow(slnd_summary), ncol = ncol(slnd_summary)))

# Combine the results into a single data frame
results_df <- data.frame(
  Model = rep(c("log-normal", "Correlated log-normal"), each = max_rows),
  Coefficients = c(rownames(stn_summary), rownames(slnd_summary)),
  Estimate = c(stn_summary[,1], slnd_summary[,1]),
  StdError = c(stn_summary[,2], slnd_summary[,2]),
  zValue = c(stn_summary[,3], slnd_summary[,3]),
  pValue = c(stn_summary[,4], slnd_summary[,4])
)

# Save the data frame to a CSV file
write.csv(results_df, "model_results.csv", row.names = FALSE)

##### Adaptation regression with rio markers #####

Rio_data_adaptation <-left_join(data_adaptation, Rio_data_adaptation, by= c("Year", "ProviderISO", "RecipientISO"))

Rio_data_adaptation <- Rio_data_adaptation%>%
  select(-c(Sector)) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy))%>%
  ungroup()%>%
  unique()

Rio_data_adaptation$NDC15[Rio_data_adaptation$NDC15%in%1 & !Rio_data_adaptation$Year%in%2015]<-0
Rio_data_adaptation$NDC15[Rio_data_adaptation$NDC16%in%1 & !Rio_data_adaptation$Year%in%2016]<-0
Rio_data_adaptation$NDC15[Rio_data_adaptation$NDC17%in%1 & !Rio_data_adaptation$Year%in%2017]<-0
Rio_data_adaptation$NDC15[Rio_data_adaptation$NDC18%in%1 & !Rio_data_adaptation$Year%in%2018]<-0

## In Rio_data_adaptation the amount of Adapt and Mitigation are defined as NA
## after constructing the Rio_data_adaptationset, change the adaptation and mitigation to zeros for NA cells

Rio_data_adaptation[is.na(Rio_data_adaptation[,42]),42]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

Rio_data_adaptation[Rio_data_adaptation[,42]>0&Rio_data_adaptation[,42]<0.1,42]<-0


summary(Rio_data_adaptation)

## assign NA International Agreement as zero
Rio_data_adaptation$InvestAgree[is.na(Rio_data_adaptation$InvestAgree)]<-0

regRio_data_adaptation1<-Rio_data_adaptation[,c(1:3,8:17,20:39,42)]


regRio_data_adaptation1$ProviderGDPCur<-log(regRio_data_adaptation1$ProviderGDPCur)
regRio_data_adaptation1$RecipientGDPCur<-log(regRio_data_adaptation1$RecipientGDPCur)
regRio_data_adaptation1$ProviderPop<-regRio_data_adaptation1$ProviderPop/10^9
regRio_data_adaptation1$RecipientPop<-regRio_data_adaptation1$RecipientPop/10^9

regRio_data_adaptation1$ProviderfisB<-regRio_data_adaptation1$ProviderfisB/100
regRio_data_adaptation1$RecipientfisB<-regRio_data_adaptation1$RecipientfisB/100

regRio_data_adaptation1$Providerdebt<-regRio_data_adaptation1$Providerdebt/100
regRio_data_adaptation1$Recipientdebt<-regRio_data_adaptation1$Recipientdebt/100

regRio_data_adaptation1$distw<-log(regRio_data_adaptation1$distw)

## change the NA for MDB to zero
regRio_data_adaptation1$MDBDummy[is.na(regRio_data_adaptation1$MDBDummy)]<-0

summary(regRio_data_adaptation1)

regRio_data_adaptationAdapt<-regRio_data_adaptation1[complete.cases(regRio_data_adaptation1),]

names(regRio_data_adaptationAdapt)[34]<-"AdaptAmount"

## remove the zero provider countries
regRio_data_adaptationAdapt$ProviderISO<-as.character(regRio_data_adaptationAdapt$ProviderISO)
regRio_data_adaptationAdapt$RecipientISO<-as.character(regRio_data_adaptationAdapt$RecipientISO)

library(fastDummies)

regRio_data_adaptationAdaptD<-dummy_cols(regRio_data_adaptationAdapt, select_columns = c("RecipientISO","ProviderISO","Year"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regRio_data_adaptationAdaptD)[35:102]

reg2<-regRio_data_adaptationAdaptD[,c(3:140)]

reg2$AdaptAmount[reg2$AdaptAmount!=0]<-log(reg2$AdaptAmount[reg2$AdaptAmount!=0]*1000)
reg2$AdaptAmount[is.na(reg2$AdaptAmount)]<-0

reg2$MDBAdapt[reg2$MDBAdapt!=0]<-log(reg2$MDBAdapt[reg2$MDBAdapt!=0]*1000)

## choose the benchmark sectors mannually
## drop these variables in the regression

reg2$NDCGHG<-0
reg2$Year_2016<-0
reg2$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
reg2$RecipientISO_ARM<-0
#reg2$RecipientISO_ERI<-0
#reg2$RecipientISO_WSM<-0
reg2$IncomeGroup<-as.character(reg2$IncomeGroup)
reg2$IncomeGroup[reg2$IncomeGroup=="UMICs"]<-"BaseUM"


library(mhurdle)
library(texreg)

## hurdle 1

reg2$WRI<-reg2$RecipientWRIExpo*reg2$RecipientWRIVul
reg2$ProviderGDPtot<-log(exp(reg2$ProviderGDPCur)*reg2$ProviderPop)
reg2$RecipientGDPtot<-log(exp(reg2$RecipientGDPCur)*reg2$RecipientPop)

names(reg2)
## remove ProviderISO_LVA 0.0002 
## LTU 0.002
## introduce provider country fixed effect in hurdle one would bring non-identification challenges
## lapack routine dgesv: system is exactly singular: U[43,43] = 0

Stn2 <- mhurdle(AdaptAmount ~ WRI+CPIAPublicAdm+CPIAbudget+
                  NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+
                  distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                  ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                  ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                  ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                  ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                  ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+
                  ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                  ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE
                | 
                  WRI+CPIAPublicAdm+CPIAbudget+
                  distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                  ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                  ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+
                  RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+
                  RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+
                  RecipientISO_CIV+RecipientISO_CMR+
                  #RecipientISO_COG+
                  RecipientISO_COM+RecipientISO_CPV+
                  RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+
                  RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+
                  RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+
                  RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+
                  RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+
                  RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+
                  RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+
                  RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+
                  RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+
                  RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+
                  RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE+
                  ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                  ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                  ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                  ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+
                  ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                  ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE+
                  Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018
                ,
                reg2,
                dist = "n", h2 = TRUE, corr = FALSE, method = "Bhhh", print.level = 0,finalHessian = TRUE)

summary(Stn2)

Slnd2 <- update(Stn2, corr = TRUE)
coef(summary(Slnd2), "corr")

# Extract the summaries of both models
summary_Stn2 <- summary(Stn2)
summary_Slnd2 <- summary(Slnd2)

# Convert the coefficients from both summaries into data frames
coef_df_Stn2 <- as.data.frame(summary_Stn2$coefficients)
coef_df_Slnd2 <- as.data.frame(summary_Slnd2$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn2 <- coef_df_Stn2 %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd2 <- coef_df_Slnd2 %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn2$Variable <- rownames(coef_df_Stn2)
coef_df_Slnd2$Variable <- rownames(coef_df_Slnd2)

# Merge both data frames by the 'Variable' column
combined_df2 <- merge(coef_df_Stn2, coef_df_Slnd2, by = "Variable", all = TRUE)

# Extract additional parameters and add them as rows
# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn2)
logLik_Slnd <- logLik(Slnd2)
pseudo_R2_Stn <- summary_Stn2$r.squared["lratio"]
coef_det_Stn <- summary_Stn2$r.squared["coefdet"]
pseudo_R2_Slnd <- summary_Slnd2$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd2$r.squared["coefdet"]

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)
# Bind additional parameters to the combined coefficient table
combined_df2 <- bind_rows(combined_df2, additional_params)

# Reorder the columns to have 'Variable' first
combined_df2 <- combined_df2 %>%
  select(Variable, everything())

# Export the combined results to a CSV file
write.csv(combined_df2, "./regressions/adaptation/combined_regression_results2.csv", row.names = FALSE)


result2<-texreg(list(Stn2, Slnd2),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result2

write.table(result2, "Rio Result for Adaptation")


##### Adaptation regression with ClimateFinanceBERT #####


# Create a dataframe with unique combinations of Year, ProviderISO, and RecipientISO
climate_adaptation <- data_adaptation %>%
  select(-c("Sector", 
            "Adaptation.related.development.finance...Commitment...Current.USD.thousand",
            "Adaptation.related.development.finance...Commitment...2018.USD.thousand",
            "Mitigation.related.development.finance...Commitment...Current.USD.thousand",
            "Mitigation.related.development.finance...Commitment...2018.USD.thousand")) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy))%>%
  ungroup()%>%
  unique() %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(climate_class = "Climate Adaptation")

resilience <- climate_adaptation %>%
  mutate(climate_class = "Resilience")

# Combine the two datasets
unique_combinations <- bind_rows(climate_adaptation, resilience)%>%
  unique()

# mutate(climate_class = ifelse(row_number() == 1, "Climate Adaptation", "Resilience"))

BERT_data_adaptation <-left_join(unique_combinations, ClimateBERT_Adaptation%>%
                                   group_by(Year,ProviderISO,RecipientISO, climate_class)%>%
                                   summarise(ClimateBERT_Adapt_Commitment=sum(ClimateBERT_Adapt_Commitment,na.rm=T)), by= c("Year", "ProviderISO", "RecipientISO","climate_class"))

BERT_data_adaptation <- BERT_data_adaptation %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  filter(!(all(is.na(ClimateBERT_Adapt_Commitment))))


BERT_data_adaptation$NDC15[BERT_data_adaptation$NDC15%in%1 & !BERT_data_adaptation$Year%in%2015]<-0
BERT_data_adaptation$NDC15[BERT_data_adaptation$NDC16%in%1 & !BERT_data_adaptation$Year%in%2016]<-0
BERT_data_adaptation$NDC15[BERT_data_adaptation$NDC17%in%1 & !BERT_data_adaptation$Year%in%2017]<-0
BERT_data_adaptation$NDC15[BERT_data_adaptation$NDC18%in%1 & !BERT_data_adaptation$Year%in%2018]<-0

## In BERT_data_adaptation the amount of Adapt and Mitigation are defined as NA
## after constructing the Rio_data_adaptationset, change the adaptation and mitigation to zeros for NA cells

BERT_data_adaptation[is.na(BERT_data_adaptation[,37]),37]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

BERT_data_adaptation[BERT_data_adaptation[,37]>0&BERT_data_adaptation[,37]<0.1,37]<-0


summary(BERT_data_adaptation)

## assign NA International Agreement as zero
BERT_data_adaptation$InvestAgree[is.na(BERT_data_adaptation$InvestAgree)]<-0

regBERT_data_adaptation1<-BERT_data_adaptation[,c(1:13,16:37)]


regBERT_data_adaptation1$ProviderGDPCur<-log(regBERT_data_adaptation1$ProviderGDPCur)
regBERT_data_adaptation1$RecipientGDPCur<-log(regBERT_data_adaptation1$RecipientGDPCur)
regBERT_data_adaptation1$ProviderPop<-regBERT_data_adaptation1$ProviderPop/10^9
regBERT_data_adaptation1$RecipientPop<-regBERT_data_adaptation1$RecipientPop/10^9

regBERT_data_adaptation1$ProviderfisB<-regBERT_data_adaptation1$ProviderfisB/100
regBERT_data_adaptation1$RecipientfisB<-regBERT_data_adaptation1$RecipientfisB/100

regBERT_data_adaptation1$Providerdebt<-regBERT_data_adaptation1$Providerdebt/100
regBERT_data_adaptation1$Recipientdebt<-regBERT_data_adaptation1$Recipientdebt/100

regBERT_data_adaptation1$distw<-log(regBERT_data_adaptation1$distw)

## change the NA for MDB to zero
regBERT_data_adaptation1$MDBDummy[is.na(regBERT_data_adaptation1$MDBDummy)]<-0

summary(regBERT_data_adaptation1)

regBERT_data_adaptatioAdapt<-regBERT_data_adaptation1[complete.cases(regBERT_data_adaptation1),]

names(regBERT_data_adaptatioAdapt)[35]<-"AdaptAmount"

## remove the zero provider countries
regBERT_data_adaptatioAdapt$ProviderISO<-as.character(regBERT_data_adaptatioAdapt$ProviderISO)
regBERT_data_adaptatioAdapt$RecipientISO<-as.character(regBERT_data_adaptatioAdapt$RecipientISO)

library(fastDummies)

regBERT_data_adaptatioAdaptD<-dummy_cols(regBERT_data_adaptatioAdapt, select_columns = c( "RecipientISO","ProviderISO","Year","climate_class"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regBERT_data_adaptatioAdaptD)[36:125]

reg3<-regBERT_data_adaptatioAdaptD[,c(4:135)]

reg3$AdaptAmount[reg3$AdaptAmount!=0]<-log(reg3$AdaptAmount[reg3$AdaptAmount!=0]*1000)
reg3$AdaptAmount[is.na(reg3$AdaptAmount)]<-0

reg3$MDBAdapt[reg3$MDBAdapt!=0]<-log(reg3$MDBAdapt[reg3$MDBAdapt!=0]*1000)

## choose the benchmark sectors mannually

## drop these variables in the regression

reg3$NDCGHG<-0
reg3$climate_class_Resilience<-0
reg3$Year_2016<-0
reg3$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
# reg3$RecipientISO_ARM<-0
#reg1$RecipientISO_ERI<-0
#reg1$RecipientISO_WSM<-0
reg3$IncomeGroup<-as.character(reg3$IncomeGroup)
reg3$IncomeGroup[reg3$IncomeGroup=="UMICs"]<-"BaseUM"

names(reg3)[131]<-c("climate_class_Climate_Adaptation")


library(mhurdle)
library(texreg)

## hurdle 1

reg3$WRI<-reg3$RecipientWRIExpo*reg3$RecipientWRIVul
reg3$ProviderGDPtot<-log(exp(reg3$ProviderGDPCur)*reg3$ProviderPop)
reg3$RecipientGDPtot<-log(exp(reg3$RecipientGDPCur)*reg3$RecipientPop)


names(reg3)

# Step 1: Gather all ProviderISO_* and RecipientISO_* columns in a tidy format
# iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg3), value = TRUE)

iso_vars <- grep("ProviderISO_", names(reg3), value = TRUE)

# Create a long format dataset to calculate proportions for each ISO variable
iso_df <- reg3 %>%
  select(all_of(iso_vars)) %>%      # Select only the ProviderISO_* and RecipientISO_* columns
  mutate(obs_id = row_number()) %>% # Add a row identifier to keep track of original rows
  pivot_longer(cols = all_of(iso_vars), 
               names_to = "country_iso", 
               values_to = "presence", 
               values_drop_na = TRUE) %>%  # Convert to long format with country ISO as the variable
  filter(presence == 1)             # Keep only rows where the country ISO variable is 1 (present)

proportions_df <- iso_df %>%
  group_by(country_iso) %>%
  summarize(prop_obs = n() / nrow(reg3))  # Calculate proportion of total observations


# Step 3: Identify countries below the 0.005 threshold
excluded_countries <- proportions_df %>%
  filter(prop_obs <= 0.005) %>%      # Countries to exclude
  pull(country_iso)

# Print the excluded countries
cat("Excluded countries (proportion < 0.005):\n")
print(excluded_countries)

# Step 4: Identify countries with proportion >= 0.005 to keep in the model
remaining_countries <- proportions_df %>%
  filter(prop_obs > 0.005) %>%     # Countries to keep in the model
  pull(country_iso)


# Step 5: Create the new model formula excluding the low-representation countries
base_formula <- " AdaptAmount ~ WRI+CPIAPublicAdm+CPIAbudget+NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop"

# Only keep the remaining countries in the formula
country_effects <- paste(remaining_countries, collapse = " + ")

# Create the final model formula conditionally
final_formula <- if (nzchar(country_effects)) {
  paste(base_formula, country_effects, sep = " + ")
} else {
  base_formula
}

iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg3), value = TRUE)

# Create a long format dataset to calculate proportions for each ISO variable
iso_df <- reg3 %>%
  select(all_of(iso_vars)) %>%      # Select only the ProviderISO_* and RecipientISO_* columns
  mutate(obs_id = row_number()) %>% # Add a row identifier to keep track of original rows
  pivot_longer(cols = all_of(iso_vars), 
               names_to = "country_iso", 
               values_to = "presence", 
               values_drop_na = TRUE) %>%  # Convert to long format with country ISO as the variable
  filter(presence == 1)             # Keep only rows where the country ISO variable is 1 (present)

proportions_df <- iso_df %>%
  group_by(country_iso) %>%
  summarize(prop_obs = n() / nrow(reg3))  # Calculate proportion of total observations


# Step 3: Identify countries below the 0.005 threshold
excluded_countries <- proportions_df %>%
  filter(prop_obs <= 0.005) %>%      # Countries to exclude
  pull(country_iso)

# Print the excluded countries
cat("Excluded countries (proportion < 0.005):\n")
print(excluded_countries)

# Step 4: Identify countries with proportion >= 0.005 to keep in the model
remaining_countries <- proportions_df %>%
  filter(prop_obs > 0.005) %>%     # Countries to keep in the model
  pull(country_iso)

# Only keep the remaining countries in the formula
country_effects <- paste(remaining_countries, collapse = " + ")


# Second part of the hurdle formula
second_formula <- "| WRI+CPIAPublicAdm+CPIAbudget+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+climate_class_Climate_Adaptation"


# Create the final model formula conditionally
complete_second <- if (nzchar(country_effects)) {
  paste(second_formula, country_effects, sep = " + ")
} else {
  second_formula
}

# Combine both the final and second parts of the formula
complete_formula <- paste(final_formula, complete_second)
complete_formula
# Step 6: Rebuild and run the model
Stn3 <- mhurdle(as.formula(complete_formula),
                data = reg3, 
                dist = "n", 
                h2 = TRUE, 
                corr = FALSE, 
                method = "Bhhh", 
                print.level = 0, 
                finalHessian = TRUE)

summary(Stn3)


Slnd3 <- update(Stn3, corr = TRUE)
coef(summary(Slnd3), "corr")

# Extract the summaries of both models
summary_Stn3 <- summary(Stn3)
summary_Slnd3 <- summary(Slnd3)

# Convert the coefficients from both summaries into data frames
coef_df_Stn3 <- as.data.frame(summary_Stn3$coefficients)
coef_df_Slnd3 <- as.data.frame(summary_Slnd3$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn3 <- coef_df_Stn3 %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd3 <- coef_df_Slnd3 %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn3$Variable <- rownames(coef_df_Stn3)
coef_df_Slnd3$Variable <- rownames(coef_df_Slnd3)

# Merge both data frames by the 'Variable' column
combined_df3 <- merge(coef_df_Stn3, coef_df_Slnd3, by = "Variable", all = TRUE)

# Extract additional parameters and add them as rows
# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn3)
logLik_Slnd <- logLik(Slnd3)
pseudo_R2_Stn <- summary_Stn3$r.squared["lratio"]
coef_det_Stn <- summary_Stn3$r.squared["coefdet"]
pseudo_R2_Slnd <- summary_Slnd3$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd3$r.squared["coefdet"]

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)

# Bind additional parameters to the combined coefficient table
combined_df3 <- bind_rows(combined_df3, additional_params)

# Reorder the columns to have 'Variable' first
combined_df3 <- combined_df3 %>%
  select(Variable, everything())

# Export the combined results to a CSV file
write.csv(combined_df3, "./regressions/adaptation/combined_regression_results3.csv", row.names = FALSE)

result3<-texreg(list(Stn3, Slnd3),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result3 

write.table(result3, "ClimateFinanceBERT Result for Adaptation")

# Save reg1 as a CSV file
write.csv(reg1, "./Redaction/reg1.csv", row.names = FALSE)

# Save reg2 as a CSV file
write.csv(reg2, "./Redaction/reg2.csv", row.names = FALSE)

# Save reg3 as a CSV file
write.csv(reg3, "./Redaction/reg3.csv", row.names = FALSE)

compare_datasets <- function(regdata, regRio_data, regBERT_data) {
  # Helper function to safely get unique provider values
  safe_unique_provider <- function(x) {
    if (is.null(x)) return(character(0))
    if (is.data.frame(x)) {
      if ("ProviderISO" %in% names(x)) return(unique(x$ProviderISO))
    }
    return(character(0))
  }
  
  # Helper function to safely get unique recipient values
  safe_unique_recipient <- function(x) {
    if (is.null(x)) return(character(0))
    if (is.data.frame(x)) {
      if ("RecipientISO" %in% names(x)) return(unique(x$RecipientISO))
    }
    return(character(0))
  }
  
  # Safely compute intersections and differences for providers
  provider_in_data_and_rio_not_bert <- intersect(
    safe_unique_provider(regdata),
    safe_unique_provider(regRio_data)
  ) %>%
    setdiff(safe_unique_provider(regBERT_data))
  
  # Safely compute intersections and differences for recipients
  recipient_in_data_and_rio_not_bert <- intersect(
    safe_unique_recipient(regdata),
    safe_unique_recipient(regRio_data)
  ) %>%
    setdiff(safe_unique_recipient(regBERT_data))
  
  # Create summary data
  summary_data <- tibble(
    Attribute = c(
      "Number of Observations",
      "Number of Sectors",
      "Number of Provider Countries",
      "Number of Recipient Countries",
      "Provider Countries in Data & Rio but not in BERT",
      "Recipient Countries in Data & Rio but not in BERT"
    ),
    
    regdata_adaptationAdapt = c(
      if (is.data.frame(regdata)) nrow(regdata) else 0,
      if (is.data.frame(regdata) && "Sector" %in% names(regdata)) 
        length(unique(regdata$Sector)) else 0,
      length(safe_unique_provider(regdata)),
      length(safe_unique_recipient(regdata)),
      "-",  # Moved to BERT column
      "-"   # Moved to BERT column
    ),
    
    regRio_data_adaptationAdapt = c(
      if (is.data.frame(regRio_data)) nrow(regRio_data) else 0,
      1,  # No sector information for regRio_data
      length(safe_unique_provider(regRio_data)),
      length(safe_unique_recipient(regRio_data)),
      "-",  # Moved to BERT column
      "-"   # Moved to BERT column
    ),
    
    regBERT_data_adaptatioAdapt = c(
      if (is.data.frame(regBERT_data)) nrow(regBERT_data) else 0,
      if (is.data.frame(regBERT_data) && "climate_class" %in% names(regBERT_data))
        length(unique(regBERT_data$climate_class)) else 0,
      length(safe_unique_provider(regBERT_data)),
      length(safe_unique_recipient(regBERT_data)),
      ifelse(length(provider_in_data_and_rio_not_bert) > 0,
             paste(provider_in_data_and_rio_not_bert, collapse = ", "),
             "None"),
      ifelse(length(recipient_in_data_and_rio_not_bert) > 0,
             paste(recipient_in_data_and_rio_not_bert, collapse = ", "),
             "None")
    )
  )
  
  return(summary_data)
}

summary_result_adap <- compare_datasets(
  regdata_adaptationAdapt,
  regRio_data_adaptationAdapt,
  regBERT_data_adaptatioAdapt
)

# Select only the AdaptAmount column and add a Source identifier
df_reg1 <- reg1 %>% select(AdaptAmount) %>% mutate(Source = "Reg1")
df_reg2 <- reg2 %>% select(AdaptAmount) %>% mutate(Source = "Reg2")
df_reg3 <- reg3 %>% select(AdaptAmount) %>% mutate(Source = "Reg3")

# Combine the data frames
combined_df <- bind_rows(df_reg1, df_reg2, df_reg3)

# Plot the distribution
ggplot(combined_df, aes(x = AdaptAmount, fill = Source)) +
  geom_density(alpha = 0.5) +  # Density plot with transparency
  labs(title = "Distribution of AdaptAmount",
       x = "AdaptAmount",
       y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("Reg1" = "#1f77b4", "Reg2" = "#ff7f0e", "Reg3" = "#2ca02c")) +
  theme(legend.title = element_blank())

####### Mitigation #######

##### Preparing Data for Mitigation #####

#### Rio Data ####

Rio_data_Mitigation <- Rio_Data%>%
  filter(ClimateAdaptation %in% (1:2))%>%
  group_by(Year, DonorName, RecipientName)%>%
  summarise(Rio_Miti_Commitment= sum(USD_Commitment_Defl*1000, na.rm = T))

Rio_data_Mitigation$ProviderISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$DonorName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation$RecipientISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$RecipientName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation[Rio_data_Mitigation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"

#### CLimateFinanceBERT ####

#### CLimateFinanceBERT Data ####

ClimateBERT <- fread("Data/climate_finance_total.csv")

ClimateBERT_Mitigation <- ClimateBERT%>%
  filter(meta_category == "Mitigation")%>%
  group_by(Year, DonorName, RecipientName, climate_class)%>%
  summarise(ClimateBERT_miti_Commitment= sum(USD_Commitment_Defl*1000, na.rm = T))

ClimateBERT_Mitigation$ProviderISO <- countrycode::countrycode(sourcevar = ClimateBERT_Mitigation$DonorName, origin = "country.name", destination = "iso3c")
ClimateBERT_Mitigation$RecipientISO <- countrycode::countrycode(sourcevar = ClimateBERT_Mitigation$RecipientName, origin = "country.name", destination = "iso3c")
ClimateBERT_Mitigation[ClimateBERT_Mitigation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"
ClimateBERT_Mitigation[ClimateBERT_Mitigation$RecipientName %in% "TÃ¼rkiye", "Country_Code"] <- "TUR"



#### Huei Data ####

data_mitigation<-read.csv("Data/Mitigation with gravity vars amended FULL Feb 14 2023.csv")



data_mitigation$NDC15[data_mitigation$NDC15%in%1 & !data_mitigation$Year%in%2015]<-0
data_mitigation$NDC15[data_mitigation$NDC16%in%1 & !data_mitigation$Year%in%2016]<-0
data_mitigation$NDC15[data_mitigation$NDC17%in%1 & !data_mitigation$Year%in%2017]<-0
data_mitigation$NDC15[data_mitigation$NDC18%in%1 & !data_mitigation$Year%in%2018]<-0

## In data the amount of Adapt and Mitigation are defined as NA
## after constructing the dataset, change the adaptation and mitigation to zeros for NA cells

data_mitigation[is.na(data_mitigation[,5]),5]<-0
data_mitigation[is.na(data_mitigation[,6]),6]<-0
data_mitigation[is.na(data_mitigation[,7]),7]<-0
data_mitigation[is.na(data_mitigation[,8]),8]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

data_mitigation[data_mitigation[,5]>0&data_mitigation[,5]<0.1,5]<-0
data_mitigation[data_mitigation[,7]>0&data_mitigation[,7]<0.1,7]<-0

## assign NA International Agreement as zero
data_mitigation$InvestAgree[is.na(data_mitigation$InvestAgree)]<-0

regData2<-data_mitigation[,c(1:4,5,7,9:18,21:40)]

regData2$ProviderGDPCur<-log(regData2$ProviderGDPCur)
regData2$RecipientGDPCur<-log(regData2$RecipientGDPCur)
regData2$ProviderPop<-regData2$ProviderPop/10^9
regData2$RecipientPop<-regData2$RecipientPop/10^9

regData2$ProviderfisB<-regData2$ProviderfisB/100
regData2$RecipientfisB<-regData2$RecipientfisB/100

regData2$Providerdebt<-regData2$Providerdebt/100
regData2$Recipientdebt<-regData2$Recipientdebt/100

regData2$distw<-log(regData2$distw)

## change the NA for MDB to zero
regData2$MDBDummy[is.na(regData2$MDBDummy)]<-0

regDataMiti<-regData2[complete.cases(regData2),]

names(regDataMiti)[5]<-"AdaptAmount"
names(regDataMiti)[6]<-"MitiAmount"

## remove the zero provider countries
regDataMiti$ProviderISO<-as.character(regDataMiti$ProviderISO)


library(fastDummies)

regDataMitiD<-dummy_cols(regDataMiti, select_columns = c("Sector","RecipientISO","ProviderISO","Year"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regDataMitiD)[44:111]


reg4<-regDataMitiD[,c(5:149)]

reg4$MitiAmount[reg4$MitiAmount!=0]<-log(reg4$MitiAmount[reg4$MitiAmount!=0]*1000)
reg4$AdaptAmount[reg4$AdaptAmount!=0]<-log(reg4$AdaptAmount[reg4$AdaptAmount!=0]*1000)

reg4$MDBAdapt[reg4$MDBAdapt!=0]<-log(reg4$MDBAdapt[reg4$MDBAdapt!=0]*1000)
reg4$MDBMiti[reg4$MDBMiti!=0]<-log(reg4$MDBMiti[reg4$MDBMiti!=0]*1000)
## choose the benchmark sectors mannually
## drop these variables in the regression

reg4$NDCGHG<-0
reg4$Sector_Others<-0
reg4$Year_2016<-0
reg4$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
reg4$RecipientISO_ARM<-0
reg4$IncomeGroup<-as.character(reg4$IncomeGroup)
reg4$IncomeGroup[reg4$IncomeGroup=="UMICs"]<-"BaseUM"

names(reg4)[33:39]<-c("Water","Transport","Energy","Agri","EnvProtect", "MultiSec","Others")


library(mhurdle)
library(texreg)

## hurdle 1

reg4$WRI<-reg4$RecipientWRIExpo*reg4$RecipientWRIVul
reg4$ProviderGDPtot<-log(exp(reg4$ProviderGDPCur)*reg4$ProviderPop)
reg4$RecipientGDPtot<-log(exp(reg4$RecipientGDPCur)*reg4$RecipientPop)

names(reg4)

Stn1 <- mhurdle(MitiAmount ~ WRI+CPIAPublicAdm+CPIAbudget+
                 NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+
                 distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                 ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                 ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                 ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                 ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                 ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+
                 ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                 ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE
               | 
                 WRI+CPIAPublicAdm+CPIAbudget+
                 distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                 ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                 ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+
                 RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+
                 RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+
                 RecipientISO_CIV+RecipientISO_CMR+
                 #RecipientISO_COG+
                 RecipientISO_COM+RecipientISO_CPV+
                 RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+
                 RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+
                 RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+
                 RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+
                 RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+
                 RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+
                 RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+
                 RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+
                 RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+
                 RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+
                 RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE+
                 ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                 ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                 ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                 ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+
                 ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                 ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE+
                 Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+
                 Water+Transport+Agri+EnvProtect+MultiSec+Energy,
               reg4,
               dist = "n", h2 = TRUE, corr = FALSE, method = "Bhhh", print.level = 0,finalHessian = TRUE)
summary(Stn1)




Slnd1 <- update(Stn1, corr = TRUE)
coef(summary(Slnd1), "corr")

result2<-texreg(list(Stn1, Slnd1),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result2

write.table(result2, "Baseline Result for  Mitigation 103950 obs")

# Extract the summaries of both models
summary_Stn <- summary(Stn1)
summary_Slnd <- summary(Slnd1)

# Convert the coefficients from both summaries into data frames
coef_df_Stn <- as.data.frame(summary_Stn$coefficients)
coef_df_Slnd <- as.data.frame(summary_Slnd$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn <- coef_df_Stn %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd <- coef_df_Slnd %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn$Variable <- rownames(coef_df_Stn)
coef_df_Slnd$Variable <- rownames(coef_df_Slnd)

# Merge both data frames by the 'Variable' column
combined_df <- merge(coef_df_Stn, coef_df_Slnd, by = "Variable", all = TRUE)

# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn1)
logLik_Slnd <- logLik(Slnd1)
pseudo_R2_Stn <- summary_Stn$r.squared["lratio"]
coef_det_Stn <- summary_Stn$r.squared["coefdet"]
pseudo_R2_Slnd <- summary_Slnd$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd$r.squared["coefdet"]

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)

# Bind additional parameters to the combined coefficient table
combined_df <- bind_rows(combined_df, additional_params)

# Reorder the columns to have 'Variable' first
combined_df <- combined_df %>%
  select(Variable, everything())

# Export the combined results to a CSV file
write.csv(combined_df, "./regressions/combined_regression_results.csv", row.names = FALSE)

##### Adaptation regression with rio markers #####

Rio_data_Mitigation <- Rio_Data%>%
  filter(ClimateMitigation %in% (1:2))%>%
  group_by(Year, DonorName, RecipientName)%>%
  summarise(Rio_Adapt_Commitment= sum(USD_Commitment_Defl*1000, na.rm = T))

Rio_data_Mitigation$ProviderISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$DonorName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation$RecipientISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$RecipientName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation[Rio_data_Mitigation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"



Rio_data_Mitigation <-left_join(data_mitigation, Rio_data_Mitigation, by= c("Year", "ProviderISO", "RecipientISO"))

Rio_data_Mitigation <- Rio_data_Mitigation%>%
  select(-c(Sector)) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy))%>%
  ungroup()%>%
  unique()

Rio_data_Mitigation$NDC15[Rio_data_Mitigation$NDC15%in%1 & !Rio_data_Mitigation$Year%in%2015]<-0
Rio_data_Mitigation$NDC15[Rio_data_Mitigation$NDC16%in%1 & !Rio_data_Mitigation$Year%in%2016]<-0
Rio_data_Mitigation$NDC15[Rio_data_Mitigation$NDC17%in%1 & !Rio_data_Mitigation$Year%in%2017]<-0
Rio_data_Mitigation$NDC15[Rio_data_Mitigation$NDC18%in%1 & !Rio_data_Mitigation$Year%in%2018]<-0

## In Rio_data_Mitigation the amount of Adapt and Mitigation are defined as NA
## after constructing the Rio_data_adaptationset, change the adaptation and mitigation to zeros for NA cells

Rio_data_Mitigation[is.na(Rio_data_Mitigation[,42]),42]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

Rio_data_Mitigation[Rio_data_Mitigation[,42]>0&Rio_data_Mitigation[,42]<0.1,42]<-0


summary(Rio_data_Mitigation)

## assign NA International Agreement as zero
Rio_data_Mitigation$InvestAgree[is.na(Rio_data_Mitigation$InvestAgree)]<-0

regRio_data_Mitigation1<-Rio_data_Mitigation[,c(1:3,8:17,20:39,42)]


regRio_data_Mitigation1$ProviderGDPCur<-log(regRio_data_Mitigation1$ProviderGDPCur)
regRio_data_Mitigation1$RecipientGDPCur<-log(regRio_data_Mitigation1$RecipientGDPCur)
regRio_data_Mitigation1$ProviderPop<-regRio_data_Mitigation1$ProviderPop/10^9
regRio_data_Mitigation1$RecipientPop<-regRio_data_Mitigation1$RecipientPop/10^9

regRio_data_Mitigation1$ProviderfisB<-regRio_data_Mitigation1$ProviderfisB/100
regRio_data_Mitigation1$RecipientfisB<-regRio_data_Mitigation1$RecipientfisB/100

regRio_data_Mitigation1$Providerdebt<-regRio_data_Mitigation1$Providerdebt/100
regRio_data_Mitigation1$Recipientdebt<-regRio_data_Mitigation1$Recipientdebt/100

regRio_data_Mitigation1$distw<-log(regRio_data_Mitigation1$distw)

## change the NA for MDB to zero
regRio_data_Mitigation1$MDBDummy[is.na(regRio_data_Mitigation1$MDBDummy)]<-0

summary(regRio_data_Mitigation1)

regRio_data_Mitigation<-regRio_data_Mitigation1[complete.cases(regRio_data_Mitigation1),]

names(regRio_data_Mitigation)[34]<-"MitiAmount"

## remove the zero provider countries
regRio_data_Mitigation$ProviderISO<-as.character(regRio_data_Mitigation$ProviderISO)
regRio_data_Mitigation$RecipientISO<-as.character(regRio_data_Mitigation$RecipientISO)

library(fastDummies)

regRio_data_MitigationD<-dummy_cols(regRio_data_Mitigation, select_columns = c("RecipientISO","ProviderISO","Year"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regRio_data_MitigationD)[35:102]

reg5<-regRio_data_MitigationD[,c(3:140)]

reg5$MitiAmount[reg5$MitiAmount!=0]<-log(reg5$MitiAmount[reg5$MitiAmount!=0]*1000)
reg5$MitiAmount[is.na(reg5$MitiAmount)]<-0

reg5$MDBAdapt[reg5$MDBAdapt!=0]<-log(reg5$MDBAdapt[reg5$MDBAdapt!=0]*1000)

## choose the benchmark sectors mannually
## drop these variables in the regression

reg5$NDCGHG<-0
reg5$Year_2016<-0
reg5$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
reg5$RecipientISO_ARM<-0
#reg2$RecipientISO_ERI<-0
#reg2$RecipientISO_WSM<-0
reg5$IncomeGroup<-as.character(reg5$IncomeGroup)
reg5$IncomeGroup[reg5$IncomeGroup=="UMICs"]<-"BaseUM"


library(mhurdle)
library(texreg)

## hurdle 1

reg5$WRI<-reg5$RecipientWRIExpo*reg5$RecipientWRIVul
reg5$ProviderGDPtot<-log(exp(reg5$ProviderGDPCur)*reg5$ProviderPop)
reg5$RecipientGDPtot<-log(exp(reg5$RecipientGDPCur)*reg5$RecipientPop)

names(reg5)
## remove ProviderISO_LVA 0.0002 
## LTU 0.002
## introduce provider country fixed effect in hurdle one would bring non-identification challenges
## lapack routine dgesv: system is exactly singular: U[43,43] = 0

Stn2 <- mhurdle(MitiAmount ~ WRI+CPIAPublicAdm+CPIAbudget+
                  NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+
                  distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                  ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                  ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                  ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                  ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                  ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+
                  ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                  ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE
                | 
                  WRI+CPIAPublicAdm+CPIAbudget+
                  distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+
                  ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+
                  ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+
                  RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+
                  RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+
                  RecipientISO_CIV+RecipientISO_CMR+
                  #RecipientISO_COG+
                  RecipientISO_COM+RecipientISO_CPV+
                  RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+
                  RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+
                  RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+
                  RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+
                  RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+
                  RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+
                  RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+
                  RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+
                  RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+
                  RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+
                  RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE+
                  ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+
                  ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+
                  ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+
                  ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+
                  ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+
                  ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE+
                  Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018
                ,
                reg5,
                dist = "n", h2 = TRUE, corr = FALSE, method = "Bhhh", print.level = 0,finalHessian = TRUE)

summary(Stn2)

Slnd2 <- update(Stn2, corr = TRUE)
coef(summary(Slnd2), "corr")

# Extract the summaries of both models
summary_Stn2 <- summary(Stn2)
summary_Slnd2 <- summary(Slnd2)

# Convert the coefficients from both summaries into data frames
coef_df_Stn2 <- as.data.frame(summary_Stn2$coefficients)
coef_df_Slnd2 <- as.data.frame(summary_Slnd2$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn2 <- coef_df_Stn2 %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd2 <- coef_df_Slnd2 %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn2$Variable <- rownames(coef_df_Stn2)
coef_df_Slnd2$Variable <- rownames(coef_df_Slnd2)

# Merge both data frames by the 'Variable' column
combined_df2 <- merge(coef_df_Stn2, coef_df_Slnd2, by = "Variable", all = TRUE)

# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn2)
logLik_Slnd <- logLik(Slnd2)
pseudo_R2_Stn <- summary_Stn2$r.squared["lratio"]
coef_det_Stn <- summary_Stn2$r.squared["coefdet"]   
pseudo_R2_Slnd <- summary_Slnd2$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd2$r.squared["coefdet"] 

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)

# Bind additional parameters to the combined coefficient table
combined_df2 <- bind_rows(combined_df2, additional_params)

# Reorder the columns to have 'Variable' first
combined_df2 <- combined_df2 %>%
  select(Variable, everything())

# Export the combined results to a CSV file
write.csv(combined_df2, "./regressions/combined_regression_results2_mitigation.csv", row.names = FALSE)


result2<-texreg(list(Stn2, Slnd2),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result2

write.table(result2, "Rio Result for Mitigation")


##### Mitigation regression with ClimateFinanceBERT #####


# Create a dataframe with unique combinations of Year, ProviderISO, and RecipientISO
climate_mitigation <- data_mitigation %>%
  select(-c("Sector", 
            "Adaptation.related.development.finance...Commitment...Current.USD.thousand",
            "Adaptation.related.development.finance...Commitment...2018.USD.thousand",
            "Mitigation.related.development.finance...Commitment...Current.USD.thousand",
            "Mitigation.related.development.finance...Commitment...2018.USD.thousand")) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy))%>%
  ungroup()%>%
  unique() %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(climate_class = "Air Pollution Mitigation")

plants <- climate_mitigation %>%
  mutate(climate_class = "Geothermal Explr/Plants")

green <- climate_mitigation %>%
  mutate(climate_class = "Green Growth Strategies")

hydro <- climate_mitigation %>%
  mutate(climate_class = "Hydro Power Plants Rehab")

renew <- climate_mitigation %>%
  mutate(climate_class = "Renewable energy")

solar <- climate_mitigation %>%
  mutate(climate_class = "Solar PV Energy")

wind <- climate_mitigation %>%
  mutate(climate_class = "Wind power farms")

# Combine the two datasets
unique_combinations <- bind_rows(climate_mitigation, plants, green, hydro, renew, solar, wind)%>%
  unique()

# mutate(climate_class = ifelse(row_number() == 1, "Climate Adaptation", "Resilience"))

BERT_data_mitigation <-left_join(unique_combinations, ClimateBERT_Mitigation%>%
                                   group_by(Year,ProviderISO,RecipientISO, climate_class)%>%
                                   summarise(ClimateBERT_miti_Commitment=sum(ClimateBERT_miti_Commitment,na.rm=T)), by= c("Year", "ProviderISO", "RecipientISO","climate_class"))

BERT_data_mitigation <- BERT_data_mitigation %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  filter(!(all(is.na(ClimateBERT_miti_Commitment))))


BERT_data_mitigation$NDC15[BERT_data_mitigation$NDC15%in%1 & !BERT_data_mitigation$Year%in%2015]<-0
BERT_data_mitigation$NDC15[BERT_data_mitigation$NDC16%in%1 & !BERT_data_mitigation$Year%in%2016]<-0
BERT_data_mitigation$NDC15[BERT_data_mitigation$NDC17%in%1 & !BERT_data_mitigation$Year%in%2017]<-0
BERT_data_mitigation$NDC15[BERT_data_mitigation$NDC18%in%1 & !BERT_data_mitigation$Year%in%2018]<-0

## In BERT_data_mitigation the amount of Adapt and Mitigation are defined as NA
## after constructing the Rio_data_adaptationset, change the adaptation and mitigation to zeros for NA cells

BERT_data_mitigation[is.na(BERT_data_mitigation[,37]),37]<-0

## remove deals less than 100 USD
## 35 obs.for adaptation and 29 for mitigation

BERT_data_mitigation[BERT_data_mitigation[,37]<0.1,37]<-0


summary(BERT_data_mitigation)

## assign NA International Agreement as zero
BERT_data_mitigation$InvestAgree[is.na(BERT_data_mitigation$InvestAgree)]<-0

regBERT_data_mitigation1<-BERT_data_mitigation[,c(1:13,16:37)]


regBERT_data_mitigation1$ProviderGDPCur<-log(regBERT_data_mitigation1$ProviderGDPCur)
regBERT_data_mitigation1$RecipientGDPCur<-log(regBERT_data_mitigation1$RecipientGDPCur)
regBERT_data_mitigation1$ProviderPop<-regBERT_data_mitigation1$ProviderPop/10^9
regBERT_data_mitigation1$RecipientPop<-regBERT_data_mitigation1$RecipientPop/10^9

regBERT_data_mitigation1$ProviderfisB<-regBERT_data_mitigation1$ProviderfisB/100
regBERT_data_mitigation1$RecipientfisB<-regBERT_data_mitigation1$RecipientfisB/100

regBERT_data_mitigation1$Providerdebt<-regBERT_data_mitigation1$Providerdebt/100
regBERT_data_mitigation1$Recipientdebt<-regBERT_data_mitigation1$Recipientdebt/100

regBERT_data_mitigation1$distw<-log(regBERT_data_mitigation1$distw)

## change the NA for MDB to zero
regBERT_data_mitigation1$MDBDummy[is.na(regBERT_data_mitigation1$MDBDummy)]<-0

summary(regBERT_data_mitigation1)

regBERT_data_mitigation<-regBERT_data_mitigation1[complete.cases(regBERT_data_mitigation1),]

names(regBERT_data_mitigation)[35]<-"MitiAmount"

## remove the zero provider countries
regBERT_data_mitigation$ProviderISO<-as.character(regBERT_data_mitigation$ProviderISO)
regBERT_data_mitigation$RecipientISO<-as.character(regBERT_data_mitigation$RecipientISO)

library(fastDummies)

regBERT_data_mitigationD<-dummy_cols(regBERT_data_mitigation, select_columns = c( "RecipientISO","ProviderISO","Year","climate_class"), remove_first_dummy = FALSE)

## column 42 to 110 are recipient
## 69 countries 
names(regBERT_data_mitigationD)[36:131]

reg6<-regBERT_data_mitigationD[,c(4:146)]

reg6$MitiAmount[reg6$MitiAmount!=0]<-log(reg6$MitiAmount[reg6$MitiAmount!=0]*1000)
reg6$MitiAmount[is.na(reg6$MitiAmount)]<-0

reg6$MDBAdapt[reg6$MDBAdapt!=0]<-log(reg6$MDBAdapt[reg6$MDBAdapt!=0]*1000)

## choose the benchmark sectors mannually

## drop these variables in the regression

names(reg6)[137]<-c("climate_class_Air_Pollution_Mitigation")
names(reg6)[138]<-c("climate_class_Geothermal_Explr_Plants")
names(reg6)[140]<-c("climate_class_Hydro_Power_Plants")
names(reg6)[141]<-c("climate_class_Renewable_energy")
names(reg6)[142]<-c("climate_class_Solar_Energy")
names(reg6)[143]<-c("climate_class_Wind_power_farms")
names(reg6)[139]<-c("climate_class_Green_Growth_Strategies")

reg6$NDCGHG<-0
reg6$climate_class_Geothermal_Explr_Plants<-0
reg6$Year_2016<-0
reg6$ProviderISO_USA<-0
## assign zero to countries with observations <0.005
# reg3$RecipientISO_ARM<-0
#reg1$RecipientISO_ERI<-0
#reg1$RecipientISO_WSM<-0
reg6$IncomeGroup<-as.character(reg6$IncomeGroup)
reg6$IncomeGroup[reg6$IncomeGroup=="UMICs"]<-"BaseUM"



library(mhurdle)
library(texreg)

## hurdle 1

reg6$WRI<-reg6$RecipientWRIExpo*reg6$RecipientWRIVul
reg6$ProviderGDPtot<-log(exp(reg6$ProviderGDPCur)*reg6$ProviderPop)
reg6$RecipientGDPtot<-log(exp(reg6$RecipientGDPCur)*reg6$RecipientPop)


names(reg6)

# Step 1: Gather all ProviderISO_* and RecipientISO_* columns in a tidy format
# iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg3), value = TRUE)

iso_vars <- grep("ProviderISO_", names(reg3), value = TRUE)

# Create a long format dataset to calculate proportions for each ISO variable
iso_df <- reg6 %>%
  select(all_of(iso_vars)) %>%      # Select only the ProviderISO_* and RecipientISO_* columns
  mutate(obs_id = row_number()) %>% # Add a row identifier to keep track of original rows
  pivot_longer(cols = all_of(iso_vars), 
               names_to = "country_iso", 
               values_to = "presence", 
               values_drop_na = TRUE) %>%  # Convert to long format with country ISO as the variable
  filter(presence == 1)             # Keep only rows where the country ISO variable is 1 (present)

proportions_df <- iso_df %>%
  group_by(country_iso) %>%
  summarize(prop_obs = n() / nrow(reg6))  # Calculate proportion of total observations


# Step 3: Identify countries below the 0.005 threshold
excluded_countries <- proportions_df %>%
  filter(prop_obs <= 0.005) %>%      # Countries to exclude
  pull(country_iso)

# Print the excluded countries
cat("Excluded countries (proportion < 0.005):\n")
print(excluded_countries)

# Step 4: Identify countries with proportion >= 0.005 to keep in the model
remaining_countries <- proportions_df %>%
  filter(prop_obs > 0.005) %>%     # Countries to keep in the model
  pull(country_iso)


# Step 5: Create the new model formula excluding the low-representation countries
base_formula <- " MitiAmount ~ WRI+CPIAPublicAdm+CPIAbudget+NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop"

# Only keep the remaining countries in the formula
country_effects <- paste(remaining_countries, collapse = " + ")

# Create the final model formula conditionally
final_formula <- if (nzchar(country_effects)) {
  paste(base_formula, country_effects, sep = " + ")
} else {
  base_formula
}

iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg6), value = TRUE)

# Create a long format dataset to calculate proportions for each ISO variable
iso_df <- reg6 %>%
  select(all_of(iso_vars)) %>%      # Select only the ProviderISO_* and RecipientISO_* columns
  mutate(obs_id = row_number()) %>% # Add a row identifier to keep track of original rows
  pivot_longer(cols = all_of(iso_vars), 
               names_to = "country_iso", 
               values_to = "presence", 
               values_drop_na = TRUE) %>%  # Convert to long format with country ISO as the variable
  filter(presence == 1)             # Keep only rows where the country ISO variable is 1 (present)

proportions_df <- iso_df %>%
  group_by(country_iso) %>%
  summarize(prop_obs = n() / nrow(reg6))  # Calculate proportion of total observations


# Step 3: Identify countries below the 0.005 threshold
excluded_countries <- proportions_df %>%
  filter(prop_obs <= 0.005) %>%      # Countries to exclude
  pull(country_iso)

# Print the excluded countries
cat("Excluded countries (proportion < 0.005):\n")
print(excluded_countries)

# Step 4: Identify countries with proportion >= 0.005 to keep in the model
remaining_countries <- proportions_df %>%
  filter(prop_obs > 0.005) %>%     # Countries to keep in the model
  pull(country_iso)

# Only keep the remaining countries in the formula
country_effects <- paste(remaining_countries, collapse = " + ")


# Second part of the hurdle formula
second_formula <- "| WRI+CPIAPublicAdm+CPIAbudget+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+climate_class_Air_Pollution_Mitigation+climate_class_Hydro_Power_Plants+climate_class_Renewable_energy+climate_class_Solar_Energy+climate_class_Wind_power_farms+climate_class_Green_Growth_Strategies"

# Create the final model formula conditionally
complete_second <- if (nzchar(country_effects)) {
  paste(second_formula, country_effects, sep = " + ")
} else {
  second_formula
}

# Combine both the final and second parts of the formula
complete_formula <- paste(final_formula, complete_second)
complete_formula
# Step 6: Rebuild and run the model
Stn3 <- mhurdle(as.formula(complete_formula),
                data = reg6, 
                dist = "n", 
                h2 = TRUE, 
                corr = FALSE, 
                method = "Bhhh", 
                print.level = 0, 
                finalHessian = TRUE)

summary(Stn3)


Slnd3 <- update(Stn3, corr = TRUE)
coef(summary(Slnd3), "corr")

# Extract the summaries of both models
summary_Stn3 <- summary(Stn3)
summary_Slnd3 <- summary(Slnd3)

# Convert the coefficients from both summaries into data frames
coef_df_Stn3 <- as.data.frame(summary_Stn3$coefficients)
coef_df_Slnd3 <- as.data.frame(summary_Slnd3$coefficients)

# Rename the columns to differentiate between the models
coef_df_Stn3 <- coef_df_Stn3 %>%
  rename_all(~paste0("Uncorrelated_", .))
coef_df_Slnd3 <- coef_df_Slnd3 %>%
  rename_all(~paste0("Correlated_", .))

# Add a common 'Variable' column to both data frames for easier merging
coef_df_Stn3$Variable <- rownames(coef_df_Stn3)
coef_df_Slnd3$Variable <- rownames(coef_df_Slnd3)

# Merge both data frames by the 'Variable' column
combined_df3 <- merge(coef_df_Stn3, coef_df_Slnd3, by = "Variable", all = TRUE)

# Calculate log-likelihoods and pseudo-R2 for both models
logLik_Stn <- logLik(Stn3)
logLik_Slnd <- logLik(Slnd3)
pseudo_R2_Stn <- summary_Stn3$r.squared["lratio"]
coef_det_Stn <- summary_Stn3$r.squared["coefdet"]
pseudo_R2_Slnd <- summary_Slnd3$r.squared["lratio"]
coef_det_Slnd <- summary_Slnd3$r.squared["coefdet"]

# Create a data frame for additional parameters
additional_params <- data.frame(
  Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R²)"),
  Uncorrelated_Estimate = c(logLik_Stn, pseudo_R2_Stn, coef_det_Stn),
  Correlated_Estimate = c(logLik_Slnd, pseudo_R2_Slnd, coef_det_Slnd)
)

# Bind additional parameters to the combined coefficient table
combined_df3 <- bind_rows(combined_df3, additional_params)

# Reorder the columns to have 'Variable' first
combined_df3 <- combined_df3 %>%
  select(Variable, everything())

summary_result_miti <- compare_datasets(
  regDataMiti,
  regRio_data_Mitigation,
  regBERT_data_mitigation
)

summary_sample <- left_join(summary_result_adap, summary_result_miti, by="Attribute")

write_csv(summary_sample, "./Redaction/summary_sample.csv")

# Export the combined results to a CSV file
write.csv(combined_df3, "./regressions/combined_regression_results3_mitigation.csv", row.names = FALSE)

result3<-texreg(list(Stn3, Slnd3),
                custom.model.names = c("log-normal", "Correlated log-normal"),
                caption = "Estimation of double hurdle selection models",
                label = "tab:sep", pos = "ht", digits =3)
result3 

write.table(result3, "ClimateFinanceBERT Result for Mitigation")

# Save reg1 as a CSV file
write.csv(reg4, "./Redaction/reg1_mitigation.csv", row.names = FALSE)

# Save reg2 as a CSV file
write.csv(reg5, "./Redaction/reg2_mitigation.csv", row.names = FALSE)

# Save reg3 as a CSV file
write.csv(reg6, "./Redaction/reg3_mitigation.csv", row.names = FALSE)

# Select only the MitiAmount column and add a Source identifier
df_reg4 <- reg4 %>% select(MitiAmount) %>% mutate(Source = "Reg4")
df_reg5 <- reg5 %>% select(MitiAmount) %>% mutate(Source = "Reg5")
df_reg6 <- reg6 %>% select(MitiAmount) %>% mutate(Source = "Reg6")

# Combine the data frames
combined_df <- bind_rows(df_reg4, df_reg5, df_reg6)
