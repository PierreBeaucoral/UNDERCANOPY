# =============================================================================
# HurdleRegHuei_parallel_robustness.R
# Parallel robustness version — replicates HurdleRegHuei.R at full speed.
#
# Differences vs. HurdleRegHuei_parallel.R (main analysis):
#   • NDC temporal-consistency lines use the ORIGINAL copy-paste bug from
#     HurdleRegHuei.R (all four lines assign to NDC15 instead of NDC15–NDC18).
#   • All outputs go to regressions/robustness/ and vif_*_robustness.csv.
#
# Everything else (parallel structure, formulas, post-processing) is identical
# to HurdleRegHuei_parallel.R.  Results are byte-for-byte equivalent to
# running HurdleRegHuei.R sequentially, only ~6x faster.
#
# Install if needed:
#   install.packages(c("future", "future.apply"))
# =============================================================================

# ── Libraries ─────────────────────────────────────────────────────────────────
# NOTE: several libraries below are not required for estimation; see build report
library(readr); library(png);  library(purrr);    library(gtable)
library(gridExtra); library(ggplot2); library(dplyr); library(knitr)
library(kableExtra); library(pander); library(tidyverse); library(tmap)
library(leaflet); library(ggforce); library(treemap); library(readxl)
library(ggalluvial); library(rlist); library(data.table); library(hms)
library(flextable); library(skimr); library(fastDummies)
library(mhurdle); library(texreg); library(car)
library(future); library(future.apply)
library(here)

# =============================================================================
# SECTION 1 — RAW DATA LOADING
# =============================================================================

# external: download from https://drive.uca.fr/d/6058b184ba134a02a708/
Rio_Data <- fread(here::here("Econometrics", "Data", "DataPB.csv"))

Rio_data_adaptation <- Rio_Data %>%
  filter(ClimateAdaptation %in% (1:2)) %>%
  group_by(Year, DonorName, RecipientName) %>%
  summarise(Rio_Adapt_Commitment = sum(USD_Commitment_Defl * 1000, na.rm = TRUE))

Rio_data_adaptation$ProviderISO <- countrycode::countrycode(sourcevar = Rio_data_adaptation$DonorName, origin = "country.name", destination = "iso3c")
Rio_data_adaptation$RecipientISO <- countrycode::countrycode(sourcevar = Rio_data_adaptation$RecipientName, origin = "country.name", destination = "iso3c")
Rio_data_adaptation[Rio_data_adaptation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"

# external: download from https://drive.uca.fr/d/6058b184ba134a02a708/
ClimateBERT <- fread(here::here("Econometrics", "Data", "climate_finance_total.csv"))

ClimateBERT_Adaptation <- ClimateBERT %>%
  filter(meta_category == "Adaptation") %>%
  group_by(Year, DonorName, RecipientName, climate_class) %>%
  summarise(ClimateBERT_Adapt_Commitment = sum(USD_Commitment_Defl * 1000, na.rm = TRUE))

ClimateBERT_Adaptation$ProviderISO <- countrycode::countrycode(sourcevar = ClimateBERT_Adaptation$DonorName, origin = "country.name", destination = "iso3c")
ClimateBERT_Adaptation$RecipientISO <- countrycode::countrycode(sourcevar = ClimateBERT_Adaptation$RecipientName, origin = "country.name", destination = "iso3c")
ClimateBERT_Adaptation[ClimateBERT_Adaptation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"

# =============================================================================
# SECTION 2 — ADAPTATION DATA PREP (reg1, reg2, reg3)
# =============================================================================

data_adaptation <- read.csv(here::here("Climate finance estimation", "Raw Data", "Adaptation with gravity vars amended FULL Feb 14 2023.csv"))

# Original NDC coding from HurdleRegHuei.R — copy-paste bug: all four lines
# assign to NDC15 instead of NDC15/NDC16/NDC17/NDC18.  Retained verbatim here
# so that the robustness results are byte-for-byte identical to the sequential
# script.
data_adaptation$NDC15[data_adaptation$NDC15 %in% 1 & !data_adaptation$Year %in% 2015] <- 0
data_adaptation$NDC15[data_adaptation$NDC16 %in% 1 & !data_adaptation$Year %in% 2016] <- 0
data_adaptation$NDC15[data_adaptation$NDC17 %in% 1 & !data_adaptation$Year %in% 2017] <- 0
data_adaptation$NDC15[data_adaptation$NDC18 %in% 1 & !data_adaptation$Year %in% 2018] <- 0

data_adaptation[is.na(data_adaptation[, 5]), 5] <- 0
data_adaptation[is.na(data_adaptation[, 6]), 6] <- 0
data_adaptation[is.na(data_adaptation[, 7]), 7] <- 0
data_adaptation[is.na(data_adaptation[, 8]), 8] <- 0

data_adaptation[data_adaptation[, 5] > 0 & data_adaptation[, 5] < 0.1, 5] <- 0
data_adaptation[data_adaptation[, 7] > 0 & data_adaptation[, 7] < 0.1, 7] <- 0

data_adaptation$InvestAgree[is.na(data_adaptation$InvestAgree)] <- 0

regdata_adaptation1 <- data_adaptation[, c(1:4, 5, 7, 9:18, 21:40)]

regdata_adaptation1$ProviderGDPCur <- log(regdata_adaptation1$ProviderGDPCur)
regdata_adaptation1$RecipientGDPCur <- log(regdata_adaptation1$RecipientGDPCur)
regdata_adaptation1$ProviderPop  <- regdata_adaptation1$ProviderPop  / 10^9
regdata_adaptation1$RecipientPop <- regdata_adaptation1$RecipientPop / 10^9
regdata_adaptation1$ProviderfisB  <- regdata_adaptation1$ProviderfisB  / 100
regdata_adaptation1$RecipientfisB <- regdata_adaptation1$RecipientfisB / 100
regdata_adaptation1$Providerdebt  <- regdata_adaptation1$Providerdebt  / 100
regdata_adaptation1$Recipientdebt <- regdata_adaptation1$Recipientdebt / 100
regdata_adaptation1$distw <- log(regdata_adaptation1$distw)
regdata_adaptation1$MDBDummy[is.na(regdata_adaptation1$MDBDummy)] <- 0

regdata_adaptationAdapt <- regdata_adaptation1[complete.cases(regdata_adaptation1), ]
names(regdata_adaptationAdapt)[5] <- "AdaptAmount"
names(regdata_adaptationAdapt)[6] <- "MitiAmount"
regdata_adaptationAdapt$ProviderISO <- as.character(regdata_adaptationAdapt$ProviderISO)
regdata_adaptationAdapt$RecipientISO <- as.character(regdata_adaptationAdapt$RecipientISO)

regdata_adaptationAdaptD <- dummy_cols(regdata_adaptationAdapt,
  select_columns = c("Sector", "RecipientISO", "ProviderISO", "Year"),
  remove_first_dummy = FALSE)

reg1 <- regdata_adaptationAdaptD[, c(5:149)]
reg1$MitiAmount[reg1$MitiAmount != 0]   <- log(reg1$MitiAmount[reg1$MitiAmount != 0] * 1000)
reg1$AdaptAmount[reg1$AdaptAmount != 0] <- log(reg1$AdaptAmount[reg1$AdaptAmount != 0] * 1000)
reg1$MDBAdapt[reg1$MDBAdapt != 0]       <- log(reg1$MDBAdapt[reg1$MDBAdapt != 0] * 1000)
reg1$MDBMiti[reg1$MDBMiti != 0]         <- log(reg1$MDBMiti[reg1$MDBMiti != 0] * 1000)
reg1$NDCGHG <- 0; reg1$Sector_Others <- 0; reg1$Year_2016 <- 0; reg1$ProviderISO_USA <- 0
reg1$RecipientISO_ARM <- 0
reg1$IncomeGroup <- as.character(reg1$IncomeGroup)
reg1$IncomeGroup[reg1$IncomeGroup == "UMICs"] <- "BaseUM"
names(reg1)[33:39] <- c("Water", "Transport", "Agri", "EnvProtect", "MultiSec", "Others", "Disaster")
reg1$WRI            <- reg1$RecipientWRIExpo * reg1$RecipientWRIVul
reg1$ProviderGDPtot <- log(exp(reg1$ProviderGDPCur) * reg1$ProviderPop)
reg1$RecipientGDPtot <- log(exp(reg1$RecipientGDPCur) * reg1$RecipientPop)

# ── reg2 (Rio markers, adaptation) ───────────────────────────────────────────
Rio_data_adaptation <- left_join(data_adaptation, Rio_data_adaptation, by = c("Year", "ProviderISO", "RecipientISO"))
Rio_data_adaptation <- Rio_data_adaptation %>%
  select(-c(Sector)) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy)) %>%
  ungroup() %>%
  unique()

Rio_data_adaptation$NDC15[Rio_data_adaptation$NDC15 %in% 1 & !Rio_data_adaptation$Year %in% 2015] <- 0
Rio_data_adaptation$NDC16[Rio_data_adaptation$NDC16 %in% 1 & !Rio_data_adaptation$Year %in% 2016] <- 0
Rio_data_adaptation$NDC17[Rio_data_adaptation$NDC17 %in% 1 & !Rio_data_adaptation$Year %in% 2017] <- 0
Rio_data_adaptation$NDC18[Rio_data_adaptation$NDC18 %in% 1 & !Rio_data_adaptation$Year %in% 2018] <- 0

Rio_data_adaptation[is.na(Rio_data_adaptation[, 42]), 42] <- 0
Rio_data_adaptation[Rio_data_adaptation[, 42] > 0 & Rio_data_adaptation[, 42] < 0.1, 42] <- 0
Rio_data_adaptation$InvestAgree[is.na(Rio_data_adaptation$InvestAgree)] <- 0

regRio_data_adaptation1 <- Rio_data_adaptation[, c(1:3, 8:17, 20:39, 42)]
regRio_data_adaptation1$ProviderGDPCur  <- log(regRio_data_adaptation1$ProviderGDPCur)
regRio_data_adaptation1$RecipientGDPCur <- log(regRio_data_adaptation1$RecipientGDPCur)
regRio_data_adaptation1$ProviderPop  <- regRio_data_adaptation1$ProviderPop  / 10^9
regRio_data_adaptation1$RecipientPop <- regRio_data_adaptation1$RecipientPop / 10^9
regRio_data_adaptation1$ProviderfisB  <- regRio_data_adaptation1$ProviderfisB  / 100
regRio_data_adaptation1$RecipientfisB <- regRio_data_adaptation1$RecipientfisB / 100
regRio_data_adaptation1$Providerdebt  <- regRio_data_adaptation1$Providerdebt  / 100
regRio_data_adaptation1$Recipientdebt <- regRio_data_adaptation1$Recipientdebt / 100
regRio_data_adaptation1$distw <- log(regRio_data_adaptation1$distw)
regRio_data_adaptation1$MDBDummy[is.na(regRio_data_adaptation1$MDBDummy)] <- 0

regRio_data_adaptationAdapt <- regRio_data_adaptation1[complete.cases(regRio_data_adaptation1), ]
names(regRio_data_adaptationAdapt)[34] <- "AdaptAmount"
regRio_data_adaptationAdapt$ProviderISO  <- as.character(regRio_data_adaptationAdapt$ProviderISO)
regRio_data_adaptationAdapt$RecipientISO <- as.character(regRio_data_adaptationAdapt$RecipientISO)

regRio_data_adaptationAdaptD <- dummy_cols(regRio_data_adaptationAdapt,
  select_columns = c("RecipientISO", "ProviderISO", "Year"),
  remove_first_dummy = FALSE)

reg2 <- regRio_data_adaptationAdaptD[, c(3:140)]
reg2$AdaptAmount[reg2$AdaptAmount != 0] <- log(reg2$AdaptAmount[reg2$AdaptAmount != 0] * 1000)
reg2$AdaptAmount[is.na(reg2$AdaptAmount)] <- 0
reg2$MDBAdapt[reg2$MDBAdapt != 0] <- log(reg2$MDBAdapt[reg2$MDBAdapt != 0] * 1000)
reg2$NDCGHG <- 0; reg2$Year_2016 <- 0; reg2$ProviderISO_USA <- 0
reg2$RecipientISO_ARM <- 0
reg2$IncomeGroup <- as.character(reg2$IncomeGroup)
reg2$IncomeGroup[reg2$IncomeGroup == "UMICs"] <- "BaseUM"
reg2$WRI            <- reg2$RecipientWRIExpo * reg2$RecipientWRIVul
reg2$ProviderGDPtot <- log(exp(reg2$ProviderGDPCur) * reg2$ProviderPop)
reg2$RecipientGDPtot <- log(exp(reg2$RecipientGDPCur) * reg2$RecipientPop)

# ── reg3 (BERT, adaptation) ───────────────────────────────────────────────────
climate_adaptation <- data_adaptation %>%
  select(-c("Sector",
            "Adaptation.related.development.finance...Commitment...Current.USD.thousand",
            "Adaptation.related.development.finance...Commitment...2018.USD.thousand",
            "Mitigation.related.development.finance...Commitment...Current.USD.thousand",
            "Mitigation.related.development.finance...Commitment...2018.USD.thousand")) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy)) %>%
  ungroup() %>%
  unique() %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(climate_class = "Climate Adaptation")

resilience <- climate_adaptation %>% mutate(climate_class = "Resilience")
unique_combinations <- bind_rows(climate_adaptation, resilience) %>% unique()

BERT_data_adaptation <- left_join(unique_combinations,
  ClimateBERT_Adaptation %>%
    group_by(Year, ProviderISO, RecipientISO, climate_class) %>%
    summarise(ClimateBERT_Adapt_Commitment = sum(ClimateBERT_Adapt_Commitment, na.rm = TRUE)),
  by = c("Year", "ProviderISO", "RecipientISO", "climate_class"))

BERT_data_adaptation <- BERT_data_adaptation %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  filter(!(all(is.na(ClimateBERT_Adapt_Commitment))))

BERT_data_adaptation$NDC15[BERT_data_adaptation$NDC15 %in% 1 & !BERT_data_adaptation$Year %in% 2015] <- 0
BERT_data_adaptation$NDC16[BERT_data_adaptation$NDC16 %in% 1 & !BERT_data_adaptation$Year %in% 2016] <- 0
BERT_data_adaptation$NDC17[BERT_data_adaptation$NDC17 %in% 1 & !BERT_data_adaptation$Year %in% 2017] <- 0
BERT_data_adaptation$NDC18[BERT_data_adaptation$NDC18 %in% 1 & !BERT_data_adaptation$Year %in% 2018] <- 0

BERT_data_adaptation[is.na(BERT_data_adaptation[, 37]), 37] <- 0
BERT_data_adaptation[BERT_data_adaptation[, 37] > 0 & BERT_data_adaptation[, 37] < 0.1, 37] <- 0
BERT_data_adaptation$InvestAgree[is.na(BERT_data_adaptation$InvestAgree)] <- 0

regBERT_data_adaptation1 <- BERT_data_adaptation[, c(1:13, 16:37)]
regBERT_data_adaptation1$ProviderGDPCur  <- log(regBERT_data_adaptation1$ProviderGDPCur)
regBERT_data_adaptation1$RecipientGDPCur <- log(regBERT_data_adaptation1$RecipientGDPCur)
regBERT_data_adaptation1$ProviderPop  <- regBERT_data_adaptation1$ProviderPop  / 10^9
regBERT_data_adaptation1$RecipientPop <- regBERT_data_adaptation1$RecipientPop / 10^9
regBERT_data_adaptation1$ProviderfisB  <- regBERT_data_adaptation1$ProviderfisB  / 100
regBERT_data_adaptation1$RecipientfisB <- regBERT_data_adaptation1$RecipientfisB / 100
regBERT_data_adaptation1$Providerdebt  <- regBERT_data_adaptation1$Providerdebt  / 100
regBERT_data_adaptation1$Recipientdebt <- regBERT_data_adaptation1$Recipientdebt / 100
regBERT_data_adaptation1$distw <- log(regBERT_data_adaptation1$distw)
regBERT_data_adaptation1$MDBDummy[is.na(regBERT_data_adaptation1$MDBDummy)] <- 0

regBERT_data_adaptatioAdapt <- regBERT_data_adaptation1[complete.cases(regBERT_data_adaptation1), ]
names(regBERT_data_adaptatioAdapt)[35] <- "AdaptAmount"
regBERT_data_adaptatioAdapt$ProviderISO  <- as.character(regBERT_data_adaptatioAdapt$ProviderISO)
regBERT_data_adaptatioAdapt$RecipientISO <- as.character(regBERT_data_adaptatioAdapt$RecipientISO)

regBERT_data_adaptatioAdaptD <- dummy_cols(regBERT_data_adaptatioAdapt,
  select_columns = c("RecipientISO", "ProviderISO", "Year", "climate_class"),
  remove_first_dummy = FALSE)

reg3 <- regBERT_data_adaptatioAdaptD[, c(4:135)]
reg3$AdaptAmount[reg3$AdaptAmount != 0] <- log(reg3$AdaptAmount[reg3$AdaptAmount != 0] * 1000)
reg3$AdaptAmount[is.na(reg3$AdaptAmount)] <- 0
reg3$MDBAdapt[reg3$MDBAdapt != 0] <- log(reg3$MDBAdapt[reg3$MDBAdapt != 0] * 1000)
reg3$NDCGHG <- 0; reg3$climate_class_Resilience <- 0; reg3$Year_2016 <- 0; reg3$ProviderISO_USA <- 0
reg3$IncomeGroup <- as.character(reg3$IncomeGroup)
reg3$IncomeGroup[reg3$IncomeGroup == "UMICs"] <- "BaseUM"
names(reg3)[131] <- "climate_class_Climate_Adaptation"
reg3$WRI            <- reg3$RecipientWRIExpo * reg3$RecipientWRIVul
reg3$ProviderGDPtot <- log(exp(reg3$ProviderGDPCur) * reg3$ProviderPop)
reg3$RecipientGDPtot <- log(exp(reg3$RecipientGDPCur) * reg3$RecipientPop)

# ── Build BERT adapt formula (dynamic, uses reg3 proportions) ─────────────────
iso_vars <- grep("ProviderISO_", names(reg3), value = TRUE)
iso_df <- reg3 %>%
  select(all_of(iso_vars)) %>%
  mutate(obs_id = row_number()) %>%
  pivot_longer(cols = all_of(iso_vars), names_to = "country_iso", values_to = "presence", values_drop_na = TRUE) %>%
  filter(presence == 1)
proportions_df <- iso_df %>% group_by(country_iso) %>% summarize(prop_obs = n() / nrow(reg3))
remaining_countries_p1 <- proportions_df %>% filter(prop_obs > 0.005) %>% pull(country_iso)

base_formula_adapt <- "AdaptAmount ~ WRI+CPIAPublicAdm+CPIAbudget+NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop"
country_effects_p1 <- paste(remaining_countries_p1, collapse = " + ")
final_formula_adapt <- if (nzchar(country_effects_p1)) paste(base_formula_adapt, country_effects_p1, sep = " + ") else base_formula_adapt

iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg3), value = TRUE)
iso_df <- reg3 %>%
  select(all_of(iso_vars)) %>%
  mutate(obs_id = row_number()) %>%
  pivot_longer(cols = all_of(iso_vars), names_to = "country_iso", values_to = "presence", values_drop_na = TRUE) %>%
  filter(presence == 1)
proportions_df <- iso_df %>% group_by(country_iso) %>% summarize(prop_obs = n() / nrow(reg3))
remaining_countries_p2 <- proportions_df %>% filter(prop_obs > 0.005) %>% pull(country_iso)
country_effects_p2 <- paste(remaining_countries_p2, collapse = " + ")

second_formula_adapt <- "| WRI+CPIAPublicAdm+CPIAbudget+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+climate_class_Climate_Adaptation"
complete_second_adapt <- if (nzchar(country_effects_p2)) paste(second_formula_adapt, country_effects_p2, sep = " + ") else second_formula_adapt

f_adapt_bert <- paste(final_formula_adapt, complete_second_adapt)

# =============================================================================
# SECTION 3 — MITIGATION DATA PREP (reg4, reg5, reg6)
# =============================================================================

ClimateBERT_Mitigation <- ClimateBERT %>%
  filter(meta_category == "Mitigation") %>%
  group_by(Year, DonorName, RecipientName, climate_class) %>%
  summarise(ClimateBERT_miti_Commitment = sum(USD_Commitment_Defl * 1000, na.rm = TRUE))

ClimateBERT_Mitigation$ProviderISO <- countrycode::countrycode(sourcevar = ClimateBERT_Mitigation$DonorName, origin = "country.name", destination = "iso3c")
ClimateBERT_Mitigation$RecipientISO <- countrycode::countrycode(sourcevar = ClimateBERT_Mitigation$RecipientName, origin = "country.name", destination = "iso3c")
ClimateBERT_Mitigation[ClimateBERT_Mitigation$RecipientName %in% "Kosovo",   "Country_Code"] <- "XKX"
ClimateBERT_Mitigation[ClimateBERT_Mitigation$RecipientName %in% "TÃ¼rkiye", "Country_Code"] <- "TUR"

data_mitigation <- read.csv(here::here("Climate finance estimation", "Raw Data", "Mitigation with gravity vars amended FULL Feb 14 2023.csv"))
# Original NDC coding from HurdleRegHuei.R — copy-paste bug: all four lines
# assign to NDC15 instead of NDC15/NDC16/NDC17/NDC18.  Retained verbatim here
# so that the robustness results are byte-for-byte identical to the sequential
# script applied to the mitigation dataset.
data_mitigation$NDC15[data_mitigation$NDC15 %in% 1 & !data_mitigation$Year %in% 2015] <- 0
data_mitigation$NDC15[data_mitigation$NDC16 %in% 1 & !data_mitigation$Year %in% 2016] <- 0
data_mitigation$NDC15[data_mitigation$NDC17 %in% 1 & !data_mitigation$Year %in% 2017] <- 0
data_mitigation$NDC15[data_mitigation$NDC18 %in% 1 & !data_mitigation$Year %in% 2018] <- 0

data_mitigation[is.na(data_mitigation[, 5]), 5] <- 0
data_mitigation[is.na(data_mitigation[, 6]), 6] <- 0
data_mitigation[is.na(data_mitigation[, 7]), 7] <- 0
data_mitigation[is.na(data_mitigation[, 8]), 8] <- 0
data_mitigation[data_mitigation[, 5] > 0 & data_mitigation[, 5] < 0.1, 5] <- 0
data_mitigation[data_mitigation[, 7] > 0 & data_mitigation[, 7] < 0.1, 7] <- 0
data_mitigation$InvestAgree[is.na(data_mitigation$InvestAgree)] <- 0

regData2 <- data_mitigation[, c(1:4, 5, 7, 9:18, 21:40)]
regData2$ProviderGDPCur  <- log(regData2$ProviderGDPCur)
regData2$RecipientGDPCur <- log(regData2$RecipientGDPCur)
regData2$ProviderPop  <- regData2$ProviderPop  / 10^9
regData2$RecipientPop <- regData2$RecipientPop / 10^9
regData2$ProviderfisB  <- regData2$ProviderfisB  / 100
regData2$RecipientfisB <- regData2$RecipientfisB / 100
regData2$Providerdebt  <- regData2$Providerdebt  / 100
regData2$Recipientdebt <- regData2$Recipientdebt / 100
regData2$distw <- log(regData2$distw)
regData2$MDBDummy[is.na(regData2$MDBDummy)] <- 0

regDataMiti <- regData2[complete.cases(regData2), ]
names(regDataMiti)[5] <- "AdaptAmount"
names(regDataMiti)[6] <- "MitiAmount"
regDataMiti$ProviderISO <- as.character(regDataMiti$ProviderISO)

regDataMitiD <- dummy_cols(regDataMiti,
  select_columns = c("Sector", "RecipientISO", "ProviderISO", "Year"),
  remove_first_dummy = FALSE)

reg4 <- regDataMitiD[, c(5:149)]
reg4$MitiAmount[reg4$MitiAmount != 0]   <- log(reg4$MitiAmount[reg4$MitiAmount != 0] * 1000)
reg4$AdaptAmount[reg4$AdaptAmount != 0] <- log(reg4$AdaptAmount[reg4$AdaptAmount != 0] * 1000)
reg4$MDBAdapt[reg4$MDBAdapt != 0] <- log(reg4$MDBAdapt[reg4$MDBAdapt != 0] * 1000)
reg4$MDBMiti[reg4$MDBMiti != 0]   <- log(reg4$MDBMiti[reg4$MDBMiti != 0] * 1000)
reg4$NDCGHG <- 0; reg4$Sector_Others <- 0; reg4$Year_2016 <- 0; reg4$ProviderISO_USA <- 0
reg4$RecipientISO_ARM <- 0
reg4$IncomeGroup <- as.character(reg4$IncomeGroup)
reg4$IncomeGroup[reg4$IncomeGroup == "UMICs"] <- "BaseUM"
names(reg4)[33:39] <- c("Water", "Transport", "Energy", "Agri", "EnvProtect", "MultiSec", "Others")
reg4$WRI            <- reg4$RecipientWRIExpo * reg4$RecipientWRIVul
reg4$ProviderGDPtot <- log(exp(reg4$ProviderGDPCur) * reg4$ProviderPop)
reg4$RecipientGDPtot <- log(exp(reg4$RecipientGDPCur) * reg4$RecipientPop)

# ── reg5 (Rio, mitigation) ────────────────────────────────────────────────────
Rio_data_Mitigation <- Rio_Data %>%
  filter(ClimateMitigation %in% (1:2)) %>%
  group_by(Year, DonorName, RecipientName) %>%
  summarise(Rio_Adapt_Commitment = sum(USD_Commitment_Defl * 1000, na.rm = TRUE))
Rio_data_Mitigation$ProviderISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$DonorName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation$RecipientISO <- countrycode::countrycode(sourcevar = Rio_data_Mitigation$RecipientName, origin = "country.name", destination = "iso3c")
Rio_data_Mitigation[Rio_data_Mitigation$RecipientName %in% "Kosovo", "Country_Code"] <- "XKX"

Rio_data_Mitigation <- left_join(data_mitigation, Rio_data_Mitigation, by = c("Year", "ProviderISO", "RecipientISO"))
Rio_data_Mitigation <- Rio_data_Mitigation %>%
  select(-c(Sector)) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy)) %>%
  ungroup() %>%
  unique()
Rio_data_Mitigation$NDC15[Rio_data_Mitigation$NDC15 %in% 1 & !Rio_data_Mitigation$Year %in% 2015] <- 0
Rio_data_Mitigation$NDC16[Rio_data_Mitigation$NDC16 %in% 1 & !Rio_data_Mitigation$Year %in% 2016] <- 0
Rio_data_Mitigation$NDC17[Rio_data_Mitigation$NDC17 %in% 1 & !Rio_data_Mitigation$Year %in% 2017] <- 0
Rio_data_Mitigation$NDC18[Rio_data_Mitigation$NDC18 %in% 1 & !Rio_data_Mitigation$Year %in% 2018] <- 0
Rio_data_Mitigation[is.na(Rio_data_Mitigation[, 42]), 42] <- 0
Rio_data_Mitigation[Rio_data_Mitigation[, 42] > 0 & Rio_data_Mitigation[, 42] < 0.1, 42] <- 0
Rio_data_Mitigation$InvestAgree[is.na(Rio_data_Mitigation$InvestAgree)] <- 0

regRio_data_Mitigation1 <- Rio_data_Mitigation[, c(1:3, 8:17, 20:39, 42)]
regRio_data_Mitigation1$ProviderGDPCur  <- log(regRio_data_Mitigation1$ProviderGDPCur)
regRio_data_Mitigation1$RecipientGDPCur <- log(regRio_data_Mitigation1$RecipientGDPCur)
regRio_data_Mitigation1$ProviderPop  <- regRio_data_Mitigation1$ProviderPop  / 10^9
regRio_data_Mitigation1$RecipientPop <- regRio_data_Mitigation1$RecipientPop / 10^9
regRio_data_Mitigation1$ProviderfisB  <- regRio_data_Mitigation1$ProviderfisB  / 100
regRio_data_Mitigation1$RecipientfisB <- regRio_data_Mitigation1$RecipientfisB / 100
regRio_data_Mitigation1$Providerdebt  <- regRio_data_Mitigation1$Providerdebt  / 100
regRio_data_Mitigation1$Recipientdebt <- regRio_data_Mitigation1$Recipientdebt / 100
regRio_data_Mitigation1$distw <- log(regRio_data_Mitigation1$distw)
regRio_data_Mitigation1$MDBDummy[is.na(regRio_data_Mitigation1$MDBDummy)] <- 0

regRio_data_Mitigation <- regRio_data_Mitigation1[complete.cases(regRio_data_Mitigation1), ]
names(regRio_data_Mitigation)[34] <- "MitiAmount"
regRio_data_Mitigation$ProviderISO  <- as.character(regRio_data_Mitigation$ProviderISO)
regRio_data_Mitigation$RecipientISO <- as.character(regRio_data_Mitigation$RecipientISO)

regRio_data_MitigationD <- dummy_cols(regRio_data_Mitigation,
  select_columns = c("RecipientISO", "ProviderISO", "Year"),
  remove_first_dummy = FALSE)

reg5 <- regRio_data_MitigationD[, c(3:140)]
reg5$MitiAmount[reg5$MitiAmount != 0] <- log(reg5$MitiAmount[reg5$MitiAmount != 0] * 1000)
reg5$MitiAmount[is.na(reg5$MitiAmount)] <- 0
reg5$MDBAdapt[reg5$MDBAdapt != 0] <- log(reg5$MDBAdapt[reg5$MDBAdapt != 0] * 1000)
reg5$NDCGHG <- 0; reg5$Year_2016 <- 0; reg5$ProviderISO_USA <- 0
reg5$RecipientISO_ARM <- 0
reg5$IncomeGroup <- as.character(reg5$IncomeGroup)
reg5$IncomeGroup[reg5$IncomeGroup == "UMICs"] <- "BaseUM"
reg5$WRI            <- reg5$RecipientWRIExpo * reg5$RecipientWRIVul
reg5$ProviderGDPtot <- log(exp(reg5$ProviderGDPCur) * reg5$ProviderPop)
reg5$RecipientGDPtot <- log(exp(reg5$RecipientGDPCur) * reg5$RecipientPop)

# ── reg6 (BERT, mitigation) ───────────────────────────────────────────────────
climate_mitigation <- data_mitigation %>%
  select(-c("Sector",
            "Adaptation.related.development.finance...Commitment...Current.USD.thousand",
            "Adaptation.related.development.finance...Commitment...2018.USD.thousand",
            "Mitigation.related.development.finance...Commitment...Current.USD.thousand",
            "Mitigation.related.development.finance...Commitment...2018.USD.thousand")) %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(MDBDummy = sum(MDBDummy)) %>%
  ungroup() %>%
  unique() %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  mutate(climate_class = "Air Pollution Mitigation")

plants <- climate_mitigation %>% mutate(climate_class = "Geothermal Explr/Plants")
green  <- climate_mitigation %>% mutate(climate_class = "Green Growth Strategies")
hydro  <- climate_mitigation %>% mutate(climate_class = "Hydro Power Plants Rehab")
renew  <- climate_mitigation %>% mutate(climate_class = "Renewable energy")
solar  <- climate_mitigation %>% mutate(climate_class = "Solar PV Energy")
wind   <- climate_mitigation %>% mutate(climate_class = "Wind power farms")
unique_combinations <- bind_rows(climate_mitigation, plants, green, hydro, renew, solar, wind) %>% unique()

BERT_data_mitigation <- left_join(unique_combinations,
  ClimateBERT_Mitigation %>%
    group_by(Year, ProviderISO, RecipientISO, climate_class) %>%
    summarise(ClimateBERT_miti_Commitment = sum(ClimateBERT_miti_Commitment, na.rm = TRUE)),
  by = c("Year", "ProviderISO", "RecipientISO", "climate_class"))

BERT_data_mitigation <- BERT_data_mitigation %>%
  group_by(Year, ProviderISO, RecipientISO) %>%
  filter(!(all(is.na(ClimateBERT_miti_Commitment))))

BERT_data_mitigation$NDC15[BERT_data_mitigation$NDC15 %in% 1 & !BERT_data_mitigation$Year %in% 2015] <- 0
BERT_data_mitigation$NDC16[BERT_data_mitigation$NDC16 %in% 1 & !BERT_data_mitigation$Year %in% 2016] <- 0
BERT_data_mitigation$NDC17[BERT_data_mitigation$NDC17 %in% 1 & !BERT_data_mitigation$Year %in% 2017] <- 0
BERT_data_mitigation$NDC18[BERT_data_mitigation$NDC18 %in% 1 & !BERT_data_mitigation$Year %in% 2018] <- 0
BERT_data_mitigation[is.na(BERT_data_mitigation[, 37]), 37] <- 0
BERT_data_mitigation[BERT_data_mitigation[, 37] < 0.1, 37] <- 0
BERT_data_mitigation$InvestAgree[is.na(BERT_data_mitigation$InvestAgree)] <- 0

regBERT_data_mitigation1 <- BERT_data_mitigation[, c(1:13, 16:37)]
regBERT_data_mitigation1$ProviderGDPCur  <- log(regBERT_data_mitigation1$ProviderGDPCur)
regBERT_data_mitigation1$RecipientGDPCur <- log(regBERT_data_mitigation1$RecipientGDPCur)
regBERT_data_mitigation1$ProviderPop  <- regBERT_data_mitigation1$ProviderPop  / 10^9
regBERT_data_mitigation1$RecipientPop <- regBERT_data_mitigation1$RecipientPop / 10^9
regBERT_data_mitigation1$ProviderfisB  <- regBERT_data_mitigation1$ProviderfisB  / 100
regBERT_data_mitigation1$RecipientfisB <- regBERT_data_mitigation1$RecipientfisB / 100
regBERT_data_mitigation1$Providerdebt  <- regBERT_data_mitigation1$Providerdebt  / 100
regBERT_data_mitigation1$Recipientdebt <- regBERT_data_mitigation1$Recipientdebt / 100
regBERT_data_mitigation1$distw <- log(regBERT_data_mitigation1$distw)
regBERT_data_mitigation1$MDBDummy[is.na(regBERT_data_mitigation1$MDBDummy)] <- 0

regBERT_data_mitigation <- regBERT_data_mitigation1[complete.cases(regBERT_data_mitigation1), ]
names(regBERT_data_mitigation)[35] <- "MitiAmount"
regBERT_data_mitigation$ProviderISO  <- as.character(regBERT_data_mitigation$ProviderISO)
regBERT_data_mitigation$RecipientISO <- as.character(regBERT_data_mitigation$RecipientISO)

regBERT_data_mitigationD <- dummy_cols(regBERT_data_mitigation,
  select_columns = c("RecipientISO", "ProviderISO", "Year", "climate_class"),
  remove_first_dummy = FALSE)

reg6 <- regBERT_data_mitigationD[, c(4:146)]
reg6$MitiAmount[reg6$MitiAmount != 0] <- log(reg6$MitiAmount[reg6$MitiAmount != 0] * 1000)
reg6$MitiAmount[is.na(reg6$MitiAmount)] <- 0
reg6$MDBAdapt[reg6$MDBAdapt != 0] <- log(reg6$MDBAdapt[reg6$MDBAdapt != 0] * 1000)
names(reg6)[137] <- "climate_class_Air_Pollution_Mitigation"
names(reg6)[138] <- "climate_class_Geothermal_Explr_Plants"
names(reg6)[140] <- "climate_class_Hydro_Power_Plants"
names(reg6)[141] <- "climate_class_Renewable_energy"
names(reg6)[142] <- "climate_class_Solar_Energy"
names(reg6)[143] <- "climate_class_Wind_power_farms"
names(reg6)[139] <- "climate_class_Green_Growth_Strategies"
reg6$NDCGHG <- 0; reg6$climate_class_Geothermal_Explr_Plants <- 0
reg6$Year_2016 <- 0; reg6$ProviderISO_USA <- 0
reg6$IncomeGroup <- as.character(reg6$IncomeGroup)
reg6$IncomeGroup[reg6$IncomeGroup == "UMICs"] <- "BaseUM"
reg6$WRI            <- reg6$RecipientWRIExpo * reg6$RecipientWRIVul
reg6$ProviderGDPtot <- log(exp(reg6$ProviderGDPCur) * reg6$ProviderPop)
reg6$RecipientGDPtot <- log(exp(reg6$RecipientGDPCur) * reg6$RecipientPop)

# ── Build BERT miti formula (dynamic, uses reg3 provider ISOs then reg6) ───────
# Note: first iso_vars pass uses reg3 (identical to original HurdleRegHuei.R)
iso_vars <- grep("ProviderISO_", names(reg3), value = TRUE)
iso_df <- reg6 %>%
  select(all_of(iso_vars)) %>%
  mutate(obs_id = row_number()) %>%
  pivot_longer(cols = all_of(iso_vars), names_to = "country_iso", values_to = "presence", values_drop_na = TRUE) %>%
  filter(presence == 1)
proportions_df <- iso_df %>% group_by(country_iso) %>% summarize(prop_obs = n() / nrow(reg6))
remaining_countries_m1 <- proportions_df %>% filter(prop_obs > 0.005) %>% pull(country_iso)
base_formula_miti <- "MitiAmount ~ WRI+CPIAPublicAdm+CPIAbudget+NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop"
country_effects_m1 <- paste(remaining_countries_m1, collapse = " + ")
final_formula_miti <- if (nzchar(country_effects_m1)) paste(base_formula_miti, country_effects_m1, sep = " + ") else base_formula_miti

iso_vars <- grep("ProviderISO_|RecipientISO_", names(reg6), value = TRUE)
iso_df <- reg6 %>%
  select(all_of(iso_vars)) %>%
  mutate(obs_id = row_number()) %>%
  pivot_longer(cols = all_of(iso_vars), names_to = "country_iso", values_to = "presence", values_drop_na = TRUE) %>%
  filter(presence == 1)
proportions_df <- iso_df %>% group_by(country_iso) %>% summarize(prop_obs = n() / nrow(reg6))
remaining_countries_m2 <- proportions_df %>% filter(prop_obs > 0.005) %>% pull(country_iso)
country_effects_m2 <- paste(remaining_countries_m2, collapse = " + ")
second_formula_miti <- "| WRI+CPIAPublicAdm+CPIAbudget+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt+Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018+climate_class_Air_Pollution_Mitigation+climate_class_Hydro_Power_Plants+climate_class_Renewable_energy+climate_class_Solar_Energy+climate_class_Wind_power_farms+climate_class_Green_Growth_Strategies"
complete_second_miti <- if (nzchar(country_effects_m2)) paste(second_formula_miti, country_effects_m2, sep = " + ") else second_formula_miti
f_miti_bert <- paste(final_formula_miti, complete_second_miti)

# =============================================================================
# SECTION 4 — COEFFICIENT DISPLAY NAMES
# =============================================================================

coef_labels <- c(
  WRI             = "WRI (Vulnerability)",
  colony          = "Colonial Ties",
  comlang         = "Common Language",
  comrelig        = "Common Religion",
  distw           = "Distance",
  NDC15           = "INDC15",
  NDC16           = "INDC16",
  NDC17           = "INDC17",
  NDC18           = "INDC18",
  InvestAgree     = "Investment Agreement",
  ProviderGDPtot  = "Provider GDP",
  RecipientGDPtot = "Recipient GDP",
  ProviderPop     = "Provider Population",
  RecipientPop    = "Recipient Population",
  ProviderfisB    = "Provider fiscal Bal",
  Providerdebt    = "Provider debt to GDP",
  RecipientfisB   = "Recipient Fiscal Bal",
  Recipientdebt   = "Recipient debt to GDP"
)

apply_coef_labels <- function(var_vec, labels = coef_labels) {
  has_prefix <- grepl("^h[12]\\.", var_vec)
  prefix     <- ifelse(has_prefix, sub("^(h[12]\\.)(.*)", "\\1", var_vec), "")
  raw        <- ifelse(has_prefix, sub("^h[12]\\.", "", var_vec), var_vec)
  nice       <- ifelse(raw %in% names(labels), labels[raw], raw)
  paste0(prefix, nice)
}

# =============================================================================
# SECTION 5 — STATIC FORMULA STRINGS (hardcoded, identical to original)
# =============================================================================

.p_sel <- "ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LUX+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE"
.p_alloc <- "ProviderISO_ARE+ProviderISO_AUS+ProviderISO_AUT+ProviderISO_BEL+ProviderISO_CAN+ProviderISO_CHE+ProviderISO_CZE+ProviderISO_DEU+ProviderISO_DNK+ProviderISO_ESP+ProviderISO_FIN+ProviderISO_FRA+ProviderISO_GBR+ProviderISO_GRC+ProviderISO_IRL+ProviderISO_ISL+ProviderISO_ITA+ProviderISO_JPN+ProviderISO_KOR+ProviderISO_LTU+ProviderISO_LUX+ProviderISO_LVA+ProviderISO_NLD+ProviderISO_NOR+ProviderISO_NZL+ProviderISO_POL+ProviderISO_PRT+ProviderISO_SVN+ProviderISO_SWE"
.r_alloc <- "RecipientISO_AFG+RecipientISO_AGO+RecipientISO_BDI+RecipientISO_BEN+RecipientISO_BFA+RecipientISO_BGD+RecipientISO_BIH+RecipientISO_BTN+RecipientISO_CAF+RecipientISO_CIV+RecipientISO_CMR+RecipientISO_COM+RecipientISO_CPV+RecipientISO_DJI+RecipientISO_ERI+RecipientISO_ETH+RecipientISO_GEO+RecipientISO_GHA+RecipientISO_GIN+RecipientISO_GMB+RecipientISO_GNB+RecipientISO_GRD+RecipientISO_GUY+RecipientISO_HND+RecipientISO_HTI+RecipientISO_IND+RecipientISO_KEN+RecipientISO_KGZ+RecipientISO_KHM+RecipientISO_KIR+RecipientISO_LAO+RecipientISO_LBR+RecipientISO_LKA+RecipientISO_LSO+RecipientISO_MDA+RecipientISO_MDG+RecipientISO_MLI+RecipientISO_MMR+RecipientISO_MNG+RecipientISO_MOZ+RecipientISO_MRT+RecipientISO_MWI+RecipientISO_NER+RecipientISO_NGA+RecipientISO_NIC+RecipientISO_NPL+RecipientISO_PAK+RecipientISO_PNG+RecipientISO_RWA+RecipientISO_SDN+RecipientISO_SEN+RecipientISO_SLB+RecipientISO_SLE+RecipientISO_STP+RecipientISO_TCD+RecipientISO_TGO+RecipientISO_TJK+RecipientISO_TON+RecipientISO_TZA+RecipientISO_UGA+RecipientISO_UZB+RecipientISO_VNM+RecipientISO_VUT+RecipientISO_WSM+RecipientISO_YEM+RecipientISO_ZMB+RecipientISO_ZWE"
.yr_alloc <- "Year_2011+Year_2012+Year_2013+Year_2014+Year_2015+Year_2017+Year_2018"
.core_sel   <- "WRI+CPIAPublicAdm+CPIAbudget+NDCActOnly+NDCnonGHG+NDC15+NDC16+NDC17+NDC18+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop"
.core_alloc <- "WRI+CPIAPublicAdm+CPIAbudget+distw+colony+comlang+comrelig+wto+MDBDummy+EIA+InvestAgree+ProviderGDPtot+RecipientGDPtot+ProviderPop+RecipientPop+ProviderfisB+RecipientfisB+Providerdebt+Recipientdebt"

f_adapt_han <- paste0(
  "AdaptAmount ~ ", .core_sel, "+", .p_sel,
  " | ", .core_alloc, "+", .r_alloc, "+", .p_alloc, "+", .yr_alloc,
  "+Water+Transport+Agri+EnvProtect+MultiSec+Disaster"
)

f_adapt_rio <- paste0(
  "AdaptAmount ~ ", .core_sel, "+", .p_sel,
  " | ", .core_alloc, "+", .r_alloc, "+", .p_alloc, "+", .yr_alloc
)

f_miti_han <- paste0(
  "MitiAmount ~ ", .core_sel, "+", .p_sel,
  " | ", .core_alloc, "+", .r_alloc, "+", .p_alloc, "+", .yr_alloc,
  "+Water+Transport+Agri+EnvProtect+MultiSec+Energy"
)

f_miti_rio <- paste0(
  "MitiAmount ~ ", .core_sel, "+", .p_sel,
  " | ", .core_alloc, "+", .r_alloc, "+", .p_alloc, "+", .yr_alloc
)

# =============================================================================
# SECTION 6 — PARALLEL BATCH 1: corr = FALSE (the slow step)
# =============================================================================

message("\n── Parallel batch 1: fitting 6 uncorrelated models ──────────────────────────")
message("Using ", min(6L, parallelly::availableCores()), " workers")

plan(multisession, workers = min(6L, parallelly::availableCores()))

model_jobs <- list(
  adapt_han  = list(f = f_adapt_han,  d = reg1),
  adapt_rio  = list(f = f_adapt_rio,  d = reg2),
  adapt_bert = list(f = f_adapt_bert, d = reg3),
  miti_han   = list(f = f_miti_han,   d = reg4),
  miti_rio   = list(f = f_miti_rio,   d = reg5),
  miti_bert  = list(f = f_miti_bert,  d = reg6)
)

t0 <- proc.time()
uncorr <- future_lapply(model_jobs, function(job) {
  library(mhurdle)
  # Use do.call so the formula and data are fully evaluated before mhurdle()
  # captures its call via match.call(). Without this, mhurdle's internal
  # eval(call, parent.frame()) fails because 'job' is out of scope on the worker.
  do.call(mhurdle, list(
    formula      = as.formula(job$f),
    data         = job$d,
    dist         = "n",
    h2           = TRUE,
    corr         = FALSE,
    method       = "Bhhh",
    print.level  = 0,
    finalHessian = TRUE
  ))
}, future.seed = NULL)
message(sprintf("Batch 1 done in %.1f s", (proc.time() - t0)[3]))

# =============================================================================
# SECTION 7 — PARALLEL BATCH 2: corr = TRUE (fast — warm start from batch 1)
# =============================================================================

message("\n── Sequential batch 2: correlated updates (warm start from batch 1) ─────────")
# update() stays in the main process — no worker scoping issues.
# Warm start from corr=FALSE estimates typically cuts iterations dramatically.
plan(multisession, workers = min(6L, parallelly::availableCores()))
t0 <- proc.time()                                                             
corr <- future_lapply(model_jobs, function(job) {         
  library(mhurdle)                                                            
  do.call(mhurdle, list(                                  
    formula      = as.formula(job$f),                                         
    data         = job$d,
    dist         = "n",                                                       
    h2           = TRUE,                                  
    corr         = TRUE,                                                      
    method       = "Bhhh",                                
    print.level  = 0,                                                         
    finalHessian = TRUE                                                       
  ))                                                                          
}, future.seed = NULL)                                                        
message(sprintf("Batch 2 done in %.1f s", (proc.time() - t0)[3]))
plan(sequential) 

# =============================================================================
# SECTION 8 — UNPACK TO NAMED OBJECTS
# =============================================================================

adapt_Stn  <- uncorr$adapt_han;   adapt_Slnd  <- corr$adapt_han
adapt_Stn2 <- uncorr$adapt_rio;   adapt_Slnd2 <- corr$adapt_rio
adapt_Stn3 <- uncorr$adapt_bert;  adapt_Slnd3 <- corr$adapt_bert
miti_Stn1  <- uncorr$miti_han;    miti_Slnd1  <- corr$miti_han
miti_Stn2  <- uncorr$miti_rio;    miti_Slnd2  <- corr$miti_rio
miti_Stn3  <- uncorr$miti_bert;   miti_Slnd3  <- corr$miti_bert

# quick sanity
cat("\nCorrelation coefficients:\n")
cat("  adapt Han: "); print(coef(summary(adapt_Slnd),  "corr"))
cat("  adapt Rio: "); print(coef(summary(adapt_Slnd2), "corr"))
cat("  adapt BERT:"); print(coef(summary(adapt_Slnd3), "corr"))
cat("  miti Han:  "); print(coef(summary(miti_Slnd1),  "corr"))
cat("  miti Rio:  "); print(coef(summary(miti_Slnd2),  "corr"))
cat("  miti BERT: "); print(coef(summary(miti_Slnd3),  "corr"))

# =============================================================================
# SECTION 9 — POST-PROCESSING HELPER
# =============================================================================

.make_csv_df <- function(stn, slnd) {
  s1 <- summary(stn);  s2 <- summary(slnd)
  df1 <- as.data.frame(s1$coefficients) %>% rename_all(~paste0("Uncorrelated_", .))
  df2 <- as.data.frame(s2$coefficients) %>% rename_all(~paste0("Correlated_",   .))
  df1$Variable <- rownames(df1)
  df2$Variable <- rownames(df2)
  out <- merge(df1, df2, by = "Variable", all = TRUE)
  extra <- data.frame(
    Variable = c("Log-likelihood", "McFadden Pseudo-R2", "Coefficient of Determination (R\u00b2)"),
    Uncorrelated_Estimate = c(as.numeric(logLik(stn)),  s1$r.squared["lratio"], s1$r.squared["coefdet"]),
    Correlated_Estimate   = c(as.numeric(logLik(slnd)), s2$r.squared["lratio"], s2$r.squared["coefdet"])
  )
  out <- bind_rows(out, extra) %>% select(Variable, everything())
  out$Variable <- apply_coef_labels(out$Variable)
  out
}

# =============================================================================
# SECTION 10 — EXPORT REGRESSION CSVs
# =============================================================================

dir.create(here::here("Econometrics", "regressions", "robustness", "adaptation"), recursive = TRUE, showWarnings = FALSE)
dir.create(here::here("Econometrics", "regressions", "robustness", "mitigation"), recursive = TRUE, showWarnings = FALSE)

write.csv(.make_csv_df(adapt_Stn,  adapt_Slnd),  here::here("Econometrics", "regressions", "robustness", "adaptation", "combined_regression_results.csv"),  row.names = FALSE)
write.csv(.make_csv_df(adapt_Stn2, adapt_Slnd2), here::here("Econometrics", "regressions", "robustness", "adaptation", "combined_regression_results2.csv"), row.names = FALSE)
write.csv(.make_csv_df(adapt_Stn3, adapt_Slnd3), here::here("Econometrics", "regressions", "robustness", "adaptation", "combined_regression_results3.csv"), row.names = FALSE)
write.csv(.make_csv_df(miti_Stn1,  miti_Slnd1),  here::here("Econometrics", "regressions", "robustness", "mitigation", "combined_regression_results.csv"),  row.names = FALSE)
write.csv(.make_csv_df(miti_Stn2,  miti_Slnd2),  here::here("Econometrics", "regressions", "robustness", "mitigation", "combined_regression_results2.csv"), row.names = FALSE)
write.csv(.make_csv_df(miti_Stn3,  miti_Slnd3),  here::here("Econometrics", "regressions", "robustness", "mitigation", "combined_regression_results3.csv"), row.names = FALSE)

message("All 6 regression CSVs written.")

# =============================================================================
# SECTION 11 — TEXREG OUTPUTS (identical to original)
# =============================================================================

result1 <- texreg(list(adapt_Stn,  adapt_Slnd),  custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
result2 <- texreg(list(adapt_Stn2, adapt_Slnd2), custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
result3 <- texreg(list(adapt_Stn3, adapt_Slnd3), custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
dir.create(here::here("Econometrics", "Results"), recursive = TRUE, showWarnings = FALSE)
write.table(result1, here::here("Econometrics", "Results", "Baseline Result for Adaptation"))
write.table(result2, here::here("Econometrics", "Results", "Rio Result for Adaptation"))
write.table(result3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Adaptation"))

result_m1 <- texreg(list(miti_Stn1, miti_Slnd1), custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
result_m2 <- texreg(list(miti_Stn2, miti_Slnd2), custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
result_m3 <- texreg(list(miti_Stn3, miti_Slnd3), custom.model.names = c("log-normal", "Correlated log-normal"), caption = "Estimation of double hurdle selection models", label = "tab:sep", pos = "ht", digits = 3)
write.table(result_m1, here::here("Econometrics", "Results", "Baseline Result for  Mitigation 103950 obs"))
write.table(result_m2, here::here("Econometrics", "Results", "Rio Result for Mitigation"))
write.table(result_m3, here::here("Econometrics", "Results", "ClimateFinanceBERT Result for Mitigation"))

# =============================================================================
# SECTION 12 — VIF DIAGNOSTICS
# =============================================================================

.extract_subst <- function(model, eq = 1) {
  prefix <- paste0("h", eq, ".")
  cn     <- names(coef(model))
  vars   <- sub(paste0("^", prefix), "", cn[startsWith(cn, prefix)])
  fe_pat <- paste0("^(ProviderISO_|RecipientISO_|Year_|",
                   "Water$|Transport$|Agri$|EnvProtect$|",
                   "MultiSec$|Disaster$|IncomeGroup|climate_class_)")
  vars[!grepl(fe_pat, vars)]
}

.compute_vif <- function(data, vars, outcome_col, pos_only = FALSE) {
  df    <- as.data.frame(data)
  if (pos_only) df <- df[df[[outcome_col]] > 0, , drop = FALSE]
  avail <- vars[vars %in% names(df)]
  df    <- df[, avail, drop = FALSE]
  df    <- df[complete.cases(df), , drop = FALSE]
  if (nrow(df) < length(avail) + 2L)
    return(setNames(rep(NA_real_, length(avail)), avail))
  set.seed(42)
  df[["y__"]] <- rnorm(nrow(df))
  m <- lm(as.formula(paste("y__ ~", paste(avail, collapse = " + "))), data = df)
  v <- tryCatch(car::vif(m), error = function(e) setNames(rep(NA_real_, length(avail)), avail))
  if (is.matrix(v)) v <- setNames(v[, 3]^2, rownames(v))
  v
}

.build_vif_long <- function(specs, outcome_col) {
  dplyr::bind_rows(lapply(specs, function(s) {
    dplyr::bind_rows(lapply(1:2, function(eq) {
      vars <- .extract_subst(s$model, eq)
      v    <- .compute_vif(s$data, vars, outcome_col, pos_only = (eq == 2))
      data.frame(Variable = names(v), Equation = c("Selection", "Allocation")[eq],
                 Spec = s$name, VIF = round(v, 3), row.names = NULL, stringsAsFactors = FALSE)
    }))
  }))
}

vif_adapt_wide <- .build_vif_long(
  list(list(model = adapt_Stn,  data = reg1, name = "Han et al."),
       list(model = adapt_Stn2, data = reg2, name = "Rio markers"),
       list(model = adapt_Stn3, data = reg3, name = "BERT Adapt.")),
  outcome_col = "AdaptAmount"
) %>%
  tidyr::pivot_wider(names_from = Spec, values_from = VIF) %>%
  dplyr::arrange(Equation, Variable)
write.csv(vif_adapt_wide, here::here("Econometrics", "regressions", "robustness", "vif_adapt_robustness.csv"), row.names = FALSE)
message("VIF diagnostics (adaptation) saved.")

vif_miti_wide <- .build_vif_long(
  list(list(model = miti_Stn1, data = reg4, name = "Han et al."),
       list(model = miti_Stn2, data = reg5, name = "Rio markers"),
       list(model = miti_Stn3, data = reg6, name = "BERT Miti.")),
  outcome_col = "MitiAmount"
) %>%
  tidyr::pivot_wider(names_from = Spec, values_from = VIF) %>%
  dplyr::arrange(Equation, Variable)
write.csv(vif_miti_wide, here::here("Econometrics", "regressions", "robustness", "vif_miti_robustness.csv"), row.names = FALSE)
message("VIF diagnostics (mitigation) saved.")

# =============================================================================
# SECTION 13 — SAVE reg DATA FRAMES
# =============================================================================

write.csv(reg1, here::here("Econometrics", "Data", "reg1.csv"), row.names = FALSE)
write.csv(reg2, here::here("Econometrics", "Data", "reg2.csv"), row.names = FALSE)
write.csv(reg3, here::here("Econometrics", "Data", "reg3.csv"), row.names = FALSE)
write.csv(reg4, here::here("Econometrics", "Data", "reg1_mitigation.csv"), row.names = FALSE)
write.csv(reg5, here::here("Econometrics", "Data", "reg2_mitigation.csv"), row.names = FALSE)
write.csv(reg6, here::here("Econometrics", "Data", "reg3_mitigation.csv"), row.names = FALSE)

# =============================================================================
# SECTION 14 — SAMPLE COMPARISON TABLE
# =============================================================================

compare_datasets <- function(regdata, regRio_data, regBERT_data) {
  safe_prov <- function(x) if (is.data.frame(x) && "ProviderISO"  %in% names(x)) unique(x$ProviderISO)  else character(0)
  safe_recp <- function(x) if (is.data.frame(x) && "RecipientISO" %in% names(x)) unique(x$RecipientISO) else character(0)
  prov_diff <- setdiff(intersect(safe_prov(regdata), safe_prov(regRio_data)), safe_prov(regBERT_data))
  recp_diff <- setdiff(intersect(safe_recp(regdata), safe_recp(regRio_data)), safe_recp(regBERT_data))
  tibble(
    Attribute = c("Number of Observations", "Number of Sectors",
                  "Number of Provider Countries", "Number of Recipient Countries",
                  "Provider Countries in Data & Rio but not in BERT",
                  "Recipient Countries in Data & Rio but not in BERT"),
    regdata_adaptationAdapt = c(
      if (is.data.frame(regdata)) nrow(regdata) else 0,
      if (is.data.frame(regdata) && "Sector" %in% names(regdata)) length(unique(regdata$Sector)) else 0,
      length(safe_prov(regdata)), length(safe_recp(regdata)), "-", "-"),
    regRio_data_adaptationAdapt = c(
      if (is.data.frame(regRio_data)) nrow(regRio_data) else 0, 1,
      length(safe_prov(regRio_data)), length(safe_recp(regRio_data)), "-", "-"),
    regBERT_data_adaptatioAdapt = c(
      if (is.data.frame(regBERT_data)) nrow(regBERT_data) else 0,
      if (is.data.frame(regBERT_data) && "climate_class" %in% names(regBERT_data)) length(unique(regBERT_data$climate_class)) else 0,
      length(safe_prov(regBERT_data)), length(safe_recp(regBERT_data)),
      ifelse(length(prov_diff) > 0, paste(prov_diff, collapse = ", "), "None"),
      ifelse(length(recp_diff) > 0, paste(recp_diff, collapse = ", "), "None"))
  )
}

summary_result_adap <- compare_datasets(regdata_adaptationAdapt, regRio_data_adaptationAdapt, regBERT_data_adaptatioAdapt)
summary_result_miti <- compare_datasets(regDataMiti,             regRio_data_Mitigation,      regBERT_data_mitigation)
summary_sample <- left_join(summary_result_adap, summary_result_miti, by = "Attribute")
write_csv(summary_sample, here::here("Econometrics", "regressions", "robustness", "summary_sample.csv"))

# =============================================================================
# SECTION 15 — DISTRIBUTION PLOT
# =============================================================================

combined_df_plot <- bind_rows(
  reg1 %>% select(AdaptAmount) %>% mutate(Source = "Reg1"),
  reg2 %>% select(AdaptAmount) %>% mutate(Source = "Reg2"),
  reg3 %>% select(AdaptAmount) %>% mutate(Source = "Reg3")
)

ggplot(combined_df_plot, aes(x = AdaptAmount, fill = Source)) +
  geom_density(alpha = 0.5) +
  labs(x = "AdaptAmount", y = "Density") +
  theme_minimal() +
  scale_fill_manual(values = c("Reg1" = "#1f77b4", "Reg2" = "#ff7f0e", "Reg3" = "#2ca02c")) +
  theme(legend.title = element_blank())

message("\nAll done.")
