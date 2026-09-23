# =============================================================================
# Climate finance estimation/Figures/PVCCImap.R
# -----------------------------------------------------------------------------
# Purpose
#   Map of the Physical Vulnerability to Climate Change Index (PVCCI, FERDI).
#   Default output reproduces paper Figure A1 (country-level map).
#
# Inputs
#   Climate finance estimation/Data/pvcciNational.csv   (shipped; FERDI PVCCI,
#     country level)
#   Country boundaries: Natural Earth (via rnaturalearth / rnaturalearthdata,
#     no download needed).
#
# Output
#   Climate finance estimation/Figures/Graphs/vulnerability_map.png  (= Figure A1)
#
# Optional (not needed for the paper): an ADM2 sub-national map, written to
#   Climate finance estimation/Figures/Graphs/PVCCImap_adm2.png
#   only when BOTH external files below are present (neither is shipped):
#     Climate finance estimation/Data/external/gadm_410-levels.gpkg
#       (GADM 4.1, "levels" GeoPackage, https://gadm.org)
#     Climate finance estimation/Data/external/pvcciSubNational.csv
#       (FERDI sub-national PVCCI, columns country, adm2, PVCCI)
#
# Run from anywhere inside the repository: Rscript "Climate finance estimation/Figures/PVCCImap.R"
# Paths resolve through here::here() (repository root is marked by `.here`).
# No stochastic element: no seed needed.
# =============================================================================

library(here)
library(sf)
library(dplyr)
library(data.table)
library(ggplot2)
library(viridis)
library(rnaturalearth)
library(countrycode)

data_dir   <- here::here("Climate finance estimation", "Data")
output_dir <- here::here("Climate finance estimation", "Figures", "Graphs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# Figure A1: country-level PVCCI map
# -----------------------------------------------------------------------------
world <- ne_countries(scale = "medium", returnclass = "sf") %>%
  filter(sovereignt != "Antarctica")

vulnerability_data <- fread(file = file.path(data_dir, "pvcciNational.csv"))
vulnerability_data$iso3 <- countrycode(
  sourcevar   = vulnerability_data$country,
  origin      = "country.name",
  destination = "iso3c"
)

world_vulnerability <- merge(world, vulnerability_data,
                             by.x = "adm0_a3", by.y = "iso3", all.x = TRUE)

# Title, subtitle and caption are kept because they appear in the published
# Figure A1 image.
map_national <- ggplot() +
  geom_sf(data = world_vulnerability, aes(fill = PVCCI)) +
  scale_fill_viridis(
    name      = "Vulnerability Score",
    option    = "magma",
    direction = -1,
    na.value  = "grey80"
  ) +
  coord_sf(crs = 3395) +  # Mercator (EPSG:3395)
  theme(
    panel.background      = element_rect(fill = "white"),
    plot.title            = element_text(hjust = 0.5, size = 20),
    plot.subtitle         = element_text(hjust = 0.5, size = 16),
    legend.position       = "right",
    plot.caption.position = "plot",
    plot.caption          = element_text(hjust = 0.5, size = 14)
  ) +
  labs(
    title    = "Global Climate Vulnerability Index",
    subtitle = "The Physical Vulnerability to Climate Change Index (PVCCI). FERDI.",
    caption  = "PVCCI measures countries' exposure to climatic shocks and can guide climate adaptation funding."
  )

ggsave(file.path(output_dir, "vulnerability_map.png"), plot = map_national,
       width = 12, height = 8, dpi = 1000, units = "in", device = "png")
message("Wrote ", file.path(output_dir, "vulnerability_map.png"))

# -----------------------------------------------------------------------------
# Optional: ADM2 sub-national map (not in the paper)
# -----------------------------------------------------------------------------
gadm_path  <- file.path(data_dir, "external", "gadm_410-levels.gpkg")
subnat_csv <- file.path(data_dir, "external", "pvcciSubNational.csv")

if (file.exists(gadm_path) && file.exists(subnat_csv)) {
  vulnerability_adm2 <- fread(file = subnat_csv)
  admin2_sf <- st_read(gadm_path, layer = "ADM_2", quiet = TRUE) %>%
    select(country = COUNTRY, ADM2 = NAME_2, geom)

  merged_adm2 <- merge(admin2_sf, vulnerability_adm2,
                       by.x = c("country", "ADM2"), by.y = c("country", "adm2"),
                       all.x = TRUE)

  map_adm2 <- ggplot() +
    geom_sf(data = merged_adm2, aes(fill = PVCCI)) +
    scale_fill_viridis(name = "Vulnerability Score", option = "magma",
                       direction = -1, na.value = "grey80") +
    coord_sf(crs = 3395) +
    theme_minimal() +
    theme(panel.background = element_rect(fill = "white"),
          legend.position  = "right")

  ggsave(file.path(output_dir, "PVCCImap_adm2.png"), plot = map_adm2,
         width = 15, height = 10, dpi = 300)
  message("Wrote ", file.path(output_dir, "PVCCImap_adm2.png"))
} else {
  message("Skipping optional ADM2 map: GADM / sub-national PVCCI files not found in ",
          file.path(data_dir, "external"))
}
