# =============================================================================
# Climate finance estimation/Figures/pipeline_diagram.R
# -----------------------------------------------------------------------------
# Purpose
#   Draws the two-stage ClimateFinanceBERT classification flowchart (paper
#   Figure 1). The node counts are hard-coded in the diagram below; they come
#   from the classification run on the full CRS corpus (Classify.py, meta.py):
#   1,312,202 unique descriptions; 105,342 relevant; Mitigation 34,301,
#   Adaptation 13,547, Environment 57,494.
#
# Output
#   Climate finance estimation/Figures/Graphs/pipeline_diagram.png  (= Figure 1)
#
# No data input, no stochastic element.
# =============================================================================

library(here)
library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

output_dir <- here::here("Climate finance estimation", "Figures", "Graphs")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

climate_finance_diagram <- grViz("
  digraph climate_finance {
    layout = 'fdp'
    
    # Graph settings
    graph [K = 2.1, rankdir = TB, fontsize = 18, fontname = Helvetica, splines = true]
    
    # Node settings
    node [shape = rectangle, style = filled, fillcolor = lightblue, 
          fontname = Helvetica, fontsize = 18, width = 2.5, height = 1.5]
    
    # Define nodes
    A [label = 'OECD CRS dataset\\n-\\nunique textual\\nprojects description\\n-\\nN = 1,312,202', fillcolor= '#FFDDC1', labelpos = 'c', labelfloat = true]
    B [label = 'Climate Relevant?\\n-\\nRe-trained ClimateBERT', shape = diamond, style = filled, fillcolor = '#FFDDC1', labelpos = 'c', labelfloat = true]
    D [label = 'Exclude from Sample\\n-\\nN = 1,206,860', fillcolor = '#FF4040', labelpos = 'c', labelfloat = true]
    
    # Define high-level node for Climate Projects
    ClimateProjects [label = 'Climate Projects\\n-\\nN = 105,342\\n-\\nPassed through multiclassifier', shape = rectangle, 
                     style = filled, fillcolor = '#2E8B57', fontsize = 18, width = 2.5, height = 1.5]
    
    # Categories with circular clusters
    subgraph cluster_0 {
      label = 'MITIGATION\\nN = 34,301'
      style = 'filled,rounded'
      fillcolor = '#2ca25f'
      color = '#2C3E50'
      fontsize = 18
      penwidth = 1.5
      
      C1 [label = 'Solar PV Energy', fillcolor = '#FFDDC1']
      C2 [label = 'Wind Power Farms', fillcolor = '#FFDDC1']
      C3 [label = 'Other clusters...', fillcolor = '#FFDDC1']
      C4 [label = 'Renewable Energy', fillcolor = '#FFDDC1']
    }
    
    subgraph cluster_1 {
      label = 'ADAPTATION\\nN = 13,547'
      style = 'filled,rounded'
      fillcolor = '#d95f0e'
      color = '#2C3E50'
      fontsize = 18
      penwidth = 1.5
      
      C5 [label = 'Climate Adaptation', fillcolor = '#FFDDC1']
      C6 [label = 'Resilience', fillcolor = '#FFDDC1']
    }
    
    subgraph cluster_2 {
      label = 'ENVIRONMENT\\nN = 57,494'
      style = 'filled,rounded'
      fillcolor = '#2b8cbe'
      color = '#2C3E50'
      fontsize = 18
      penwidth = 1.5
      
      C7 [label = 'Other clusters...', fillcolor = '#FFDDC1']
      C8 [label = 'Biodiv Conserv Prog', fillcolor = '#FFDDC1']
      C9 [label = 'Wildlife Conservation', fillcolor = '#FFDDC1']
      C10 [label = 'Marine-Coastal Protected\\nAreas Mgmt', fillcolor = '#FFDDC1']
    }
    
    # Define edges for primary flow
    edge [fontsize = 16, fontname = Helvetica, labelfontsize = 16, labelfontname = Helvetica, labelangle = -45]
    A -> B [label = 'Passed through\\nfirst classifier', labelpos = 'c', labelfloat = true]
    B -> D [label = 'Climate-irrelevant\\ndescriptions', labelpos = 'c', labelfloat = true]
    B -> ClimateProjects [label = 'Climate-relevant\\ndescriptions', labelpos = 'c', labelfloat = true]
    
    # Connect ClimateProjects to clusters instead of individual nodes
    ClimateProjects -> C1
    ClimateProjects -> C2
    ClimateProjects -> C3
    ClimateProjects -> C4
    ClimateProjects -> C5
    ClimateProjects -> C6
    ClimateProjects -> C7
    ClimateProjects -> C8
    ClimateProjects -> C9
    ClimateProjects -> C10
    
    # Rank settings for hierarchy
    {rank = min; A;}
    {rank = same; B;}
    {rank = same; D; ClimateProjects;}
    {rank = max; C1; C2; C3; C4; C5; C6; C7; C8; C9; C10;}
  }
")

climate_finance_diagram |>
  export_svg() |>
  charToRaw() |>
  rsvg_png(file.path(output_dir, "pipeline_diagram.png"))
message("Wrote ", file.path(output_dir, "pipeline_diagram.png"))
