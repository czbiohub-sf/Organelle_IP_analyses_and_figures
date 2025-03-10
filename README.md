# Global organelle profiling defines a sub-cellular map of the human proteome

This repository serves as a comprehensive resource for exploring and understanding our manuscript, [Global organelle profiling reveals subcellular localization and remodeling at proteome scale](https://www.biorxiv.org/content/10.1101/2023.12.18.572249v1). Inside, you'll find detailed Jupyter notebooks and scripts that were pivotal in our data analysis process and in generating the figures.

Our aim is to provide an in-depth, transparent view into our research methods and findings. Dive into our notebooks to see how we transformed raw data into meaningful insights, or explore our scripts to understand the technical underpinnings of our figure generation. We hope this repository will be a useful tool in your own research and learning journey.

## Usage
### Set up code, data and environment
```sh
git clone https://github.com/czbiohub-sf/Organelle_IP_analyses_and_figures.git
cd Organelle_IP_analyses_and_figures
```

To run the notebooks (except supplementary figure 3 panel E) use the provided [conda](https://docs.conda.io/en/latest/) environment:
```sh
conda env create -n OrgIP -f environment.yml
conda activate OrgIP
```

### Reproducing figures
To reproduce the figures, set `USE_FROZEN = True` in both `notebooks/0.Set_timestamp.ipynb` and `notebooks/Fig5/0.Set_fig5_timestamp.ipynb`, and execute these  notebooks. The notebooks will search for the frozen enrichment tables that were used to generate the figures.

### Fresh run
To begin a distinct run, set `USE_FROZEN = False` in both `notebooks/0.Set_timestamp.ipynb` and `notebooks/Fig5/0.Set_fig5_timestamp.ipynb`, and execute these notebooks. This will generate a new timestamp, which subsequent notebooks will use to save their output files. The notebooks will then utilize the MaxQuant output file (refer to the `data/` section) to regenerate the enrichment tables with slight variations in the random imputations of missing values.

### Supplementary figure 6 panel C
Supplementary figure 6 panel C uses a separate conda environment specification.
```sh
conda env create -n xgb2 -f notebooks/Supplementary_figures/Suppl_fig6/panel_C/environment.yml
conda activate xgb2
```

## What's in this repo
### `data/`
This directory contains various external and processed datasets required to make the figures. Note that some of these datasets are from external sources; these are found in the `data/external/` subdirectory. The remaining datasets are all original datasets generated by or derived from this project. Note that the MaxQuant output file `proteinGroups.txt` is too large to host on GitHub, and it is available through the PRIDE repository under the identifier PXD046440.


### `notebooks/`
This directory includes a series of Jupyter notebooks that detail the analytical processes and methodologies used in the creation of each figure. These notebooks serve as the principal guide for understanding the application of the scripts and Python modules located in the scripts/ directory, specifically focusing on their roles in analysis and figure generation.

We provide a structured tree diagram (below) representing the organization of the notebooks directory. This diagram delineates the specific notebooks responsible for generating each panel of the figures.

It is important to execute the notebooks sequentially, following the top-to-bottom order presented in the tree diagram. This ensures a smooth workflow, as many notebooks requires the output of their predecessors.

```
notebooks
|
├── 0.Set_timestamp.ipynb
├── enrichment
|   ├── 1.QC_filter_and_impute.ipynb
|   ├── 2.Batch_selection.ipynb
|   ├── 3.correlation_filter.ipynb
|   ├── 4.NOC_processing.ipynb
|   └── output
|       ├── correlation_tables
|       |   └── ..
|       ├── enrichment_and_volcano_tables
|       |   └── ..
|       └── preprocessing
|           └── ..
├── Fig1
│   └── panel_L
│       ├── Fig1_L_enrichment_heatmap.ipynb
│       └── output
│           └── ..
├── Fig2
│   ├── panel_B
│   │   ├── Fig2_B_heatmap.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_C
│   │   ├── Fig2_C_consensus_annotation.ipynb
│   │   └── output
│   │       └── ..
│   └── panel_D
│       ├── Fig2_D_umap.ipynb
│       └── output
│           └── ..
├── Fig3
│   ├── panels_A_B_F
│   │   ├── Fig3_A_B_F_local_k-NN_network.ipynb
│   │   └── output
│   |       └── ..
│   └── panels_C_D
│       ├── Fig3_C_D_cluster_connectivity_and_Jaccard_coeff.ipynb
│       └── output
│           └── ..
├── Fig4
│   └── panel_D
│       └── Please_read.txt
├── Fig5
│   ├── 0.Set_fig5_timestamp.ipynb
│   ├── panel_A
│   │   ├── 1.infected_enrichment
│   │   │   ├── 1.QC_filter_and_impute.ipynb
│   │   │   ├── 2.Batch_selection.ipynb
│   │   │   ├── 3.correlation_filter.ipynb
│   │   │   ├── 4.NOC_processing.ipynb
│   │   │   └── output
│   │   │       └── ..
│   │   ├── 2.control_enrichment
│   │   │   ├── 1.QC_filter_and_impute.ipynb
│   │   │   ├── 2.Batch_selection.ipynb
│   │   │   ├── 3.correlation_filter.ipynb
│   │   │   ├── 4.NOC_processing.ipynb
│   │   │   └── output
│   │   │       └── ..
│   │   └── 3.aligned_umap
│   │       ├── Fig5_A_aligned_umap.ipynb
│   │       └── output
│   │           └── ..
│   ├── panel_B
│   │   ├── Fig5_B_remodeling_score.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_C
│   │   ├── Fig5_C_umap_with_leiden_labels.ipynb
│   │   └── output
│   │       └── ..
│   ├── panel_D
│   │   ├── Fig5_D_trajectory.ipynb
│   │   └── output
│   │       └── ..
│   └── panel_E
│       ├── Fig5_E_Sankey_plot.ipynb
│       └── output
│           └── ..
└─── Supplementary_figures
    ├── Suppl_fig1
    │   ├── Suppl_fig1_marker_expression_in_cell_lines.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig2
    │   └── README.md
    ├── Suppl_fig3
    │   ├── Suppl_fig3_faceted_volcano_plots.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig4
    │   ├── Suppl_fig4_enrichment_heatmap_all_IPs.ipynb
    │   └── output
    │       └── ..
    ├── Suppl_fig5
    │   ├── panel_A
    │   │   ├── Suppl_fig5_A_IP_correlation_vs_interaction_stoi.ipynb
    │   │   └── output
    │   │       └── ..
    │   ├── panel_B
    │   │   ├── Suppl_fig5_B_enrichment_entropy.ipynb
    │   │   └── output
    │   │       └── ..
    │   └── panel_C
    │       ├── Suppl_fig5_C_tenary_plots.ipynb
    │       └── output
    │           └── ..
    ├── Suppl_fig6
    │   ├── panel_B
    │   │   └── README.md
    │   ├── panel_C
    │   │   ├── Suppl_fig6_C_XGBoost_classifier.ipynb
    │   │   ├── environment.yml
    │   │   └── output
    │   │       └── ..
    │   └── panel_F_G
    │       ├── Suppl_fig6_F_G_sankey_and_confusion.ipynb
    │       └── output
    │           └── ..
    ├── Suppl_fig7
    │   └── panel_C
    │       └── Suppl_fig7_C_upset_plot.ipynb
    ├── Suppl_fig8
    │   ├── panel_A
    │   │   ├── Suppl_fig8_A_external_UMAPs.ipynb
    │   │   ├── output
    │   │   │   └── ..
    │   │   └── process_external_spatial_datasets
    │   │       └── ..
    │   └── panel_B
    │       ├── Suppl_fig8_B_score_external_dataset.ipynb
    │       ├── Suppl_fig8_B_score_internal_dataset.ipynb
    │       ├── output
    │       │   └── ..
    │       └── process_internal_variant
    │           └── ..
    ├── Suppl_fig9
    │   └── README.md
    ├── Suppl_fig10
    │   └── README.md
    ├── Suppl_fig11
    │   └── panel_C_D
    │       └── Suppl_fig11_C_D_UMAP_with_virus_proteins.ipynb
    │   
    └── Suppl_fig12
        └── README.md
```


### `scripts/`
These are Python modules that contain the bulk of the code used for data analysis and figure generation. They are used directly by the Jupyter notebooks discussed above. Please note that these scripts are explicitly written for, and specific to this project and/or the OpenCell project. They are __not__ intended to form a stand-alone or general-purpose Python package.



# License
This project is licensed under the BSD 3-Clause license - see the LICENSE file for details.
