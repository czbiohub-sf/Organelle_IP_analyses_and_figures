# Global organelle profiling defines a sub-cellular map of the human proteome

This repository serves as a comprehensive resource for exploring and understanding our manuscript, "Global Organelle Profiling Defines a Sub-Cellular Map of the Human Proteome". Inside, you'll find detailed Jupyter notebooks and scripts that were pivotal in our data analysis process and in generating the figures.

Our aim is to provide an in-depth, transparent view into our research methods and findings. Dive into our notebooks to see how we transformed raw data into meaningful insights, or explore our scripts to understand the technical underpinnings of our figure generation. We hope this repository will be a useful tool in your own research and learning journey.
## What's in this repo
### `data/`
This directory contains various external and processed datasets required to make the figures. Note that some of these datasets are from external sources; these are found in the `data/external/` subdirectory. The remaining datasets are all original datasets generated by or derived from this project. Note that some datasets, including the MaxQuant output is too large to host on GitHub. These datasets are available [on FigShare](https://figshare.com/). 


### `notebooks/`
These are Jupyter notebooks that document how the figures were generated using the Python modules in `scripts/`. The notebooks used for each figure panel are [specified below](#where-to-find-the-code-and-data-used-to-generate-each-figure). These notebooks are the primary documentation for how the scripts in `scripts/` were used for analysis and figure generation. The notebooks should be executed in the order specified below, as the output of one notebook is used as input for the next.

```
notebooks
|
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
│       ├── Fig1_L_heatmap.ipynb
│       └── output
|           └── ..
├── Fig2
│   ├── panel_B
│   │   ├── Fig2_B_heatmap.ipynb
│   │   └── output
|   |       └── ..
│   └── panel_D
│       ├── Fig2_D_umap.ipynb
│       └── output
|           └── ..
├── Fig3
│   ├── panels_A_B_F
│   │   ├── Fig3_A_B_F_local_k-NN_network.ipynb
│   │   └── output
|   |       └── ..
│   └── panels_C_D
│      ├── Fig3_C_D_cluster_connectivity.ipynb
│      └── output
|          └── ..
├── Fig4
│   └── panel_D
│       └── Please_read.txt
├── Fig5
│   ├── panel_A
│   │   ├── infected_enrichment
|   |   |   ├── 1.QC_filter_and_impute.ipynb
|   |   |   ├── 2.Batch_selection.ipynb
|   |   |   ├── 3.correlation_filter.ipynb
|   |   |   ├── 4.NOC_processing.ipynb
|   |   |   └── output
|   |   |       └── ..
│   │   ├── control_enrichment
|   |   |   ├── 1.QC_filter_and_impute.ipynb
|   |   |   ├── 2.Batch_selection.ipynb
|   |   |   ├── 3.correlation_filter.ipynb
|   |   |   ├── 4.NOC_processing.ipynb
|   |   |   └── output
|   |   |       └── ..
│   │   └── aligned_umap
│   │       ├── Fig5_A_aligned_umap.ipynb
│   │       └── output
|   |           └── ..
│   ├── panel_B
│   │   ├── Fig5_B_remodeling_score.ipynb
│   │   └── output
|   |       └── ..
│   ├── panel_C
│   │   ├── Fig5_C_umap_with_leiden_labels.ipynb
│   │   └── output
|   |       └── ..
│   ├── panel_D
│   │   ├── Fig5_D_trajectory.ipynb
│   │   └── output
|   |       └── ..
│   └── panel_E
│   │   ├── Fig5_E_Sankey_plot.ipynb
│   │   └── output
|   |       └── ..

```


### `scripts/`
These are Python modules that contain the bulk of the code used for data analysis and figure generation. They are used directly by the Jupyter notebooks discussed above. Please note that these scripts are explicitly written for, and specific to this projection and/or the OpenCell project. They are __not__ intended to form a stand-alone or general-purpose Python package.



# License
Chan Zuckerberg Biohub Software License

This software license is the 2-clause BSD license plus a third clause
that prohibits redistribution and use for commercial purposes without further
permission.

Copyright © 2023. Chan Zuckerberg Biohub.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1.	Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2.	Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3.	Redistributions and use for commercial purposes are not permitted without
the Chan Zuckerberg Biohub's written permission. For purposes of this license,
commercial purposes are the incorporation of the Chan Zuckerberg Biohub's
software into anything for which you will charge fees or other compensation or
use of the software to perform a commercial service for a third party.
Contact ip@czbiohub.org for commercial licensing opportunities.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
