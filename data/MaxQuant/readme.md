The MaxQuant output file "proteinGroups.txt" is over 100MB and cannot be uploaded to GitHub. It is available through the PRIDE repository under the identifier PXD046440. You can access it via the following FTP link: ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2023/12/PXD046440/proteinGroups.txt.  
  
"proteinGroups.txt" is needed for calculating enrichment values.  

you can skip enrichment calculation by setting `USE_FROZEN = True` in `notebooks/0.Set_timestamp.ipynb` and executing the notebook. Then proceed to `notebooks/Fig1/` (skipping `notebooks/enrichment/`). Precomputed enrichment values in 
 `notebooks/enrichment/output/enrichment_and_volcano_tables/2023-10-21-imp5-for-figures_enrichment_table_NOC_prop.csv` will be used.
