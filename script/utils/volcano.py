import re
import pandas as pd
from utils.label_processing import attach_annotations

def custom_sort(item):
    try :
         return (item.endswith("[p]"), item)
    except AttributeError:
        print(item)
   

def load_volcano_data(csv_path, label_path):
    volcano_df = pd.read_csv(csv_path, header=[0,1], index_col=0)

    # append annotations
    # attach canonical gene names
    labels_csv = label_path
    lookup_table = pd.read_csv(labels_csv)
    to_df = volcano_df["metadata"].copy()
    list_of_cols_to_add = reversed(["Gene_name_canonical", "consensus_graph_annnotation"])
    for c in list_of_cols_to_add:
        new_col_data = attach_annotations(from_df=lookup_table, to_df=to_df, anno_col=c , from_on="Majority protein IDs", to_on="Majority protein IDs")
        volcano_df[("metadata", c)] = new_col_data

    pulldowns = []
    for i in volcano_df.columns:
        if i[0] != "metadata":
            pulldowns.append(i[0])

    #find duplicated gene names and append the first majority protein ID to the gene name
    dup_bool = volcano_df[('metadata','Gene_name_canonical')].duplicated(keep=False)
    volcano_df.loc[dup_bool, ('metadata',"Gene_name_canonical")] = volcano_df.loc[dup_bool, ('metadata',"Gene_name_canonical")] + " (" + volcano_df.loc[dup_bool, ('metadata', "Majority protein IDs")].str.split(";").str[0] + ")"

    #Gene_name_canonical = sorted(volcano_df[('metadata','Gene_name_canonical')].unique(), key=custom_sort)
    Gene_name_canonical = volcano_df[('metadata','Gene_name_canonical')]
    
    # rename columns
    winning_pulldowns = ["12-YWHAQ","12-ACTB","09-ATG101","17-MAP1LC3B","11-CEP350","10-VPS35","07-CLTA","11-EEA1","05-NCLN",
    "06-CCDC47","12-RTN4","12-SEC61B","02-COPE","03-SEC23A","11-SEC31A","14-RAB1A","07-COG8","11-GPR107",
    "13-GOLGA2","03-HSPA1B","09-HSP90AA1","12-LAMP1","14-RAB7A","12-TOMM20","02-DCP1A","05-EDC4","09-PEX3",
    "05-CAV1","17-ATP1B3","17-SLC30A2","09-PSMB7","13-RAB14","14-RAB11A","17-RPL36","17-CAPRIN1","17-G3BP1"]
    assert all([i in pulldowns for i in winning_pulldowns]), "Not all winning pulldowns are in the data" 

    compartments = ["14-3-3","actin","autophagosome","autophagosome","centrosome","endo/lysosome","endosome","endosome","ER",
    "ER","ER","ER","ER/Golgi","ER/Golgi","ER/Golgi","ER/Golgi","Golgi","Golgi","Golgi","HSP70 chaperone",
    "HSP90 chaperone","lysosome","lysosome","mitochondria","p-body","p-body","peroxisome","plasma membrane",
    "plasma membrane","plasma membrane","proteasome","recycling endosome","recycling endosome","ribosome",
    "stress granule","stress granule"]


    highlights = [["14-3-3_scaffold"],
                        ["actin_cytoskeleton"],
                        ["ER", "ERGIC"],
                        ["lysosome"],
                        ["centrosome"],
                        ["early_endosome"],
                        ["early_endosome", "trans-Golgi"],
                        ["early_endosome"],
                        ["ER"],
                        ["ER"],
                        ["ER"],
                        ["ER"],
                        ["ERGIC", "Golgi", "ER"],
                        ["ERGIC", "Golgi", "ER"],
                        ["ERGIC", "Golgi"],
                        ["ERGIC", "Golgi", "ER"],
                        ["Golgi"],
                        ["Golgi"],
                        ["Golgi"],
                        ["cytosol"],
                        ["cytosol", "mitochondrion"],
                        ["lysosome"],
                        ["lysosome"],
                        ["mitochondrion"],
                        ["p-body"],
                        ["p-body"],
                        ["peroxisome"],
                        ["plasma_membrane", "actin_cytoskeleton"],
                        ["plasma_membrane", "actin_cytoskeleton"],
                        ["plasma_membrane", "actin_cytoskeleton"],
                        ["proteasome"],
                        ["early_endosome", "trans-Golgi"],
                        ["recycling_endosome", "trans-Golgi"],
                        ["translation"],
                        ["stress_granule"],
                        ["stress_granule"]]

    name_with_comparts = [f"{p} ({c})" for p, c in zip(winning_pulldowns, compartments)]
    name_mapping = dict(zip(winning_pulldowns, name_with_comparts))

    # drop non-metadata columns where it is not a winning pulldown
    for col in volcano_df.columns:
        if col[0] != "metadata":
            if col[0] not in winning_pulldowns:
                #print(f"dropping {col}")
                volcano_df.drop(col, axis=1, inplace=True)

    # rename the columns (adding the compartment name)
    volcano_df.rename(columns=name_mapping, level = 0, inplace=True)
    
    # remove experiment number from the column names
    volcano_df.columns = pd.MultiIndex.from_tuples([(re.sub(r'^\d+-', '', col[0]), col[1]) for col in volcano_df.columns])
    # update the pulldown names
    name_with_comparts = [f"{p.split('-')[1]} ({c})" for p, c in zip(winning_pulldowns, compartments)]
            
    #volcano_df.columns = [(col[0].split("-")[1], col[1]) if len(col[0].split("-"))>1 else col for col in volcano_df.columns]

    pval_df = volcano_df.xs("pvals", axis=1, level=1, drop_level=True)
    pval_df = pval_df[name_with_comparts]
    pval_df["Gene_name_canonical"] = volcano_df[('metadata', 'Gene_name_canonical')]
    pval_df.set_index("Gene_name_canonical", inplace=True)
    
    enrich_df = volcano_df.xs("enrichment", axis=1, level=1, drop_level=True)
    enrich_df = enrich_df[name_with_comparts]
    enrich_df["Gene_name_canonical"] = volcano_df[('metadata', 'Gene_name_canonical')]
    enrich_df.set_index("Gene_name_canonical", inplace=True)
    
    return volcano_df, name_with_comparts, Gene_name_canonical, highlights
