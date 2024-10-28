import time
import pandas as pd
import requests


def upload_foreground(gene_list):
    gene_list = [i.split(";")[0] for i in gene_list] # enrichr gene lists can't have ";" in them
    gene_list = [i.split("[")[0] for i in gene_list]

    description = "foreground gene list"
    base_url = "https://maayanlab.cloud/speedrichr"
    res = requests.post(
        base_url+'/api/addList',
        files=dict(
            list=(None, '\n'.join(gene_list)),
            description=(None, description),
        )
    )
    if res.ok:
        response = res.json()
        return response
    else:
        print('Error uploading gene list')
        print(res.text)
        return None
    
def enrichment_analysis(bg, fg, library):
    '''run enrichment analysis on a foreground list and a background list
    bg: response of background list uploaded to enrichr
    fg: response of foreground list uploaded to enrichr
    library: enrichment library
    '''
    base_url = "https://maayanlab.cloud/speedrichr"
    res = requests.post(
        base_url+'/api/backgroundenrich',
        data=dict(
            userListId = fg['userListId'],
            backgroundid = bg['backgroundid'],
            backgroundType = f"{library}",
        )
    )
    if res.ok:
        #response = res.json()
        # convert json format to a dataframe
        df = pd.DataFrame(res.json()[library], columns = ["Rank", "Term name", "P-value", 
                                                                    "Odds ratio", "Combined score", "Overlapping genes", 
                                                                    "Adjusted p-value", "Old p-value", "Old adjusted p-value"])
        return df
    else:
        print('Error running enrichment analysis')
        print(res.text)
        return None
    
    
def run_enrichr(cluster_col, df, background_response, verbose=True):
    '''
    cluster_col: column name of the cluster labels
    df: dataframe with two columns: Gene_name_canonical and cluster_col
    background_response: response of background list uploaded to enrichr

    return: a dictionary with keys being the cluster labels and values being the enrichment results
    '''

    # initialize dataframes to store results (by databases)
    df_GO_CC = pd.DataFrame()
    df_JCOMP = pd.DataFrame()

    # initialize a dictionary to store the results (by clusters)
    enrichr_results = {}

    clusters = df[cluster_col].unique()
    print(f"Performing enrichment analysis for:") if verbose else None

    for clu in clusters:
        print(f"- {clu}", end="", flush=True) if verbose else None
        # get gene symbols
        genes = df[df[cluster_col] == clu]['Gene_name_canonical'].values
        print(f" ({len(genes)} genes) ... ", end="", flush=True) if verbose else None
        
        # upload foreground
        foreground_response = upload_foreground(genes)

        # GO CC
        _df_GO_CC = enrichment_analysis(background_response, foreground_response, "GO_Cellular_Component_2023")
        _df_GO_CC.insert(0, cluster_col, clu)
        _df_GO_CC.insert(0, "Database", "GO_CC")
        time.sleep(0.6) # pause for 1 second to avoid overloading the server

        # Jesen COMPARTMENTS
        _df_JCOMP = enrichment_analysis(background_response, foreground_response, "Jensen_COMPARTMENTS")
        _df_JCOMP.insert(0, cluster_col, clu)
        _df_JCOMP.insert(0, "Database", "Jensen_COMPARTMENTS")
        time.sleep(0.6) # pause for 1 second to avoid overloading the server

        # filter for p-value < 0.01
        _df_GO_CC = _df_GO_CC[_df_GO_CC["P-value"] < 0.01]
        _df_JCOMP = _df_JCOMP[_df_JCOMP["P-value"] < 0.01]

        # drop unnecessary columns
        _df_GO_CC.drop(columns = ["Rank", "Combined score", "Adjusted p-value", "Old p-value", "Old adjusted p-value"], inplace = True)
        _df_JCOMP.drop(columns = ["Rank", "Combined score", "Adjusted p-value", "Old p-value", "Old adjusted p-value"], inplace = True)

        # concatenate the results
        df_GO_CC = pd.concat([df_GO_CC, _df_GO_CC])
        df_JCOMP = pd.concat([df_JCOMP, _df_JCOMP])

        # save a copy of the results to dictioary
        enrichr_results[clu] = {}
        enrichr_results[clu]["GO_CC"] = _df_GO_CC
        enrichr_results[clu]["JCOMP"] = _df_JCOMP

        print("done", flush=True) if verbose else None

    return enrichr_results, df_GO_CC, df_JCOMP

def remove_clu_num_and_map_names(df, col1 ,col2):
    '''
    Consolidate and standardize compartment annotation names by grouping detailed components under their corresponding high-level compartment names.
    this step will be fully automated in the new python package we are developing for Organelle IP analysis, 
    and will be available at https://github.com/czbiohub-sf/orgipy
    '''
    df["cluster_annot_uninf"] = df[col1].map(lambda x: x.split(":")[1])
    df["cluster_annot_inf"] = df[col2].map(lambda x: x.split(":")[1])

    df["cluster_annot_uninf"].replace("Nucleolus", "nucleus", inplace=True)
    df["cluster_annot_inf"].replace("Nucleolus", "nucleus", inplace=True)
    df["cluster_annot_uninf"].replace("Nuclear Lumen", "nucleus", inplace=True)
    df["cluster_annot_inf"].replace("Nuclear Lumen", "nucleus", inplace=True)
    df["cluster_annot_uninf"].replace("Nucleus", "nucleus", inplace=True)
    df["cluster_annot_inf"].replace("Nucleus", "nucleus", inplace=True)
    df["cluster_annot_inf"].replace("Cilium","nucleus", inplace=True)
    
    df["cluster_annot_uninf"].replace("cullin-RING Ubiquitin Ligase Complex", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("cullin-RING Ubiquitin Ligase Complex", "cytosol", inplace=True)
    df["cluster_annot_uninf"].replace("Cytoplasmic Vesicle Lumen", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("Cytoplasmic Vesicle Lumen", "cytosol", inplace=True)
    df["cluster_annot_uninf"].replace("Secretory Granule Lumen", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("Secretory Granule Lumen", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("Ficolin-1-Rich Granule Lumen", "cytosol", inplace=True)
    df["cluster_annot_uninf"].replace("Ficolin-1-Rich Granule Lumen", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("Protein Serine/Threonine Phosphatase Complex", "cytosol", inplace=True)
    df["cluster_annot_uninf"].replace("Protein Serine/Threonine Phosphatase Complex", "cytosol", inplace=True)
    df["cluster_annot_inf"].replace("RNA Polymerase III Complex", "cytosol", inplace=True)
    df["cluster_annot_uninf"].replace("RNA Polymerase III Complex", "cytosol", inplace=True)

    df["cluster_annot_uninf"].replace("Endoplasmic Reticulum Membrane", "ER", inplace=True)
    df["cluster_annot_inf"].replace("Endoplasmic Reticulum Membrane", "ER", inplace=True)
    df["cluster_annot_uninf"].replace("Golgi Membrane", "ERGIC/Golgi", inplace=True)
    df["cluster_annot_inf"].replace("Golgi Membrane", "ERGIC/Golgi", inplace=True)

    df["cluster_annot_uninf"].replace("Mitochondrial Membrane", "mitochondrion", inplace=True)
    df["cluster_annot_inf"].replace("Mitochondrial Membrane", "mitochondrion", inplace=True)

    df["cluster_annot_inf"].replace("Cell-Cell Junction", "plasma memb. & actin", inplace=True)
    df["cluster_annot_uninf"].replace("Cell-Cell Junction", "plasma memb. & actin", inplace=True)
    df["cluster_annot_uninf"].replace("Cell-Substrate Junction", "plasma memb. & actin", inplace=True)
    df["cluster_annot_inf"].replace("Basolateral Plasma Membrane", "plasma memb. & actin", inplace=True)
    df["cluster_annot_uninf"].replace("Basolateral Plasma Membrane", "plasma memb. & actin", inplace=True)
    df["cluster_annot_uninf"].replace("Actin Cytoskeleton", "plasma memb. & actin", inplace=True)
    df["cluster_annot_inf"].replace("Actin Cytoskeleton", "plasma memb. & actin", inplace=True)
    df["cluster_annot_inf"].replace("Cytoskeleton", "plasma memb. & actin", inplace=True)
    df["cluster_annot_uninf"].replace("Cytoskeleton", "plasma memb. & actin", inplace=True)

    df["cluster_annot_uninf"].replace("COPI Vesicle Coat", "COPI vesicle", inplace=True)
    df["cluster_annot_inf"].replace("COPI Vesicle Coat", "COPI vesicle", inplace=True)

    df["cluster_annot_uninf"].replace("Lysosome", "endo-lysosome & trans-Golgi", inplace=True)
    df["cluster_annot_inf"].replace("Lysosome", "endo-lysosome & trans-Golgi", inplace=True)

    df["cluster_annot_uninf"].replace("Ribosome", "translation/RNA granules", inplace=True)
    df["cluster_annot_inf"].replace("Ribosome", "translation/RNA granules", inplace=True)

    df["cluster_annot_uninf"].replace("Peroxisome", "peroxisome", inplace=True)
    df["cluster_annot_inf"].replace("Peroxisome", "peroxisome", inplace=True)

    df["cluster_annot_uninf"].replace("Mitotic Spindle Microtubule", "centrosome", inplace=True)
    df["cluster_annot_inf"].replace("Mitotic Spindle Microtubule", "centrosome", inplace=True)
    df["cluster_annot_inf"].replace("Protein Phosphatase Type 2A Complex", "centrosome", inplace=True)
    df["cluster_annot_uninf"].replace("Protein Phosphatase Type 2A Complex", "centrosome", inplace=True)

    df["cluster_annot_uninf"].replace("NA", "cytosol", inplace=True) # unlabelled is signaling scaffold, which take the same color as cytosol

    df["cluster_annot_uninf"].replace("Cilium", "unlabelled", inplace=True)
    
    df["cluster_annot_inf"].replace("NA", "unlabelled", inplace=True)
    df["cluster_annot_uninf"].fillna("unlabelled", inplace=True)
    df["cluster_annot_inf"].fillna("unlabelled", inplace=True)
    return df