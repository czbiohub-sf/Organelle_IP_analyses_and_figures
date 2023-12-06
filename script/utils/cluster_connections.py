def clusterwise_connection(df_annot,nei_df, annot_col_name = "Graph-based_localization_annotation",
                            cluster1_name = "early_endosome", cluster2_name="plasma_membrane", normalize_conn = False,
                            restrict_to_interfacial = False, interfacial_proteins = None):
    """ 
    Computes the number of connections between two clusters
    normalize_conn: boolean, if True, normalize the number of connections by the number of neighbors
    """
    if restrict_to_interfacial and interfacial_proteins is None:
        raise ValueError("interfacial_proteins must be provided if restrict_to_interfacial is True")

    bool_idx1 = df_annot[annot_col_name] == cluster1_name
    bool_idx2 = df_annot[annot_col_name] == cluster2_name

    c1_genes = df_annot[bool_idx1]["Gene_name_canonical"].tolist()
    c2_genes = df_annot[bool_idx2]["Gene_name_canonical"].tolist()

    c1_to_c2_count = 0
    c2_to_c1_count = 0

    # count the number of connections from cluster 1 ->  cluster 2
    # for each gene in cluster 1 count neighbors in cluster 2
    for g in c1_genes:

        # skip current gene if interfacial required and it's not interfacial
        if restrict_to_interfacial and g not in interfacial_proteins:
            continue

        neighbors = get_neighbors_from_df(nei_df, g).iloc[0,2] # get the firs row, second column
        if isinstance(neighbors, str):
            neighbors = eval(neighbors) # convert string to dictionary
        
        for nei in neighbors.keys(): # for each neighbor, check if it is in cluster 2
            if nei in c2_genes:
                if normalize_conn:
                    c1_to_c2_count += 1/len(neighbors)
                else:
                    c1_to_c2_count += 1

    for g in c2_genes:

        # skip current gene if interfacial required and it's not interfacial
        if restrict_to_interfacial and g not in interfacial_proteins:
            continue

        neighbors = get_neighbors_from_df(nei_df, g).iloc[0,2] # get the firs row, second column
        if isinstance(neighbors, str):
            neighbors = eval(neighbors) # convert string to dictionary
        
        for nei in neighbors.keys(): # for each neighbor, check if it is in cluster 2
            if nei in c1_genes:
                if normalize_conn:
                    c2_to_c1_count += 1/len(neighbors)
                else:
                    c2_to_c1_count += 1
    return c1_to_c2_count, c2_to_c1_count

def get_neighbors_from_df(df, target_gene, keep_top_n=None):
    bool_idx = df["Gene_names_canonical"] == target_gene
    df = df[bool_idx]
    return df