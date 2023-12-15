import numpy as np

def find_all_indices(mylist, value):
    ''' for attaching labels to the dataframes'''
    indices = []
    for i, x in enumerate(mylist):
        if x == value:
            indices.append(i)
    return indices


def attach_annotations(from_df, to_df, anno_col="v1_clusters_seed_2078", from_on="Protein IDs", to_on="Protein IDs"):
    ''' for attaching labels to the dataframes'''
    identifier_list = from_df[from_on].to_list()
    new_col_data = []
    for i in to_df[(to_on)]:
        indices = find_all_indices(identifier_list, i)
        if len(indices) == 0:
            new_col_data.append(np.nan)  # take the first one 
        else:
            new_col_data.append(from_df[anno_col][indices[0]])  # take the first one 
    return new_col_data