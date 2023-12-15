import copy
import sys
from collections import Counter
from pathlib import Path

script_path = Path.cwd().parent.parent.parent / "script"
sys.path.append(str(script_path))
print(Path.cwd())
from utils.knn import get_neighbors


def Jaccard_Coeff_mod(mylist, label_total_counts, norm_degrees_to_def_top_partites=True, min_partite_deg=3, verbose=True):
    '''compute modified Jaccard Coefficient.
    modifications:
    - use normalized degrees to define top partites ( norm = degrees / total number of labels)
    - minimum counts of labels for partite_2
      for a list, count the number of unique elements, and then take the top n counts, return the multiplication product of the top n counts
        the output is a candidate metric for measuring interfacialness
    input:
      mylist: list of labels (with duplicates)
      verbose: print the sorted counts
    output: Jaccard_Coeff, connectivity of the top 2 labels, labels and label_total_counts of the top 2 labels, all labels counts
        
    '''
    counts = Counter(mylist)
    sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
    sorted_counts_prefilter = copy.deepcopy(sorted_counts) # save a copy of the sorted counts prefiltering
    sorted_counts = {k:v for k,v in sorted_counts.items() if v >= min_partite_deg} # filter out labels with counts < min_partite_2
    print(sorted_counts) if verbose else None

    if norm_degrees_to_def_top_partites:
      sorted_counts_norm = {k:v/label_total_counts[k] for k,v in sorted_counts.items()}
      sorted_counts_norm = dict(sorted(sorted_counts_norm.items(), key=lambda item: item[1], reverse=True)) # sort by normalized counts
      print(sorted_counts_norm) if verbose else None
      
      if len(sorted_counts_norm) >= 2:
          key1 = list(sorted_counts_norm.keys())[0] # define partite 1 with normalized counts
          key2 = list(sorted_counts_norm.keys())[1] # define partite 2 with normalized counts
          d1 = sorted_counts[key1]
          d2 = sorted_counts[key2]
          J_coeff = (d1 + d2) / (label_total_counts[key1] + label_total_counts[key2] - (d1 + d2))
          return J_coeff, d1, d2, key1, key2, label_total_counts[key1], label_total_counts[key2], sorted_counts_prefilter
      
      else: # requested higher n than the number of unique annotations in the list
          if len(sorted_counts_norm) == 1: # if only one partite
              key1 = list(sorted_counts_norm.keys())[0] # define partite 1 with normalized counts
              d1 = sorted_counts[key1]
              return 0, d1, "NA", key1, "NA", label_total_counts[key1], "NA", sorted_counts_prefilter
          else:
            return 0, "NA", "NA", "NA", "NA", "NA", "NA", sorted_counts_prefilter

    else: # use raw counts to define top partites
      if len(sorted_counts) >= 2:
          key1 = list(sorted_counts.keys())[0]
          key2 = list(sorted_counts.keys())[1]
          d1 = sorted_counts[key1]
          d2 = sorted_counts[key2]
          J_coeff = (d1 + d2)/ (label_total_counts[key1] + label_total_counts[key2] - (d1 + d2))
          return J_coeff, d1, d2, key1, key2, label_total_counts[key1], label_total_counts[key2], sorted_counts_prefilter
      
      else: # requested higher n than the number of unique annotations in the list
          if len(sorted_counts) == 1: # if only one partite
              key1 = list(sorted_counts.keys())[0]
              d1 = sorted_counts[key1]
              return 0, d1, "NA", key1, "NA", label_total_counts[key1], "NA", sorted_counts_prefilter
          else:
            return 0, "NA", "NA", "NA", "NA", "NA", "NA", sorted_counts_prefilter

def annotate_gene(annot_df=None, gene_name="AP2B1", gene_name_col="Gene_name_canonical", annot_col="Graph-based_localization_annotation"):
    return annot_df[annot_df[gene_name_col] == gene_name][annot_col].to_list()

def neighbor_df_to_dict(df):
    '''Convert neighbor df to dictionary, keys being the neighbor names, values being the connectivity
        collapse the duplicated neighbors, and take the highest connectivity
    '''
    nei_dict = {}
    for idx, row in df.iterrows():
        for name,value in row.iteritems():
            if name in nei_dict:
                if value > nei_dict[name]: # update the value if the new value is higher
                    nei_dict[name] = round(value,4)
            else:
                nei_dict[name] = round(value,4)
    return nei_dict

def gene_neighbor_annots(gene_name="VPS11", adata=None, annot_df= None, gene_name_col="Gene_name_canonical", 
                         annot_col="Graph-based_localization_annotation", top_n_neighbors=None):
    '''for a given gene, extract the 1-st degree neighbors, annotate them
    input:
        gene_name: the gene name to query
        adata: the AnnData object containing the neighbors
        gene_name_col: the column name in the metadata table that contains the gene names
        annot_col: the column name in the metadata table that contains the annotations
    output:
        list of neighbors
        list of neighbor annotations
    '''
    nei_list_annot = []
    nei_list = []

    if top_n_neighbors is None:
        neighbors_result = get_neighbors(adata, gene_name, gene_name_col)
    else:
        neighbors_result = get_neighbors(adata, gene_name, gene_name_col, keep_top_n=top_n_neighbors)

    for nei in neighbors_result.columns:
        annot_list = annotate_gene(gene_name=nei, annot_df=annot_df, gene_name_col=gene_name_col, annot_col=annot_col)
        for i in annot_list:
            if i not in (None, "NaN", ""): # take the first annotation that is not empty
                nei_list.append(nei)
                nei_list_annot.append(i)
                break
    return nei_list, nei_list_annot

# test
if __name__ == "__main__":
    # test
    test_list = ['trans-Golgi', 'plasma_membrane', 'trans-Golgi',  'plasma_membrane', 'recycling_endosome', 'plasma_membrane', 'recycling_endosome',
        'actin_cytoskeleton', 'plasma_membrane', 'plasma_membrane', 'lysosome', 'lysosome',  'plasma_membrane',
        'trans-Golgi',  'trans-Golgi', 'trans-Golgi', 'plasma_membrane', 'plasma_membrane', 'plasma_membrane', 'actin_cytoskeleton', 'early_endosome', 'lysosome',
        'early_endosome', 'plasma_membrane', 'cytosol']
    label_total_counts = {'nucleus': 1525,
                            'Golgi': 192,
                            'cytosol': 1800,
                            'trans-Golgi': 107,
                            'early_endosome': 208,
                            'stress_granule': 253,
                            'mixed': 884,
                            'centrosome': 54,
                            'ER': 749,
                            'recycling_endosome': 84,
                            'plasma_membrane': 648,
                            'mitochondrion': 863,
                            'lysosome': 165,
                            'actin_cytoskeleton': 283,
                            'translation': 138,
                            'nucleolus': 208,
                            '14-3-3_scaffold': 129,
                            'peroxisome': 39,
                            'p-body': 17,
                            'ERGIC': 55,
                            'proteasome': 71,
                            'spindle': 69}
    Jaccard_Coeff_mod(test_list, label_total_counts, verbose=True)