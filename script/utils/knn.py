import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_neighbors(_adata, target_gene, gene_name_col = "Gene_name_canonical", keep_top_n=None):
    '''helper function to extract neighbors and their connectivity
    Input:
        adata: an anndata object that contains precomputed connectivities
        target_gene: a string that is the name of the gene of interest
    Output:
        a dataframe that contains the connectivities of the target gene to its neighbors
        Note that this dataframe may contain multiple rows if the target gene duplicated in the dataset (e.g. isoforms)
    '''
    connectivies = np.array(_adata.obsp["connectivities"].todense())
    gene_names = _adata.obs[gene_name_col]

    # subset the connectivity matrix to only include the query gene
    bool_idx = gene_names == target_gene
    connectivity_arr = np.array(connectivies)
    gene_arr = connectivity_arr[bool_idx, :]

    # get the gene names of the neighbors
    bool_array = (gene_arr == 0).all(axis=0)
    indices = np.where(bool_array == False)[0]
    neighbors = gene_names.iloc[indices]

    # subset the connectivity matrix to only include the neighbors of the query gene
    gene_arr = gene_arr[:,~(gene_arr == 0).all(axis=0) ]

    df = pd.DataFrame(gene_arr, columns=neighbors, index =[target_gene]*gene_arr.shape[0])

    #collapse isoforms
    df = pd.DataFrame(df.max()).T
    df.index = [target_gene]

    #eliminate self-connections (caused by isoforms)
    df = df.loc[:,~df.columns.isin(df.index)]


    if keep_top_n is not None:
        df = df.T.sort_values(by=target_gene, ascending=False).head(keep_top_n).T

    return df

def get_neighbors_of_neighbors(_adata, _neighbors, keep_top_n=None):
    '''helper function to extract neighbors and their connectivity
    Input:
        adata: an anndata object that contains precomputed connectivities
        neighbors: a list of neighbors
    Output:
        a dataframe that contains the connectivities of the target gene to its neighbors
    '''
    connectivies = np.array(_adata.obsp["connectivities"].todense())
    gene_names = _adata.obs["Gene_name_canonical"]

    result = {} # setup the output variable

    for target_gene in _neighbors:
        # subset the connectivity matrix to only include the query gene
        bool_idx = gene_names == target_gene
        connectivity_arr = np.array(connectivies)
        gene_arr = connectivity_arr[bool_idx, :]

        # get the gene names of the neighbors
        bool_array = (gene_arr == 0).all(axis=0)
        indices = np.where(bool_array == False)[0]
        _neighbors = gene_names.iloc[indices]

        # subset the connectivity matrix to only include the neighbors of the query gene
        gene_arr = gene_arr[:,~(gene_arr == 0).all(axis=0) ]

        df = pd.DataFrame(gene_arr, columns=_neighbors, index =[target_gene]*gene_arr.shape[0])

        #collapse isoforms
        df = pd.DataFrame(df.max()).T
        df.index = [target_gene]

        if keep_top_n is not None:
            df = df.T.sort_values(by=target_gene, ascending=False).head(keep_top_n).T

        for idx, val in df.loc[target_gene].items():
            if idx != target_gene:
                result[frozenset([target_gene, idx])] = val

    return result


def remove_extra_degrees(result_dict, keep_top_n, query_gene=None):
    # get node with more connections than allowed
    lst = []
    for i in result_dict:
        lst.append(list(i)[0])
        lst.append(list(i)[1])
    c = Counter(lst)
    c = {k: v for k, v in c.items() if v > keep_top_n}
    c
    # trim the graph
    for i in c.keys():
        keys_with_protein = [key for key in result_dict.keys() if i in list(key)]
        keys_with_protein_sorted = sorted(keys_with_protein, key=lambda k: result_dict[k], reverse=True)
        # Select the top n pairs
        top_keys = keys_with_protein_sorted[:keep_top_n]            
        keys_to_remove = [key for key in keys_with_protein if key not in top_keys]
        query_gene_flag = False
        for key in keys_to_remove:
            if query_gene is not None and query_gene in key:
                query_gene_flag = True
                continue
            del result_dict[key]
        # in the case that the query gene is in the list of keys to remove, we need to remove the next best pair
        if query_gene_flag:
            for key in reversed(keys_with_protein_sorted):
                if not query_gene in key and key in result_dict.keys():
                    del result_dict[key] 
                    break
    return result_dict

def prune_single_connection_nodes(result_dict, keep_1st_order_neighbors=True, nei_df=None, gene=None):
    '''remove edges that are connected to only one node
    Input:
        result_dict: a dictionary of edges and their connectivities
    Output:
        a (pruned) dictionary of edges and their connectivities
    '''
    all_edges = {}
    gene_edge_count = {}
    # tally edges
    for key, value in result_dict.items():
        g1 = list(key)[0]
        g2 = list(key)[1]

        all_edges[frozenset([g1, g2])] = value

        if g1 not in gene_edge_count:
            gene_edge_count[g1] = 0
        if g2 not in gene_edge_count:
            gene_edge_count[g2] = 0

        gene_edge_count[g1] += 1
        gene_edge_count[g2] += 1
    # process neighbors
    allow_list = []
    if keep_1st_order_neighbors and nei_df is not None and gene is not None:
        first_order_neighbors = nei_df.columns.to_list()
        allow_list = [gene] + first_order_neighbors
    # prune
    pruned_dict = {}
    for key, value in result_dict.items():
        g1 = list(key)[0]
        g2 = list(key)[1]

        if gene_edge_count[g1] > 1 and gene_edge_count[g2] > 1: #  keep the edge if both nodes have more than one connection,
            pruned_dict[frozenset([g1, g2])] = value
        elif g1 in allow_list and g2 in allow_list: # keep edges that are connecting query to first order neighbors
            pruned_dict[frozenset([g1, g2])] = value
            
    return pruned_dict

def convert_color_to_rgba(color,alpha=1.0):
    # Convert named colors and hex to rgba
    if isinstance(color, str):
        try:
            # Convert from named color or hex to RGB
            rgb_float = mcolors.to_rgb(color)
            # Scale RGB to 0-255 range
            rgba_255 = tuple(int(val * 255) for val in rgb_float) + (alpha,)
            # Format as a string for Plotly
            return f'rgba{rgba_255}'

        except ValueError:
            return "Invalid color name or hex value"
    else:
        return "Input must be a string"
    
def prepare_plotly_network_graph(_G, annot_dict, node_color_by, category_colors):

    edge_x = []
    edge_y = []
    for edge in _G.edges():
        x0, y0 = _G.nodes[edge[0]]['pos']
        x1, y1 = _G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False)

    node_x = []
    node_y = []
    for node in _G.nodes():
        x, y = _G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    if node_color_by == "connections":
        node_adjacencies = []
        node_annot = []
        node_text = []
        hover_info = []
        for node, adjacencies in enumerate(_G.adjacency()):
            node_name = adjacencies[0]
            node_annot.append(annot_dict.get(node_name, ''))
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(node_name)
            hover_info.append(f"{annot_dict.get(node_name, '')}")
            
        # create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            #hovertemplate='%{text}<br><br># of connections: %{marker.color}',
            hovertemplate='%{text}<br># of connections: %{marker.color}<br>%{hovertext}',
            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2),
                showlegend=False)
        node_trace.marker.color = node_adjacencies

    category_color_map = {} # need this variable for function return
    if node_color_by == "compartment":

        node_adjacencies = []
        node_annot = []
        node_text = []
        hover_info = []
        for node, adjacencies in enumerate(_G.adjacency()):
            node_name = adjacencies[0]
            node_annot.append(annot_dict.get(node_name, ''))
            node_adjacencies.append(len(adjacencies[1]))
            node_text.append(node_name)
            hover_info.append(len(adjacencies[1]))

        # make a mapping from compartment to color
        for i, category in enumerate(list(set(node_annot))):
            category_color_map[category] = category_colors[i % len(category_colors)]

        # get node colors
        node_colors = []
        for node, adjacencies in enumerate(_G.adjacency()):
            node_name = adjacencies[0]
            node_colors.append(category_color_map[annot_dict.get(node_name, '')])

        # create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            #hovertemplate='%{text}<br><br># of connections: %{marker.color}',
            hovertemplate='%{text}<br># of connections: %{hovertext}<br>%{customdata}',
            customdata = node_annot,
            marker=dict(
                size=10,
                line_width=2),
            showlegend=False)
        node_trace.marker.color = node_colors   

    node_trace.text = node_text
    node_trace.textposition = 'top center'
    node_trace.hovertext = hover_info

    return node_trace, edge_trace, category_color_map