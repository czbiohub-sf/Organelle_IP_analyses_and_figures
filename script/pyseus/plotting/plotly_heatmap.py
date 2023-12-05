import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib
from numbers import Number
import numpy as np
import pandas as pd
from pyseus import basic_processing as pys
import plotly.offline
from plotly import graph_objs as go
import seaborn as sns
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans
import time
import pdb


def subtract_prey_median(imputed_df, features, metadata, mad_mod=True, mad_factor=1):
    """As an option to visualize clustering so that each intensity
    is subtracted by the prey group median, this function
    alters the base dataframe with the transformation"""

    transformed = imputed_df[features].copy()



    transformed = transformed.T

    # Get a list of the columns (baits or preys)
    cols = list(transformed)

    # go through each prey (now in columns) and subtract median
    for col in cols:
        transformed[col] = transformed[col] - transformed[col].median()
        if mad_mod:
            mad = transformed[col].mad() * mad_factor
            transformed[col] = transformed[col].apply(lambda x: x if x > mad else 0)

    transformed = transformed.T
    # if mad_mod:
    #     t_cols = list(transformed)
    #     for col in t_cols:
    #         mad = transformed[col].mad() * mad_factor
    #         transformed[col] = transformed[col].apply(lambda x: x if x > mad else 0)

    # transpose back to original shape and add the info columns again
    for col in metadata:
        transformed[col] = imputed_df[col]

    return transformed


def prey_kmeans(imputed_df, k=20, method='single', ordering=True, verbose=True):
    """Create a large k clustered groups, and sort them by average group intensity.
    Return a list of Protein IDs after the sort

    rtype: dendro_side plotly figurefactory
    rtype: dendro_leaves list"""

    if verbose:
        print("Generating prey hierarchies and dendrogram...")
        start_time = time.time()

    # Create a median_df, taking median of all replicates
    median_df = pys.median_replicates(imputed_df, save_info=True, col_str='')
    median_df.drop(columns=['Protein names', 'Gene names',
    'Majority protein IDs'], inplace=True)

    # Protein IDs will be the reference to retrieve the correct order of preys
    median_df.set_index('Protein IDs', inplace=True)

    # Conduct K means clustering
    kmeans_model = KMeans(n_clusters=k).fit(median_df)
    kmeans_clusters = kmeans_model.predict(median_df)

    median_df['cluster'] = kmeans_clusters

    # Sort clusters by cluster average intensity
    grouped_df = median_df.groupby(['cluster'])
    cluster_intensities = grouped_df.mean()

    # Create a hierarchy of the clusters
    cluster_linkage = linkage(cluster_intensities, method=method,
        optimal_ordering=ordering)

    # List of clusters to be plotted sequentially
    cluster_leaves = leaves_list(cluster_linkage)

    # list of preys to be populated from cluster sequence
    leaves = []

    # sort thrugh clusters and populate with hierarchy of individual leaves
    for cluster in cluster_leaves:
        cluster_df = median_df[median_df['cluster'] == cluster]
        cluster_df.drop(columns=['cluster'], inplace=True)

        if cluster_df.shape[0] > 1:
            # Use plotly function to generate a linkage
            prey_linkage = linkage(cluster_df, method=method, optimal_ordering=ordering)

            # Retrieve the order of preys in the new linkage
            prey_leaves = leaves_list(prey_linkage)
            prey_leaves = [list(cluster_df.index)[x] for x in prey_leaves]

        else:
            prey_leaves = list(cluster_df.index)

        # add to the master list of leaves
        leaves = leaves + prey_leaves

    if verbose:
        end_time = np.round(time.time() - start_time, 2)
        print("Finished generating linkage in " + str(end_time) + " seconds.")

    return leaves


def bait_leaves(imputed_df, features, method='average', distance='cosine',
        grouped=True, verbose=True):
    """Calculate the prey linkage and return the list of
    prey plotting sequence to use for heatmap. Use prey_kmeans for better performance
    rtype: prey_leaves list"""

    if verbose:
        print("Generating bait linkage...")
        start_time = time.time()


    if grouped:
        # Create a median_df, taking median of all replicates
        median_df = pys.median_replicates(imputed_df, save_info=True, col_str='')
        median_df = median_df[features].copy()

    else:
        median_df = imputed_df[features]
    # Transpose to get linkages of baits
    median_df = median_df.T

    bait_linkage = linkage(median_df, method=method, optimal_ordering=True)

    # Retreieve the order of baits in the new linkage
    bait_leaves = leaves_list(bait_linkage)
    bait_leaves = [list(median_df.index)[x] for x in bait_leaves]

    if verbose:
        end_time = np.round(time.time() - start_time, 2)
        print("Finished generating linkage in " + str(end_time) + " seconds.")

    return bait_leaves


def prey_leaves(imputed_df, features, method='average', distance='cosine', grouped=True,
        index_id='Protein IDs', verbose=True):
    """Calculate the prey linkage and return the list of
    prey plotting sequence to use for heatmap. Use prey_kmeans for better performance.

    rtype: prey_leaves list"""
    if verbose:
        print("Generating prey linkage...")
        start_time = time.time()

    # Set index
    features = features.copy()
    features.append(index_id)

    if grouped:
        # Create a median_df, taking median of all replicates
        median_df = pys.median_replicates(imputed_df, save_info=True, col_str='')
        median_df = median_df[features]

    else:
        median_df = imputed_df[features]


    # Index IDs will be the reference to retrieve the correct order of preys
    median_df.set_index(index_id, inplace=True)


    prey_linkage = linkage(median_df, method=method)

    # Retrieve the order of preys in the new linkage
    prey_leaves = leaves_list(prey_linkage)
    prey_leaves = [list(median_df.index)[x] for x in prey_leaves]


    if verbose:
        end_time = np.round(time.time() - start_time, 2)
        print("Finished generating linkage in " + str(end_time) + " seconds.")

    return prey_leaves


def dendro_heatmap(imputed_df, prey_leaves, hexmap, zmin, zmid, zmax, label, features,
        index_id='Protein IDs', bait_leaves=None, bait_clust=False, reverse=False, verbose=True,
        prey_clust=False):
    """ From the dendro_leaves data, generate a properly oriented
    heatmap

    rtype fig pyplot Fig"""

    if verbose:
        print("Generating Heatmap...")
        start_time = time.time()

    plot_df = imputed_df.copy()


    # Set index to Protein IDs to match the dendro leaves
    plot_df.set_index(index_id, inplace=True)

    if prey_clust:
        # Correctly order the plot df according to dendro leaves
        plot_df = plot_df.T[prey_leaves].T

    # Reset index to set label
    plot_df.set_index(label, inplace=True)

    # Informational columns are unnecessary now, drop them
    plot_df = plot_df[features]

    # Reorder columns based on bait_leaves
    if bait_clust:
        plot_df = plot_df[bait_leaves]

    if hexmap in ['RdBu', 'Temps', 'Tropic']:
        # force vals in zmin and zmax range because plotly doesnt allow zmid with zmin/zmax
        mask = plot_df.copy()
        mask = mask.applymap(lambda x: x if x >= zmin else zmin)
        mask = mask.applymap(lambda x: x if x <= zmax else zmax)


        heatmap = go.Heatmap(x=list(plot_df), y=list(plot_df.index), z=mask.values.tolist(),
            colorscale=hexmap, zmid=zmid, reversescale=reverse)

    else:
        # Generate the heatmap
        heatmap = go.Heatmap(x=list(plot_df), y=list(plot_df.index), z=plot_df.values.tolist(),
            colorscale=hexmap, zmin=zmin, zmax=zmax, reversescale=reverse)

    if verbose:
        end_time = np.round(time.time() - start_time, 2)
        print("Finished heatmap in " + str(end_time) + " seconds.")

    return heatmap


def df_min_max(df):
    """Quickly output min and max values of the df"""

    # flatten the df to a list of all values
    all_vals = df.values.flatten().tolist()
    all_vals = list(filter(lambda x: isinstance(x, Number), all_vals))

    return min(all_vals), max(all_vals)


def color_map(zmin, zmid, zmax, colors='perseus', reverse=False):
    """generate a color map, zmin, and zmax that the heatmap function will use
    Will add customization features in the future"""

    z = np.linspace(zmin, zmax, 200)
    if colors == 'perseus':

        # Use built in seaborn function to blend palette
        cmap = sns.blend_palette(('black', 'blue', 'green', 'yellow',
            'orange', 'red'), n_colors=8, as_cmap=False)
        hexmap = []
        # change to hex values that Plotly can read
        for color in cmap:
            hexmap.append(matplotlib.colors.rgb2hex(color))
    else:
        hexmap = colors

    # a range list from zmin to zmax
    y = [0]*len(z)
    # plot colorscale
    if colors in ['RdBu', 'Temps', 'Tropic']:

        # apply zmin and zmax
        fig = go.Figure(go.Heatmap(x=z, y=y, z=z, zmid=zmid,
            colorscale=hexmap, showscale=False, reversescale=reverse),
            layout=go.Layout(yaxis={'showticklabels': False}))

    else:
        fig = go.Figure(go.Heatmap(x=z, y=y, z=z, zmin=zmin, zmax=zmax,
            colorscale=hexmap, showscale=False, reversescale=reverse),
            layout=go.Layout(yaxis={'showticklabels': False}))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig, hexmap



    # Keep only numerics


#     return heatmap_fig

# def heatmap_fig(df,hexmap,cols,width=600,height=800):
#     """generate a plotly heatmap given the df and selected columns"""
#     fig = go.Figure(data=go.Heatmap(
#                 z= df[cols].values.tolist(),
#                 x= cols,
#                 y= list(df.index),
#                 colorscale = hexmap,
#                 reversescale=False,
#                 hoverongaps = False,
#                 zmin=5,
#                 zmax= 14),
#                layout = go.Layout(width = width,height = height))
#     fig.show()
