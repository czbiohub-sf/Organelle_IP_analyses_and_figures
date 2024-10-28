import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


# functions to compute distance matrices
def calc_dist_mat(data, method="euclidean"):
    if method not in ["minkowski", "seuclidean", "mahalanobis"]:
        dist_mat = pdist(data, method)
    elif method == "minkowski":
        dist_mat = pdist(data, method, p=2.0)
    elif method == "seuclidean":
        dist_mat = pdist(data, method, V=None)
    elif method == "mahalanobis":
        dist_mat = pdist(data, method, VI=None)
    # return square form of distance matrix
    return squareform(dist_mat)

def calc_all_dist_mat(
    data, method_list=["canberra", "correlation", "cosine"]
):  
    '''
    Calculate distance matrices for a given data matrix using a list of distance metrics.
    possible distance methods are: 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 
    'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
    'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 
    'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
    '''
    dist_mat_dict = {}
    for method in method_list:
        dist_mat_dict[method] = calc_dist_mat(data, method)
    return dist_mat_dict

def compute_dot_product_dist(df):
    dot_product_matrix = np.dot(df.values, df.values.T)
    # Convert the result back to a DataFrame for better readability
    dot_product_df = pd.DataFrame(dot_product_matrix, index=df.index, columns=df.index)
    return dot_product_df

def compute_spectral_cluster_distance(df, n_neighbors=20, n_clusters=5):
    import numpy as np
    from sklearn.cluster import SpectralClustering
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import kneighbors_graph
    from scipy.spatial.distance import euclidean

    # Convert the DataFrame to a numpy array
    vectors = df.values

    # Create the adjacency matrix using KNN graph
    adjacency_matrix = kneighbors_graph(vectors, n_neighbors=n_neighbors, include_self=False).toarray()

    # Perform spectral clustering
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='discretize')
    labels = spectral.fit_predict(adjacency_matrix)

    # Create the cluster centers based on the mean of vectors within each cluster
    cluster_centers = np.array([vectors[labels == i].mean(axis=0) for i in range(n_clusters)])

    # Initialize an empty DataFrame to store the pairwise distances
    pairwise_dist_df = pd.DataFrame(index=df.index, columns=df.index)

    # Calculate pairwise spectral clustering-based distances
    for i in range(len(vectors)):
        for j in range(i, len(vectors)):
            distance = euclidean(cluster_centers[labels[i]], cluster_centers[labels[j]])
            pairwise_dist_df.iloc[i, j] = distance
            pairwise_dist_df.iloc[j, i] = distance

    return pairwise_dist_df


def extract_pairwise_distances(matrix, names):
    '''function to extract pairwise distances'''
    # names: a list where the indices correspond to the names.
    n = matrix.shape[0]
    pairwise_distances = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == j or names[i] == names[j]:
                continue
            pair = frozenset([names[i], names[j]])
            pairwise_distances[pair] = matrix[i, j]
    
    return pairwise_distances