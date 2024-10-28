import math
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as mplPath
from shapely.geometry import Point
from scipy.spatial import procrustes

def calculate_distance(neighbors_mapper):
    '''Calculate (for each point) the distance between the the aligned UMAPs
    Input: neighbors_mapper object
    '''
    #extract coordinates
    JointUMAP_uninf_1 = list(neighbors_mapper.embeddings_[0].T[0])
    JointUMAP_uninf_2 = list(neighbors_mapper.embeddings_[0].T[1])
    JointUMAP_inf_1 = list(neighbors_mapper.embeddings_[1].T[0])
    JointUMAP_inf_2 = list(neighbors_mapper.embeddings_[1].T[1])

    coordinate_uninf = list(zip(JointUMAP_uninf_1, JointUMAP_uninf_2))
    coordinate_inf = list(zip(JointUMAP_inf_1, JointUMAP_inf_2))

    # calculate the distance between the two coordinates
    distances = []
    for idx, val in enumerate(coordinate_uninf):
        distance = math.dist(val, coordinate_inf[idx])
        distances.append(distance)
    return distances

def print_samples(samples):
    printout = ""
    for idx, val in enumerate(samples):
        if idx > 0:
            if val.split("-")[0] == samples[idx - 1].split("-")[0]:
                printout += f", {val}"
            else:
                printout += f"\n{val}"
        else:
            printout += f"\n{val}"
    print(printout)

def in_list_ele(ele, lst):
    for i in lst:
        if ele in i:
            return True
    return False

def compute_transformation(UMAP1_common, UMAP2_common):
    '''function to compute transformation using Procrustes analysis'''
    # perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(UMAP1_common, UMAP2_common)
    
    # compute the transformation matrix
    translation = np.mean(mtx1, axis=0) - np.mean(mtx2, axis=0)
    scale = np.std(mtx1) / np.std(mtx2)
    
    # compute the rotation matrix
    rotation_matrix = np.linalg.lstsq(mtx2, mtx1, rcond=None)[0]
    
    return translation, scale, rotation_matrix

def compute_transformation_v2(UMAP1_common, UMAP2_common):
    '''function to compute transformation using Procrustes analysis'''
    # perform Procrustes analysis
    mtx1, mtx2, disparity = procrustes(UMAP1_common, UMAP2_common)
    
    # calculate the optimal rotation matrix using SVD
    U, _, Vt = np.linalg.svd(mtx2.T @ mtx1)
    rotation_matrix = U @ Vt
    
    # calculate scaling and translation
    scale = np.std(mtx1) / np.std(mtx2)
    translation = np.mean(mtx1, axis=0) - scale * np.mean(mtx2 @ rotation_matrix, axis=0)
    
    return translation, scale, rotation_matrix

def apply_transformation(points, translation, scale, rotation_matrix):
    # apply rotation and scaling
    transformed_points = points @ rotation_matrix * scale
    
    # apply translation
    transformed_points += translation
    
    return transformed_points

def create_circles(points, radius=0.1):
    '''function to create circle shapes around points, for plotting related to aligned umap'''
    circles = [Point(xy).buffer(radius) for xy in points]
    return circles

def polygon_to_patch(polygon, **kwargs):
    '''function to create circle shapes around points, for plotting related to aligned umap'''
    if polygon.is_empty:
        return None
    vertices = []
    codes = []
    # Exterior ring
    x, y = polygon.exterior.xy
    coords = np.column_stack((x, y))
    vertices.extend(coords)
    codes.extend([mplPath.MOVETO] + [mplPath.LINETO]*(len(coords)-2) + [mplPath.CLOSEPOLY])
    # Interiors (holes)
    for interior in polygon.interiors:
        x, y = interior.xy
        coords = np.column_stack((x, y))
        vertices.extend(coords)
        codes.extend([mplPath.MOVETO] + [mplPath.LINETO]*(len(coords)-2) + [mplPath.CLOSEPOLY])
    path = mplPath(vertices, codes)
    patch = PathPatch(path, **kwargs)
    return patch

def rescale_list(input_list, original_min, original_max, desired_min, desired_max):
    return [desired_min + ((x - original_min) * (desired_max - desired_min) / (original_max - original_min)) for x in input_list]