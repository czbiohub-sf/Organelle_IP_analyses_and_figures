import urllib.parse
import urllib.request
import sys
import pdb
import collections
import multiprocessing
import itertools
import scipy
import random
import re
import pandas as pd
import numpy as np

from pyseus import primary_analysis as pa

from multiprocessing import Queue
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import repeat
from multiprocessing import Pool
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity



class Validation():
    """
    Validation class takes as input standard_hits_table from
    AnalysisTables class (from primary_analysis.py) as input tables for various
    post-processing or validation methods
    """

    def __init__(self, hit_table, target_col, prey_col, corum=None, localization_table=None,
            interaction_table=None):
        """
        initiate class with a hit table (without interactions called) and other
        necessary tables required for precision-recall analysis =
        """

        self.hit_table = hit_table
        self.target = target_col
        self.prey = prey_col
        self.corum = corum
        self.localization_table = localization_table
        self.interaction_table = interaction_table

    def static_fdr(self, curvature, offset, negative=False):
        """
        Call significant interactors from standard hits table with user input
        offset and curvature for thresholding
        """

        hits = self.hit_table.copy()
        hits['fdr'] = None
        hits['fdr'] = hits['fdr'].apply(lambda x: [curvature, offset])

        bait_pval = hits['pvals']
        if negative:
            enrichment = hits['enrichment'].apply(np.abs)
        else:
            enrichment = hits['enrichment']

        threshold = enrichment.apply(pa.calc_thresh, args=[curvature, offset])
        hits['interaction'] = np.where((bait_pval > threshold), True, False)

        self.called_table = hits
        self.interaction_table = hits[hits['interaction']]

    def pval_threshold(self, pval_thresh, metric='pvals'):
        """
        Call significant interactors based on p-val threshold.
        Used primarily in context of generating aprecision-recall curve
        """
        hits = self.hit_table.copy()
        bait_pval = hits[metric]
        hits['interaction'] = np.where((bait_pval > pval_thresh), True, False)

        self.called_table = hits
        self.interaction_table = hits[hits['interaction']]


    def hawaii_fdr(self, perc=10, curvature=3, offset_seed=2.5, experiment=True):
        """
        compute dynamic FDR for all samples pooled in all_hits table
        all_hits: DataFrame, output of hit_calling_validations.get_all_interactors
        perc: threshold for get_fdr5_seed specifying how much % of false positives
            are allowed before calling a FDR threshold
        RETURN:
        DataFrame, dataframe containing all the major and minor FDR thresholds
            for each bait-plate group
        DataFrame, all_hits df with fdr threshold columns added
        """
        hits = self.hit_table.copy()


        if not experiment:
            hits['experiment'] = 'N/A'


        # parallel processing for calculating FDR seed

        seed = hawaii_find_thresh(hits, perc, curvature, offset_seed)

        fdr = [curvature, seed]

        enrichment = hits['enrichment']
        pvals = hits['pvals']
        thresh = enrichment.apply(pa.calc_thresh,
            args=[fdr[0], fdr[1]])

        hits['interaction'] = np.where(
            (pvals > thresh), True, False)

        hits['fdr'] = [fdr] * hits.shape[0]

        self.called_table = hits
        self.interaction_table = hits[hits['interaction']].copy()


    def dynamic_fdr(self, perc=10, curvature=3, offset_seed=2.5, experiment=True):
        """
        compute dynamic FDR for each plate-bait group in all_hits table
        all_hits: DataFrame, output of hit_calling_validations.get_all_interactors
        perc: threshold for get_fdr5_seed specifying how much % of false positives
            are allowed before calling a FDR threshold
        RETURN:
        DataFrame, dataframe containing all the major and minor FDR thresholds
            for each bait-plate group
        DataFrame, all_hits df with fdr threshold columns added
        """
        hits = self.hit_table.copy()

        # group hits table by experiment & target and place them into a bin of lists
        selects = []
        if not experiment:
            hits['experiment'] = 'N/A'
        grouped = hits.groupby(['experiment', 'target'])
        baits = []
        experiments = []

        for bait, group in grouped:
            experiments.append(bait[0])
            baits.append(bait[1])
            selects.append(group)

        # parallel processing for calculating FDR seed

        p = Pool()
        seeds = p.starmap(dfdr_find_thresh, zip(selects, baits,
            repeat(perc), repeat(curvature), repeat(offset_seed)))
        p.close()
        p.join()

        fdr_full = [[curvature, seed] for seed in seeds]

        fdr_df = pd.DataFrame()
        fdr_df['experiment'] = experiments
        fdr_df['target'] = baits
        fdr_df['fdr'] = fdr_full
        # fdr_df.set_index('bait', inplace=True)

        # Have to reconcile proper hits table format
        new_groups = []
        for bait, group in grouped:
            group = group.copy()
            experiment = bait[0]
            target = bait[1]
            fdr = fdr_df[(fdr_df['experiment'] == experiment)
                & (fdr_df['target'] == target)].fdr.item()


            bait_pval = group['pvals']
            enrichment = group['enrichment']

            thresh = enrichment.apply(pa.calc_thresh,
                args=[fdr[0], fdr[1]])

            group['interaction'] = np.where(
                (bait_pval > thresh), True, False)

            group['fdr'] = [fdr] * group.shape[0]
            new_groups.append(group)

        interaction_table = pd.concat(new_groups)
        self.dynamic_fdr_table = fdr_df
        self.called_table = interaction_table
        self.interaction_table = interaction_table[interaction_table['interaction']]

    def convert_to_unique_interactions(self, target_match=False, get_edge=False, edge='pvals'):
        """
        convert bait/prey interactions to unique, directionless interactions using gene names
        """
        dataset = self.interaction_table.copy()
        target_col = self.target
        prey_col = self.prey

        # remove interactions where target and prey are the same proteins
        if not target_match:
            dataset = dataset[dataset[target_col] != dataset[prey_col]]
        original = self.interaction_table.copy()

        # combine values from two columns to a list and sort alphabetically
        dataset = dataset[[target_col, prey_col]]

        # force values in target and prey columns to be string
        dataset[target_col] = dataset[target_col].astype(str)
        dataset[prey_col] = dataset[prey_col].astype(str)

        combined = pd.Series(dataset.values.tolist())

        combined = combined.apply(sorted)

        combined_list = combined.to_list()

        # Unzip the sorted interactions and create them into two lists
        unzipped = list(zip(*combined_list))

        first, second = unzipped[0], unzipped[1]

        # Generate a sorted interaction dataframe and drop duplicates
        interactions = pd.DataFrame()
        interactions['prot_1'] = first
        interactions['prot_2'] = second

        interactions.drop_duplicates(inplace=True)

        if get_edge:
            vals = []
            for i, row in interactions.iterrows():
                prot_1 = row.prot_1
                prot_2 = row.prot_2
                prots = [prot_1, prot_2]
                selection = original[
                    (original[target_col].isin(prots)) & original[prey_col].isin(prots)]
                max_edge = selection[edge].max()
                vals.append(max_edge)
            interactions[edge] = vals
        interactions.reset_index(drop=True, inplace=True)

        self.unique_interaction_table = interactions


    def corum_interaction_coverage(self, distance=False, directional=False):
        """
        calculate db's coverage of possible corum interactions
        if distance = True, an interaction is a true positive if a target's
        second neighbor is a corum interactor.
        if directional = True, recall considers all possible corum interactors for each target
        if directional = False, recall considers a corum interactor covered if the interaction appears
        in either direction
        """
        target_col = self.target
        prey_col = self.prey
        network = self.interaction_table[[target_col, prey_col]]
        corum = self.corum.copy()

        # get a list of all unique targets in the ppi network
        targets = set(network[target_col].to_list())

        # count all the corum interactions possible within targets if directional is False
        if directional:
            overlap_sum = 0
        else:
            overlap_corum = corum[(corum['prot_1'].isin(targets)) | (corum['prot_2'].isin(targets))]
            overlap_sum = overlap_corum.shape[0]

        coverage_sum = 0
        for target in targets:
            # get all the corum interactions possible with the targets
            left_corum = corum[corum['prot_1'] == target]
            right_corum = corum[corum['prot_2'] == target]

            # if directional, add all the possible CORUM interaction counts
            if directional:
                overlap_sum += left_corum.shape[0] + right_corum.shape[0]

            network_target = network[network[target_col] == target]
            network_preys = set(network_target[prey_col].to_list())

            left_covered = left_corum[left_corum['prot_2'].isin(network_preys)]
            right_covered = right_corum[right_corum['prot_1'].isin(network_preys)]

            # the distance option searches for second neighbors of a target, for the preys
            # of the target that were corum interactors.
            if distance:
                left_preys = left_covered['prot_2'].to_list()
                right_preys = right_covered['prot_1'].to_list()

                left_targets = targets.intersection(left_preys)
                right_targets = targets.intersection(right_preys)

                new_targets = left_targets.union(right_targets)
                new_targets.add(target)

                expanded_target = network[network[target_col].isin(new_targets)]
                expanded_preys = set(expanded_target[prey_col].to_list())

                left_covered = left_corum[left_corum['prot_2'].isin(expanded_preys)]
                right_covered = right_corum[right_corum['prot_1'].isin(expanded_preys)]


            # calculate how many corum interactions were covered by the target in network
            coverage_sum += left_covered.shape[0] + right_covered.shape[0]

            if not directional:
                # delete covered interactions
                left_idxs = left_covered.index.to_list()
                right_idxs = right_covered.index.to_list()
                drop_idxs = left_idxs + right_idxs
                corum.drop(drop_idxs, inplace=True)

        self.recall = coverage_sum / overlap_sum


    def colocalization_precision(self):
        """
        merge mnc_classifier from localization table
        and return pandas df where target and prey are colocalized
        """

        # use unique interactions for co-localization analysis
        try:
            network = self.unique_interaction_table.copy()
        except AttributeError:
            self.convert_to_unique_interactions()
            network = self.unique_interaction_table.copy()

        target_col = 'prot_1'
        prey_col = 'prot_2'

        network = network[[target_col, prey_col]].copy()
        # force string type
        network[target_col] = network[target_col].astype(str)
        network[prey_col] = network[prey_col].astype(str)

        network = network[network[target_col] != network[prey_col]]
        localization = self.localization_table.copy()

        localization['mnc_classifier'] = localization['mnc_classifier'].apply(
            lambda x: x.split('/'))

        # make target and prey merges with localization data
        # inner merge network data with localization on targets
        merge1 = network.merge(localization.rename(
            columns={'gene_names': target_col, 'mnc_classifier': 'target_localization'}),
            on=target_col, how='inner')
        # then inner merge the table with localization on preys
        merge2 = merge1.merge(localization.rename(
            columns={'gene_names': prey_col, 'mnc_classifier': 'prey_localization'}),
            on=prey_col, how='inner')

        # find interactions where there is at least one mutual localization between
        # target and prey. Save the indices of the colocalized interactions
        intersections = []
        for i, row in merge2.iterrows():
            target = set(row.target_localization)
            prey = set(row.prey_localization)
            if len(target.intersection(prey)) > 0:
                intersections.append(i)
            elif 'B' in target or 'B' in prey:
                intersections.append(i)

        self.precision = merge2.loc[intersections].shape[0] / merge2.shape[0]


def dfdr_find_thresh(select, bait, perc=10, curvature=3, seed=2.5):
    """
    Find the proper p-val/enrichment threshold for a bait
    """

    # filter for negative hits
    neg_select = select[select['enrichment'] < 0]
    pos_select = select[select['enrichment'] > 0]

    # calcuate initial hit count by given curvature and seed
    hit = hit_count(neg_select, curvature, seed)

    # Find a threshold that lies just outside one hit detection
    if hit > 0:
        while hit > 0 and seed < 10:
            if seed > 4.2:
                seed += 0.2
            else:
                seed += 0.1
            hit = hit_count(neg_select, curvature, seed)
    else:
        while hit == 0:
            seed -= 0.1
            hit = hit_count(neg_select, curvature, seed)
        seed += 0.1

    # With the calculated seed, find the threshold that meets the
    # requirement of less than 2 neg hits or less than designated % of positive hits
    neg_hit = hit_count(neg_select, curvature, seed)
    pos_hit = hit_count(pos_select, curvature, seed)
    if pos_hit == 0:
        pos_perc = 0
    else:
        pos_perc = 100 * neg_hit / pos_hit

    while (neg_hit < 2 or pos_perc < perc) and seed > 0.1:

        seed -= 0.1
        neg_hit = hit_count(neg_select, 3, seed)
        pos_hit = hit_count(pos_select, 3, seed)
        if pos_hit == 0:
            pos_perc = 0
        else:
            pos_perc = 100 * neg_hit / pos_hit

    if pos_perc > perc:
        seed += 0.1

    return round(seed, 2)


def hawaii_find_thresh(select, perc=10, curvature=3, seed=2.5):
    """
    Find the proper p-val/enrichment threshold for a bait
    """

    # filter for negative hits
    neg_select = select[select['enrichment'] < 0]
    pos_select = select[select['enrichment'] > 0]

    # With the calculated seed, find the threshold that meets the
    # requirement of less than 2 neg hits or less than designated % of positive hits
    neg_hit = hit_count(neg_select, curvature, seed)
    pos_hit = hit_count(pos_select, curvature, seed)
    if pos_hit == 0:
        pos_perc = 0
    else:
        pos_perc = 100 * neg_hit / pos_hit

    if pos_perc < perc:
        while pos_perc < perc and seed > 0.1:
            seed -= 0.1
            neg_hit = hit_count(neg_select, 3, seed)
            pos_hit = hit_count(pos_select, 3, seed)
            if pos_hit == 0:
                pos_perc = 0
            else:
                pos_perc = 100 * neg_hit / pos_hit

        if pos_perc > perc:
            seed += 0.1
    else:
        while pos_perc > perc and seed < 30:
            if seed > 5:
                seed += 0.5
            else:
                seed == 0.2
            neg_hit = hit_count(neg_select, 3, seed)
            pos_hit = hit_count(pos_select, 3, seed)
            if pos_hit == 0:
                pos_perc = 0
            else:
                pos_perc = 100 * neg_hit / pos_hit

        if pos_perc < perc:
            seed -= 0.2



    return round(seed, 2)


def hit_count(bait_series, curvature, offset):
    """
    Count # of hits possible in a bait series with a given curvature and offset
    """
    bait_series = bait_series.copy()
    thresh = bait_series['enrichment'].apply(calc_thresh, args=[curvature, offset])
    hit = np.where(bait_series['pvals'] > thresh, True, False)

    return hit.sum()


def calc_thresh(enrich, curvature, offset):
    """simple function to get FCD thresh to recognize hits"""
    if abs(enrich) < offset:
        return np.inf
    else:
        return curvature / (abs(enrich) - offset)
