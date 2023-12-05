import multiprocessing
import sys
import itertools
import scipy
import random
import re
import pandas as pd
import numpy as np
import anndata as ad


from pyseus import basic_processing as bp
from pyseus import primary_analysis as pa
from pyseus import validation_analysis as va
from external import clustering_workflows

from multiprocessing import Queue
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import repeat
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class ContrastTables(pa.AnalysisTables):
    """
    For experiments where there is a specific contrast to make between control and
    experiment samples. Allows automatic contrast selection and t-tests.
    Inherits AnalysisTables class in primary_analysis module
    """

    def set_up_contrast_matrix(self, condition='-infected'):
        """
        set up the contrast matrix to only include experiment condition vs controls
        """

        # exclusion matrix and condition label
        mat = self.exclusion_matrix.copy()
        self.exp_condition = condition

        # get a list of samples in the exp-condition
        samples = list(mat)
        samples.remove('Samples')

        # case insensitive selection of condition
        exp_samples = [x for x in samples if condition.lower() in x.lower()]

        # create a new matrix for made for exp-condition samples,
        # testing only against its control counterpart
        new_mat = pd.DataFrame()
        new_mat['Samples'] = samples

        # Create a new matrix for the experiments , testing only against its
        # control counterpart
        for exp_sample in exp_samples:
            # remove the condition tag
            control = exp_sample.replace(condition, '')

            new_controls = []
            # Boolean filter for the exclusion matrixd
            for sample in samples:
                # pick the control sample
                if ((control in sample) or (sample in control)) and (
                        condition not in sample):

                    new_controls.append(True)

                else:
                    new_controls.append(False)
            # assign a new column in the matrix if there is a respective control
            if any(new_controls):
                new_mat[exp_sample] = new_controls

        self.contrast_matrix = new_mat.copy()


    def contrast_pval_enrichment(self, std_enrich=False):
        """
        use the contrast matrix for the pval calculations
        and conver to standard table
        """

        # run simple pval with the custom contrast matrix
        self.simple_pval_enrichment(std_enrich=std_enrich, custom=True,
            exclusion_mat=self.contrast_matrix)
        # standardizec
        self.convert_to_standard_table(experiment=False, simple_analysis=True, perseus=False)


    def call_hits(self, curvature=2.6, offset=1, negative=True):
        """
        Use Validations class to call hits by given parameters and save the table
        """
        # initiate a Validation class to call hits
        vali = va.Validation(hit_table=self.standard_hits_table, target_col='target',
            prey_col='Gene names')
        # call hits based on a set curvature and offset
        vali.static_fdr(curvature=curvature, offset=offset, negative=negative)
        self.hits_table = vali.called_table.copy()
        self.interaction_table = vali.interaction_table.copy()


    def contrast_filters(self, mask_insignificants, sig_cols=['interaction'],
            mask_enrichment=False, enrichment_thresh=5, exclude_wt_sigs=False):
        """
        use various filters to return a 'wide' enrichment table where
        insignificant / filtered enrichment values are 0, and protein rows
        are removed if no protein passes the filters
        """

        diff_table = self.hits_table.copy()

        # if a filtering column - such as 'interaction' column is boolean
        # use the column as a filter to convert enrichments to 0
        if mask_insignificants:
            # turn enrichments into zero
            for col in sig_cols:
                diff_table.loc[diff_table[col] is False, 'enrichment'] = 0

        # mask enrichments that dont meet a threshold to 0
        if mask_enrichment:
            diff_table['abs_enrichment'] = diff_table['enrichment'].apply(np.absolute)
            diff_table.loc[diff_table['abs_enrichment'] <= enrichment_thresh, 'enrichment'] = 0

        targets = diff_table['target'].unique()
        targets.sort()



        # create 'wide' tables from the hits table
        for i, target in enumerate(targets):
            selection = diff_table[diff_table['target'] == target].reset_index(drop=True)

            if i == 0:
                # create a template enrichment table
                enrich_table = pd.DataFrame()
                enrich_table['Protein IDs'] = selection['Protein IDs'].to_list()
                enrich_table['Gene names'] = selection['Gene names'].to_list()



            enrich_table[target] = selection['enrichment'].to_list()

        # remove protein rows with all zeros (did not pass previous filters)
        if mask_insignificants:
            # remove rows with all zeros
            sums = enrich_table[targets].sum(axis=1)
            zeros = sums[sums == 0].index.to_list()
            enrich_table.drop(zeros, axis=0, inplace=True)
            enrich_table.reset_index(drop=True, inplace=True)

        # remove rows that contained hits in the wildtypes
        if exclude_wt_sigs:
            wts = [target for target in targets if 'wt' in target.lower()]
            wt_sums = enrich_table[wts].sum(axis=1)
            excludes = wt_sums[wt_sums > 0].index.to_list()
            enrich_table.drop(excludes, axis=0, inplace=True)
            enrich_table.drop(wts, axis=1, inplace=True)
            enrich_table.reset_index(drop=True, inplace=True)

        # convert to standard pyseus webapp format
        enrich_table = standard_pyseus_headers(enrich_table)

        self.contrast_filtered_table = enrich_table


    def order_table(self, order_list, index_id='Protein IDs'):

        df = self.contrast_filtered_table.copy()
        df = df.droplevel(0, axis=1)

        # get only Protein IDs that exist in table
        index_cols = df[index_id].to_list()
        order_list = [x for x in order_list if x in index_cols]

        df.set_index(index_id, inplace=True)
        df = df.T[order_list].T
        df.reset_index(drop=False, inplace=True)

        df = standard_pyseus_headers(df)

        return df


def standard_pyseus_headers(table, meta_cols=['Protein IDs',
        'Majority protein IDs', 'Gene names']):
    """
    add headers as a multicolumn for standard pyseus Webapp format
    """

    table = table.copy()
    new_cols = []
    for col in list(table):
        if col in meta_cols:
            new_cols.append(('metadata', col))
        else:
            new_cols.append(('sample', col))

    table.columns = pd.MultiIndex.from_tuples(new_cols)

    return table
