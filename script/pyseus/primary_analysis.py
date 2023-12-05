import urllib.parse
import urllib.request
import sys
import pdb
import collections
import multiprocessing
import itertools
import scipy
import random
import pickle
import re
import pandas as pd
import numpy as np
import os
from pyseus import basic_processing as bp
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


class AnalysisTables:
    """
    Analysis Tables contains DataFrame objects, functions, and metadata that cover
    essential analysis including enrichment/significance testing to call interactions
    and stoichiometry calculations
    """

    def __init__(
            self,
            root=None,
            analysis=None,
            imputed_table=None,
            grouped_table=None,
            exclusion_matrix=None,
            auto_group=False):

        # initiate class that cover essential metadata and imputed table
        # from RawTables class.

        self.root = root
        self.analysis = analysis
        self.imputed_table = imputed_table
        self.exclusion_matrix = exclusion_matrix

        # if the preprocessed table is not grouped, use group function
        if grouped_table is not None:
            self.grouped_table = grouped_table
        elif auto_group:
            # use RawTables class to group replicates
            features = list(imputed_table['sample'])
            labels = list(imputed_table['metadata'])
            processing = bp.RawTables(file_designated=True)
            processing.transformed_table = imputed_table.droplevel(0, axis=1)
            processing.sample_cols = features
            processing.info_cols = labels
            processing.group_replicates(reg_exp=r'(.*)_\d+$')
            processing.generate_export_bait_matrix()

            self.grouped_table = processing.grouped_table.copy()
            self.generate_export_bait_matrix()


    def generate_export_bait_matrix(self):
        """
        Generates and saves a Boolean bait matrix that will be used for control
        exclusion in p-val and enrichment analysis.
        """
        grouped = self.grouped_table.copy()
        baits = list(set(grouped.columns.get_level_values('Samples').to_list()))
        baits.remove('metadata')
        baits.sort()
        bait_df = pd.DataFrame()
        bait_df['Samples'] = baits
        bait_df.reset_index(drop=True, inplace=True)

        # Create a boolean table
        for bait in baits:
            bait_bools = [True if x != bait else False for x in baits]
            bait_df[bait] = bait_bools

        self.exclusion_matrix = bait_df.copy()

    def restore_default_exclusion_matrix(self):
        """
        Restore exclusion matrix to default - No exclusion
        """

        exclusion = self.exclusion_matrix.copy()
        baits = list(exclusion)
        baits.remove('Samples')
        for bait in baits:
            exclusion[bait] = True

        self.exclusion_matrix = exclusion


    def load_exclusion_matrix(self, alt_text=''):
        """
        Load user-defined exclusion_matrix for simple analysis
        """
        exclusion_matrix = pd.read_csv(self.root + self.analysis
            + '/analysis_exclusion_matrix' + alt_text + '.csv')

        self.exclusion_matrix = exclusion_matrix

    def print_baits(self):
        """
        Show dataframe of all possible baits
        """
        return self.exclusion_matrix['Samples']

    def print_controls(self, bait):
        """
        Print all the selected controls for an input bait
        """

        excluded = self.exclusion_matrix.copy()
        excluded = excluded[['Samples', bait]]
        excluded = excluded[excluded[bait]]

        if excluded.shape[0] > 0:
            return excluded[['Samples']]
        else:
            print("No control baits selected as control")

    def print_excluded_controls(self, bait):
        """
        Print all the excluded controls for an input bait
        """

        excluded = self.exclusion_matrix.copy()
        excluded = excluded[['Samples', bait]]
        excluded = excluded[~excluded[bait]]

        if excluded.shape[0] > 0:
            return excluded[['Samples']]
        else:
            print("No excluded baits in control")

    def select_wildtype_controls(self, wt_re='_WT'):
        """
        Using string operation, select only wildtypes to use as controls
        and exclude all others. Since this is based on string,
        one can customize any label to use for control samples.

        Does not override default excluded controls.
        """

        exclusion = self.exclusion_matrix.copy()
        exclusion.set_index('Samples', inplace=True)
        exclusion = exclusion.T
        baits = list(exclusion)

        for bait in baits:
            if wt_re in bait:
                continue
            else:
                exclusion[bait] = False

        exclusion = exclusion.T
        exclusion.reset_index(inplace=True)

        self.exclusion_matrix = exclusion

    def select_matching_WT_control(self, wt_re='_WT'):
        """
        Using string operation, select only wildtypes of matching experiments to use as controls
        (e.g. use 14-WT only for 14)
        and exclude all others. Since this is based on string,

        Does not override default excluded controls.
        """

        exclusion = self.exclusion_matrix.copy()
        exclusion.set_index('Samples', inplace=True)
        exclusion = exclusion.T
        baits = list(exclusion)

        # go through each pair, and for the sample experiment ID, check if there is a matching WT
        for row in exclusion.index:
            for col in exclusion.columns:
                exclusion.at[row, col] = False #default to false
                if col.split('-')[0] == row.split('-')[0]: # matching experiment IDs
                    if (wt_re in col) != (wt_re in row): # if one is WT and the other is not
                        exclusion.at[row, col] = True
                    else:
                        exclusion.at[row, col] = False

        # check if there are expriments with no matching WT
        for row in exclusion.index:
            if (exclusion.loc[row] == False).all():
                for col in exclusion.columns:
                    if (wt_re in col):
                        exclusion.at[row, col] = True
        for col in exclusion.columns:
            if (exclusion[col] == False).all():
                for row in exclusion.index:
                    if (wt_re in row):
                        exclusion.at[row, col] = True

        exclusion = exclusion.T
        exclusion.reset_index(inplace=True)

        self.exclusion_matrix = exclusion


    def simple_pval_enrichment(self, std_enrich=True, mean=False, custom=False, exclusion_mat=None):
        """
        Calculate enrichment and pvals for each bait, no automatic removal
        """
        imputed = self.grouped_table.copy()
        if custom:
            exclusion = exclusion_mat.copy()
        else:
            exclusion = self.exclusion_matrix.copy()


        imputed.reset_index(drop=True, inplace=True)

        if custom:
            bait_list = list(exclusion)
            bait_list.remove('Samples')


        else:
            # iterate through each cluster to generate neg con group
            bait_list = [col[0] for col in list(imputed) if col[0] != 'metadata']
            bait_list = list(set(bait_list))

        multi_args = zip(bait_list, repeat(imputed),
        repeat(exclusion), repeat(std_enrich), repeat(mean), repeat(True))

        p = Pool()
        print("P-val calculations..")
        outputs = p.starmap(calculate_pval, multi_args)
        p.close()
        p.join()
        print("Finished!")

        master_df = pd.concat(outputs, axis=1)

        # join metadata to the df
        meta_cols = [col for col in list(imputed) if col[0] == 'metadata']
        metadata = imputed[meta_cols].copy()


        master_df = pd.concat([master_df, metadata], axis=1, join='inner')

        self.simple_pval_table = master_df



        # gene_names = imputed[[('metadata', 'Protein IDs'), ('metadata', 'Gene names')]].copy()
        # gene_names.set_index(('metadata', 'Protein IDs'), drop=True, inplace=True)
        # gene_names.rename(columns={'metadata': 'gene_names'}, inplace=True)
        # gene_names.rename(columns={'Gene names': 'gene_names'}, inplace=True)


    def two_step_bootstrap_pval_enrichment(self, std_enrich=True, mean=False, thresh=0.001,
            bootstrap_rep=100):
        """
        The two-step bootstrap pval/enrichment calculations does not use
        an exclusion table of user defined controls. It automatically
        drops statistically significant outliers from the prey pool on the
        first round, and uses the distribution with dropped outliers
        to calculate bootstrapped null distribution. The null distribution
        of the preys are then used in the second round for pval and enrichment
        calculation. Uses multi-processing for faster runtime
        """

        imputed = self.grouped_table.copy()
        imputed.reset_index(drop=True, inplace=True)
        bait_list = [col[0] for col in list(imputed) if col[0] != 'metadata']
        bait_list = list(set(bait_list))

        multi_args = zip(bait_list, repeat(imputed), repeat(None), repeat(std_enrich),
            repeat(mean), repeat(False), repeat(True), repeat(False), repeat(thresh), repeat(False),
            repeat(None))

        p = Pool()
        print("First round p-val calculations..")
        neg_dfs = p.starmap(calculate_pval, multi_args)
        p.close()
        p.join()
        master_neg = pd.concat(neg_dfs, axis=1)
        print("First round finished!")

        self.second_round_neg_control = master_neg.copy()
        master_neg.reset_index(inplace=True, drop=True)

        multi_args2 = zip(bait_list, repeat(imputed), repeat(None), repeat(std_enrich),
            repeat(mean), repeat(False), repeat(False), repeat(True), repeat(thresh), repeat(True),
            repeat(master_neg))

        print("Second round p-val calculations...")
        p = Pool()
        outputs = p.starmap(calculate_pval, multi_args2)

        master_df = pd.concat(outputs, axis=1)
        print("Second round finished!")

        # # join gene names to the df
        # gene_names = imputed[[('metadata', 'Protein IDs'), ('metadata', 'Gene names')]].copy()
        # gene_names.set_index(('metadata', 'Protein IDs'), drop=True, inplace=True)
        # gene_names.rename(columns={'metadata': 'gene_names'}, inplace=True)
        # gene_names.rename(columns={'Gene names': 'gene_names'}, inplace=True)

        # master_df = pd.concat([master_df, gene_names], axis=1, join='inner')

        # join metadata to the df
        meta_cols = [col for col in list(imputed) if col[0] == 'metadata']
        metadata = imputed[meta_cols].copy()

        master_df = pd.concat([master_df, metadata], axis=1, join='inner')


        self.two_step_pval_table = master_df.copy()

    def convert_to_enrichment_table(self, enrichment='enrichment', simple_analysis=True):
        """
        this function takes only the enrichment (or pvals) from the pval table
        and creates a standard table compatible with Pyseus webapp
        """

        try:
            if simple_analysis:
                pvals = self.simple_pval_table.copy()
                # protein_ids = pvals.index.to_list()
            else:
                pvals = self.two_step_pval_table.copy()
                # protein_ids = pvals.index.to_list()
        except AttributeError:
            print("pval table not calculated")
            return

        # process download table for UMAP
        take_cols = []
        for col in list(pvals):
            if (col[0] == 'metadata') or (col[1] == enrichment):
                take_cols.append(col)
            else:
                continue

        enrichs = pvals[take_cols].copy()
        new_cols = []
        for col in list(enrichs):
            if col[1] == enrichment:
                new_cols.append(('sample', col[0]))
            else:
                new_cols.append(col)

        enrichs.columns = pd.MultiIndex.from_tuples(new_cols)

        self.enrichment_table = enrichs


    def convert_to_standard_table(self, metrics=['pvals', 'enrichment'],
            experiment=True, simple_analysis=True, perseus=True):
        """
        the standard table no longer uses column organization for baits.
        It follows a more SQL-like form where bait information is provided in
        separate columns.
        """
        try:
            if simple_analysis:
                pvals = self.simple_pval_table.copy()
                # protein_ids = pvals.index.to_list()
            else:
                pvals = self.two_step_pval_table.copy()
                # protein_ids = pvals.index.to_list()
        except AttributeError:
            print("pval table not calculated")
            return

        # pvals.set_index(('gene_names', 'gene_names'), inplace=True)
        # pvals.index.name = 'gene_names'

        bait_list = [col[0] for col in list(pvals) if col[0] != 'metadata']
        meta_list = [col for col in list(pvals) if col[0] == 'metadata']
        metas = [col[1] for col in meta_list]

        targets = list(set(bait_list))

        all_hits = []
        # Get all hits and minor hits along with the metric data
        for target in targets:
            target_pvs = pvals[target].copy()
            # copy each metadata
            for meta in meta_list:
                target_pvs[meta[1]] = pvals[meta].to_list()

            # return target_pvs

            hits = target_pvs.copy()
            if experiment:
                hits['experiment'] = target.split('_')[0]
                hits['target'] = target.split('_')[1]
            else:
                hits['target'] = target
            if perseus:
                hits.rename(columns={'Gene names': 'prey'}, inplace=True)
                hits.rename(columns={'Protein IDs': 'protein_ids'}, inplace=True)
            hits.reset_index(drop=True, inplace=True)
            all_hits.append(hits)
        if perseus:
            metas.remove('Gene names')
            metas.remove('Protein IDs')
            metas.insert(0, 'prey')
            metas.insert(0, 'protein_ids')

        all_hits = pd.concat(all_hits, axis=0)
        if experiment:
            col_order = ['experiment', 'target'] + metas + metrics
        else:
            col_order = ['target'] + metas + metrics
        all_hits = all_hits[col_order].copy()
        all_hits.reset_index(drop=True, inplace=True)

        self.standard_hits_table = all_hits.copy()


    def save(self, option_str=''):
        """
        save class to a designated directory
        """
        analysis_dir = self.root + self.analysis
        try:
            print("Saving standard hits table..")
            self.standard_hits_table.to_csv(analysis_dir + '/standard_hits_table.csv')
        except AttributeError:
            pass
        if len(option_str) > 0:
            option_str = '_' + option_str
        file_dir = analysis_dir + "/pval_tables" + option_str + '.pkl'
        if not os.path.isdir(analysis_dir):
            print(analysis_dir)
            print('Directory does not exist! Creating new directory')
            os.mkdir(analysis_dir)

        print("Saving to: " + file_dir)
        with open(file_dir, 'wb') as file_:
            pickle.dump(self, file_, -1)


def calculate_pval(bait, df, exclusion, std_enrich=True, mean=False,
        simple=True, first_round=False, second_round=False, thresh=0.001, bagging=False,
        second_round_neg_control=None):
    """ General script for pval calculations - encompasses options for
    simple and two-step bootstrap calculations """

    df = df.copy()
    if simple:
        excluded = exclusion.copy()

    # # initiate other variables required for the fx
    # gene_list = df[('metadata', 'Protein IDs')].tolist()

    # construct a negative control
    temporary = df.copy()
    temporary.drop('metadata', level='Samples', inplace=True, axis=1)
    if second_round:
        neg_control = second_round_neg_control.copy()
    else:
        neg_control = df.copy()
        neg_control.drop('metadata', level='Samples', inplace=True, axis=1)

    if simple:
        # Get a list of excluded genes
        excluded = excluded[['Samples', bait]]
        excluded = excluded[~excluded[bait]]



        if excluded.shape[0] > 0:
            exclude_list = excluded['Samples'].to_list()

        # Convert all values in same groups as np.nans
        for gene in exclude_list:
            neg_control[gene] = neg_control[gene].where(
                neg_control[gene] > 100, np.nan)



    # combine values of replicates into one list
    bait_series = temporary[bait].values.tolist()

    if first_round:
        # copy a bait series that will be returned with removed hits
        neg_series = temporary[bait].copy()
        # neg_series.index = gene_list
        neg_series.columns = pd.MultiIndex.from_product([[bait], neg_series.columns])

    # add an index value to the list for locating neg_control indices
    for i in np.arange(len(bait_series)):
        bait_series[i].append(i)

    # perform the p value calculations
    pval_series = pd.Series(bait_series, name='pvals')
    # pval_series = pd.Series(bait_series, index=gene_list, name='pvals')

    if simple:
        pval_series = pval_series.apply(get_pvals, args=[neg_control.T, std_enrich, mean])
    else:
        pval_series = pval_series.apply(get_pvals, args=[neg_control.T, std_enrich, mean, bagging])

    pvals, enrichment = pval_series.apply(lambda x: x[0]), pval_series.apply(lambda x: x[1])
    pvals.name = 'pvals'
    enrichment.name = 'enrichment'


    pe_df = pd.concat([pvals, enrichment], axis=1)

    # Find positive hits from enrichment and pval calculations to exclude on second round
    if first_round:

        pe_df = pe_df[pe_df['enrichment'] > 0]
        pe_df['hits'] = np.where((pe_df['pvals'] > thresh), True, False)

        # Get genes names of all the hits
        hits = set(pe_df[pe_df['hits']].index.tolist())

        # Remove hits from the negative control
        replicates = list(neg_series)

        for rep in replicates:
            for hit in hits:
                neg_series[rep][hit] = np.nan

        return neg_series


    output = pd.concat([pe_df[['enrichment', 'pvals']]], keys=[bait],
        names=['baits', 'values'], axis=1)

    return output



def get_pvals(x, control_df, std_enrich, mean=False, bagging=False, bootstrap_rep=100):
    """This is an auxillary function to calculate p values
    that is used in enrichment_pval_dfs function

    rtype: pval float"""

    # get the index to access the right set of control intensities
    row = x[-1]
    neg_con = np.array(control_df[row].values.tolist())

    if bagging:
        orig_len = len(neg_con)
        con_dropped = tuple(neg_con[~np.isnan(neg_con)])
        dropped_con = list(con_dropped)

        # bootstrap sampling
        bagged_means = []
        bagged_stds = []
        for _ in np.arange(bootstrap_rep):
            bagged_con = np.random.choice(dropped_con, size=orig_len)
            bagged_means.append(np.mean(bagged_con))
            bagged_stds.append(np.std(bagged_con))

        bootstrap_mean = np.mean(bagged_means)
        bootstrap_std = np.mean(bagged_stds)

        neg_con = np.random.normal(loc=bootstrap_mean, scale=bootstrap_std, size=orig_len)


    pval = scipy.stats.ttest_ind(x[:-1], neg_con,
    nan_policy='omit')[1]

    # negative log of the pvals
    pval = -1 * np.log10(pval)


    # calculate enrichment
    if std_enrich:
        std = np.nanstd(neg_con)
        if mean:
            enrichment = (np.nanmean(x[:-1]) - np.nanmean(neg_con)) / std
        else:
            enrichment = (np.nanmedian(x[:-1]) - np.nanmedian(neg_con)) / std

    else:
        if mean:
            enrichment = (np.nanmean(x[:-1]) - np.nanmean(neg_con))
        else:
            enrichment = (np.nanmedian(x[:-1]) - np.nanmedian(neg_con))

    return [pval, enrichment]


def calc_thresh(enrich, curvature, offset):
    """simple function to get FCD thresh to recognize hits"""

    if enrich <= offset:
        return np.inf

    elif (enrich == 0) & (offset == 0):
        return np.inf

    else:
        return curvature / (abs(enrich) - offset)


def standard_to_wide_table(df, metric='enrichment', bait='target',
        metadata=['Protein IDs', 'Gene names']):

    df = df.copy()
    targets = df[bait].unique()
    targets.sort()

    to_concat = []
    for i, target in enumerate(targets):
        selection = df[df[bait] == target]
        # save metadata
        if i == 0:
            meta = selection[metadata].reset_index(drop=True).copy()
            to_concat.append(meta)
        series = selection[[metric]].reset_index(drop=True).copy()
        series.rename(columns={metric: target}, inplace=True)
        to_concat.append(series)

    new_wide = pd.concat(to_concat, axis=1)
    new_wide.reset_index(drop=True, inplace=True)

    return new_wide
