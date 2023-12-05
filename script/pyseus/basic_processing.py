import urllib.parse
import urllib.request
import sys
import multiprocessing
import os
import re
import pickle
import pandas as pd
import numpy as np
from itertools import repeat
from multiprocessing import Pool


class RawTables:
    """
    Raw Tables class contains DataFrame objects, functions, and metadata that cover
    multiple pre-processing steps to create a final processed imputed table.

    """

    # initiate raw table by importing from data directory
    def __init__(self, experiment_dir='', analysis='', pg_file='proteinGroups.txt', info_cols=None,
            sample_cols=None, intensity_type='Intensity ', proteingroup=None,
            file_designated=True):
        # set up root folders for the experiment and standard for comparison
        self.root = experiment_dir
        self.intensity_type = intensity_type
        self.analysis = analysis
        if file_designated:
            self.pg_table = proteingroup
        else:
            self.pg_table = pd.read_csv(
                self.root + pg_file, sep='\t', low_memory=False)
        if info_cols is None:
            self.info_cols = [
                'Protein IDs',
                'Majority protein IDs',
                'Protein names',
                'Gene names']
        else:
            self.info_cols = info_cols
        self.sample_cols = sample_cols

    def save(self, option_str=''):
        """
        save class to a designated directory
        """
        analysis_dir = self.root + self.analysis
        if len(option_str) > 0:
            option_str = '_' + option_str
        file_dir = analysis_dir + "/preprocessed_tables" + option_str + '.pkl'
        if not os.path.isdir(analysis_dir):
            print(analysis_dir)
            print('Directory does not exist! Creating new directory')
            os.mkdir(analysis_dir)

        print("Saving to: " + file_dir)
        with open(file_dir, 'wb') as file_:
            pickle.dump(self, file_, -1)

    def filter_table(self, select_intensity=True, skip_filter=False, verbose=True):
        """filter rows that do not meet the QC (contaminants, reverse seq, only identified by site)
        Also filter non-intensity columns that will not be used for further processing"""

        try:
            ms_table = self.renamed_table.copy()
        except AttributeError:
            # if table has not been renamed use the raw pg table
            ms_table = self.pg_table.copy()

        if not skip_filter:
            pre_filter = ms_table.shape[0]

            # remove rows with potential contaminants
            ms_table = ms_table[ms_table['Potential contaminant'].isna()]

            # remove rows only identified by site
            ms_table = ms_table[ms_table['Only identified by site'].isna()]

            # remove rows that are reverse seq
            ms_table = ms_table[ms_table['Reverse'].isna()]

            filtered = pre_filter - ms_table.shape[0]
            if verbose:
                print("Filtered " + str(filtered) + ' of '
                    + str(pre_filter) + ' rows. Now '
                    + str(ms_table.shape[0]) + ' rows.')

        # select necessary columns
        if select_intensity:
            all_cols = list(ms_table)
            int_cols, sample_cols = select_intensity_cols(all_cols, self.intensity_type)
            rename = {i: j for i, j in zip(int_cols, sample_cols)}

            ms_table = ms_table.rename(columns=rename)
            info_cols = self.info_cols
            self.sample_cols = sample_cols
        else:
            info_cols = self.info_cols
            sample_cols = self.sample_cols
        ms_table = ms_table[info_cols + sample_cols]


        self.filtered_table = ms_table



    def rename_columns(self, RE, replacement_RE, repl_search=False):
        """
        change intensity column names to a readable format. More specifically,
        search a column name from an input RE and substitute matches with another
        input substitute strings or REs.
            col_names: list, a list of column names from raw_df
            RE: list, a list of regular expressions to search in column names
            replacement_RE: list, a list of strs/REs that substitute the original expression
            repl_search: boolean, if True, elements in replacement_RE are treated as regular
                expressions used in search, and all specified groups are used in substitution

        """
        try:
            df = self.filtered_table.copy()
        except AttributeError:
            # if table has not been filtered yet use the raw pg table
            df = self.pg_table.copy()
        sample_cols = self.sample_cols

        # start a new col list
        new_cols = []

        # Loop through cols and make qualifying subs
        for col in sample_cols:
            for i in np.arange(len(RE)):
                if re.search(RE[i], col, flags=re.IGNORECASE):
                    replacement = replacement_RE[i]
                    if (repl_search) & (len(replacement) > 1):
                        rep_search = re.search(replacement, col,
                                    flags=re.IGNORECASE)
                        replacement = ''
                        for group in rep_search.groups():
                            replacement += group

                    col = re.sub(RE[i], replacement, col, flags=re.IGNORECASE)
            new_cols.append(col)

        self.sample_cols = new_cols
        rename = {i: j for i, j in zip(sample_cols, new_cols)}

        renamed = df.rename(columns=rename)

        self.renamed_table = renamed
        self.filtered_table = renamed



    def transform_intensities(self, func=np.log2):
        """transform intensity values in the dataframe to a given function"""

        try:
            filtered = self.filtered_table.copy()
        except AttributeError:
            print(
                "Raw table has not been filtered yet, use filter_table() method"
                "before transforming intensities")
            return

        filtered = self.filtered_table.copy()
        sample_cols = self.sample_cols

        # for each intensity column, transform the values
        for int_col in sample_cols:
            # if transformation is log2, convert 0s to nans
            # (faster in one apply step than 2)
            filtered[int_col] = filtered[int_col].astype(float)
            if func == np.log2:
                filtered[int_col] = filtered[int_col].apply(lambda x: np.nan
                    if x == 0 else func(x))
            else:
                filtered[int_col] = filtered[int_col].apply(func)
                # Replace neg inf values is np.nan
                filtered[int_col] = filtered[int_col].apply(
                    lambda x: np.nan if np.isneginf(x) else x)

        self.transformed_table = filtered

    def group_replicates(self, reg_exp=r'(.*_.*)_\d+$'):
        """Group the replicates of intensities into replicate groups"""

        try:
            self.transformed_table
            transformed = self.transformed_table.copy()
        except AttributeError:
            print(
                'Intensity values have not been transformed yet from '
                'filtered table,\nwe recommend using transform_intensities() '
                'method before grouping replicates.\n')

            try:
                print("Using filtered_table to group replicates.")
                transformed = self.filtered_table.copy()
            except AttributeError:
                print('Please filter raw table first using filter_table()\
                    method.')
                return


        # get col names
        col_names = list(transformed)

        # using a dictionary, group col names into replicate groups
        sample_group_names = []
        group_dict = {}
        for col in col_names:
            # if intensity col, get the group name and add to the group dict
            # use groups from re.search to customize group names
            if col in self.sample_cols:
                group_search = re.search(reg_exp, col, flags=re.IGNORECASE)
                group_name = ''
                #print("processing " + col)
                for re_group in group_search.groups():
                    group_name += re_group
                sample_group_names.append(group_name)
                group_dict[col] = group_name

            # if not, group into 'metadata'
            else:
                group_dict[col] = 'metadata'

        sample_groups = list(set(sample_group_names))
        sample_groups.sort()
        self.sample_groups = sample_groups

        # pd function to add the replicate group to the columns
        grouped = pd.concat(dict((*transformed.groupby(group_dict, 1),)), axis=1)

        grouped.columns = grouped.columns.rename("Samples", level=0)
        grouped.columns = grouped.columns.rename("Replicates", level=1)

        self.grouped_table = grouped



    def remove_invalid_rows(self, verbose=True):
        """Remove rows that do not have at least one group that has values
        in all triplicates"""

        try:
            grouped = self.grouped_table.reset_index(drop=True).copy()
        except AttributeError:
            print("Replicates need to be grouped before this method."
                "Please use group_replicates() to group replicates under same sample")
            return

        # reset index
        grouped = self.grouped_table.reset_index(drop=True).copy()
        unfiltered = self.grouped_table.shape[0]

        # Get a list of all groups in the df
        group_list = list(set([col[0] for col in list(grouped) if col[0] != 'metadata']))

        # booleans for if there is a valid value
        filtered = grouped[group_list].apply(np.isnan)
        # loop through each group, and filter rows that have valid values
        for group in group_list:
            # filter all rows that qualify as all triplicates having values
            filtered = filtered[filtered[group].any(axis=1)]

        # a list containing all the rows to delete
        del_list = list(filtered.index)

        # create a new df, dropping rows with invalid data
        filtered_df = grouped.drop(del_list)
        filtered_df.reset_index(drop=True, inplace=True)
        filtered = filtered_df.shape[0]

        if verbose:
            print("Removed invalid rows. " + str(filtered) + " from "
                + str(unfiltered) + " rows remaining.")

        self.preimpute_table = filtered_df


    def remove_invalid_rows_custom(self, mygroup_list, verbose=True):
        """Remove rows that do not have at least one group that has values in at least *two* replicates
           Specify a custom group list
           This function udates the grouped_table, and does not produce preimputa_table
           This means that remove_invalid_rows() cannot be skipped after this method
             """
        try:
            grouped = self.grouped_table.reset_index(drop=True).copy()
        except AttributeError:
            print("Replicates need to be grouped before this method."
                "Please use group_replicates() to group replicates under same sample")
            return

        # reset index
        grouped = self.grouped_table.reset_index(drop=True).copy()
        unfiltered = self.grouped_table.shape[0]

        # Get a list of all groups in the df
        group_list = list(set([col[0] for col in list(grouped) if col[0] != 'metadata']))

        for g in mygroup_list:
            if g not in group_list:
                print(g + " is not found in the grouped table")
                return

        print("Removing invalid rows for " + str(len(mygroup_list)) + " groups")

        group_list = mygroup_list
 
         # booleans for if there is a valid value
        filtered = grouped[group_list].apply(np.isnan)
        # loop through each group, and filter rows that have valid values
        for group in group_list:
            # filter all rows that qualify as all triplicates having values
            filtered = filtered[(filtered[group] == True).sum(axis=1) >= 2 ] # if a row has two or more nan values in a group, keep it in to to_delete list

        # a list containing all the rows to delete
        del_list = list(filtered.index)

        # create a new df, dropping rows with invalid data
        filtered_df = grouped.drop(del_list)
        filtered_df.reset_index(drop=True, inplace=True)
        filtered = filtered_df.shape[0]

        if verbose:
            print("Removed invalid rows. " + str(filtered) + " from "
                + str(unfiltered) + " rows remaining.")

        self.grouped_table = filtered_df


    def bait_impute(self, distance=1.8, width=0.3, local=True):
        """
        bait-imputation for sets of data without enough samples.
        This fx imputes a value from a normal distribution of the left-tail
        of a bait’s capture distribution for the undetected preys using
        multi-processing.
            distance: float, distance in standard deviation from the
            mean of the sample distribution upon which to impute. Default = 0
            width: float, width of the distribution to impute in standard deviations. Default = 0.3
        """

        try:
            imputed = self.preimpute_table.copy()
        except AttributeError:
            try:
                imputed = self.grouped_table.copy()
            except AttributeError:
                print('Please group replicates first using group_replicates()\
                    method.')
                return

        self.bait_impute_params = {'distance': distance, 'width': width}

        # Retrieve all col names that are not classified as metadata
        bait_names = [col[0] for col in list(imputed) if col[0] != 'metadata']
        baits = list(set(bait_names))
        bait_series = [imputed[bait].copy() for bait in baits]
        if local:
            global_mean = 0
            global_stdev = 0

        else:
            # if not using columnwise imputation, calculate global mean and stdev
            all_intensities = imputed[[col for col in list(imputed) if col[0] != 'metadata']].copy()
            global_mean = all_intensities.droplevel('Samples', axis=1).stack().mean()
            global_stdev = all_intensities.droplevel('Samples', axis=1).stack().std()




        bait_params = zip(
            bait_series, repeat(distance), repeat(width), repeat(local),
            repeat(global_mean), repeat(global_stdev))

        # Use multiprocessing pool to parallel impute
        p = Pool()
        impute_list = p.starmap(pool_impute, bait_params)
        p.close()
        p.join()

        for i, bait in enumerate(baits):
            imputed[bait] = impute_list[i]

        self.bait_imputed_table = imputed

    def prey_impute(self, distance=0, width=0.3, thresh=100):
        """
        default mode of imputation. For protein groups with less than threshold number
        of sample number, impute a value from a normal distribution of the prey’s capture
        distribution using multi-processing. Note- most protein groups do not need imputation
        with 12-plate MBR

            distance: float, distance in standard deviation from the mean of the
                sample distribution upon which to impute. Default = 0
            width: float, width of the distribution to impute in standard deviations.
                Default = 0.3
            threshold: int, max number of samples required for imputation
        """

        try:
            imputed = self.preimpute_table.copy()
        except AttributeError:
            try:
                imputed = self.grouped_table.copy()
            except AttributeError:
                print('Please group replicates first using group_replicates()\
                    method.')
                return

        imputed = self.preimpute_table.copy()
        imputed.drop(columns='metadata', inplace=True)
        imputed = imputed.T
        self.prey_impute_params = {'distance': distance, 'width': width,
            'thresh': thresh}

        # Retrieve all col names that are not classified as metadata
        baits = list(imputed)
        bait_series = [imputed[bait].copy() for bait in baits]
        bait_params = zip(
            bait_series, repeat(distance), repeat(width), repeat(thresh))

        # Use multiprocessing pool to parallel impute
        p = Pool()
        impute_list = p.starmap(pool_impute_prey, bait_params)
        p.close()
        p.join()

        for i, bait in enumerate(baits):
            imputed[bait] = impute_list[i]

        imputed = imputed.T

        info_cols = [x for x in list(self.preimpute_table) if x[0] == 'metadata']
        for col in info_cols:
            imputed[col] = self.preimpute_table[col]

        self.prey_imputed_table = imputed


    def generate_export_bait_matrix(self, export=False):
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
        self.bait_list = bait_df.copy()
        bait_df2 = bait_df.copy()
        bait_df2['plot'] = True
        if export:
            bait_df2.to_csv(self.root + self.analysis + '/plotting_exclusion_list.csv',
            index=False)

        # Create a boolean table
        bools = []
        for bait in baits:
            bait_bools = [True if x != bait else False for x in baits]
            temp_bools = pd.DataFrame()
            temp_bools[bait] = bait_bools
            bools.append(temp_bools)
        bait_df = pd.concat([bait_df] + bools, axis=1)
        self.bait_matrix = bait_df.copy()
        if export:
            self.bait_matrix.to_csv(self.root + self.analysis + '/analysis_exclusion_matrix.csv',
                index=False)


def opencell_initial_processing(root, analysis, pg_file='proteinGroups.txt',
        intensity_type='LFQ intensity', impute='bait', distance=1.8, width=0.3,
        thresh=100, local=True, group_regexp=r'(P\d{3})(?:.\d{2})?(_.*)_\d{2}'):

    """
    wrapper script for all the pre-processing up to imputation using
    PyseusRawTables Class. Saves and returns the PyseusRawTables in the
    designated analysis directory

    impute options: 'bait',
    """
    # make directory for analysis folder
    analysis_dir = root + analysis

    if not os.path.isdir(analysis_dir):
        os.mkdir(analysis_dir)

    # Run all the processing methods
    pyseus_tables = RawTables(experiment_dir=root,
        intensity_type=intensity_type, pg_file=pg_file, file_designated=False)
    pyseus_tables.filter_table()
    pyseus_tables.transform_intensities(func=np.log2)
    pyseus_tables.group_replicates(reg_exp=group_regexp)
    pyseus_tables.remove_invalid_rows()
    if impute == 'bait':
        pyseus_tables.bait_impute(distance=distance, width=width, local=local)
    elif impute == 'prey':
        pyseus_tables.prey_impute(distance=distance, width=width, thresh=thresh)
    pyseus_tables.generate_export_bait_matrix()
    # pyseus_tables.save()
    return pyseus_tables


def load_raw_tables(file_dir):
    """
    use pickle to load RawTables class
    """
    return pickle.load(open(file_dir, 'rb', -1))


def select_intensity_cols(orig_cols, intensity_type, spacer=' '):
    """from table column names, return a list of only intensity cols
    rtype: intensity_cols list """
    # new list of intensity cols
    intensity_cols = []
    rename_cols = []

    # create a regular expression that can distinguish between
    # intensity and LFQ intensity
    re_intensity = '(^' + intensity_type + ')'

    # for loop to include all the intensity col names
    intensity_type = intensity_type
    for col in orig_cols:

        # check if col name has intensity str
        if re.search(re_intensity, col):
            sample = re.search(re_intensity + spacer + '(.*)', col).groups()[1]

            intensity_cols.append(col)
            rename_cols.append(sample)

    return intensity_cols, rename_cols


def pool_impute(bait_group, distance=1.8, width=0.3, local=True, global_mean=0, global_stdev=0):
    """target for multiprocessing pool from multi_impute_nans"""

    if local:
        all_vals = bait_group.stack()
        mean = all_vals.mean()
        stdev = all_vals.std()
        # get imputation distribution mean and stdev
        imp_mean = mean - distance * stdev
        imp_stdev = stdev * width
    else:
        # use global mean and stdev
        imp_mean = global_mean - distance * global_stdev
        imp_stdev = global_stdev * width


    # copy a df of the group to impute values
    bait_df = bait_group.copy()

    # loop through each column in the group
    for col in list(bait_df):
        bait_df[col] = bait_df[col].apply(random_imputation_val,
            args=(imp_mean, imp_stdev))
    return bait_df


def pool_impute_prey(bait_group, distance=0, width=0.3, thresh=100):
    """target for multiprocessing pool from multi_impute_nans"""

    if bait_group.count() > thresh:
        return bait_group


    mean = bait_group.mean()
    stdev = bait_group.std()

    # get imputation distribution mean and stdev
    imp_mean = mean - distance * stdev
    imp_stdev = stdev * width

    # copy a df of the group to impute values
    bait_df = bait_group.copy()


    bait_df = bait_df.apply(random_imputation_val,
            args=(imp_mean, imp_stdev))
    return bait_df


def random_imputation_val(x, mean, std):
    """from a normal distribution take a random sample if input is
    np.nan. For real values, round to 4th decimal digit.
    Floats with longer digits will be 'barcoded' by further digits

    rtype: float"""

    if np.isnan(x):
        return np.random.normal(mean, std, 1)[0]
    else:
        return np.round(x, 4)


def sample_rename(col_names, RE, replacement_RE, repl_search=False):
    """
    method to change column names for previewing in notebook
    """

    # start a new col list
    new_cols = []

    # Loop through cols and make quaifying subs
    for col in col_names:
        for i in np.arange(len(RE)):
            if re.search(RE[i], col, flags=re.IGNORECASE):
                replacement = replacement_RE[i]
                if (repl_search) & (len(replacement) > 1):
                    rep_search = re.search(replacement, col,
                                flags=re.IGNORECASE)
                    replacement = ''
                    for group in rep_search.groups():
                        replacement += group

                col = re.sub(RE[i], replacement, col, flags=re.IGNORECASE)
        new_cols.append(col)
    return new_cols


def median_replicates(imputed_df, mean=False, save_info=True, col_str=''):
    """For each bait group, calculate the median of the replicates
    and returns a df of median values

    rtype: median_df pd dataframe"""

    imputed_df = imputed_df.copy()
    # retrieve bait names
    bait_names = [col[0] for col in list(imputed_df) if col[0] != 'metadata']
    bait_names = list(set(bait_names))


    # for each bait calculate medain across replicates and add
    # to the new df
    medians = []
    for bait in bait_names:
        # initiate a new df for medians
        median_cut = pd.DataFrame()
        if mean:
            bait_median = imputed_df[bait].mean(axis=1)
        else:
            bait_median = imputed_df[bait].median(axis=1)
        new_col_name = col_str + bait
        median_cut[new_col_name] = bait_median
        medians.append(median_cut)

    median_df = pd.concat(medians, axis=1)

    if save_info:
        # get info columns into the new df
        info = imputed_df['metadata']
        median_df = pd.concat([median_df, info], axis=1)

    return median_df


def dash_output_table(data_table, sample_cols, metadata_cols):
    """
    Method to force any tables from RawTables class to fit into a standard
    output table used in the custon DASH app.
    Strips any multi-level columns, and create a new level of columns
    that specify samples and metadata
    """

    data_table = data_table.copy()

    if data_table.columns.nlevels > 1:
        # strip the grouping multi-level columns
        data_table = data_table.droplevel('Samples', axis=1)
        data_table.columns.name = None

    # divide into sample table and meta table
    sample_table = data_table[sample_cols].copy()
    meta_table = data_table[metadata_cols].copy()

    # Add appropriate column headings to both
    sample_table = pd.concat([sample_table], keys=['sample'], axis=1)
    meta_table = pd.concat([meta_table], keys=['metadata'], axis=1)

    # join the tables again
    data_table = pd.concat([meta_table, sample_table], axis=1)

    return data_table
