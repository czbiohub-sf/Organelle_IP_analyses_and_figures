from curses import reset_shell_mode
import multiprocessing
import sys
import itertools
import scipy
import random
import re
import pandas as pd
import numpy as np
import anndata as ad
import ternary
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt


from pyseus import basic_processing as bp
from pyseus import primary_analysis as pa
from pyseus import validation_analysis as va
from external import clustering_workflows

from multiprocessing import Queue
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



class SpatialTables():
    """
    SpatialTables class takes as input standard_table from
    AnalysisTables class and performs various processing tools
    that enhance the datasets
    """

    def __init__(self, preprocessed_table=None, hit_table=None, enrichment_table=None,
            target_col='target', prey_col='prey', control_mat=None):
        """
        initiate class with a standard enrichment table

        preprocessed_table: df, the standard output of imputed table from 
        either basic_processing.py or DashWebapp pre-processing

        hit_table: df, SQL-style output table with p-values and enrichments
        enrichment_table: df, wide-style table of enrichments

        target_col: str, column name for the bait or the sample pulldown
        prey_col: str, column name for the prey (usually gene names)



        """
        self.preprocessed_table = preprocessed_table
        self.hit_table = hit_table
        self.target = target_col
        self.prey = prey_col
        self.enrichment_table = enrichment_table
        self.corr = None

        if (hit_table) and (enrichment_table is None):
            self.create_enrichment_table()

        # generate a control mat
        if control_mat is None:
            self.create_default_con_mat()
        else:
            self.control_mat = control_mat

    def create_enrichment_table(self):
        """
        many of the spatial functions require an enrichment table,
        which is a wide-version of the hits table with just enrichments (excluding pvals)
        """

        # create an AnalysisTables class which handles the enrichment conversion
        analysis = pa.AnalysisTables(auto_group=False)

        analysis.simple_pval_table = self.hit_table
        analysis.convert_to_enrichment_table(enrichment='enrichment', simple_analysis=True)

        self.control_mat = analysis.exclusion_matrix.copy()
        self.enrichment_table = analysis.enrichment_table.copy()


    def create_default_con_mat(self):
        """
        generate a default exclusion matrix from the enrichment table
        """

        enrichments = self.enrichment_table.copy()

        baits = list(enrichments['sample'])
        baits.sort()
        bait_df = pd.DataFrame()
        bait_df['Samples'] = baits
        bait_df.reset_index(drop=True, inplace=True)

        # Create a boolean table
        for bait in baits:
            bait_bools = [True if x != bait else False for x in baits]
            bait_df[bait] = bait_bools

        self.control_mat = bait_df.copy()



    def enrichment_corr_control_mat(self, corr=0.4, low_filter=False):
        """
        create a control matrix that has a correlation filter between samples,
        effectively removing samples that are closely related for pval-calculations
        """

        enrichments = self.enrichment_table.copy()
        enrichments = enrichments['sample'].copy()
        if self.control_mat is None:
            print("Please assign control_mat to the Class!")
            return
        else:
            control_mat = self.control_mat
        cols = list(enrichments)

        # get correlation tables
        sample_corrs = enrichments[cols].corr()
        self.sample_corrs = sample_corrs.copy() #save the correlation table
        #print(sample_corrs)

        # get the list of columns in mat table
        mat_samples = control_mat['Samples'].to_list()

        # create a new mat and filter contrasts by correlation
        new_mat = control_mat.copy()
        for col in cols:
            truths = []
            if low_filter:
                passing_corrs = sample_corrs[col][sample_corrs[col] >= corr]
            else:
                passing_corrs = sample_corrs[col][sample_corrs[col] <= corr]
            for sample in mat_samples:
                if sample in passing_corrs:
                    truths.append(True)
                else:
                    truths.append(False)
            new_mat[col] = truths

        self.corr = corr
        self.corr_mat = new_mat
        self.raw_corrs = sample_corrs
        
        #print(self.corr_mat.shape)
        #print(self.corr_mat)


    def new_corr_ARI(self, labels, reference, repeat=5, label_col='organelle',
            merge_col='protein', quant_cols=None, gene_col='Gene names',
            balance_num=68, just_enrichment=False, table_provided=False, def_res=None,
            n_neighbors=None, std_enrich=True):
        """
        
        this is a wrapper function for two main functions:
        1) creating an enrichment table with a given correlation filter matrix (see functions above)
        2) calculating ARI of the enrichment table. 
        The ARI is calculated by umap-nearest neighbors matrix generation, leiden clustering,
        and calculating grid-search ARI maximum

        Theoretically it should be divided into two separate functions, but the two parts
        are separated but the boolean table_provided. 

        labels: df, the ground truth table to be used

        reference: list of strs, all the ground truth labels from the reference table
            to be used in the ARI scoring.
        

        repeat: int, number of repetitions for ARI calculationß

        label_col: str, column name of the reference table that contain the ground truth labels
        merge_col: str, column name of the reference table that contain the gene name identifier

        quant_cols: list of strs, the features (or samples) that are used in leiden clustering and
        subsequent ARI clustering

        gene_col: str, column name of the enrichment table that contain the gene name identifier

        balance_num: int, a cap for how many total observations a ground truth label can have.
            this is used for 'semi-balancing'

        just_enrichment: boolean, if True, just return the enrichment table after correlation filter
            enrichment/pval calculation

        table_provided: skips the corr-filter enrichment/pval calculation as enrichment talbe is provided

        def_res: float, for cases where resolution of leiden clustering is specified by the user, skipping part
            of the grid search

        n_neighbors: list of ints, includes the list of n_neighbors parameter to try in the grid search

        std_enrich: boolean, in corr-filter enrichment calculation, designates whether absolute/relative enrichment
            to calculate


        """

        if table_provided is False:
            # calculate pvals, load the calculated correlation filter matrix
            analysis = pa.AnalysisTables(grouped_table=self.preprocessed_table,
                auto_group=False, exclusion_matrix=self.corr_mat)

            # remove all other samples not necesary for pval table
            grouped = analysis.grouped_table.copy()
            samples = list(self.corr_mat)
            samples.remove('Samples')
            grouped = grouped[samples + ['metadata']]
            analysis.grouped_table = grouped


            # run the pval/enrichment steps
            analysis.simple_pval_enrichment(std_enrich=std_enrich)
            analysis.convert_to_enrichment_table(enrichment='enrichment', simple_analysis=True)
            analysis.convert_to_standard_table(experiment=False, perseus=False)

            enrichments = analysis.enrichment_table.copy()
            self.corr_pval_table = analysis.simple_pval_table.copy()
            self.corr_enrichment_table = enrichments
            self.corr_standard_table = analysis.standard_hits_table.copy()
            
            # end function
            if just_enrichment:
                return
        else:
            enrichments = self.enrichment_table.copy()
            # drop NA, as some future steps require absence of NaN values
            q_cols = [x for x in list(enrichments) if x[1] in quant_cols]
            enrichments = enrichments.dropna(subset=q_cols) 

        # empty lists to fill in through the for loop
        maxes = []
        reses = []
        neighbors = []

        # for loop for repetition of the ground truth sampling + grid search +  max ARI calculation
        for i in np.arange(repeat):
            metadata = enrichments['metadata'].copy()

            # balance labels
            new_labels = []
            label_counts = labels[label_col].value_counts()

            for label in reference:
                label_sampling = labels[labels[label_col] == label]
                if label_counts[label] > balance_num:
                    # this is the semi balancing to cap the number of labels
                    label_sampling = label_sampling.sample(balance_num)
                new_labels.append(label_sampling)

            new_labels = pd.concat(new_labels).reset_index(drop=True)

            # change enrichment table's gene_col to a standard name for merge
            enrichments.rename({gene_col: 'gene_names'}, axis=1, inplace=True)
            metadata = enrichments['metadata'].copy()

            # merge organelle labels by finding all matching proteins in proteingroups
            organelles = []
            for _, row in metadata.iterrows():
                # combinations of gene names are attached by semicolons
                genes = str(row.gene_names).split(';')
                label_match = new_labels[new_labels[merge_col].isin(genes)]
                if label_match.shape[0] > 0:
                    # take (usually only) matching organelle
                    organelles.append(label_match[label_col].iloc[0])
                else:
                    organelles.append('none')

            # hard coding label name for subsequent functions
            metadata['organelles'] = organelles

            enrichments.reset_index(drop=True, inplace=True)
            metadata.reset_index(drop=True, inplace=True)


            # calculate the max ARI resolution
            max, res, neighbor = self.max_ari_summary(enrichments, metadata,
                quant_cols=quant_cols, def_res=def_res, n_neighbors=n_neighbors)

            # after the first grid search, it is very unlikely the discovered parameters
            # will change, save those parameters for the next repetitions
            if def_res is None:
                def_res = res
            if n_neighbors is None:
                n_neighbors = [neighbor]

            # append max_ari and parameters to the list
            maxes.append(max)
            reses.append(res)
            neighbors.append(neighbor)

        corr_str = ''
        if self.corr is not None:
            corr_str = str(self.corr) + '_'

        # wrap the results in a dataframe
        summary_table = pd.DataFrame()
        summary_table[corr_str + 'max_ARI'] = maxes
        summary_table[corr_str + 'res'] = reses
        summary_table[corr_str + 'neighbors'] = neighbors

        self.summary_table = summary_table.copy()


    def max_ari_summary(self, enrichments, metadata, quant_cols=None,
            n_neighbors=None, def_res=None):
        """
        This is a script to generate AnnData from the enrichment table
        and the metadata table, and search through n_neighbors parameter 
        to find the maximum ARI. 

        enrichments: df, standard enrichment table from pyseus output
        metadata: df, metadata selection from the enrichment table

        quant_cols: list of strs, the features (or samples) that are used in leiden clustering and
        subsequent ARI clustering

        n_neighbors: list of ints, includes the list of n_neighbors parameter to try 
            in the grid search

        def_res: float, for cases where resolution of leiden clustering is specified by the user, 
            skipping part of the grid search

        """


        quants = enrichments['sample'].copy()
        if quant_cols is not None:
            quants = quants[quant_cols].copy()

        # if n_neighbors is unspecified, use the default list given below
        if n_neighbors is None:
            n_neighbors = [2, 5, 10, 20, 50]

        adata = ad.AnnData(quants, obs=metadata)
        

        # outputs to save in the for loop
        aris = []
        resolutions = []

        # if there is only one element in n_neighbor, process it without parallel pool
        if len(n_neighbors) == 1:
            output = calculate_max_ari(
                adata, n_neighbors[0], 'organelles', res=0.4, n_random_states=1, def_res=def_res)
            aris.append(output[0])
            resolutions.append(output[1])
        else:
            # multiprocessing Pool to search through n_neighbors parameter
            p = Pool()
            # note the hard coded 'organelles' as the reference column from new_corr_ARI() function
            multiargs = zip(repeat(adata), n_neighbors, repeat('organelles'),
                repeat(0.4), repeat(2), repeat(def_res))
            # using calculate_max_ari function for the pool
            outputs = p.starmap(calculate_max_ari, multiargs)
            p.close()
            p.join()
            aris = [x[0] for x in outputs]
            resolutions = [x[1] for x in outputs]

        # find the max ARI and the corresponding n_neighbor / leiden resolution 
        max = np.max(aris)
        max_idx = aris.index(max)
        res = resolutions[max_idx]
        n_neighbor = n_neighbors[max_idx]

        return max, res, n_neighbor


    def grouped_reference_testing(self, labels, condition='-infected',
            merge_col='Gene names', label_col='organelle'):
        """
        Using the enrichment table, test the organellar difference between
        a contrast and control group, return a pval/enrichment table.
        """

        enrichment = self.enrichment_table.copy()
        labels = labels[[merge_col, label_col]].copy()

        # parse samples that are experiment controls
        samples = list(enrichment['sample'])
        control_samples = [x for x in samples if condition not in x]
        condition_samples = [x for x in samples if condition in x]
        control_samples.sort()
        print(condition_samples)

        # control-experiment pair dictionary
        sample_dict = {}
        for sample in control_samples:
            condition_pair = [x for x in condition_samples if sample in x]
            if len(condition_pair) > 0:
                sample_dict[sample] = condition_pair[0]

        # merge enrichment table with labels
        enrichment = enrichment.droplevel(0, axis=1)
        merged = enrichment.merge(labels, on=merge_col, how='left')

        # get unique labels except nans
        orgs = merged[label_col].unique()[1:]
        pvals = pd.DataFrame()
        pvals['organelle'] = orgs

        # find t-test significance by each organelle
        for sample in sample_dict.keys():
            cond_sample = sample_dict[sample]

            control = merged[[sample, label_col]].copy()
            conditioned = merged[[cond_sample, label_col]].copy()

            pvs = []
            for org in orgs:
                control_orgs = control[control[label_col] == org]
                condition_orgs = conditioned[conditioned[label_col] == org]
                pval = scipy.stats.ttest_ind(
                    control_orgs[sample], condition_orgs[cond_sample])[1]
                pval = np.round(-1 * np.log10(pval), 2)
                pvs.append(pval)

            pvals[sample] = pvs

        self.ref_pvals_table = pvals


class RForestScoring():
    """
    This is a class for scoring of a dataset based on default RandomForest
    prediction algorithm, practically testing how much information is in the
    dataset for prediction power.

    """
    def __init__(self, dataset, labels, reference, dataset_gene_col, reference_gene_col,
            reference_col, quant_cols):
        """
        dataset: df, standard enrichment table from pyseus
        reference: df, reference table with ground truths
        dataset_gene_col: str, column name in dataset that encompasses gene identifiers to merge
            with the reference
        reference_gene_col: str, column name in reference that encompasses gene identifiers to merge
            with the reference

        reference_col: str, column name that has the reference labels
        quant_cols: list of strs, all the features from the dataset to be used

        """

        # remove the sample / metadata headers
        dataset = dataset.droplevel(0, axis=1).copy()
        # change referenge gene col to match merge
        labels = labels.rename(columns={reference_gene_col: dataset_gene_col}).copy()
        labels = labels[labels[reference_col].isin(reference)]

        dataset = dataset.merge(labels, how='left', on=dataset_gene_col
            ).drop_duplicates()


        self.table = dataset
        self.ref_col = reference_col
        self.quant_cols = quant_cols


    def multiclass_balance_scale_split_predict(self, max_sample=150, predict=False):
        # initiate variables
        table = self.table.copy()
        ref_col = self.ref_col
        quant_cols = self.quant_cols.copy()

        all_cols = quant_cols + [ref_col]
        table = table[all_cols]

        # find value counts of a label
        if predict:
            table2 = table.copy()
            X_all = table2[quant_cols].values


        table.dropna(inplace=True)
        num_labels = table[ref_col].nunique()
        refs = table[ref_col].unique()
        refs.sort()

        # limit samples to max number alloted, this is semi-balancing
        samples = []
        for ref in refs:
            sample = table[table[ref_col] == ref]
            if sample.shape[0] <= max_sample:
                samples.append(sample)
            else:
                random_sample = sample.sample(max_sample)
                samples.append(random_sample)

        # concat all the reference samples
        balanced = pd.concat(samples).reset_index(drop=True)

        labels = pd.factorize(balanced[ref_col])
        definitions = labels[1]

        balanced['label'] = labels[0]

        # scale the quant columns with a StandardScaler
        X = balanced[quant_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        y = balanced['label'].values

        # split and standard scale
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


        # Random Forest Classifier
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        # Reverse factorizers
        reversefactor = dict(zip(range(num_labels), definitions))

        # return predictions
        if predict:
            X_all_scaled = scaler.fit_transform(X_all)

            y_pred = classifier.predict(X_all_scaled)
            y_predicted = np.vectorize(reversefactor.get)(y_pred)
            return y_predicted

        y_pred = classifier.predict(X_test)



        y_tested = np.vectorize(reversefactor.get)(y_test)
        y_predicted = np.vectorize(reversefactor.get)(y_pred)


        return y_tested, y_predicted, classifier


    def one_balance_scale_split_predict(self, ref, return_x_y=False, balance_multiplier=1,
            max_sample=100):
        # initiate variables
        table = self.table.copy()
        ref_col = self.ref_col
        quant_cols = self.quant_cols.copy()

        # count sample size for the reference being tested
        ref_sample = table[table[ref_col] == ref]
        sample_size = ref_sample.shape[0]

        # balance reference with all other samples
        # balance other samples by annotation too
        rest = table[table[ref_col] != ref]
        rest = rest.dropna(subset=[ref_col])

        # balance the rest of the annotations for training/test
        refs = rest[ref_col].unique()
        refs.sort()

        # limit samples to max number alloted, this is semi-balancing
        samples = []
        for ref in refs:
            sample = rest[rest[ref_col] == ref]
            if sample.shape[0] <= max_sample:
                samples.append(sample)
            else:
                random_sample = sample.sample(max_sample)
                samples.append(random_sample)

        # concat all the reference samples
        others_balanced = pd.concat(samples).reset_index(drop=True)



        others = others_balanced.sample(sample_size * balance_multiplier)
        others[ref_col] = 'Others'

        balanced = pd.concat([ref_sample, others]).reset_index(drop=True)

        labels = pd.factorize(balanced[ref_col])
        definitions = labels[1]

        balanced['label'] = labels[0]


        X = balanced[quant_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        y = balanced['label'].values

        if return_x_y:
            return X_scaled, y, definitions

        # split and standard scale
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        # Random Forest Classifier
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        # Reverse factorizers
        reversefactor = dict(zip(range(2), definitions))

        y_tested = np.vectorize(reversefactor.get)(y_test)
        y_predicted = np.vectorize(reversefactor.get)(y_pred)

        return y_tested, y_predicted, classifier


    def repeat_collect_tests(self, one_vs_all=False, ref=None, max_sample=150, repeats=100):
        """
        Repeat either 1 v all or multiclass random forest tests and return
        confusion tables for precision and recall
        """
        tests = []
        predictions = []

        # repeat random forest prediction tests and save prediction results
        for _ in np.arange(repeats):
            if one_vs_all:
                y_tested, y_predicted, _ = self.one_balance_scale_split_predict(ref=ref)
            else:
                y_tested, y_predicted, _ = self.multiclass_balance_scale_split_predict(
                    max_sample=max_sample)
            tests.append(y_tested)
            predictions.append(y_predicted)

        # concatenate all prediction results
        all_tests = np.concatenate(tests)
        all_preds = np.concatenate(predictions)

        # generate confusion matrix
        recall_table, precision_table = self.confusion_precision_recall(
            all_tests, all_preds)

        return recall_table, precision_table


    def confusion_precision_recall(self, tests, predictions, exp=''):
        """
        class method to create confusion chart from the output of repeat_collect_tests
        function
        """

        tests = tests
        predictions = predictions
        cross_recall = pd.crosstab(
            tests, predictions, rownames=['Actual Compartment'],
            colnames=['Predicted Compartment']).apply(
                lambda r: np.round(r/r.sum(), 3), axis=1)

        cross_precision = pd.crosstab(
            tests, predictions, rownames=['Actual Compartment'],
            colnames=['Predicted Compartment']).apply(
                lambda r: np.round(r/r.sum(), 3), axis=0)

        orgs = list(cross_recall)
        recall_table = {}
        precision_table = {}
        for org in orgs:
            recall_table[org] = cross_recall[org][org]
            precision_table[org] = cross_precision[org][org]

        recall_table = pd.DataFrame(recall_table, index=[exp])
        precision_table = pd.DataFrame(precision_table, index=[exp])

        return recall_table, precision_table


class InfectedViz():
    """
    This is a class used for visualisations of TIC at organelle,
    N/O/C, and whole cell level.
    """

    def __init__(self, diff_table=None, whole_cell=None, whole_medians=None, nocs=None,
            query_prot=None):
        self.diff_table = diff_table
        self.whole_cell = whole_cell
        self.whole_meds = whole_medians
        self.nocs = nocs
        self.query = query_prot

    def single_prot_orgip_bars(self, pval_thresh=2, wt=False,
            ylim=[-8, 4], figsize=(10, 5), full_name=False, style='ggplot'):
        """
        create a bar chart for a single protein's enrichment/depletion in org_IPs
        """
        diffs = self.diff_table.copy()

        # query org_ip rows with the matching target name
        search = diffs[diffs['Gene names'] == self.query]

        # methods if there is no exact match
        if search.shape[0] == 0:
            search = diffs[diffs['Gene names'].apply(lambda x: self.query in x)]

            if search.shape[0] == 0:
                print("no exact or similar match found, try another search.")
                return
        else:
            # if there is more than one match
            found_preys = search['Gene names'].unique()
            if len(found_preys) > 1:
                print('Multiple gene names found in that search, try one of the following:')
                print(found_preys)
                return

        search.reset_index(drop=True, inplace=True)

        if wt is False:
            # remove WTs
            samples = search[~search['target'].apply(lambda x: 'WT' in x)].copy()
        else:
            samples = search.copy()

        if not full_name:
            samples['target'] = samples['target'].apply(lambda x: x.split('-')[1])

        # find org-IPs where the protein was significantly and save the data
        sigs = samples[samples['pvals'] >= pval_thresh]
        sig_pts = []
        for i, row in sigs.iterrows():
            idx = row.name
            enrich = row.enrichment
            sig_pts.append([idx, enrich])

        # generate figure
        plt.style.use(style)
        fig = samples.plot.bar(x='target', y='enrichment', figsize=figsize)
        fig.set_ylim(ylim)
        fig.set_ylabel('Differential Abundance (log2)', fontsize=14)
        fig.set_xlabel('Org-IP Pulldown', fontsize=14)
        plt.yticks(fontsize=13)
        _ = plt.xticks(rotation=45, fontsize=12)
        fig.get_legend().remove()
        sig_lim = 10 ** (-1 * pval_thresh)
        fig.annotate('* : p-val < ' + str(sig_lim), xy=(0.8, 0.95), xycoords='figure fraction',
                    size=14, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='w'))

        for sig in sig_pts:
            if sig[1] > 0:
                _ = plt.text(x=sig[0] - .15, y=sig[1] + .2, s='*', fontsize=20)
            if sig[1] < 0:
                _ = plt.text(x=sig[0] - .15, y=sig[1] - 1, s='*', fontsize=20)

        return fig

    def whole_cell_protein_metrics(self, show=True):
        """
        return bar plots and histograms on whole cell abundances of the
        specific protein
        """
        whole = self.whole_cell.copy()

        # query org_ip rows with the matching target name
        search = whole[whole['Gene names'] == self.query]

        # methods if there is no exact match
        if search.shape[0] == 0:
            search = whole[whole['Gene names'].apply(lambda x: self.query in x)]

            if search.shape[0] == 0:
                print("no exact or similar match found, try another search.")
                return
        else:
            # if there is more than one match
            found_preys = search['Gene names'].unique()
            if len(found_preys) > 1:
                print('Multiple gene names found in that search, try one of the following:')
                print(found_preys)
                return
        search.reset_index(drop=True, inplace=True)

        # this is a very hacky way of finding infected vs uninfected
        # will address later
        infecteds = [x for x in list(search) if x[:3] == 'inf']
        uninfs = [x for x in list(search) if x[:3] == 'uni']

        infected = search[infecteds].T.stack().values
        uninfected = search[uninfs].T.stack().values

        inf_mean = np.mean(infected)
        inf_std = np.std(infected)
        uninf_mean = np.mean(uninfected)
        uninf_std = np.std(uninfected)

        # save the calculations for uses in other methods
        self.whole_cell_infected_mean = inf_mean
        self.whole_cell_infected_std = inf_std
        self.whole_cell_uninfected_mean = uninf_mean
        self.whole_cell_uninfected_std = uninf_std

        means = [uninf_mean, inf_mean]
        errs = [uninf_std, inf_std]

        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(4, 5))
        barlist = ax.bar([0, 1], means, yerr=errs, align='center',
            alpha=0.9, ecolor='black', capsize=10, width=0.5)

        barlist[0].set_color('#00a5ff')
        barlist[1].set_color('#dc0000b2')

        ax.yaxis.grid(True)

        # Save the figure and show
        plt.title(self.query + ' Abundance')
        plt.ylabel('Abundance (log2 intensity)', fontsize=13)
        plt.xticks([0, 1], ['Uninfected', 'Infected'], fontsize=14)
        plt.xlim(-0.5, 1.5)
        plt.ylim([0, 30])
        plt.tight_layout()
        if show is False:
            plt.close(fig)

        return fig

    def whole_prot_vs_hists(self, diffs=False):
        """
        plot the single protein data on top of the proteome histogram
        """
        whole_meds = self.whole_meds.copy()
        inf_mean = self.whole_cell_infected_mean
        uninf_mean = self.whole_cell_uninfected_mean

        if diffs is True:
            fig = whole_meds['diffs'].plot.hist(bins=200, alpha=0.6, density=True,
                figsize=(9, 5), color='#8491b4b2', label='All proteins')
            plt.xlim(-8, 8)
            plt.title('Distribution of the Infected vs Control abundance')
            plt.ylabel('Frequency', fontsize=14)
            plt.yticks([])
            plt.xlabel('Difference (log2)', fontsize=14)
            plt.xticks(fontsize=14)
            plt.axvline(inf_mean - uninf_mean,
                color='#dc0000b2', label='Diff. in ' + self.query, linewidth=3)
            _ = plt.legend(fontsize=13)
            return fig

        else:
            fig = whole_meds['uninfected-whole'].plot.hist(bins=70, alpha=0.8, density=True,
                figsize=(9, 5), color='#8491b4b2', label='Uninfected')
            plt.title('Intensity distribution of the HEK293 proteome')
            plt.ylabel('Frequency', fontsize=14)
            plt.yticks([])
            plt.xlabel('Abundance (log2 intensity)', fontsize=14)
            plt.axvline(inf_mean, color='#dc0000b2', label=self.query + ' - infected', linewidth=3)
            plt.axvline(uninf_mean, color='#00a5ff', label=self.query + ' - uninfected', linewidth=3)
            plt.legend(fontsize=13)
            _ = plt.xticks(fontsize=14)

            return fig

    def ternary_prot(self):

        nocs = self.nocs.copy()
        # query org_ip rows with the matching target name
        search = nocs[nocs['Gene names'] == self.query]

        # methods if there is no exact match
        if search.shape[0] == 0:
            search = nocs[nocs['Gene names'].apply(lambda x: self.query in x)]

            if search.shape[0] == 0:
                print("no exact or similar match found, try another search.")
                return
        else:
            # if there is more than one match
            found_preys = search['Gene names'].unique()
            if len(found_preys) > 1:
                print('Multiple gene names found in that search, try one of the following:')
                print(found_preys)
                return
        search.reset_index(drop=True, inplace=True)

        # get nocs for uninfected and infected
        uninf_nocs = search[['NOC2-cytosolic', 'NOC2-organellar',
            'NOC2-nuclear']].T.stack().values
        inf_nocs = search[['NOC2-infected-cytosolic', 'NOC2-infected-organellar',
            'NOC2-infected-nuclear']].T.stack().values

        plt.style.use('default')
        # Boundary and Gridlines
        scale = 1
        figure, tax = ternary.figure(scale=scale)


        figure.set_figheight(6)
        figure.set_figwidth(6.6)

        tax.boundary(linewidth=2.0)
        tax.gridlines(color="black", multiple=0.2, linewidth=1)
        tax.gridlines(color="black", multiple=0.1, linewidth=0.25)

        # Set ticks
        tax.ticks(axis='lbr', multiple=0.2, linewidth=2, offset=0.03,
        tick_formats="%.1f", fontsize=14)

        fontsize = 16
        # axis labels
        tax.left_axis_label("1K Fraction", fontsize=fontsize, position=[-.12, 0.22, 0.4])
        tax.right_axis_label("24K Fraction", fontsize=fontsize, position=[0.06, 1.12, 0])
        tax.bottom_axis_label("Supernatant", fontsize=fontsize, position=[1, -.055, 0.5])
        tax._redraw_labels()

        # scatterplot C, N, O
        tax.scatter([uninf_nocs], s=140, color='#8491b4b2', label='Uninfected', alpha=1)
        tax.scatter([inf_nocs], s=140, color='#e64b35b2', label='Infected', alpha=1)

        # legend
        tax.legend(loc='upper left', fontsize=12)

        # organellar line
        p1 = (0.15, 0.3, 0)
        p2 = (0, 0.3, 0)
        tax.line(p1, p2, linewidth=3, linestyle='--')
        p3 = (0.15, 0.85, 0)
        p4 = (0.15, 0.3, 0)

        tax.line(p3, p4, linewidth=3, linestyle='--')

        # nuclear line
        tax.right_parallel_line(0.85, linewidth=3, color='#dc0000b2', linestyle='--')
        tax.left_parallel_line(0.85, linewidth=3, color='#7cae00', linestyle='--')

        plt.box(on=None)
        tax.clear_matplotlib_ticks()

        return tax




def calculate_max_ari(adata, n_neighbors, ground_truth_label, res, n_random_states, def_res=None):
    """
    Calculate max ARI using leiden clustering and UMAP n-neighbor metric

    adata: AnnData of the enrichment table from Pyseus

    n_neighbors: int, number of neighbors to use for nearest neighbor matrix generation

    ground_truth_label: str, column name in AnnData that contain ground truth labels

    res: float, parameter used to generate range of resolutions to search for max ARI

    n_random_states: specific number of random_states to test for in leiden clustering

    """
    adata = adata.copy()

    # start clusteringworkflow class / preprocessing
    cluster = clustering_workflows.ClusteringWorkflow(adata=adata)
    cluster.preprocess(n_pcs=None)
    # umap neighbor calculation and ARI calculation
    cluster.calculate_neighbors(n_pcs=None, n_neighbors=n_neighbors)
    ari_table = cluster.calculate_ari(ground_truth_label=ground_truth_label,
        res=res, n_random_states=n_random_states, def_res=def_res)

    # find max ARI, and corresponding resolution
    res_groups = ari_table.groupby('resolution').mean()
    max_ari = res_groups['ari'].max()
    # resolution where ARI is max
    ari_res = res_groups['ari'].idxmax()

    return [max_ari, ari_res]
