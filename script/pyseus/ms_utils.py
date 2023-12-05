import re
import hashlib
import numpy as np
from datetime import datetime


def find_mismatching_target_names(plates_df, hits_df):
    """
    Identify and print gene names that are mismatching between the plates
    dataframe and the hits dataframe so that the names could be manually changed
    """

    # get target names from hits_df
    hits_genes = set([x[0].split('_', 1)[1] for x in list(hits_df) if 'P0' in x[0]])

    # get target names from plates_df
    plate_genes = set(plates_df['target_name'].values.tolist())
    return (hits_genes - plate_genes)


def fdr_mismatching_target_names(plates_df, fdr_df):
    """
    Identify and print gene names that are mismatching between the plates
    dataframe and the hits dataframe so that the names could be manually changed
    """

    # get target names from hits_df
    fdr_genes = set(fdr_df['target'].to_list())

    # get target names from plates_df
    plate_genes = set(plates_df['target_name'].values.tolist())
    return (fdr_genes - plate_genes)


def format_ms_plate(plate_id, source='CZBMPI'):
    """
    Format MS Plate IDs to 'CZBMPI_%04d' % plate_number
    Also allow for 'CZBMPI%04d.d format
    """
    plate_id = str(plate_id)

    result = re.match('^' + source + '_[0-9]{4}(.[0-9])?$',
        plate_id)
    if result is None:
        sub_plate = None
        plate_number = None

        # If the plate ends with subheading such as 0009.2, denoted by the
        # decimal, save the decimal
        sub_result = re.search(r'(\.[0-9]+$)', plate_id)
        if sub_result:
            sub_plate = sub_result.groups()[0]

        # Check for plate numbering
        result = re.search(r'([0-9]+)', plate_id)
        if result:
            plate_number = result.groups()[0]
        try:
            plate_number = int(plate_number)
        except TypeError:
            return None

        plate_id = (source + '_%04d' % plate_number)
        if sub_plate and sub_plate != '.0':
            plate_id = plate_id + sub_plate

    return plate_id


def reformat_pulldown_table(pulldown_df):
    """
    combine multiple rows with different replicates into a single row,
    and add columns for well info for different replicates
    """
    abridged = pulldown_df.copy()

    # drop replicate and pulldown_well_id from abridged, and drop replicates
    abridged.drop(
        columns=['replicate', 'pulldown_well_id'],
        inplace=True
    )
    abridged = abridged.dropna(how='any', subset=['design_id'])
    abridged.drop_duplicates(inplace=True)
    return abridged


def create_protein_group_id(uniprot_ids):
    """
    In the mass spec datasets, protein groups are ID'd by a string
    of semicolon-separated uniprot IDs.
    This function sorts these IDs alphabetically, then hashes them using hashlib.
    The purpose of this is to generate a unique ID from a unique set of uniprot IDs,
    which can then be used as a primary key for the MassSpecProteinGroup table.
    """

    # split the string into a list
    uniprot_ids = sorted(uniprot_ids.split(';'))

    # serialize the sorted list
    serialized_uniprot_ids = str(uniprot_ids)

    # hash the serialized list
    hashed_uniprot_ids = hashlib.sha256(serialized_uniprot_ids.encode('utf-8')).hexdigest()

    return hashed_uniprot_ids, uniprot_ids
