import gzip
from mltoolkit.mlutils.helpers.formatting.general import unescape
from preprocessing.fields import OutputFields, AmazonFields, YelpFields
from mltoolkit.mlutils.helpers.paths_and_files import get_file_name, \
    safe_mkfdir, \
    comb_paths
import csv
import os
import json


def read_yelp_data(path):
    """Reads Yelp data, formats, and adds a dummy category attribute (for cons).

    Args:
        path (str): data path to a file with Yelp reviews.

    Returns: an iterator over pairs of group_id and list of data-units (reviews
        with attributes).

    """
    yelp_to_output_map = {
        YelpFields.BUS_ID: OutputFields.GROUP_ID,
        YelpFields.REV_TEX: OutputFields.REV_TEXT,
        YelpFields.STARS: OutputFields.RATING
    }
    prev_business_id = None
    dus = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            du = json.loads(line)
            business_id = du[YelpFields.BUS_ID]

            du = {yelp_to_output_map[attr]: du[attr] for attr
                  in yelp_to_output_map.keys()}

            du[OutputFields.REV_TEXT] = clean_text(du[OutputFields.REV_TEXT])
            du[OutputFields.CAT] = 'business'

            if prev_business_id is not None and prev_business_id != business_id:
                yield prev_business_id, dus
                dus = []

            prev_business_id = business_id
            dus.append(du)

    if len(dus):
        yield prev_business_id, dus


def read_amazon_data(path, max_revs=None, replace_xml=False):
    """Reads AmazonFields data, formats and enriches by adding the category attribute.

    Args:
        path (str): data path to a file with AmazonFields reviews.
        max_revs (int): the maximum number of reviews to read.
        replace_xml (bool): if set to True will replace XML/HTML symbols with
            proper strings.

    Returns: an iterator over pairs of group_id and list of data-units (reviews
        with attributes).

    """
    amazon_to_output_map = {
        AmazonFields.PROD_ID: OutputFields.GROUP_ID,
        AmazonFields.REV_TEXT: OutputFields.REV_TEXT,
        AmazonFields.OVERALL: OutputFields.RATING
    }
    dus = []
    prev_prod_id = None
    for indx, du in enumerate(parse(path)):
        if any((du_key not in du for du_key in amazon_to_output_map.keys())):
            continue

        prod_id = du[AmazonFields.PROD_ID]

        if replace_xml:
            du[AmazonFields.REV_TEXT] = unescape(du[AmazonFields.REV_TEXT])
        du = {amazon_to_output_map[attr]: du[attr] for attr
              in amazon_to_output_map.keys()}

        # adding the category attribute based on the file name
        du[OutputFields.CAT] = get_file_name(path).lower()

        du[OutputFields.REV_TEXT] = clean_text(du[OutputFields.REV_TEXT])

        if prev_prod_id is not None and prod_id != prev_prod_id:
            yield prev_prod_id, dus
            dus = []

        prev_prod_id = prod_id
        dus.append(du)

        if max_revs and indx >= max_revs - 1:
            break
    if len(dus):
        yield prev_prod_id, dus


def read_csv_file(file_path, sep='\t'):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=sep)
        for item in reader:
            yield item


def write_groups_to_csv(out_dir_path, group_id_to_units, sep='\t'):
    for group_id, group_units in group_id_to_units.items():
        full_file_name = "%s.csv" % group_id
        out_file_path = comb_paths(out_dir_path, full_file_name)
        write_group_to_csv(out_file_path, group_units, sep=sep)


def write_group_to_csv(out_file_path, units, sep="\t"):
    """Writes data units into a CSV file.

    Args:
        out_file_path (str): self-explanatory.
        units (list): list with dicts (review texts and other attributes).
        sep (str): separation in the output csv files.

    Returns: None.

    """
    safe_mkfdir(out_file_path)
    with open(out_file_path, 'w', encoding='utf-8') as f:
        header = None
        for du in units:
            if header is None:
                header = du.keys()
                f.write(sep.join(header) + "\n")
            str_to_write = sep.join([str(du[attr]) for attr in header])
            f.write(str_to_write + '\n')


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def get_act_out_dir_path(out_dir_path, inp_file_path, middle_path):
    """Creates the final/actual output directory path specific to a step."""
    out_file_path = os.path.join(out_dir_path, middle_path,
                                 get_file_name(inp_file_path))
    return out_file_path


def partition(groups, train_part=0.8, val_part=0.1, test_part=0.1):
    """Splits groups into training, validation, and test partitions.

    Args:
        groups (list): list of units (e.g. dicts).
        train_part (float): proportion in [0, 1] of units for training.
        val_part (float): self-explanatory.
        test_part (float): self-explanatory.

    Returns: lists of data-chunks for each.

    """
    assert train_part + val_part + test_part == 1.

    total_size = len(groups)
    train_part_end = int(total_size * train_part)
    val_part_end = train_part_end + int(total_size * val_part)

    train_groups = groups[:train_part_end]
    val_groups = groups[train_part_end:val_part_end]
    if test_part == 0.:
        val_groups += groups[val_part_end:]
        test_groups = []
    else:
        test_groups = groups[val_part_end:]

    return train_groups, val_groups, test_groups


def clean_text(text_str):
    return text_str.replace("\t", '').replace('\n', '')
