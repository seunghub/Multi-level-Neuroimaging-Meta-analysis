# %%
import re
from functools import partial

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from loguru import logger
from sklearn import preprocessing

from src.constants import DATA_PATH, FMRI_DATA_PATH

preprocessing_on_samples = partial(
    preprocessing.scale, with_mean=True, with_std=True, axis=1
)


def pad_group(peaks, max_number_of_peaks):
    padded_group = np.pad(
        peaks[["X", "Y", "Z"]].values, ((0, max_number_of_peaks - len(peaks)), (0, 0))
    )
    mask = np.concatenate(
        [
            np.ones((len(peaks), 1)),
            np.zeros((max_number_of_peaks - len(peaks), 1)),
        ],
        axis=0,
    )

    return padded_group, mask


def pad_peaks(peaks_df, max_number_of_peaks):
    groups = peaks_df.groupby("image_path")

    images_paths = []
    padded_peaks = []
    masks = []
    for image_path, group in groups:
        if len(group) >= max_number_of_peaks:
            logger.info(
                f"Image {image_path} contains {len(group)} peaks\n which is more than {max_number_of_peaks} peaks. Ignoring it."
            )
            continue
        images_paths.append(image_path)
        padded_group, mask = pad_group(group, max_number_of_peaks)
        padded_peaks.append(padded_group)
        masks.append(mask)

    return images_paths, np.array(padded_peaks), np.array(masks)


def mask_rows(mask, *args):
    return (arg[mask] for arg in args)


def highly_corr_cols(df, thresh=1.0, return_indices=False, verbose=False):
    df_corr = df.corr()
    to_keep = []
    for i, label in enumerate(df.columns):
        strong_correlates = np.where(np.abs(df_corr.values[i, :i]) >= thresh)[0]
        if len(strong_correlates):
            if verbose:
                print(
                    df.columns[i],
                    "is strongly correlated to",
                    list(df.columns[strong_correlates]),
                    "-> removing it",
                )
        else:
            to_keep += [i]

    if return_indices:
        return to_keep
    else:
        return df.iloc[:, to_keep]


def highly_corr_cols_np(
    np_array, cols, thresh=1.0, return_indices=False, verbose=False
):
    df = pd.DataFrame(np_array, columns=cols)
    if return_indices:
        return highly_corr_cols(df, thresh, return_indices, verbose)
    else:
        return highly_corr_cols(df, thresh, return_indices, verbose).values


def lookup(
    pattern,
    df,
    axis=0,
    case=True,
    regex=True,
    col_white_list=None,
    col_black_list=None,
    verbose=False,
):
    """
    Looks for a given term (string) in a dataframe
    and returns the mask (on the chosen axis) where it was found.

    Parameters:
    -----------
    :pattern: string
        The string (can be a regex) to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :axis: int (0 or 1) or None
        The axis of the desired mask:
            - 0 to get the lines where the term was found
            - 1 to get the columns
            - None to get a 2D mask
    :case: boolean
        If True, the lookup is case sensitive.
    :regex: boolean
        If True, the pattern is matched as a regex.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """

    # select the proper columns where to look for the term
    if col_white_list:
        df_explored = df[col_white_list]
    else:
        df_explored = df.copy()

    if col_black_list:
        df_explored = df_explored.drop(col_black_list, axis=1)

    df_explored = df_explored.select_dtypes(include=["object"])

    if verbose:
        print(
            "> The term '" + pattern + "' will be looked for in the following columns:",
            df_explored.columns,
        )

    # does the lookup
    mask = np.column_stack(
        [
            df_explored[col]
            .astype(str)
            .str.contains(pattern, case=case, regex=regex, na=False)
            for col in df_explored
        ]
    )
    if verbose:
        print("> Found values:", mask.sum())

    if axis is not None:
        if axis == 0:
            mask = mask.any(axis=1)
        else:
            mask = mask.any(axis=0)
        if verbose:
            print("> Found entries along axis", axis, ":", mask.sum())

    return mask


def unit_tagger(
    pattern,
    df,
    tag=None,
    label_col=None,
    reset=False,
    case=False,
    regex=False,
    col_white_list=None,
    col_black_list=None,
    verbose=False,
):
    """
    Looks for a given term (string) in a dataframe
    and adds a corresponding tag to the rows where it is found.

    Parameters:
    -----------
    :pattern: string
        The string (can be a regex) to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :tag: string or None
        The tag to add if the pattern is found.
        If None, the pattern is used as the tag.
        (try not to use pattern with complex regex as column name)
    :label_col: string or None
        The name of the column where the tag should be added.
        If None, a new column with the name of the tag is created
        and the tag presence is reported as a boolean.
    :reset: boolean
        If True (only relevant if label_col is not None),
        the tag column is set to an empty string.
        (this happens inplace, be careful not to delete useful data)
    :case: boolean
        If True, the lookup is case sensitive.
    :regex: boolean
        If True, the pattern is matched as a regex.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """
    df_labelled = df
    if tag is None:
        tag = pattern

    if (label_col is not None) and ((label_col not in df.columns) or reset):
        df_labelled.loc[:, label_col] = ""

    mask = lookup(
        pattern=pattern,
        df=df,
        axis=0,
        case=case,
        regex=regex,
        col_white_list=col_white_list,
        col_black_list=col_black_list,
        verbose=verbose,
    )

    if verbose:
        print(">>> Number of tags found for the tag '{}': {}".format(tag, mask.sum()))

    if label_col is not None:
        df_labelled.loc[mask, label_col] = df_labelled.loc[mask, label_col] + tag + ","
    else:
        df_labelled.loc[:, tag] = mask

    return df_labelled


def vocab_tagger(
    vocab,
    df,
    label_col=None,
    reset=False,
    case=False,
    col_white_list=None,
    col_black_list=None,
    verbose=False,
):
    """
    Looks for a given term (string) in a dataframe
    and adds a corresponding tag to the rows where it is found.

    Parameters:
    -----------
    :vocab: list of strings
        The strings to look for.
    :df: pandas.DataFrame
        The dataframe where the string is looked for.
    :label_col: string or None
        The name of the column where the tags should be added.
        If None, a new column with the name of the tag is created for each tag
        and the tag presence is reported as a boolean.
    :reset: boolean
        If True (only relevant if label_col is not None),
        the tag column is set to an empty string.
        (this happens inplace, be careful not to delete useful data)
    :case: boolean
        If True, the lookup is case sensitive.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """
    df_labelled = df

    for tag in vocab:
        df_labelled = unit_tagger(
            tag,
            df_labelled,
            tag=None,
            label_col=label_col,
            reset=reset,
            case=case,
            regex=False,
            col_white_list=col_white_list,
            col_black_list=col_black_list,
            verbose=verbose,
        )

    return df_labelled


def parallel_vocab_tagger(n_jobs, vocab, df, **kwargs):
    """Parallel version of the `vocab_tagger` function."""
    df_splits = np.array_split(df, n_jobs * 3)

    vocab_tagger_function = partial(vocab_tagger, vocab=vocab, **kwargs)
    df_splits_processed = Parallel(n_jobs)(
        delayed(vocab_tagger_function)(df=df) for df in df_splits
    )

    return pd.concat(df_splits_processed, axis=0)


def dumb_tagger(
    df,
    split_regex=r"[\s-_]+",
    label_col="tags",
    min_chars=3,
    vocab=None,
    keep_figures=False,
    col_white_list=None,
    col_black_list=None,
    verbose=False,
):
    """Takes the str columns of a dataframe, splits them
     and considers each separate token as a tag.

    Parameters:
    -----------
    :df: pandas.DataFrame
        The dataframe that will be labelled.
    :split_regex: string
        The regex used to split the strings.
        Ex: r"[\W_]+" (default) to split character strings
                separated by any non-letter/figure character
            r",[\s]*" to split multi-words strings separated by commas
    :label_col: string or None
        The name of the column where the tags should be added.
        If None, a new column is created FOR EACH TAG encountered
        and the tag presence is reported as a boolean.
        (be careful, the None value can result in huge dataframes)
    :min_chars: int > 0
        The minimal number (included) of characters for a tag to be kept.
        (should be at least 1 since you might get empty strings)
    :vocab: list or None
        If not None : the vocabulary the tags should be extracted from
    :keep_figures: boolean
        If True, purely numerical tags are kept, else they are removed.
    :col_white_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns the lookup will be restricted to.
    :col_black_list: List of strings (or any iterable that returns strings)
        If not None: the names of the columns where the lookup will not occur.
    :verbose: boolean
        If True, information is printed about the lookup.
    """

    # select the proper columns where to look for the term
    if col_white_list:
        df_labelled = df[col_white_list]
    else:
        df_labelled = df.copy()

    if col_black_list:
        df_labelled = df_labelled.drop(col_black_list, axis=1, inplace=True)

    df_labelled = df_labelled.select_dtypes(include=["object"])

    df_res = pd.DataFrame(index=df_labelled.index)

    if verbose:
        print(
            "> The term will be looked for in the folowing columns:",
            df_labelled.columns,
        )

    # Concatenation of all columns into a single one separated by spaces
    full_text = df_labelled.apply(" ".join, axis=1)
    full_text = full_text.str.lower()

    # Splitting with chosen regex
    tags = full_text.str.split(split_regex)

    # Tags cleaning according to criteria
    # remove figures-only tags
    if not keep_figures:

        def figures_only(x):
            return re.fullmatch("[0-9]+", x) is None

        tags = tags.apply(lambda x: list(filter(figures_only, x)))

    # remove too-short tags
    def long_enough(x):
        return len(x) >= min_chars

    tags = tags.apply(lambda x: list(filter(long_enough, x)))

    # trim tags (removes spaces at the beginning/end)
    tags = tags.apply(lambda label_list: [tag.strip() for tag in label_list])

    # remove tags outside of authorized vocabulary
    if vocab is not None:
        in_vocab = lambda x: x in vocab
        tags = tags.apply(lambda x: list(filter(in_vocab, x)))

    # returns tags either as lists within a single "tag" column
    #                  or as single booleans in per-tag columns
    if label_col:
        df_res[label_col] = tags
    else:
        labels_dummies = pd.get_dummies(tags.apply(pd.Series).stack()).sum(level=0)
        df_res = labels_dummies

    return df_res

# %% All functions that are useful to this experiment
def convert_labels_to_one_hot(labels):
    with open(FMRI_DATA_PATH / "cogatlas_concepts.txt", encoding="utf-8") as f:
        concept_names = [line.rstrip("\n") for line in f]
        concept_names = sorted(
            [concept_name.strip().lower() for concept_name in concept_names]
        )

    Y = dumb_tagger(labels, split_regex=r",\s*", vocab=concept_names, label_col=None)

    # Extract vocabulary of labels present in the dataset
    vocab_orig = np.array(Y.columns)
    logger.info(
        f"Number of labels in the whole dataset (TRAIN+TEST): {len(vocab_orig)}"
    )

    # Convert Y to np.array of int
    Y = Y.values * 1

    return Y, vocab_orig


def filter_empty_labels(*arrays, labels, meta_fmris):
    mask_labelled = ~labels.iloc[:, 0].isna()
    *arrays, labels, meta_fmris = mask_rows(
        mask_labelled,
        *arrays,
        labels,
        meta_fmris,
    )

    return *arrays, labels, meta_fmris


def filter_fmris(*arrays, labels, meta_fmris):
    # TODO: Split again the following filters
    # (at least one positive label, blacklist)
    *arrays, labels, meta_fmris = filter_empty_labels(
        *arrays,
        labels=labels,
        meta_fmris=meta_fmris,
    )
    Y, vocab_orig = convert_labels_to_one_hot(labels)

    # In case the labels did not come from the proper vocabulary,
    #   remove the fmris without any label
    mask_label_checked = Y.sum(axis=1) != 0
    meta_fmris, *arrays, Y = mask_rows(mask_label_checked, meta_fmris, *arrays, Y)

    # Remove maps from blacklist if present
    mask_not_blacklisted = np.full(len(meta_fmris), True)
    blacklist = {"collection_id": [4343]}
    for blacklist_key in blacklist:
        mask_not_blacklisted = mask_not_blacklisted & ~meta_fmris[blacklist_key].isin(
            blacklist[blacklist_key]
        )

    meta_fmris, *arrays, Y = mask_rows(mask_not_blacklisted, meta_fmris, *arrays, Y)

    logger.info(f"Number of fMRIs with labels: {len(meta_fmris)}")
    logger.info(f"Number of labels in Train: {len(vocab_orig)}")

    return *arrays, Y, meta_fmris, vocab_orig


def filter_rare_labels(*arrays, Y, meta_fmris, vocab_orig, test_collections):
    # Filtering labels with too few instances in train
    mask_test = meta_fmris["collection_id"].isin(test_collections)

    min_train_label = 10
    colmask_lab_in_train = Y[~mask_test].sum(axis=0) >= min_train_label

    number_of_rare_labels = len(vocab_orig) - int(colmask_lab_in_train.sum())
    logger.info(f"Removed {number_of_rare_labels} labels that were too rare")

    # updating X and Y
    Y = Y[:, colmask_lab_in_train]
    mask_lab_in_train = np.sum(Y, axis=1) != 0
    meta_fmris, *arrays, Y = mask_rows(mask_lab_in_train, meta_fmris, *arrays, Y)

    # updating vocab mask
    vocab_current = vocab_orig[colmask_lab_in_train]

    return *arrays, Y, meta_fmris, vocab_current


def get_rare_labels(Y, meta_fmris, vocab_orig, test_collections):
    # Filtering labels with too few instances in train
    mask_test = meta_fmris["collection_id"].isin(test_collections)

    min_train_label = 10
    colmask_rare_labels_in_train = Y[~mask_test].sum(axis=0) < min_train_label

    number_of_rare_labels = len(vocab_orig) - int(colmask_rare_labels_in_train.sum())
    logger.info(f"Selecting {number_of_rare_labels} labels that are too rare")

    # updating X and Y
    Y = Y[:, colmask_rare_labels_in_train]

    # updating vocab mask
    vocab_current = vocab_orig[colmask_rare_labels_in_train]

    return Y, vocab_current


def filter_correlated_labels(*arrays, Y, meta_fmris, vocab_current):
    # Remove almost fully correlated columns
    labels_low_corr_indices = highly_corr_cols_np(Y, vocab_current, 0.95, True)

    number_of_too_correlated_labels = Y.shape[1] - len(labels_low_corr_indices)
    logger.info(
        f"Removed {number_of_too_correlated_labels} labels that were too correlated"
    )

    Y = Y[:, labels_low_corr_indices]
    vocab_current = vocab_current[labels_low_corr_indices]

    # Update of data and testset mask after highly correlated labels removal
    mask_has_low_corr_lab = np.sum(Y, axis=1) != 0
    meta_fmris, *arrays, Y = mask_rows(mask_has_low_corr_lab, meta_fmris, *arrays, Y)

    return *arrays, Y, meta_fmris, vocab_current


def experiment_filtering(
    *arrays,
    labels,
    meta_fmris,
    test_collection_ids,
    filter_rare=True,
):
    *arrays, Y, meta_fmris, vocab_current = filter_fmris(
        *arrays,
        labels=labels,
        meta_fmris=meta_fmris,
    )

    # In the very near future, we should remove this rare labels filter
    if filter_rare:
        (*arrays, Y, meta_fmris, vocab_current,) = filter_rare_labels(
            *arrays,
            Y=Y,
            meta_fmris=meta_fmris,
            vocab_current=vocab_current,
            test_collections=test_collection_ids,
        )

    (*arrays, Y, meta_fmris, vocab_current,) = filter_correlated_labels(
        *arrays,
        Y=Y,
        meta_fmris=meta_fmris,
        vocab_current=vocab_current,
    )

    return *arrays, Y, meta_fmris, vocab_current


def train_test_collection_split(
    *arrays, meta_fmris, test_collections, train_collections=None
):
    assert len(arrays[0]) == len(
        meta_fmris
    ), "Arrays and meta_fmris should have the same size"
    mask_test = meta_fmris["collection_id"].isin(test_collections)

    mask_train = (
        meta_fmris["collection_id"].isin(train_collections)
        if train_collections
        else ~mask_test
    )

    arrays_train = mask_rows(mask_train, *arrays)
    arrays_test = mask_rows(mask_test, *arrays)

    return (
        *arrays_train,
        *arrays_test,
    )
