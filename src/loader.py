from pathlib import Path

import joblib
import pandas as pd
from neuroquery import datasets as nq_datasets
from scipy import sparse

from src.constants import CACHE_PATH, DATA_PATH, FMRI_DATA_PATH, FMRI_DECODING_PATH
from src.nnod import experiment_filtering, pad_peaks


class NeuroqueryCoordinatesLoader:
    def __init__(self, data_dir=DATA_PATH, n_jobs=None):
        self.data_dir = Path(nq_datasets.fetch_neuroquery_model(data_dir=data_dir))

        self.n_jobs = n_jobs if n_jobs else joblib.cpu_count()
        self.load_valid_coordinates()

    def load_valid_coordinates(self):
        self.corpus_metadata = pd.read_csv(str(self.data_dir / "corpus_metadata.csv"))

        self.tfidf = sparse.load_npz(str(self.data_dir / "corpus_tfidf.npz"))
        self.coordinates = pd.read_csv(nq_datasets.fetch_peak_coordinates())



def load_neurovault(
    preprocessing_dir=FMRI_DATA_PATH / "preprocessing",
    hash="6618_e46692580e0aa89bac35a072f2776d1f",
):
    hash_dir = preprocessing_dir / hash

    neurovault_normalized_difumo = pd.read_csv(
        preprocessing_dir / "difumo_1024_from_normalized_images.csv"
    ).set_index("image_path")
    neurovault_difumo = pd.read_csv(
        preprocessing_dir / "difumo_1024_from_raw_images.csv"
    ).set_index("image_path")
    neurovault_gaussian = pd.read_csv(
        hash_dir / "difumo_1024_from_normalized_gaussians_from_difumo_peaks.csv"
    ).set_index("image_path")

    neurovault_normalized_difumo = neurovault_normalized_difumo.loc[
        neurovault_gaussian.index
    ]
    neurovault_difumo = neurovault_difumo.loc[
        neurovault_gaussian.index
    ]

    return (
        neurovault_difumo,
        neurovault_normalized_difumo,
        neurovault_gaussian,
    )


NNOD_PREPROCESSING_DIR = FMRI_DECODING_PATH / "scripts" / "preprocessing" / "nnod"
memory = joblib.Memory(CACHE_PATH / "nnod")


@memory.cache()
def load_neurovault_decoding(
    preprocessing_dir=FMRI_DATA_PATH / "preprocessing",
    hash="6618_e46692580e0aa89bac35a072f2776d1f",
    test_collection_ids=None,
    train_collection_ids=None,
    max_number_of_peaks=150,
    dimension=1024,
):
    hash_dir = preprocessing_dir / hash

    peaks = pd.read_csv(hash_dir / "peaks_from_difumo_1024.csv")

    difumo_from_images = pd.read_csv(
        preprocessing_dir / f"difumo_{dimension}_from_raw_images.csv"
    ).set_index("image_path")

    difumo_from_normalized_images = pd.read_csv(
        preprocessing_dir / f"difumo_{dimension}_from_normalized_images.csv"
    ).set_index("image_path")

    difumo_from_normalized_gaussians_from_difumo_peaks = pd.read_csv(
        hash_dir / f"difumo_{dimension}_from_normalized_gaussians_from_difumo_peaks.csv"
    ).set_index("image_path")

    meta_fmris_path = NNOD_PREPROCESSING_DIR / "assets" / "meta" / "fmris_meta.csv"
    meta_fmris = (
        pd.read_csv(meta_fmris_path, index_col=0)
        .loc[lambda df: df.kept]
        .set_index("absolute_path")
    )

    labels_path = NNOD_PREPROCESSING_DIR / "labels" / "a7_label_all_syn_hyp.csv"
    labels = (
        pd.read_csv(labels_path, index_col=0)
        .assign(absolute_path=meta_fmris.index)
        .set_index("absolute_path")
    )

    if train_collection_ids is not None and test_collection_ids is not None:
        raise AttributeError(
            "Specify either `train_collection_ids` or `test_collection_ids`."
        )

    if train_collection_ids is not None:
        test_collection_ids = [
            collection_id
            for collection_id in meta_fmris.collection_id.unique()
            if collection_id not in train_collection_ids
        ]

    images_paths = list(
        set(difumo_from_normalized_gaussians_from_difumo_peaks.index)
        .intersection(set(peaks.image_path.unique()))
    )
    difumo_from_normalized_gaussians_from_difumo_peaks = (
        difumo_from_normalized_gaussians_from_difumo_peaks.loc[images_paths]
    )
    difumo_from_normalized_images = difumo_from_normalized_images.loc[
        images_paths
    ]
    difumo_from_images = difumo_from_images.loc[images_paths]
    peaks = peaks.set_index("image_path").loc[images_paths].reset_index()

    images_paths, padded_peaks, masks = pad_peaks(
        peaks, max_number_of_peaks=max_number_of_peaks
    )

    gaussian = difumo_from_normalized_gaussians_from_difumo_peaks.loc[
        images_paths
    ].values
    raw_difumo = difumo_from_images.loc[images_paths].values
    difumo = difumo_from_normalized_images.loc[images_paths].values
    meta_fmris = meta_fmris.loc[images_paths]
    labels = labels.loc[images_paths]

    # Filtering
    (
        raw_difumo,
        difumo,
        gaussian,
        padded_peaks,
        masks,
        Y,
        meta_fmris,
        vocab_current,
    ) = experiment_filtering(
        raw_difumo,
        difumo,
        gaussian,
        padded_peaks,
        masks,
        labels=labels,
        meta_fmris=meta_fmris,
        test_collection_ids=test_collection_ids,
        filter_rare=False,
    )

    return (
        raw_difumo,
        difumo,
        gaussian,
        Y,
        padded_peaks,
        masks,
        meta_fmris,
        vocab_current,
    )
